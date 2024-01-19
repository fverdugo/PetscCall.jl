
function init(;args=String[],file="",help="",finalize_atexit=true)
    if !MPI.Initialized()
        MPI.Init()
    end
    if finalize_atexit
        atexit(finalize)
    end
    finalize()
    new_args = ["PETSC"]
    append!(new_args,args)
    @check_error_code PetscInitializeNoPointers(length(new_args),new_args,file,help)
    nothing
end

function initialized()
    flag = Ref{PetscBool}()
    @check_error_code PetscInitialized(flag)
    flag[] == PETSC_TRUE
end

function finalize()
    if initialized()
        @check_error_code PetscFinalize()
    end
    nothing
end

function with(f;kwargs...)
    init(;kwargs...)
    out = f()
    finalize()
    out
end

function VecCreateSeqWithArray_args(v::AbstractVector)
    T = Vector{PetscScalar}
    w = convert(T,v)
    VecCreateSeqWithArray_args(w)
end

function VecCreateSeqWithArray_args(v::Vector{PetscScalar},block_size=1)
    n = length(v)
    (MPI.COMM_SELF,block_size,n,v)
end

function VecCreateSeqWithArray_args!(args,w::AbstractVector)
    v = args[end]
    if v !== w
        v .= w
    end
    nothing
end

function VecCreateSeqWithArray_args_reversed!(w::AbstractVector,args)
    v = args[end]
    if v !== w
        w .= v
    end
    nothing
end

function MatCreateSeqAIJWithArrays_args(A::AbstractMatrix)
  m, n = size(A)
  i = [PetscInt(n*(i-1)) for i=1:m+1]
  j = [PetscInt(j-1) for i=1:m for j=1:n]
  v = [ PetscScalar(A[i,j]) for i=1:m for j=1:n]
  (MPI.COMM_SELF,m,n,i,j,v)
end

function MatCreateSeqAIJWithArrays_args(A::SplitMatrix)
    MatCreateSeqAIJWithArrays_args(A.blocks.own_own)
end

function MatCreateSeqAIJWithArrays_args(A::AbstractSparseMatrix)
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  csr = convert(Tm,A)
  MatCreateSeqAIJWithArrays_args(csr)
end

function MatCreateSeqAIJWithArrays_args(csr::SparseMatrixCSR{0,PetscScalar,PetscInt})
  m, n = size(csr); i = csr.rowptr; j = csr.colval; v = csr.nzval
  (MPI.COMM_SELF,m,n,i,j,v)
end

function ksp_handles()
    ksp = Ref{KSP}()
    mat = Ref{Mat}()
    vec_b = Ref{Vec}()
    vec_x = Ref{Vec}()
    (ksp,mat,vec_b,vec_x)
end

function ksp_destroy_handles!(handles)
    (ksp,mat,vec_b,vec_x) = handles
    @check_error_code KSPDestroy(ksp)
    @check_error_code MatDestroy(mat)
    @check_error_code VecDestroy(vec_b)
    @check_error_code VecDestroy(vec_x)
    nothing
end

function default_ksp_low_level_setup(ksp)
    @check_error_code KSPSetFromOptions(ksp[])
    nothing
end

function default_ksp_low_level_postpro(ksp)
    niters_ref = Ref{PetscInt}()
    @check_error_code KSPGetIterationNumber(ksp[],niters_ref)
    niters = niters_ref[]
    (;niters)
end

## KSP for sequential matrices (default for all matrix types)

mutable struct KspSeqSetup{A,B,C,D}
    handles::A
    ownership::B
    low_level_setup::C
    low_level_postpro::D
end

function ksp_setup(x,A,b;
    handles = ksp_handles(),
    low_level_setup = default_ksp_low_level_setup,
    low_level_postpro = default_ksp_low_level_postpro,
    )
    (ksp,mat,vec_b,vec_x) = handles
    args_A = MatCreateSeqAIJWithArrays_args(A)
    args_b = VecCreateSeqWithArray_args(copy(b))
    args_x = VecCreateSeqWithArray_args(copy(x))
    @check_error_code MatCreateSeqAIJWithArrays(args_A...,mat)
    @check_error_code MatAssemblyBegin(mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code MatAssemblyEnd(mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code VecCreateSeqWithArray(args_b...,vec_b)
    @check_error_code VecCreateSeqWithArray(args_x...,vec_x)
    @check_error_code KSPCreate(MPI.COMM_SELF,ksp)
    @check_error_code KSPSetOperators(ksp[],mat[],mat[])
    low_level_setup(ksp)
    @check_error_code KSPSetUp(ksp[])
    ownership = (args_A,args_b,args_x)
    setup = KspSeqSetup(handles,ownership,low_level_setup,low_level_postpro)
    setup
end

function ksp_solve!(x,setup,b)
    (args_A,args_b,args_x) = setup.ownership 
    (ksp,mat,vec_b,vec_x) = setup.handles
    VecCreateSeqWithArray_args!(args_b,b)
    @check_error_code KSPSolve(ksp[],vec_b[],vec_x[])
    VecCreateSeqWithArray_args_reversed!(x,args_x)
    results = setup.low_level_postpro(ksp)
    results
end

function ksp_setup!(setup,A)
    (args_A,args_b,args_x) = setup.ownership 
    (ksp,mat,vec_b,vec_x) = setup.handles
    @check_error_code MatDestroy(mat)
    args_A = MatCreateSeqAIJWithArrays_args(A)
    @check_error_code MatCreateSeqAIJWithArrays(args_A...,mat)
    @check_error_code MatAssemblyBegin(mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code MatAssemblyEnd(mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code KSPSetOperators(ksp[],mat[],mat[])
    setup.ownership = (args_A,args_b,args_x)
    setup
end

function ksp_destroy_setup!(setup::KspSeqSetup)
    ksp_destroy_handles!(setup.handles)
end

# KSP for PartitionedArrays (default for any back-end)
# This one is not efficient. It is simple collecting the
# arrays in the main process and applying the sequential version

function ksp_setup(x::PVector,A::PSparseMatrix,b::PVector;kwargs...)
    ksp_setup_parallel_impl(partition(A),x,A,b;kwargs...)
end

struct KspPartitionedSetup{A,B,C}
    setup_in_main::A
    arrays_in_main::B
    caches::C
end

function ksp_setup_parallel_impl(::AbstractArray,x,A,b;kwargs...)
    function local_setup(args...)
        ksp_setup(args...;kwargs...)
    end
    m,n = size(A)
    ranks = linear_indices(partition(A))
    rows_trivial = trivial_partition(ranks,m)
    cols_trivial = trivial_partition(ranks,n)
    A_in_main, cacheA = repartition(A,rows_trivial,cols_trivial;reuse=true) |> fetch
    b_in_main, cacheb = repartition(b,partition(axes(A_in_main,1));reuse=true) |> fetch
    x_in_main, cachex = repartition(x,partition(axes(A_in_main,2));reuse=true) |> fetch
    xp = partition(x_in_main)
    Ap = partition(A_in_main)
    bp = partition(b_in_main)
    setup_in_main = map_main(local_setup,xp,Ap,bp)
    arrays_in_main = (x_in_main,A_in_main,b_in_main)
    caches = (cachex,cacheA,cacheb)
    setup = KspPartitionedSetup(setup_in_main,arrays_in_main,caches)
    setup
end

function ksp_solve!(x,setup::KspPartitionedSetup,b)
    setup_in_main = setup.setup_in_main
    (x_in_main,A_in_main,b_in_main) = setup.arrays_in_main
    (cachex,cacheA,cacheb) = setup.caches
    repartition!(b_in_main,b,cacheb) |> wait
    xp = partition(x_in_main)
    bp = partition(b_in_main)
    results_in_main = map_main(ksp_solve!,xp,setup_in_main,bp)
    repartition!(x,x_in_main,cachex;reversed=true) |> wait
    PartitionedArrays.getany(multicast(results_in_main))
end

function ksp_setup!(setup::KspPartitionedSetup,A)
    setup_in_main = setup.setup_in_main
    (x_in_main,A_in_main,b_in_main) = setup.arrays_in_main
    (cachex,cacheA,cacheb) = setup.caches
    repartition!(A_in_main,A,cacheA) |> wait
    map_main(ksp_setup!,setup_in_main,partition(A_in_main))
    setup
end

function ksp_destroy_setup!(setup::KspPartitionedSetup)
    map_main(ksp_destroy_setup!,setup.setup_in_main)
    nothing
end

## KSP for PartitionedArrays (MPI back-end)

function MatCreateMPIAIJWithSplitArrays_args(a::PSparseMatrix)
    @assert a.assembled
    @assert isa(partition(a),MPIArray)
    # TODO not asserted assumptions:
    # Assumes that global ids are ordered and split format
    rows, cols = axes(a)
    values = partition(a)
    comm = values.comm
    M = length(rows)
    N = length(cols)
    function setup(a,rows,cols)
        Tm  = SparseMatrixCSR{0,PetscScalar,PetscInt}
        own_own = convert(Tm,a.blocks.own_own)
        own_ghost = convert(Tm,a.blocks.own_ghost)
        i = own_own.rowptr; j = own_own.colval; v = own_own.nzval
        oi = own_ghost.rowptr; oj = own_ghost.colval; ov = own_ghost.nzval
        if a.blocks.own_ghost === own_ghost
            oj = copy(oj)
        end
        u = PetscInt(1)
        oj .+= u
        map_ghost_to_global!(oj,cols)
        oj .-= u
        m = own_length(rows)
        n = own_length(cols)
        (comm,m,n,M,N,i,j,v,oi,oj,ov)
    end
    args = map(setup,partition(a),partition(rows),partition(cols))
    args.item
end

function VecCreateMPIWithArray_args(v::PVector,block_size=1)
    @assert isa(partition(v),MPIArray)
    # TODO not asserted assumptions:
    # Assumes that global ids are ordered and that the vector is assembled
    rows = axes(v,1)
    values = partition(v)
    comm = values.comm
    N = length(rows)
    function setup(v_own)
        n = length(v_own)
        T = Vector{PetscScalar}
        array = convert(T,v_own)
        (comm,block_size,n,N,array)
    end
    args = map(setup,own_values(v))
    args.item
end

function VecCreateMPIWithArray_args!(args,w::PVector)
    v = args[end]
    map(own_values(w)) do w
        if v !== w
            v .= w
        end
    end
    nothing
end

function VecCreateMPIWithArray_args_reversed!(w::PVector,args)
    v = args[end]
    map(own_values(w)) do w
        if v !== w
            w .= v
        end
    end
    nothing
end

function ksp_setup_parallel_impl(::MPIArray,x,A,b;
    handles = ksp_handles(),
    low_level_setup = default_ksp_low_level_setup,
    low_level_postpro = default_ksp_low_level_postpro,
    )

    (ksp,mat,vec_b,vec_x) = handles
    args_A = MatCreateMPIAIJWithSplitArrays_args(A)
    args_b = VecCreateMPIWithArray_args(copy(b))
    args_x = VecCreateMPIWithArray_args(copy(x))
    @check_error_code MatCreateMPIAIJWithSplitArrays(args_A...,mat)
    @check_error_code MatAssemblyBegin(mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code MatAssemblyEnd(mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code VecCreateMPIWithArray(args_b...,vec_b)
    @check_error_code VecCreateMPIWithArray(args_x...,vec_x)
    @check_error_code KSPCreate(MPI.COMM_SELF,ksp)
    @check_error_code KSPSetOperators(ksp[],mat[],mat[])
    low_level_setup(ksp)
    @check_error_code KSPSetUp(ksp[])
    ownership = (args_A,args_b,args_x)
    setup = KspMPISetup(handles,ownership,low_level_setup,low_level_postpro)
    setup
end

mutable struct KspMPISetup{A,B,C,D}
    handles::A
    ownership::B
    low_level_setup::C
    low_level_postpro::D
end

function ksp_solve!(x,setup::KspMPISetup,b)
    (args_A,args_b,args_x) = setup.ownership 
    (ksp,mat,vec_b,vec_x) = setup.handles
    VecCreateMPIWithArray_args!(args_b,b)
    @check_error_code KSPSolve(ksp[],vec_b[],vec_x[])
    VecCreateMPIWithArray_args_reversed!(x,args_x)
    results = setup.low_level_postpro(ksp)
    results
end

function ksp_setup!(setup::KspMPISetup,A)
    (args_A,args_b,args_x) = setup.ownership 
    (ksp,mat,vec_b,vec_x) = setup.handles
    @check_error_code MatDestroy(mat)
    args_A = MatCreateMPIAIJWithSplitArrays_args(A)
    @check_error_code MatCreateMPIAIJWithSplitArrays(args_A...,mat)
    @check_error_code MatAssemblyBegin(mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code MatAssemblyEnd(mat[],PETSC.MAT_FINAL_ASSEMBLY)
    @check_error_code KSPSetOperators(ksp[],mat[],mat[])
    setup.ownership = (args_A,args_b,args_x)
    setup
end

function ksp_destroy_setup!(setup::KspMPISetup)
    ksp_destroy_handles!(setup.handles)
end


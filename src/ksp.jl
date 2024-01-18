
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
    (_,_,_,v) = args
    if v !== w
        v .= w
    end
    nothing
end

function VecCreateSeqWithArray_args_reversed!(w::AbstractVector,args)
    (_,_,_,v) = args
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

function MatCreateSeqAIJWithArrays_args(A::AbstractSparseMatrix)
  Tm = SparseMatrixCSR{0,PetscScalar,PetscInt}
  csr = convert(Tm,A)
  MatCreateSeqAIJWithArrays_args(csr)
end

function MatCreateSeqAIJWithArrays_args(csr::SparseMatrixCSR{0,PetscScalar,PetscInt})
  m, n = size(csr); i = csr.rowptr; j = csr.colval; v = csr.nzval
  (MPI.COMM_SELF,m,n,i,j,v)
end

function ksp_create_handles()
    ksp = Ref{KSP}()
    mat = Ref{Mat}()
    vec_b = Ref{Vec}()
    vec_x = Ref{Vec}()
    (ksp,mat,vec_b,vec_x)
end

function ksp_destroy_handles!(handlers)
    (ksp,mat,vec_b,vec_x) = handlers
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

function ksp_setup(x,A,b,handlers;
    low_level_setup = default_ksp_low_level_setup,
    low_level_postpro = default_ksp_low_level_postpro,
    )
    (ksp,mat,vec_b,vec_x) = handlers
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
    setup = (handlers,ownership,low_level_postpro)
    setup
end

function ksp_setup!(setup,A)
    (handlers,ownership,low_level_postpro) = setup
    (args_A,args_b,args_x) = ownership 
    (ksp,mat,vec_b,vec_x) = handlers
    # TODO
end

function ksp_solve!(x,b,setup)
    (handlers,ownership,low_level_postpro) = setup
    (args_A,args_b,args_x) = ownership 
    (ksp,mat,vec_b,vec_x) = handlers
    VecCreateSeqWithArray_args!(args_b,b)
    @check_error_code KSPSolve(ksp[],vec_b[],vec_x[])
    VecCreateSeqWithArray_args_reversed!(x,args_x)
    results = low_level_postpro(ksp)
    results
end


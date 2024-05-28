module Defs

using PartitionedArrays
using PetscCall
using LinearAlgebra
using Test
using SparseArrays

function spmv_petsc!(b,A,x)
    # Convert the input to petsc objects
    mat = Ref{PetscCall.Mat}()
    vec_b = Ref{PetscCall.Vec}()
    vec_x = Ref{PetscCall.Vec}()
    parts = linear_indices(partition(x))
    petsc_comm = PetscCall.setup_petsc_comm(parts)
    args_A = PetscCall.MatCreateMPIAIJWithSplitArrays_args(A,petsc_comm)
    args_b = PetscCall.VecCreateMPIWithArray_args(copy(b),petsc_comm)
    args_x = PetscCall.VecCreateMPIWithArray_args(copy(x),petsc_comm)
    ownership = (args_A,args_b,args_x)
    PetscCall.@check_error_code PetscCall.MatCreateMPIAIJWithSplitArrays(args_A...,mat)
    PetscCall.@check_error_code PetscCall.MatAssemblyBegin(mat[],PetscCall.MAT_FINAL_ASSEMBLY)
    PetscCall.@check_error_code PetscCall.MatAssemblyEnd(mat[],PetscCall.MAT_FINAL_ASSEMBLY)
    PetscCall.@check_error_code PetscCall.VecCreateMPIWithArray(args_b...,vec_b)
    PetscCall.@check_error_code PetscCall.VecCreateMPIWithArray(args_x...,vec_x)
    # This line does the actual product
    PetscCall.@check_error_code PetscCall.MatMult(mat[],vec_x[],vec_b[])
    # Move the result back to julia
    PetscCall.VecCreateMPIWithArray_args_reversed!(b,args_b)
    # Cleanup
    GC.@preserve ownership PetscCall.@check_error_code PetscCall.MatDestroy(mat)
    GC.@preserve ownership PetscCall.@check_error_code PetscCall.VecDestroy(vec_b)
    GC.@preserve ownership PetscCall.@check_error_code PetscCall.VecDestroy(vec_x)
    b
end

function test_spmm_petsc(A,B)
    parts = linear_indices(partition(A))
    petsc_comm = PetscCall.setup_petsc_comm(parts)
    C1, cacheC = spmm(A,B,reuse=true)
    mat_A = Ref{PetscCall.Mat}()
    mat_B = Ref{PetscCall.Mat}()
    mat_C = Ref{PetscCall.Mat}()
    args_A = PetscCall.MatCreateMPIAIJWithSplitArrays_args(A,petsc_comm)
    args_B = PetscCall.MatCreateMPIAIJWithSplitArrays_args(B,petsc_comm)
    ownership = (args_A,args_B)
    PetscCall.@check_error_code PetscCall.MatCreateMPIAIJWithSplitArrays(args_A...,mat_A)
    PetscCall.@check_error_code PetscCall.MatCreateMPIAIJWithSplitArrays(args_B...,mat_B)
    PetscCall.@check_error_code PetscCall.MatProductCreate(mat_A[],mat_B[],C_NULL,mat_C)
    PetscCall.@check_error_code PetscCall.MatProductSetType(mat_C[],PetscCall.MATPRODUCT_AB)
    PetscCall.@check_error_code PetscCall.MatProductSetFromOptions(mat_C[])
    PetscCall.@check_error_code PetscCall.MatProductSymbolic(mat_C[])
    PetscCall.@check_error_code PetscCall.MatProductNumeric(mat_C[])
    PetscCall.@check_error_code PetscCall.MatProductReplaceMats(mat_A[],mat_B[],C_NULL,mat_C[])
    PetscCall.@check_error_code PetscCall.MatProductNumeric(mat_C[])
    PetscCall.@check_error_code PetscCall.MatProductClear(mat_C[])
    GC.@preserve ownership PetscCall.@check_error_code PetscCall.MatDestroy(mat_A)
    GC.@preserve ownership PetscCall.@check_error_code PetscCall.MatDestroy(mat_B)
    GC.@preserve ownership PetscCall.@check_error_code PetscCall.MatDestroy(mat_C)
end

function petsc_coo(petsc_comm,I,J,V,rows,cols)
    m = own_length(rows)
    n = own_length(cols)
    M = PetscCall.PETSC_DECIDE
    N = PetscCall.PETSC_DECIDE
    I .= I .- 1
    J .= J .- 1
    ownership = (I,J,V)
    ncoo = length(I)
    A = Ref{PetscCall.Mat}()
    PetscCall.@check_error_code PetscCall.MatCreate(petsc_comm,A)
    PetscCall.@check_error_code PetscCall.MatSetType(A[],PetscCall.MATMPIAIJ)
    PetscCall.@check_error_code PetscCall.MatSetSizes(A[],m,n,M,N)
    PetscCall.@check_error_code PetscCall.MatSetFromOptions(A[])
    PetscCall.@check_error_code PetscCall.MatSetPreallocationCOO(A[],ncoo,I,J)
    PetscCall.@check_error_code PetscCall.MatSetValuesCOO(A[],V,PetscCall.ADD_VALUES)
    #PetscCall.@check_error_code PetscCall.MatAssemblyBegin(A[],PetscCall.MAT_FINAL_ASSEMBLY)
    #PetscCall.@check_error_code PetscCall.MatAssemblyEnd(A[],PetscCall.MAT_FINAL_ASSEMBLY)
    GC.@preserve ownership PetscCall.@check_error_code PetscCall.MatDestroy(A)
end

function generate_coo(args...)
    A = PartitionedArrays.laplace_matrix(args...)
    row_partition = partition(axes(A,1))
    col_partition = partition(axes(A,2))
    (I,J,V) = map(partition(A),row_partition,col_partition) do myA,rows,cols
        Id,Jd,Vd = findnz(myA.blocks.own_own)
        Io,Jo,Vo = findnz(myA.blocks.own_ghost)
        myI = vcat(map_own_to_global!(Id,rows),map_ghost_to_global!(Io,rows))
        myJ = vcat(map_own_to_global!(Jd,cols),map_ghost_to_global!(Jo,cols))
        myV = vcat(Vd,Vo)
        (myI,myJ,myV)
    end |> tuple_of_arrays
    I,J,V,row_partition,col_partition
end

function main(distribute,params)
    nodes_per_dir = params.nodes_per_dir
    parts_per_dir = params.parts_per_dir
    np = prod(parts_per_dir)
    ranks = LinearIndices((np,)) |> distribute
    A = PartitionedArrays.laplace_matrix(nodes_per_dir,parts_per_dir,ranks)
    rows = partition(axes(A,1))
    cols = partition(axes(A,2))
    x = pones(cols)
    b1 = pzeros(rows)
    b2 = pzeros(rows)
    mul!(b1,A,x)
    if ! PetscCall.initialized()
        PetscCall.init()
    end
    spmv_petsc!(b2,A,x)
    c = b1-b2
    tol = 1.0e-12
    @test norm(b1) > tol
    @test norm(b2) > tol
    @test norm(c)/norm(b1) < tol
    B = 2*A
    test_spmm_petsc(A,B)
    I,J,V,row_partition,col_partition = generate_coo(nodes_per_dir,parts_per_dir,ranks)
    petsc_comm = PetscCall.setup_petsc_comm(ranks)
    map(I,J,V,row_partition,col_partition) do args...
        petsc_coo(petsc_comm,args...)
    end
end

end #module

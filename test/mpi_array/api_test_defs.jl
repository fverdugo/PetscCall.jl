module Defs

using PartitionedArrays
using PetscCall
using LinearAlgebra
using Test

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
end

end #module

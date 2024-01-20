
using SparseArrays
using Test
using PETSC
using PartitionedArrays

function ksp_tests(distribute,np)

    parts = distribute(LinearIndices((np,)))

    n = 7
    row_partition = uniform_partition(parts,np,n)
    col_partition = row_partition

    if np == 1
        I,J,V = map(parts) do part
            I = [1,2,3,1,3,4,5,5,4,5,6,7,6]
            J = [1,2,3,5,6,4,5,3,6,7,6,7,4]
            V = [9,9,9,1,1,9,9,1,1,1,9,9,1]
            I,J,Float64.(V)
        end |> tuple_of_arrays
    else
        I,J,V = map(parts) do part
            if part == 1
                I = [1,2,3,1,3]
                J = [1,2,3,5,6]
                V = [9,9,9,1,1]
            elseif part == 2
                I = [4,5,5,4,5]
                J = [4,5,3,6,7]
                V = [9,9,9,1,1]
            elseif part == 3
                I = [6,7,6]
                J = [6,7,4]
                V = [9,9,1]
            end
            I,J,Float64.(V)
        end |> tuple_of_arrays

    end

    A = psparse(I,J,V,row_partition,col_partition) |> fetch
    x = pones(partition(axes(A,2)))
    b = A*x

    options = "-ksp_type gmres -ksp_converged_reason -ksp_error_if_not_converged true -pc_type jacobi -ksp_rtol 1.0e-12"
    args = split(options)
    comm = PETSC.getcomm(parts)
    petsc_comm = PETSC.init(;comm,args)

    x2 = similar(x); x2 .= 0
    setup = PETSC.ksp_setup(x2,A,b,petsc_comm)
    results = PETSC.ksp_solve!(x2,setup,b)
    @test x ≈ x2

    b = 2*b
    results = PETSC.ksp_solve!(x2,setup,b)
    @test 2*x ≈ x2

    PETSC.ksp_setup!(setup,A)
    results = PETSC.ksp_solve!(x2,setup,b)
    @test 2*x ≈ x2

    PETSC.ksp_destroy_setup!(setup)
end



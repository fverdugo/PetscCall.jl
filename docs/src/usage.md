
# Usage

## Sequential example

    using SparseArrays
    using Test
    using PETSC
    using LinearAlgebra
    
    # Create a spare matrix and a vector in Julia
    I = [1,1,2,2,2,3,3,3,4,4]
    J = [1,2,1,2,3,2,3,4,3,4]
    V = [4,-2,-1,6,-2,-1,6,-2,-1,4]
    m = 4
    n = 4
    A = sparse(I,J,V,m,n)
    x = ones(m)
    b = A*x
    
    # PETSC options
    options = "-ksp_type gmres -ksp_monitor -pc_type ilu"
    PETSC.init(args=split(options))
    
    # Say, we want to solve A*x=b
    x2 = similar(x); x2 .= 0
    setup = PETSC.ksp_setup(x2,A,b)
    results = PETSC.ksp_solve!(x2,setup,b)
    @test x ≈ x2
    
    # Info about the solution process
    @show results
    
    # Now with the same matrix, but a different rhs
    b = 2*b
    results = PETSC.ksp_solve!(x2,setup,b)
    @test 2*x ≈ x2
    
    # Now with a different matrix, but reusing as much as possible
    # from the previous solve.
    A = 2*A
    PETSC.ksp_setup!(setup,A)
    results = PETSC.ksp_solve!(x2,setup,b)
    @test x ≈ x2
    
    # The user needs to explicitly destroy
    # the setup object. This cannot be hidden in
    # Julia finalizers since destructors in petsc are
    # collective operations (in parallel runs).
    # Julia finalizers do not guarantee this.
    PETSC.ksp_destroy_setup!(setup)
    
    # The setup object cannot be used anymore.
    # This now would be provably a code dump:
    # PETSC.ksp_solve!(x2,setup,b)

## Parallel example

First write the parallel code in a function in a file `demo.jl`.

    # File demo.jl
    using SparseArrays
    using Test
    using PETSC
    using PartitionedArrays
    
    function ksp_tests(distribute)
    
        # Create the matrix and vector with PartitionedArrays.jl
        parts = distribute(LinearIndices((np,)))
        n = 7
        np = 3
        row_partition = uniform_partition(parts,np,n)
        col_partition = row_partition
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
        A = psparse(I,J,V,row_partition,col_partition) |> fetch
        x = pones(partition(axes(A,2)))
        b = A*x
    
        # Now solve the system with petsc
        options = "-ksp_type gmres -pc_type jacobi -ksp_rtol 1.0e-12"
        PETSC.init(;args = split(options))
    
        x2 = similar(x); x2 .= 0
        setup = PETSC.ksp_setup(x2,A,b)
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

Then, test your code with the debug back-end of PartitionedArrays.jl

    include("demo.jl")
    with_debug(ksp_tests)

If it works, you can move the MPI back-end. Create your driver in file `driver.jl`.

    # File driver.jl
    using PartitionedArrays
    using MPI; MPI.Init()
    include("demo.jl")
    with_mpi(ksp_tests)

You can launch `driver.jl` with MPI.

    using MPI
    mpiexec(cmd->run(`$cmd -np 3 julia --project=. driver.jl`))


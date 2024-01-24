module PetscCallTest

using PetscCall
using Test

@testset "KSP" begin
    @testset "Sequential" begin include("ksp_test.jl") end
    @testset "PartitionedArrays: DebugArray" begin include("debug_array/ksp_test.jl") end
    @testset "PartitionedArrays: MPIArray" begin include("mpi_array/ksp_test.jl") end
end

end

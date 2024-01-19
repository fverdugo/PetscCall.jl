module KspTestsMPIArray

using PartitionedArrays
using MPI; MPI.Init()

include("../ksp_test_parallel.jl")

np = MPI.Comm_size(MPI.COMM_WORLD)
with_mpi(x->ksp_tests(x,np))

end # module

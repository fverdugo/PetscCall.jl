module KspTests

using MPI
using Test

if MPI.MPI_LIBRARY == "OpenMPI" || (isdefined(MPI, :OpenMPI) && MPI.MPI_LIBRARY == MPI.OpenMPI)
    mpiexec_cmd = `$(mpiexec()) --oversubscribe`
else
    mpiexec_cmd = mpiexec()
end

repodir = normpath(joinpath(@__DIR__,"..",".."))

code = quote
    using MPI; MPI.Init()
    using PartitionedArrays
    include(joinpath($repodir,"test","ksp_test_parallel.jl"))
    np = MPI.Comm_size(MPI.COMM_WORLD)
    with_mpi(x->ksp_tests(x,np))
end

run(`$mpiexec_cmd -np 3 $(Base.julia_cmd()) --project=$repodir -e $code`)

end # module


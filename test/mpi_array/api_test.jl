module ApiTests

using Test
using MPI
using PartitionedArrays

repodir = normpath(joinpath(@__DIR__,"..",".."))

defs = joinpath(repodir,"test","mpi_array","api_test_defs.jl")

include(defs)
params = (;nodes_per_dir=(10,10,10),parts_per_dir=(1,1,1))
with_mpi(dist->Defs.main(dist,params))

code = quote
    using MPI; MPI.Init()
    using PartitionedArrays
    include($defs)
    params = (;nodes_per_dir=(10,10,10),parts_per_dir=(2,2,2))
    with_mpi(dist->Defs.main(dist,params))
end
run(`$(mpiexec()) -np 8 $(Base.julia_cmd()) --project=$repodir -e $code`)

code = quote
    using MPI; MPI.Init()
    using PartitionedArrays
    include($defs)
    params = (;nodes_per_dir=(10,10,10),parts_per_dir=(2,4,1))
    with_mpi(dist->Defs.main(dist,params))
end
run(`$(mpiexec()) -np 8 $(Base.julia_cmd()) --project=$repodir -e $code`)

end # module

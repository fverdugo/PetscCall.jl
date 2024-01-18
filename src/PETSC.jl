module PETSC

using Preferences
using Libdl
using MPI
using PartitionedArrays

include("preferences_head.jl")
include("api.jl")
include("preferences_tail.jl")

end # module

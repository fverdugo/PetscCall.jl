module PetscCall

using Preferences
using Libdl
using MPI
using SparseArrays
using SparseMatricesCSR
using PartitionedArrays

include("preferences_head.jl")
include("api.jl")
include("preferences_tail.jl")
include("environment.jl")
include("ksp.jl")

end # module

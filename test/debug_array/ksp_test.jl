module KspTestsDebugArray

using PartitionedArrays

include("../ksp_test_parallel.jl")

np = 1
with_debug(x->ksp_tests(x,np))

np = 3
with_debug(x->ksp_tests(x,np))

end # module

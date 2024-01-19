using MPI
include("ksp_test_driver.jl")
include("run_mpi_driver.jl")
file = joinpath(@__DIR__,"ksp_test_driver.jl")
run_mpi_driver(file;procs=3)

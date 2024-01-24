```@meta
CurrentModule = PetscCall
```

# PetscCall

Documentation for [PetscCall](https://github.com/fverdugo/PetscCall.jl).

## What

The goal of this package is to provide a high-level Julia interface to solvers from the [PETSc](https://petsc.org/) library.
At this moment it wraps linear solvers from the KSP module in PETSc, but the goal is to also
provide nonlinear solvers from the SNES module in PETSc. The package also provides a low-level interface with functions
that are almost 1-to-1 to the corresponding C functions for advanced users. The low level API is mostly taken from [GridapPETSc.jl](https://github.com/gridap/GridapPETSc.jl).

## Features

- Configuration of PETSc installation using [Preferences.jl](https://github.com/JuliaPackaging/Preferences.jl).
- High-level wrapper of KSP solvers.
- Support for sequential runs.
- Support for parallel parallel runs with [PartitionedArrays.jl](https://github.com/fverdugo/PartitionedArrays.jl).
- Commonly used low-level API for KSP solvers.



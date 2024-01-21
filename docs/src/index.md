```@meta
CurrentModule = PETSC
```

# PETSC

Documentation for [PETSC](https://github.com/fverdugo/PETSC.jl).

## What

The goal of this package is to provide a high-level Julia interface to solvers from the PETSc library.
At this moment it wraps linear solvers from the KSP module in PETSc, but the goal is to also
provide nonlinear solvers from the SNES module in PETSc. The package also provides a low-level interface with functions
that are almost 1-to-1 to the corresponding C functions for advanced users.

## Why

The main difference of this package with respect to other Julia bindings to PETSc (e.g., PETSc.jl),
is that our high-level interface is based on pure Julia types.
I.e., the high-level interface only provides new functions, and the inputs and outputs of the
such functions are pure Julia types. For instance, the functions to solve systems of linear equations
take standard Julia (sparse) matrices and vectors. For parallel computations, one can use the pure
Julia parallel sparse matrices and vectors implemented in PartitionedArrays.jl.


# Configuration

This packages uses Preferences.jl for its configuration.

## Use jll binary

To use the petsc installation provided by BinaryBuilder.jl run these commands:

    using PetscCall
    PetscCall.use_petsc_jll()
    # Restart Julia now

You can also specify the integer and scalar types to use:

    using PetscCall
    PetscCall.use_petsc_jll(PetscInt=Int32,PetscScalar=Float32)
    # Restart Julia now

## Use a system binary

To use PETSc installed in your system:

    using PetscCall
    PetscCall.use_system_petsc()
    # Restart Julia now
    
This will look in `LD_LIBRARY_PATH` for a file called `libpetsc.so`.

You can also provide the full path to `libpetsc.so`.

    using PetscCall
    PetscCall.use_system_petsc(;libpetsc_path=path/to/libpetsc.so)
    # Restart Julia now


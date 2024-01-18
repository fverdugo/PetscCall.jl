

function use_system_binary(;libpetsc_path)
    Libdl.dlclose(libpetsc_handle[])
    flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL
    libpetsc_handle[] = Libdl.dlopen(libpetsc_path, flags)

    scalar_type = Ref{PetscDataType}()
    scalar_found = Ref{PetscBool}()
    scalar_size = Ref{Csize_t}()
    scalar_msg = "Could not determine PetscScalar datatype."
    @check_error_code PetscDataTypeFromString("Scalar",scalar_type,scalar_found)
    if scalar_found[] == PETSC_FALSE
        error(scalar_msg)
    end
    @check_error_code PetscDataTypeGetSize(scalar_type[],scalar_size)
    if scalar_type[] == PETSC_DOUBLE &&  scalar_size[] == 8
        PetscScalar = Float64
    elseif scalar_type[] == PETSC_DOUBLE &&  scalar_size[] == 4
        PetscScalar = Float32
    else
        error(scalar_msg)
    end

    int_type = Ref{PetscDataType}()
    int_found = Ref{PetscBool}()
    int_size = Ref{Csize_t}()
    int_msg = "Could not determine PetscInt datatype."
    @check_error_code PetscDataTypeFromString("Int",int_type,int_found)
    if int_found[] == PETSC_FALSE
        error(int_msg)
    end
    @check_error_code PetscDataTypeGetSize(int_type[],int_size)
    if int_type[] in (PETSC_INT, PETSC_DATATYPE_UNKNOWN) &&  int_size[] == 8
        PetscInt = Int64
    elseif int_type[] in (PETSC_INT, PETSC_DATATYPE_UNKNOWN) &&  int_size[] == 4
        PetscInt = Int32
    else
        error(int_msg)
    end

    libpetsc_provider = "system"
    @set_preferences!(
                      "libpetsc_provider"=>libpetsc_provider,
                      "libpetsc_path"=>libpetsc_path,
                      "PetscScalar"=>string(PetscScalar),
                      "PetscInt"=>string(PetscInt),
                     )
    msg = """
    PETSC preferences changed!
    The new preferences are:
        libpetsc_provider = $(libpetsc_provider)
        libpetsc_path = $(libpetsc_path)
        PetscScalar = $(string(PetscScalar))
        PetscInt = $(string(PetscInt))
    Restart Julia for this changes to take effect. Otherwise, expect undefined behaviour.
    """
    @info msg
    nothing
end

function use_jll_binary(;PetscScalar=Float64,PetscInt=Int64)
    libpetsc_provider = "PETSc_jll"
    @set_preferences!(
        "libpetsc_provider"=>libpetsc_provider,
        "PetscScalar"=>string(PetscScalar),
        "PetscInt"=>string(PetscInt),
       )
    msg = """
    PETSC preferences changed!
    The new preferences are:
        libpetsc_provider = $(libpetsc_provider)
        PetscScalar = $(string(PetscScalar))
        PetscInt = $(string(PetscInt))
    Restart Julia for this changes to take effect. Otherwise, expect undefined behaviour.
    """
    @info msg
    nothing
end


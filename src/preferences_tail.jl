

function use_system_petsc(;libpetsc_path=nothing)

    function findindir(route)
        files = readdir(route,join=false)
        for file in files
            if file == "libpetsc.so"
                libpetsc_path = joinpath(route,file)
                return libpetsc_path
            end
        end
        nothing
    end

    if libpetsc_path === nothing
        if haskey(ENV,"LD_LIBRARY_PATH") && ENV["LD_LIBRARY_PATH"] != ""
            routes = split(ENV["LD_LIBRARY_PATH"],':')
            for route in routes
                libpetsc_path = findindir(route)
                if libpetsc_path !== nothing
                    @info "Petsc installation found in the system at $(libpetsc_path)."
                    break
                end
            end
        end
        msg = """
        Unable to find a Petsc installation in the system.

        We looked for the library file libpetsc.so in the folders in LD_LIBRARY_PATH.

        You can also manualy specify the route to the instalation you want to use
        with the key-word argument libpetsc_path.

        Example
        =======
        
        julia> using PETSC
        julia> using PETSC.use_system_petsc(;libpetsc_path="path/to/libpetsc.so")
        """
        error(msg)
    end

    flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL
    old_libpetsc_handle = libpetsc_handle[]
    new_libpetsc_handle = Libdl.dlopen(libpetsc_path,flags,throw_error=true)
    try 
        libpetsc_handle[] = new_libpetsc_handle
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
    finally
        Libdl.dlclose(new_libpetsc_handle)
        libpetsc_handle[] = old_libpetsc_handle
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
    Restart Julia for these changes to take effect.
    """
    @info msg
    nothing
end

function use_petsc_jll(;PetscScalar=Float64,PetscInt=Int64)
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
    Restart Julia for these changes to take effect.
    """
    @info msg
    nothing
end


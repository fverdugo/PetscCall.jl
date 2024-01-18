
const preferences = (;
    libpetsc_provider = @load_preference("libpetsc_provider","PETSc_jll"),
    PetscScalar = @load_preference("PetscScalar","Float64"),
    PetscInt = @load_preference("PetscInt","Int64"),
    libpetsc_path = @load_preference("libpetsc_path")
   )

@static if preferences.PetscInt == "Int64"
    const PetscInt = Int64
elseif preferences.PetscInt == "Int32"
    const PetscInt = Int32
else
    error("Case not implemented")
end

@static if preferences.PetscScalar == "Float64"
    const PetscScalar = Float64
    const PetscReal = Float64
elseif preferences.PetscScalar == "Float32"
    const PetscScalar = Float32
    const PetscReal = Float32
elseif preferences.PetscScalar == "ComplexF64"
    const PetscScalar = Complex{Float64}
    const PetscReal = Float64
elseif preferences.PetscScalar == "ComplexF32"
    const PetscScalar = Complex{Float32}
    const PetscReal = Float32
else
    error("Case not implemented")
end

@static if preferences.libpetsc_provider == "PETSc_jll"
    using PETSc_jll
end

function __init__()
    if preferences.libpetsc_provider == "PETSc_jll"
        if preferences.PetscScalar == "Float64" && preferences.PetscInt == "Int64"
            libpetsc_handle[] = PETSc_jll.libpetsc_Float64_Real_Int64_handle
        elseif preferences.PetscScalar == "Float64" && preferences.PetscInt == "Int32"
            libpetsc_handle[] = PETSc_jll.libpetsc_Float64_Real_Int32_handle
        elseif preferences.PetscScalar == "Float32" && preferences.PetscInt == "Int64"
            libpetsc_handle[] = PETSc_jll.libpetsc_Float32_Real_Int64_handle
        elseif preferences.PetscScalar == "Float32" && preferences.PetscInt == "Int32"
            libpetsc_handle[] = PETSc_jll.libpetsc_Float32_Real_Int32_handle
        elseif preferences.PetscScalar == "ComplexF64" && preferences.PetscInt == "Int64"
            libpetsc_handle[] = PETSc_jll.libpetsc_Float64_Complex_Int64_handle
        elseif preferences.PetscScalar == "ComplexF64" && preferences.PetscInt == "Int32"
            libpetsc_handle[] = PETSc_jll.libpetsc_Float64_Complex_Int32_handle
        elseif preferences.PetscScalar == "ComplexF32" && preferences.PetscInt == "Int64"
            libpetsc_handle[] = PETSc_jll.libpetsc_Float32_Complex_Int64_handle
        elseif preferences.PetscScalar == "ComplexF32" && preferences.PetscInt == "Int32"
            libpetsc_handle[] = PETSc_jll.libpetsc_Float32_Complex_Int32_handle
        else
            error("case not implemented")
        end
    else
       flags = Libdl.RTLD_LAZY | Libdl.RTLD_DEEPBIND | Libdl.RTLD_GLOBAL
       libpetsc_handle[] = Libdl.dlopen(preferences.libpetsc_path, flags)
    end
    for (handle,sym) in PRELOADS
        new_handle = Libdl.dlsym(libpetsc_handle[],sym;throw_error=false)
        if new_handle !== nothing
            handle[] = new_handle
        end
    end
end


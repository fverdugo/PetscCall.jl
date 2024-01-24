
"""
    PetscCall.init(:kwargs...)

Document me!
"""
function init(;args=String[],file="",help="",finalize_atexit=true)
    if !MPI.Initialized()
        MPI.Init()
    end
    if finalize_atexit
        atexit(finalize)
    end
    finalize()
    new_args = ["PetscCall"]
    append!(new_args,args)
    @check_error_code PetscInitializeNoPointers(length(new_args),new_args,file,help)
    nothing
end

"""
    PetscCall.initialized()

Document me!
"""
function initialized()
    flag = Ref{PetscBool}()
    @check_error_code PetscInitialized(flag)
    flag[] == PETSC_TRUE
end

"""
    PetscCall.finalize()

Document me!
"""
function finalize()
    if initialized()
        @check_error_code PetscFinalize()
    end
    nothing
end

"""
    PetscCall.with(f;kwargs...)

Document me!
"""
function with(f;kwargs...)
    init(;kwargs...)
    out = f()
    finalize()
    out
end


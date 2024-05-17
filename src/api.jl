const libpetsc_handle = Ref{Ptr{Cvoid}}()
const PRELOADS = Tuple{Ref{Ptr{Cvoid}},Symbol}[]

docs_url = "https://petsc.org/main/manualpages"

# Utils

"""
Julia alias to `PetscErrorCode` C type.

See [PETSc manual]($docs_url/Sys/PetscErrorCode.html).
"""
const PetscErrorCode = Cint

"""
    struct PetscError <: Exception
      code::PetscErrorCode
    end

Custom `Exception` thrown by [`@check_error_code`](@ref).
"""
struct PetscError <: Exception
  code::PetscErrorCode
end
Base.showerror(io::IO, e::PetscError) = print(io, "Petsc returned with error code: $(Int(e.code)) ")

"""
    @check_error_code expr

Check if `expr` returns an error code equal to `zero(PetscErrorCode)`.
If not, throw an instance of [`PetscError`](@ref).
"""
macro check_error_code(expr)
  quote
    code = $(esc(expr))
    if code != zero(PetscErrorCode)
      throw(PetscError(code))
    end
  end
end

macro wrapper(fn,rt,argts,args,url)
  hn = Symbol("$(fn.value)_handle")
  sargs = "$(args)"
  if length(args.args) == 1
    sargs = sargs[1:end-2]*")"
  end
  if isempty(rstrip(url))
    str = """
        PetscCall.$(fn.value)$(sargs)
    """
  else
    str = """
        PetscCall.$(fn.value)$(sargs)

    See [PETSc manual]($docs_url$url).
    """
  end
  expr = quote
    const $hn = Ref(C_NULL)
    push!(PRELOADS,($hn,$fn))
    @doc $str
    @inline function $(fn.value)($(args.args...))
      @boundscheck @assert $(hn)[] != C_NULL "Missing symbol. Open a fresh julia session"
      ccall($(hn)[],$rt,$argts,$(args.args...))
    end
  end
  esc(expr)
end

"""
Julia alias to `PetscBool` C enum.

See [PETSc manual]($docs_url/Sys/PetscBool.html).
"""
@enum PetscBool PETSC_FALSE PETSC_TRUE

"""
Julia alias to `PetscDataType` C enum.

See [PETSc manual]($docs_url/Sys/PetscDataType.html).
"""
@enum PetscDataType begin
  PETSC_DATATYPE_UNKNOWN = 0
  PETSC_DOUBLE = 1
  PETSC_COMPLEX = 2
  PETSC_LONG = 3
  PETSC_SHORT = 4
  PETSC_FLOAT = 5
  PETSC_CHAR = 6
  PETSC_BIT_LOGICAL = 7
  PETSC_ENUM = 8
  PETSC_BOOL = 9
  PETSC___FLOAT128 = 10
  PETSC_OBJECT = 11
  PETSC_FUNCTION = 12
  PETSC_STRING = 13
  PETSC___FP16 = 14
  PETSC_STRUCT = 15
  PETSC_INT = 16
  PETSC_INT64 = 17
end

"""
    PetscDataTypeFromString(name,ptype,found)

See [PETSc manual]($docs_url/Sys/PetscDataTypeFromString.html).
"""
function PetscDataTypeFromString(name,ptype,found)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscDataTypeFromString),
    PetscErrorCode,(Cstring,Ptr{PetscDataType},Ptr{PetscBool}),
    name,ptype,found)
end

"""
    PetscDataTypeGetSize(ptype,size)

See [PETSc manual]($docs_url/Sys/PetscDataTypeGetSize.html).
"""
function PetscDataTypeGetSize(ptype,size)
  ccall(
    Libdl.dlsym(libpetsc_handle[],:PetscDataTypeGetSize),
    PetscErrorCode,(PetscDataType,Ptr{Csize_t}),
    ptype,size)
end

#Petsc init related functions

@wrapper(:PetscInitializeNoArguments,PetscErrorCode,(),(),"/Sys/PetscInitializeNoArguments.html")
@wrapper(:PetscInitializeNoPointers,PetscErrorCode,(Cint,Ptr{Cstring},Cstring,Cstring),(argc,args,filename,help),"")
@wrapper(:PetscFinalize,PetscErrorCode,(),(),"/Sys/PetscFinalize.html")
@wrapper(:PetscFinalized,PetscErrorCode,(Ptr{PetscBool},),(flag,),"/Sys/PetscFinalized.html")
@wrapper(:PetscInitialized,PetscErrorCode,(Ptr{PetscBool},),(flag,),"/Sys/PetscInitialized.html")

# viewer related functions

"""
Julia alias for `PetscViewer` C type.

See [PETSc manual]($docs_url/Viewer/PetscViewer.html).
"""
struct PetscViewer
  ptr::Ptr{Cvoid}
end
PetscViewer() = PetscViewer(Ptr{Cvoid}())
Base.convert(::Type{PetscViewer},p::Ptr{Cvoid}) = PetscViewer(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::PetscViewer) = v.ptr

@wrapper(:PETSC_VIEWER_STDOUT_,PetscViewer,(MPI.Comm,),(comm,),"/Viewer/PETSC_VIEWER_STDOUT_.html")
@wrapper(:PETSC_VIEWER_DRAW_,PetscViewer,(MPI.Comm,),(comm,),"/Viewer/PETSC_VIEWER_DRAW_.html")

"""
    @PETSC_VIEWER_STDOUT_SELF

See [PETSc manual]($docs_url/Viewer/PETSC_VIEWER_STDOUT_SELF.html).
"""
macro PETSC_VIEWER_STDOUT_SELF()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_SELF)
  end
end

"""
    @PETSC_VIEWER_STDOUT_WORLD

See [PETSc manual]($docs_url/Viewer/PETSC_VIEWER_STDOUT_WORLD.html).
"""
macro PETSC_VIEWER_STDOUT_WORLD()
  quote
    PETSC_VIEWER_STDOUT_(MPI.COMM_WORLD)
  end
end

"""
    @PETSC_VIEWER_DRAW_SELF

See [PETSc manual]($docs_url/Viewer/PETSC_VIEWER_DRAW_SELF.html).
"""
macro PETSC_VIEWER_DRAW_SELF()
  quote
    PETSC_VIEWER_DRAW_(MPI.COMM_SELF)
  end
end

"""
    @PETSC_VIEWER_DRAW_WORLD

See [PETSc manual]($docs_url/Viewer/PETSC_VIEWER_DRAW_WORLD.html).
"""
macro PETSC_VIEWER_DRAW_WORLD()
  quote
    PETSC_VIEWER_DRAW_(MPI.COMM_WORLD)
  end
end

# Vector related functions

"""
Julia alias for the `InsertMode` C enum.

See [PETSc manual]($docs_url/Sys/InsertMode.html).
"""
@enum InsertMode begin
  NOT_SET_VALUES
  INSERT_VALUES
  ADD_VALUES
  MAX_VALUES
  MIN_VALUES
  INSERT_ALL_VALUES
  ADD_ALL_VALUES
  INSERT_BC_VALUES
  ADD_BC_VALUES
end

"""
Julia alias for the `VecOption` C enum.

See [PETSc manual]($docs_url/Vec/VecSetOption.html).
"""
@enum VecOption begin
  VEC_IGNORE_OFF_PROC_ENTRIES
  VEC_IGNORE_NEGATIVE_INDICES
  VEC_SUBSET_OFF_PROC_ENTRIES
end

"""
Julia alias for the `NormType` C enum.

See [PETSc manual]($docs_url/Vec/NormType.html).
"""
@enum NormType begin
  NORM_1=0
  NORM_2=1
  NORM_FROBENIUS=2
  NORM_INFINITY=3
  NORM_1_AND_2=4
end

"""
Julia alias for the `Vec` C type.

See [PETSc manual]($docs_url/Vec/Vec.html).
"""
struct Vec
  ptr::Ptr{Cvoid}
end
Vec() = Vec(Ptr{Cvoid}())
Base.convert(::Type{Vec},p::Ptr{Cvoid}) = Vec(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::Vec) = v.ptr

@wrapper(:VecCreateSeq,PetscErrorCode,(MPI.Comm,PetscInt,Ptr{Vec}),(comm,n,vec),"/Vec/VecCreateSeq.html")
@wrapper(:VecCreateSeqWithArray,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscScalar},Ptr{Vec}),(comm,bs,n,array,vec),"/Vec/VecCreateSeqWithArray.html")
@wrapper(:VecCreateMPIWithArray,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscScalar},Ptr{Vec}),(comm,bs,n,N,array,vec),"/Vec/VecCreateMPIWithArray.html")
@wrapper(:VecCreateGhost,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{Vec}),(comm,n,N,nghost,ghosts,vv),"/Vec/VecCreateGhost.html")
@wrapper(:VecCreateGhostWithArray,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},Ptr{Vec}),(comm,n,N,nghost,ghosts,array,vv),"/Vec/VecCreateGhostWithArray.html")
@wrapper(:VecCreateMPI,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Vec),(comm,n,N,v),"/Vec/VecCreateMPI.html")
@wrapper(:VecDestroy,PetscErrorCode,(Ptr{Vec},),(vec,),"/Vec/VecDestroy.html")
@wrapper(:VecView,PetscErrorCode,(Vec,PetscViewer),(vec,viewer),"/Vec/VecView.html")
@wrapper(:VecSetValues,PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),(x,ni,ix,y,iora),"/Vec/VecSetValues.html")
@wrapper(:VecGetValues,PetscErrorCode,(Vec,PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),(x,ni,ix,y),"/Vec/VecGetValues.html")
@wrapper(:VecGetArray,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"/Vec/VecGetArray.html")
@wrapper(:VecGetArrayRead,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"/Vec/VecGetArrayRead.html")
@wrapper(:VecGetArrayWrite,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"/Vec/VecGetArrayWrite.html")
@wrapper(:VecRestoreArray,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"/Vec/VecRestoreArray.html")
@wrapper(:VecRestoreArrayRead,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"/Vec/VecRestoreArrayRead.html")
@wrapper(:VecRestoreArrayWrite,PetscErrorCode,(Vec,Ptr{Ptr{PetscScalar}}),(x,a),"/Vec/VecRestoreArrayWrite.html")
@wrapper(:VecGetSize,PetscErrorCode,(Vec,Ptr{PetscInt}),(vec,n),"/Vec/VecGetSize.html")
@wrapper(:VecGetLocalSize,PetscErrorCode,(Vec,Ptr{PetscInt}),(vec,n),"/Vec/VecGetLocalSize.html")
@wrapper(:VecAssemblyBegin,PetscErrorCode,(Vec,),(vec,),"/Vec/VecAssemblyBegin.html")
@wrapper(:VecAssemblyEnd,PetscErrorCode,(Vec,),(vec,),"/Vec/VecAssemblyEnd.html")
@wrapper(:VecPlaceArray,PetscErrorCode,(Vec,Ptr{PetscScalar}),(vec,array),"/Vec/VecPlaceArray.html")
@wrapper(:VecResetArray,PetscErrorCode,(Vec,),(vec,),"/Vec/VecResetArray.html")
@wrapper(:VecScale,PetscErrorCode,(Vec,PetscScalar),(x,alpha),"/Vec/VecScale.html")
@wrapper(:VecSet,PetscErrorCode,(Vec,PetscScalar),(x,alpha),"/Vec/VecSet.html")
@wrapper(:VecDuplicate,PetscErrorCode,(Vec,Ptr{Vec}),(v,newv),"/Vec/VecDuplicate.html")
@wrapper(:VecCopy,PetscErrorCode,(Vec,Vec),(x,y),"/Vec/VecCopy.html")
@wrapper(:VecAXPY,PetscErrorCode,(Vec,PetscScalar,Vec),(y,alpha,x),"/Vec/VecAXPY.html")
@wrapper(:VecAYPX,PetscErrorCode,(Vec,PetscScalar,Vec),(y,beta,x),"/Vec/VecAYPX.html")
@wrapper(:VecAXPBY,PetscErrorCode,(Vec,PetscScalar,PetscScalar,Vec),(y,alpha,beta,x),"/Vec/VecAXPBY.html")
@wrapper(:VecSetOption,PetscErrorCode,(Vec,VecOption,PetscBool),(x,op,flg),"/Vec/VecSetOption.html")
@wrapper(:VecNorm,PetscErrorCode,(Vec,NormType,Ptr{PetscReal}),(x,typ,val),"/Vec/VecNorm.html")
@wrapper(:VecGhostGetLocalForm,PetscErrorCode,(Vec,Ptr{Vec}),(g,l),"/Vec/VecGhostGetLocalForm.html")
@wrapper(:VecGhostRestoreLocalForm,PetscErrorCode,(Vec,Ptr{Vec}),(g,l),"/Vec/VecGhostRestoreLocalForm.html")
@wrapper(:VecZeroEntries,PetscErrorCode,(Vec,),(v,),"/Vec/VecZeroEntries.html")

# Matrix related functions

"""
Julia alias for the `MatAssemblyType` C enum.

See [PETSc manual]($docs_url/Mat/MatAssemblyType.html).
"""
@enum MatAssemblyType begin
  MAT_FINAL_ASSEMBLY=0
  MAT_FLUSH_ASSEMBLY=1
end

"""
Julia alias for the `MatDuplicateOption` C enum.

See [PETSc manual]($docs_url/Mat/MatDuplicateOption.html).
"""
@enum MatDuplicateOption begin
  MAT_DO_NOT_COPY_VALUES
  MAT_COPY_VALUES
  MAT_SHARE_NONZERO_PATTERN
end

"""
Julia alias for the `MatReuse` C enum.

See [PETSc manual]($docs_url/Mat/MatReuse.html).
"""
@enum MatReuse begin
  MAT_INITIAL_MATRIX
  MAT_REUSE_MATRIX
  MAT_IGNORE_MATRIX
  MAT_INPLACE_MATRIX
end

"""
Julia alias for the `MatInfoType` C enum.

See [PETSc manual]($docs_url/Mat/MatInfoType.html).
"""
@enum MatInfoType begin
  MAT_LOCAL=1
  MAT_GLOBAL_MAX=2
  MAT_GLOBAL_SUM=3
end

"""
Julia alias for the `MatStructure` C enum.

See [PETSc manual]($docs_url/Mat/MatStructure.html).
"""
@enum MatStructure begin
  DIFFERENT_NONZERO_PATTERN
  SUBSET_NONZERO_PATTERN
  SAME_NONZERO_PATTERN
  UNKNOWN_NONZERO_PATTERN
end

"""
Julia alias to `PetscLogDouble` C type.

See [PETSc manual]($docs_url/Sys/PetscLogDouble.html).
"""
const PetscLogDouble = Cdouble

"""
Julia alias for the `MatInfo` C struct.

See [PETSc manual]($docs_url/Mat/MatInfo.html).
"""
struct MatInfo
  block_size         ::PetscLogDouble
  nz_allocated       ::PetscLogDouble
  nz_used            ::PetscLogDouble
  nz_unneeded        ::PetscLogDouble
  memory             ::PetscLogDouble
  assemblies         ::PetscLogDouble
  mallocs            ::PetscLogDouble
  fill_ratio_given   ::PetscLogDouble
  fill_ratio_needed  ::PetscLogDouble
  factor_mallocs     ::PetscLogDouble
end

"""
Julia constant storing the `PETSC_DEFAULT` value.

See [PETSc manual]($docs_url/Sys/PETSC_DEFAULT.html).
"""
const PETSC_DEFAULT = Cint(-2)

"""
Julia constant storing the `PETSC_DECIDE` value.

See [PETSc manual]($docs_url/Sys/PETSC_DECIDE.html).
"""
const PETSC_DECIDE = Cint(-1)

"""
Julia constant storing the `PETSC_DETERMINE` value.

See [PETSc manual]($docs_url/Sys/PETSC_DETERMINE.html).
"""
const PETSC_DETERMINE = PETSC_DECIDE

"""
Julia alias for `MatType` C type.

See [PETSc manual]($docs_url/Mat/MatType.html).
"""
const MatType = Cstring
const MATSAME            = "same"
const MATMAIJ            = "maij"
const MATSEQMAIJ         = "seqmaij"
const MATMPIMAIJ         = "mpimaij"
const MATKAIJ            = "kaij"
const MATSEQKAIJ         = "seqkaij"
const MATMPIKAIJ         = "mpikaij"
const MATIS              = "is"
const MATAIJ             = "aij"
const MATSEQAIJ          = "seqaij"
const MATMPIAIJ          = "mpiaij"
const MATAIJCRL          = "aijcrl"
const MATSEQAIJCRL       = "seqaijcrl"
const MATMPIAIJCRL       = "mpiaijcrl"
const MATAIJCUSPARSE     = "aijcusparse"
const MATSEQAIJCUSPARSE  = "seqaijcusparse"
const MATMPIAIJCUSPARSE  = "mpiaijcusparse"
const MATAIJKOKKOS       = "aijkokkos"
const MATSEQAIJKOKKOS    = "seqaijkokkos"
const MATMPIAIJKOKKOS    = "mpiaijkokkos"
const MATAIJVIENNACL     = "aijviennacl"
const MATSEQAIJVIENNACL  = "seqaijviennacl"
const MATMPIAIJVIENNACL  = "mpiaijviennacl"
const MATAIJPERM         = "aijperm"
const MATSEQAIJPERM      = "seqaijperm"
const MATMPIAIJPERM      = "mpiaijperm"
const MATAIJSELL         = "aijsell"
const MATSEQAIJSELL      = "seqaijsell"
const MATMPIAIJSELL      = "mpiaijsell"
const MATAIJMKL          = "aijmkl"
const MATSEQAIJMKL       = "seqaijmkl"
const MATMPIAIJMKL       = "mpiaijmkl"
const MATBAIJMKL         = "baijmkl"
const MATSEQBAIJMKL      = "seqbaijmkl"
const MATMPIBAIJMKL      = "mpibaijmkl"
const MATSHELL           = "shell"
const MATDENSE           = "dense"
const MATDENSECUDA       = "densecuda"
const MATSEQDENSE        = "seqdense"
const MATSEQDENSECUDA    = "seqdensecuda"
const MATMPIDENSE        = "mpidense"
const MATMPIDENSECUDA    = "mpidensecuda"
const MATELEMENTAL       = "elemental"
const MATSCALAPACK       = "scalapack"
const MATBAIJ            = "baij"
const MATSEQBAIJ         = "seqbaij"
const MATMPIBAIJ         = "mpibaij"
const MATMPIADJ          = "mpiadj"
const MATSBAIJ           = "sbaij"
const MATSEQSBAIJ        = "seqsbaij"
const MATMPISBAIJ        = "mpisbaij"
const MATMFFD            = "mffd"
const MATNORMAL          = "normal"
const MATNORMALHERMITIAN = "normalh"
const MATLRC             = "lrc"
const MATSCATTER         = "scatter"
const MATBLOCKMAT        = "blockmat"
const MATCOMPOSITE       = "composite"
const MATFFT             = "fft"
const MATFFTW            = "fftw"
const MATSEQCUFFT        = "seqcufft"
const MATTRANSPOSEMAT    = "transpose"
const MATSCHURCOMPLEMENT = "schurcomplement"
const MATPYTHON          = "python"
const MATHYPRE           = "hypre"
const MATHYPRESTRUCT     = "hyprestruct"
const MATHYPRESSTRUCT    = "hypresstruct"
const MATSUBMATRIX       = "submatrix"
const MATLOCALREF        = "localref"
const MATNEST            = "nest"
const MATPREALLOCATOR    = "preallocator"
const MATSELL            = "sell"
const MATSEQSELL         = "seqsell"
const MATMPISELL         = "mpisell"
const MATDUMMY           = "dummy"
const MATLMVM            = "lmvm"
const MATLMVMDFP         = "lmvmdfp"
const MATLMVMBFGS        = "lmvmbfgs"
const MATLMVMSR1         = "lmvmsr1"
const MATLMVMBROYDEN     = "lmvmbroyden"
const MATLMVMBADBROYDEN  = "lmvmbadbroyden"
const MATLMVMSYMBROYDEN  = "lmvmsymbroyden"
const MATLMVMSYMBADBROYDEN = "lmvmsymbadbroyden"
const MATLMVMDIAGBROYDEN   = "lmvmdiagbroyden"
const MATCONSTANTDIAGONAL  = "constantdiagonal"
const MATHARA              = "hara"

const MATSOLVERSUPERLU          = "superlu"
const MATSOLVERSUPERLU_DIST     = "superlu_dist"
const MATSOLVERSTRUMPACK        = "strumpack"
const MATSOLVERUMFPACK          = "umfpack"
const MATSOLVERCHOLMOD          = "cholmod"
const MATSOLVERKLU              = "klu"
const MATSOLVERSPARSEELEMENTAL  = "sparseelemental"
const MATSOLVERELEMENTAL        = "elemental"
const MATSOLVERESSL             = "essl"
const MATSOLVERLUSOL            = "lusol"
const MATSOLVERMUMPS            = "mumps"
const MATSOLVERMKL_PARDISO      = "mkl_pardiso"
const MATSOLVERMKL_CPARDISO     = "mkl_cpardiso"
const MATSOLVERPASTIX           = "pastix"
const MATSOLVERMATLAB           = "matlab"
const MATSOLVERPETSC            = "petsc"
const MATSOLVERCUSPARSE         = "cusparse"

"""
Julia alias for the `Mat` C type.

See [PETSc manual]($docs_url/Mat/Mat.html).
"""
struct Mat
  ptr::Ptr{Cvoid}
end
Mat() = Mat(Ptr{Cvoid}())
Base.convert(::Type{Mat},p::Ptr{Cvoid}) = Mat(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::Mat) = v.ptr

@wrapper(:MatCreate,PetscErrorCode,(MPI.Comm,Ptr{Mat}),(comm,mat),"/Mat/MatCreate.html")
@wrapper(:MatCreateAIJ,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{Mat}),(comm,m,n,M,N,d_nz,d_nnz,o_nz,o_nnz,mat),"/Mat/MatCreateAIJ.html")
@wrapper(:MatCreateSeqAIJ,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{Mat}),(comm,m,n,nz,nnz,mat),"/Mat/MatCreateSeqAIJ.html")
@wrapper(:MatCreateSeqAIJWithArrays,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{Mat}),(comm,m,n,i,j,a,mat),"/Mat/MatCreateSeqAIJWithArrays.html")
@wrapper(:MatCreateMPIAIJWithArrays,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{Mat}),(comm,m,n,M,N,i,j,a,mat),"/Mat/MatCreateMPIAIJWithArrays.html")
@wrapper(:MatCreateMPIAIJWithSplitArrays,PetscErrorCode,(MPI.Comm,PetscInt,PetscInt,PetscInt,PetscInt,Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{PetscInt},Ptr{PetscInt},Ptr{PetscScalar},Ptr{Mat}),(comm,m,n,M,N,i,j,a,oi,oj,oa,mat),"/Mat/MatCreateMPIAIJWithSplitArrays.html")
@wrapper(:MatDestroy,PetscErrorCode,(Ptr{Mat},),(A,),"/Mat/MatDestroy.html")
@wrapper(:MatView,PetscErrorCode,(Mat,PetscViewer),(mat,viewer),"/Mat/MatView.html")
@wrapper(:MatSetType,PetscErrorCode,(Mat,MatType),(mat,matype),"/Mat/MatSetType.html")
@wrapper(:MatSetSizes,PetscErrorCode,(Mat,PetscInt,PetscInt,PetscInt,PetscInt),(A,m,n,M,N),"/Mat/MatSetSizes.html")
@wrapper(:MatSetPreallocationCOO,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},Ptr{PetscInt}),(A,ncoo,coo_i,coo_j),"/Mat/MatSetPreallocationCOO.html")
@wrapper(:MatSetValuesCOO,PetscErrorCode,(Mat,Ptr{PetscScalar},InsertMode),(A,coo_v,imode),"/Mat/MatSetValuesCOO.html")
@wrapper(:MatSetValues,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar},InsertMode),(mat,m,idxm,n,idxn,v,addv),"/Mat/MatSetValues.html")
@wrapper(:MatGetValues,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt},Ptr{PetscScalar}),(mat,m,idxm,n,idxn,v),"/Mat/MatGetValues.html")
@wrapper(:MatAssemblyBegin,PetscErrorCode,(Mat,MatAssemblyType),(mat,typ),"/Mat/MatAssemblyBegin.html")
@wrapper(:MatAssemblyEnd,PetscErrorCode,(Mat,MatAssemblyType),(mat,typ),"/Mat/MatAssemblyEnd.html")
@wrapper(:MatGetSize,PetscErrorCode,(Mat,Ptr{PetscInt},Ptr{PetscInt}),(mat,m,n),"/Mat/MatGetSize.html")
@wrapper(:MatEqual,PetscErrorCode,(Mat,Mat,Ptr{PetscBool}),(A,B,flg),"/Mat/MatEqual.html")
@wrapper(:MatMultAdd,PetscErrorCode,(Mat,Vec,Vec,Vec),(mat,v1,v2,v3),"/Mat/MatMultAdd.html")
@wrapper(:MatMult,PetscErrorCode,(Mat,Vec,Vec),(mat,x,y),"/Mat/MatMult.html")
@wrapper(:MatScale,PetscErrorCode,(Mat,PetscScalar),(mat,alpha),"/Mat/MatScale.html")
@wrapper(:MatConvert,PetscErrorCode,(Mat,MatType,MatReuse,Ptr{Mat}),(mat,newtype,reuse,M),"/Mat/MatConvert.html")
@wrapper(:MatGetInfo,PetscErrorCode,(Mat,MatInfoType,Ptr{MatInfo}),(mat,flag,info),"/Mat/MatGetInfo.html")
@wrapper(:MatZeroEntries,PetscErrorCode,(Mat,),(mat,),"/Mat/MatZeroEntries.html")
@wrapper(:MatCopy,PetscErrorCode,(Mat,Mat,MatStructure),(A,B,str),"/Mat/MatCopy.html")
@wrapper(:MatSetBlockSize,PetscErrorCode,(Mat,PetscInt),(mat,bs),"/Mat/MatSetBlockSize.html")
@wrapper(:MatMumpsSetIcntl,PetscErrorCode,(Mat,PetscInt,PetscInt),(mat,icntl,val),"/Mat/MatMumpsSetIcntl.html")
@wrapper(:MatMumpsSetCntl,PetscErrorCode,(Mat,PetscInt,PetscReal),(mat,icntl,val),"/Mat/MatMumpsSetCntl.html")
@wrapper(:MatMPIAIJSetPreallocation,PetscErrorCode,(Mat,PetscInt,Ptr{PetscInt},PetscInt,Ptr{PetscInt}),(B,d_nz,d_nnz,o_nz,o_nnz),"/Mat/MatMPIAIJSetPreallocation.html")

# Matrix products related

"""
Julia alias for the `MatProductType` C enum.

See [PETSc manual]($docs_url/Mat/MatProductType.html).
"""
@enum MatProductType begin
  MATPRODUCT_UNSPECIFIED = 0
  MATPRODUCT_AB
  MATPRODUCT_AtB
  MATPRODUCT_ABt
  MATPRODUCT_PtAP
  MATPRODUCT_RARt
  MATPRODUCT_ABC
end

"""
Julia alias for `MatProductAlgorithm` C type.

See [PETSc manual]($docs_url/Mat/MatProductAlgorithm.html).
"""
const MatProductAlgorithm = Cstring
const MATPRODUCTALGORITHMDEFAULT         = "default"
const MATPRODUCTALGORITHMSORTED          = "sorted"
const MATPRODUCTALGORITHMSCALABLE        = "scalable"
const MATPRODUCTALGORITHMSCALABLEFAST    = "scalable_fast"
const MATPRODUCTALGORITHMHEAP            = "heap"
const MATPRODUCTALGORITHMBHEAP           = "btheap"
const MATPRODUCTALGORITHMLLCONDENSED     = "llcondensed"
const MATPRODUCTALGORITHMROWMERGE        = "rowmerge"
const MATPRODUCTALGORITHMOUTERPRODUCT    = "outerproduct"
const MATPRODUCTALGORITHMATB             = "at*b"
const MATPRODUCTALGORITHMRAP             = "rap"
const MATPRODUCTALGORITHMNONSCALABLE     = "nonscalable"
const MATPRODUCTALGORITHMSEQMPI          = "seqmpi"
const MATPRODUCTALGORITHMBACKEND         = "backend"
const MATPRODUCTALGORITHMOVERLAPPING     = "overlapping"
const MATPRODUCTALGORITHMMERGED          = "merged"
const MATPRODUCTALGORITHMALLATONCE       = "allatonce"
const MATPRODUCTALGORITHMALLATONCEMERGED = "allatonce_merged"
const MATPRODUCTALGORITHMALLGATHERV      = "allgatherv"
const MATPRODUCTALGORITHMCYCLIC          = "cyclic"
const MATPRODUCTALGORITHMHYPRE           = "hypre"

@wrapper(:MatMatMult,PetscErrorCode,(Mat,Mat,MatReuse,PetscReal,Ptr{Mat}),(A,B,scall,fill,C),"/Mat/MatMatMult.html")
@wrapper(:MatProductCreate,PetscErrorCode,(Mat,Mat,Mat,Ptr{Mat}),(A,B,C,D),"/Mat/MatProductCreate.html")
@wrapper(:MatProductSetType,PetscErrorCode,(Mat,MatProductType),(mat,productype),"/Mat/MatProductSetType.html")
@wrapper(:MatProductSymbolic,PetscErrorCode,(Mat,),(mat,),"/Mat/MatProductSymbolic.html")
@wrapper(:MatProductNumeric,PetscErrorCode,(Mat,),(mat,),"/Mat/MatProductNumeric.html")
@wrapper(:MatProductReplaceMats,PetscErrorCode,(Mat,Mat,Mat,Mat),(A,B,C,D),"/Mat/MatProductReplaceMats.html")
@wrapper(:MatProductSetAlgorithm,PetscErrorCode,(Mat,MatProductAlgorithm),(mat,alg),"/Mat/MatProductSetAlgorithm.html")
@wrapper(:MatProductGetAlgorithm,PetscErrorCode,(Mat,Ptr{MatProductAlgorithm}),(mat,alg),"/Mat/MatProductGetAlgorithm.html")
@wrapper(:MatProductSetFill,PetscErrorCode,(Mat,PetscReal),(mat,fill),"/Mat/MatProductSetFill.html")
@wrapper(:MatProductSetFromOptions,PetscErrorCode,(Mat,),(mat,),"/Mat/MatProductSetFromOptions.html")
@wrapper(:MatProductClear,PetscErrorCode,(Mat,),(mat,),"/Mat/MatProductClear.html")

# Null space related

"""
Julia alias for the `MatNullSpace` C type.

See [PETSc manual]($docs_url/Mat/MatNullSpace.html).
"""
struct MatNullSpace
  ptr::Ptr{Cvoid}
end
MatNullSpace() = MatNullSpace(Ptr{Cvoid}())
Base.convert(::Type{MatNullSpace},p::Ptr{Cvoid}) = MatNullSpace(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::MatNullSpace) = v.ptr

@wrapper(:MatSetNearNullSpace,PetscErrorCode,(Mat,MatNullSpace),(mat,nullsp),"/Mat/MatSetNearNullSpace.html")
@wrapper(:MatNullSpaceCreateRigidBody,PetscErrorCode,(Vec,Ptr{MatNullSpace}),(coords,sp),"/Mat/MatNullSpaceCreateRigidBody.html")
@wrapper(:MatNullSpaceCreate,PetscErrorCode,(MPI.Comm,PetscBool,PetscInt,Ptr{Vec},Ptr{MatNullSpace}),(comm,has_cnst,n,vecs,sp),"/Mat/MatNullSpaceCreate.html")
@wrapper(:MatNullSpaceDestroy,PetscErrorCode,(Ptr{MatNullSpace},),(ns,),"/Mat/MatNullSpaceDestroy.html")

# KSP and PC related things

"""
Julia alias for `KSPType` C type.

See [PETSc manual]($docs_url/KSP/KSPType.html).
"""
const KSPType = Cstring
const KSPRICHARDSON = "richardson"
const KSPCHEBYSHEV  = "chebyshev"
const KSPCG         = "cg"
const KSPGROPPCG    = "groppcg"
const KSPPIPECG     = "pipecg"
const KSPPIPECGRR   = "pipecgrr"
const KSPPIPELCG    = "pipelcg"
const KSPPIPEPRCG   = "pipeprcg"
const KSPPIPECG2    = "pipecg2"
const KSPCGNE       = "cgne"
const KSPNASH       = "nash"
const KSPSTCG       = "stcg"
const KSPGLTR       = "gltr"
const KSPFCG        = "fcg"
const KSPPIPEFCG    = "pipefcg"
const KSPGMRES      = "gmres"
const KSPPIPEFGMRES = "pipefgmres"
const KSPFGMRES     = "fgmres"
const KSPLGMRES     = "lgmres"
const KSPDGMRES     = "dgmres"
const KSPPGMRES     = "pgmres"
const KSPTCQMR      = "tcqmr"
const KSPBCGS       = "bcgs"
const KSPIBCGS      = "ibcgs"
const KSPFBCGS      = "fbcgs"
const KSPFBCGSR     = "fbcgsr"
const KSPBCGSL      = "bcgsl"
const KSPPIPEBCGS   = "pipebcgs"
const KSPCGS        = "cgs"
const KSPTFQMR      = "tfqmr"
const KSPCR         = "cr"
const KSPPIPECR     = "pipecr"
const KSPLSQR       = "lsqr"
const KSPPREONLY    = "preonly"
const KSPQCG        = "qcg"
const KSPBICG       = "bicg"
const KSPMINRES     = "minres"
const KSPSYMMLQ     = "symmlq"
const KSPLCD        = "lcd"
const KSPPYTHON     = "python"
const KSPGCR        = "gcr"
const KSPPIPEGCR    = "pipegcr"
const KSPTSIRM      = "tsirm"
const KSPCGLS       = "cgls"
const KSPFETIDP     = "fetidp"
const KSPHPDDM      = "hpddm"

"""
Julia alias for `PCType` C type.

See [PETSc manual]($docs_url/PC/PCType.html).
"""
const PCType = Cstring
const PCNONE            = "none"
const PCJACOBI          = "jacobi"
const PCSOR             = "sor"
const PCLU              = "lu"
const PCSHELL           = "shell"
const PCBJACOBI         = "bjacobi"
const PCMG              = "mg"
const PCEISENSTAT       = "eisenstat"
const PCILU             = "ilu"
const PCICC             = "icc"
const PCASM             = "asm"
const PCGASM            = "gasm"
const PCKSP             = "ksp"
const PCCOMPOSITE       = "composite"
const PCREDUNDANT       = "redundant"
const PCSPAI            = "spai"
const PCNN              = "nn"
const PCCHOLESKY        = "cholesky"
const PCPBJACOBI        = "pbjacobi"
const PCVPBJACOBI       = "vpbjacobi"
const PCMAT             = "mat"
const PCHYPRE           = "hypre"
const PCPARMS           = "parms"
const PCFIELDSPLIT      = "fieldsplit"
const PCTFS             = "tfs"
const PCML              = "ml"
const PCGALERKIN        = "galerkin"
const PCEXOTIC          = "exotic"
const PCCP              = "cp"
const PCBFBT            = "bfbt"
const PCLSC             = "lsc"
const PCPYTHON          = "python"
const PCPFMG            = "pfmg"
const PCSYSPFMG         = "syspfmg"
const PCREDISTRIBUTE    = "redistribute"
const PCSVD             = "svd"
const PCGAMG            = "gamg"
const PCCHOWILUVIENNACL = "chowiluviennacl"
const PCROWSCALINGVIENNACL = "rowscalingviennacl"
const PCSAVIENNACL      = "saviennacl"
const PCBDDC            = "bddc"
const PCKACZMARZ        = "kaczmarz"
const PCTELESCOPE       = "telescope"
const PCPATCH           = "patch"
const PCLMVM            = "lmvm"
const PCHMG             = "hmg"
const PCDEFLATION       = "deflation"
const PCHPDDM           = "hpddm"
const PCHARA            = "hara"

"""
Julia alias for the `KSP` C type.

See [PETSc manual]($docs_url/KSP/KSP.html).
"""
struct KSP
  ptr::Ptr{Cvoid}
end
KSP() = KSP(Ptr{Cvoid}())
Base.convert(::Type{KSP},p::Ptr{Cvoid}) = KSP(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::KSP) = v.ptr

"""
Julia alias for the `PC` C type.

See [PETSc manual]($docs_url/PC/PC.html).
"""
struct PC
  ptr::Ptr{Cvoid}
end
PC() = PC(Ptr{Cvoid}())
Base.convert(::Type{PC},p::Ptr{Cvoid}) = PC(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::PC) = v.ptr

@wrapper(:KSPCreate,PetscErrorCode,(MPI.Comm,Ptr{KSP}),(comm,inksp),"/KSP/KSPCreate.html")
@wrapper(:KSPDestroy,PetscErrorCode,(Ptr{KSP},),(ksp,),"/KSP/KSPDestroy.html")
@wrapper(:KSPSetFromOptions,PetscErrorCode,(KSP,),(ksp,),"/KSP/KSPSetFromOptions.html")
@wrapper(:KSPSetOptionsPrefix,PetscErrorCode,(KSP,Cstring),(ksp,prefix),"/KSP/KSPSetOptionsPrefix.html")
@wrapper(:KSPSetUp,PetscErrorCode,(KSP,),(ksp,),"/KSP/KSPSetUp.html")
@wrapper(:KSPSetOperators,PetscErrorCode,(KSP,Mat,Mat),(ksp,Amat,Pmat),"/KSP/KSPSetOperators.html")
@wrapper(:KSPSetTolerances,PetscErrorCode,(KSP,PetscReal,PetscReal,PetscReal,PetscInt),(ksp,rtol,abstol,dtol,maxits),"/KSP/KSPSetTolerances.html")
@wrapper(:KSPSolve,PetscErrorCode,(KSP,Vec,Vec),(ksp,b,x),"/KSP/KSPSolve.html")
@wrapper(:KSPSolveTranspose,PetscErrorCode,(KSP,Vec,Vec),(ksp,b,x),"/KSP/KSPSolveTranspose.html")
@wrapper(:KSPGetIterationNumber,PetscErrorCode,(KSP,Ptr{PetscInt}),(ksp,its),"/KSP/KSPGetIterationNumber.html")
@wrapper(:KSPView,PetscErrorCode,(KSP,PetscViewer),(ksp,viewer),"/KSP/KSPView.html")
@wrapper(:KSPSetInitialGuessNonzero,PetscErrorCode,(KSP,PetscBool),(ksp,flg),"/KSP/KSPSetInitialGuessNonzero.html")
@wrapper(:KSPSetType,PetscErrorCode,(KSP,KSPType),(ksp,typ),"/KSP/KSPSetType.html")
@wrapper(:KSPGetPC,PetscErrorCode,(KSP,Ptr{PC}),(ksp,pc),"/KSP/KSPGetPC.html")
@wrapper(:PCSetType,PetscErrorCode,(PC,PCType),(pc,typ),"/PC/PCSetType.html")
@wrapper(:PCView,PetscErrorCode,(PC,PetscViewer),(pc,viewer),"/PC/PCView.html")
@wrapper(:PCFactorSetMatSolverType,PetscErrorCode,(PC,PCType),(pc,typ),"/PC/PCFactorSetMatSolverType.html")
@wrapper(:PCFactorSetUpMatSolverType,PetscErrorCode,(PC,),(pc,),"/PC/PCFactorSetUpMatSolverType.html")
@wrapper(:PCFactorGetMatrix,PetscErrorCode,(PC,Ptr{Mat}),(ksp,mat),"/PC/PCFactorGetMatrix.html")


"""
Julia alias for the `SNES` C type.

See [PETSc manual]($docs_url/SNES/SNES.html).
"""
struct SNES
  ptr::Ptr{Cvoid}
end
SNES() = SNES(Ptr{Cvoid}())
Base.convert(::Type{SNES},p::Ptr{Cvoid}) = SNES(p)
Base.unsafe_convert(::Type{Ptr{Cvoid}},v::SNES) = v.ptr

const SNESType = Cstring
const SNESNEWTONLS         = "newtonls"
const SNESNEWTONTR         = "newtontr"
const SNESPYTHON           = "python"
const SNESNRICHARDSON      = "nrichardson"
const SNESKSPONLY          = "ksponly"
const SNESKSPTRANSPOSEONLY = "ksptransposeonly"
const SNESVINEWTONRSLS     = "vinewtonrsls"
const SNESVINEWTONSSLS     = "vinewtonssls"
const SNESNGMRES           = "ngmres"
const SNESQN               = "qn"
const SNESSHELL            = "shell"
const SNESNGS              = "ngs"
const SNESNCG              = "ncg"
const SNESFAS              = "fas"
const SNESMS               = "ms"
const SNESNASM             = "nasm"
const SNESANDERSON         = "anderson"
const SNESASPIN            = "aspin"
const SNESCOMPOSITE        = "composite"
const SNESPATCH            = "patch"


@wrapper(:SNESCreate,PetscErrorCode,(MPI.Comm,Ptr{SNES}),(comm,snes),"/SNES/SNESCreate.html")
@wrapper(:SNESSetFunction,PetscErrorCode,(SNES,Vec,Ptr{Cvoid},Ptr{Cvoid}),(snes,vec,fptr,ctx),"/SNES/SNESSetFunction.html")
@wrapper(:SNESSetJacobian,PetscErrorCode,(SNES,Mat,Mat,Ptr{Cvoid},Ptr{Cvoid}),(snes,A,P,jacptr,ctx),"/SNES/SNESSetJacobian.html")
@wrapper(:SNESSolve,PetscErrorCode,(SNES,Vec,Vec),(snes,b,x),"/SNES/SNESSolve.html")
@wrapper(:SNESDestroy,PetscErrorCode,(Ptr{SNES},),(snes,),"/SNES/SNESDestroy.html")
@wrapper(:SNESSetFromOptions,PetscErrorCode,(SNES,),(snes,),"/SNES/SNESSetFromOptions.html")
@wrapper(:SNESView,PetscErrorCode,(SNES,PetscViewer),(snes,viewer),"/SNES/SNESView.html")
@wrapper(:SNESSetType,PetscErrorCode,(SNES,SNESType),(snes,type),"/SNES/SNESSetType.html")
@wrapper(:SNESGetKSP,PetscErrorCode,(SNES,Ptr{KSP}),(snes,ksp),"/SNES/SNESGetKSP.html")
@wrapper(:SNESGetIterationNumber,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,iter),"/SNES/SNESGetIterationNumber.html")
@wrapper(:SNESGetLinearSolveIterations,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,iter),"/SNES/SNESGetLinearSolveIterations.html")
@wrapper(:SNESSetCountersReset,PetscErrorCode,(SNES,PetscBool),(snes,reset),"/SNES/SNESSetCountersReset.html")
@wrapper(:SNESGetNumberFunctionEvals,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,nfuncs),"/SNES/SNESGetNumberFunctionEvals.html")
@wrapper(:SNESGetLinearSolveFailures,PetscErrorCode,(SNES,Ptr{PetscInt}),(snes,nfails),"/SNES/SNESGetLinearSolveFailures.html")

# Garbage collection of PETSc objects
@wrapper(:PetscObjectRegisterDestroy,PetscErrorCode,(Ptr{Cvoid},),(obj,),"/Sys/PetscObjectRegisterDestroy.html")
@wrapper(:PetscObjectRegisterDestroyAll,PetscErrorCode,(),(),"/Sys/PetscObjectRegisterDestroyAll.html")


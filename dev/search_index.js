var documenterSearchIndex = {"docs":
[{"location":"usage/#Usage","page":"Usage","title":"Usage","text":"","category":"section"},{"location":"usage/#Sequential-example","page":"Usage","title":"Sequential example","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"using SparseArrays\nusing Test\nusing PETSC\nusing LinearAlgebra\n\n# Create a spare matrix and a vector in Julia\nI = [1,1,2,2,2,3,3,3,4,4]\nJ = [1,2,1,2,3,2,3,4,3,4]\nV = [4,-2,-1,6,-2,-1,6,-2,-1,4]\nm = 4\nn = 4\nA = sparse(I,J,V,m,n)\nx = ones(m)\nb = A*x\n\n# PETSC options\noptions = \"-ksp_type gmres -ksp_monitor -pc_type ilu\"\nPETSC.init(args=split(options))\n\n# Say, we want to solve A*x=b\nx2 = similar(x); x2 .= 0\nsetup = PETSC.ksp_setup(x2,A,b)\nresults = PETSC.ksp_solve!(x2,setup,b)\n@test x ≈ x2\n\n# Info about the solution process\n@show results\n\n# Now with the same matrix, but a different rhs\nb = 2*b\nresults = PETSC.ksp_solve!(x2,setup,b)\n@test 2*x ≈ x2\n\n# Now with a different matrix, but reusing as much as possible\n# from the previous solve.\nA = 2*A\nPETSC.ksp_setup!(setup,A)\nresults = PETSC.ksp_solve!(x2,setup,b)\n@test x ≈ x2\n\n# The user needs to explicitly destroy\n# the setup object. This cannot be hidden in\n# Julia finalizers since destructors in petsc are\n# collective operations (in parallel runs).\n# Julia finalizers do not guarantee this.\nPETSC.ksp_finalize!(setup)\n\n# The setup object cannot be used anymore.\n# This now would be provably a code dump:\n# PETSC.ksp_solve!(x2,setup,b)","category":"page"},{"location":"usage/#Parallel-example","page":"Usage","title":"Parallel example","text":"","category":"section"},{"location":"usage/","page":"Usage","title":"Usage","text":"First write the parallel code in a function in a file demo.jl.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# File demo.jl\nusing SparseArrays\nusing Test\nusing PETSC\nusing PartitionedArrays\n\nfunction ksp_tests(distribute)\n\n    # Create the matrix and vector with PartitionedArrays.jl\n    parts = distribute(LinearIndices((np,)))\n    n = 7\n    np = 3\n    row_partition = uniform_partition(parts,np,n)\n    col_partition = row_partition\n    I,J,V = map(parts) do part\n        if part == 1\n            I = [1,2,3,1,3]\n            J = [1,2,3,5,6]\n            V = [9,9,9,1,1]\n        elseif part == 2\n            I = [4,5,5,4,5]\n            J = [4,5,3,6,7]\n            V = [9,9,9,1,1]\n        elseif part == 3\n            I = [6,7,6]\n            J = [6,7,4]\n            V = [9,9,1]\n        end\n        I,J,Float64.(V)\n    end |> tuple_of_arrays\n    A = psparse(I,J,V,row_partition,col_partition) |> fetch\n    x = pones(partition(axes(A,2)))\n    b = A*x\n\n    # Now solve the system with petsc\n    options = \"-ksp_type gmres -pc_type jacobi -ksp_rtol 1.0e-12\"\n    PETSC.init(;args = split(options))\n\n    x2 = similar(x); x2 .= 0\n    setup = PETSC.ksp_setup(x2,A,b)\n    results = PETSC.ksp_solve!(x2,setup,b)\n    @test x ≈ x2\n\n    b = 2*b\n    results = PETSC.ksp_solve!(x2,setup,b)\n    @test 2*x ≈ x2\n\n    PETSC.ksp_setup!(setup,A)\n    results = PETSC.ksp_solve!(x2,setup,b)\n    @test 2*x ≈ x2\n\n    PETSC.ksp_finalize!(setup)\nend","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"Then, test your code with the debug back-end of PartitionedArrays.jl","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"include(\"demo.jl\")\nwith_debug(ksp_tests)","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"If it works, you can move the MPI back-end. Create your driver in file driver.jl.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"# File driver.jl\nusing PartitionedArrays\nusing MPI; MPI.Init()\ninclude(\"demo.jl\")\nwith_mpi(ksp_tests)","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"You can launch driver.jl with MPI.","category":"page"},{"location":"usage/","page":"Usage","title":"Usage","text":"using MPI\nmpiexec(cmd->run(`$cmd -np 3 julia --project=. driver.jl`))","category":"page"},{"location":"advanced/#Advanced-API","page":"Advanced API","title":"Advanced API","text":"","category":"section"},{"location":"advanced/","page":"Advanced API","title":"Advanced API","text":"Modules = [PETSC]\nPages = [\"api.jl\"]","category":"page"},{"location":"advanced/#PETSC.PETSC_DECIDE","page":"Advanced API","title":"PETSC.PETSC_DECIDE","text":"Julia constant storing the PETSC_DECIDE value.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"constant"},{"location":"advanced/#PETSC.PETSC_DEFAULT","page":"Advanced API","title":"PETSC.PETSC_DEFAULT","text":"Julia constant storing the PETSC_DEFAULT value.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"constant"},{"location":"advanced/#PETSC.PETSC_DETERMINE","page":"Advanced API","title":"PETSC.PETSC_DETERMINE","text":"Julia constant storing the PETSC_DETERMINE value.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"constant"},{"location":"advanced/#PETSC.InsertMode","page":"Advanced API","title":"PETSC.InsertMode","text":"Julia alias for the InsertMode C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.KSP","page":"Advanced API","title":"PETSC.KSP","text":"Julia alias for the KSP C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.KSPType","page":"Advanced API","title":"PETSC.KSPType","text":"Julia alias for KSPType C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.Mat","page":"Advanced API","title":"PETSC.Mat","text":"Julia alias for the Mat C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.MatAssemblyType","page":"Advanced API","title":"PETSC.MatAssemblyType","text":"Julia alias for the MatAssemblyType C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.MatDuplicateOption","page":"Advanced API","title":"PETSC.MatDuplicateOption","text":"Julia alias for the MatDuplicateOption C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.MatInfo","page":"Advanced API","title":"PETSC.MatInfo","text":"Julia alias for the MatInfo C struct.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.MatInfoType","page":"Advanced API","title":"PETSC.MatInfoType","text":"Julia alias for the MatInfoType C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.MatNullSpace","page":"Advanced API","title":"PETSC.MatNullSpace","text":"Julia alias for the MatNullSpace C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.MatReuse","page":"Advanced API","title":"PETSC.MatReuse","text":"Julia alias for the MatReuse C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.MatStructure","page":"Advanced API","title":"PETSC.MatStructure","text":"Julia alias for the MatStructure C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.MatType","page":"Advanced API","title":"PETSC.MatType","text":"Julia alias for MatType C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.NormType","page":"Advanced API","title":"PETSC.NormType","text":"Julia alias for the NormType C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.PC","page":"Advanced API","title":"PETSC.PC","text":"Julia alias for the PC C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.PCType","page":"Advanced API","title":"PETSC.PCType","text":"Julia alias for PCType C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.PetscBool","page":"Advanced API","title":"PETSC.PetscBool","text":"Julia alias to PetscBool C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.PetscDataType","page":"Advanced API","title":"PETSC.PetscDataType","text":"Julia alias to PetscDataType C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.PetscError","page":"Advanced API","title":"PETSC.PetscError","text":"struct PetscError <: Exception\n  code::PetscErrorCode\nend\n\nCustom Exception thrown by @check_error_code.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.PetscErrorCode","page":"Advanced API","title":"PETSC.PetscErrorCode","text":"Julia alias to PetscErrorCode C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.PetscLogDouble","page":"Advanced API","title":"PETSC.PetscLogDouble","text":"Julia alias to PetscLogDouble C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.PetscViewer","page":"Advanced API","title":"PETSC.PetscViewer","text":"Julia alias for PetscViewer C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.SNES","page":"Advanced API","title":"PETSC.SNES","text":"Julia alias for the SNES C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.Vec","page":"Advanced API","title":"PETSC.Vec","text":"Julia alias for the Vec C type.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.VecOption","page":"Advanced API","title":"PETSC.VecOption","text":"Julia alias for the VecOption C enum.\n\nSee PETSc manual.\n\n\n\n\n\n","category":"type"},{"location":"advanced/#PETSC.KSPCreate-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.KSPCreate","text":"PETSC.KSPCreate(comm, inksp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPDestroy-Tuple{Any}","page":"Advanced API","title":"PETSC.KSPDestroy","text":"PETSC.KSPDestroy(ksp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPGetIterationNumber-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.KSPGetIterationNumber","text":"PETSC.KSPGetIterationNumber(ksp, its)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPGetPC-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.KSPGetPC","text":"PETSC.KSPGetPC(ksp, pc)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPSetFromOptions-Tuple{Any}","page":"Advanced API","title":"PETSC.KSPSetFromOptions","text":"PETSC.KSPSetFromOptions(ksp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPSetInitialGuessNonzero-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.KSPSetInitialGuessNonzero","text":"PETSC.KSPSetInitialGuessNonzero(ksp, flg)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPSetOperators-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.KSPSetOperators","text":"PETSC.KSPSetOperators(ksp, Amat, Pmat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPSetOptionsPrefix-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.KSPSetOptionsPrefix","text":"PETSC.KSPSetOptionsPrefix(ksp, prefix)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPSetTolerances-NTuple{5, Any}","page":"Advanced API","title":"PETSC.KSPSetTolerances","text":"PETSC.KSPSetTolerances(ksp, rtol, abstol, dtol, maxits)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPSetType-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.KSPSetType","text":"PETSC.KSPSetType(ksp, typ)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPSetUp-Tuple{Any}","page":"Advanced API","title":"PETSC.KSPSetUp","text":"PETSC.KSPSetUp(ksp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPSolve-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.KSPSolve","text":"PETSC.KSPSolve(ksp, b, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPSolveTranspose-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.KSPSolveTranspose","text":"PETSC.KSPSolveTranspose(ksp, b, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.KSPView-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.KSPView","text":"PETSC.KSPView(ksp, viewer)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatAssemblyBegin-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.MatAssemblyBegin","text":"PETSC.MatAssemblyBegin(mat, typ)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatAssemblyEnd-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.MatAssemblyEnd","text":"PETSC.MatAssemblyEnd(mat, typ)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatConvert-NTuple{4, Any}","page":"Advanced API","title":"PETSC.MatConvert","text":"PETSC.MatConvert(mat, newtype, reuse, M)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatCopy-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.MatCopy","text":"PETSC.MatCopy(A, B, str)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatCreateAIJ-NTuple{10, Any}","page":"Advanced API","title":"PETSC.MatCreateAIJ","text":"PETSC.MatCreateAIJ(comm, m, n, M, N, d_nz, d_nnz, o_nz, o_nnz, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatCreateMPIAIJWithArrays-NTuple{9, Any}","page":"Advanced API","title":"PETSC.MatCreateMPIAIJWithArrays","text":"PETSC.MatCreateMPIAIJWithArrays(comm, m, n, M, N, i, j, a, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatCreateMPIAIJWithSplitArrays-NTuple{12, Any}","page":"Advanced API","title":"PETSC.MatCreateMPIAIJWithSplitArrays","text":"PETSC.MatCreateMPIAIJWithSplitArrays(comm, m, n, M, N, i, j, a, oi, oj, oa, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatCreateSeqAIJ-NTuple{6, Any}","page":"Advanced API","title":"PETSC.MatCreateSeqAIJ","text":"PETSC.MatCreateSeqAIJ(comm, m, n, nz, nnz, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatCreateSeqAIJWithArrays-NTuple{7, Any}","page":"Advanced API","title":"PETSC.MatCreateSeqAIJWithArrays","text":"PETSC.MatCreateSeqAIJWithArrays(comm, m, n, i, j, a, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatDestroy-Tuple{Any}","page":"Advanced API","title":"PETSC.MatDestroy","text":"PETSC.MatDestroy(A)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatEqual-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.MatEqual","text":"PETSC.MatEqual(A, B, flg)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatGetInfo-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.MatGetInfo","text":"PETSC.MatGetInfo(mat, flag, info)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatGetSize-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.MatGetSize","text":"PETSC.MatGetSize(mat, m, n)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatGetValues-NTuple{6, Any}","page":"Advanced API","title":"PETSC.MatGetValues","text":"PETSC.MatGetValues(mat, m, idxm, n, idxn, v)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatMult-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.MatMult","text":"PETSC.MatMult(mat, x, y)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatMultAdd-NTuple{4, Any}","page":"Advanced API","title":"PETSC.MatMultAdd","text":"PETSC.MatMultAdd(mat, v1, v2, v3)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatMumpsSetCntl-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.MatMumpsSetCntl","text":"PETSC.MatMumpsSetCntl(mat, icntl, val)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatMumpsSetIcntl-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.MatMumpsSetIcntl","text":"PETSC.MatMumpsSetIcntl(mat, icntl, val)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatNullSpaceCreate-NTuple{5, Any}","page":"Advanced API","title":"PETSC.MatNullSpaceCreate","text":"PETSC.MatNullSpaceCreate(comm, has_cnst, n, vecs, sp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatNullSpaceCreateRigidBody-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.MatNullSpaceCreateRigidBody","text":"PETSC.MatNullSpaceCreateRigidBody(coords, sp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatNullSpaceDestroy-Tuple{Any}","page":"Advanced API","title":"PETSC.MatNullSpaceDestroy","text":"PETSC.MatNullSpaceDestroy(ns)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatScale-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.MatScale","text":"PETSC.MatScale(mat, alpha)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatSetBlockSize-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.MatSetBlockSize","text":"PETSC.MatSetBlockSize(mat, bs)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatSetNearNullSpace-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.MatSetNearNullSpace","text":"PETSC.MatSetNearNullSpace(mat, nullsp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatSetValues-NTuple{7, Any}","page":"Advanced API","title":"PETSC.MatSetValues","text":"PETSC.MatSetValues(mat, m, idxm, n, idxn, v, addv)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatView-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.MatView","text":"PETSC.MatView(mat, viewer)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.MatZeroEntries-Tuple{Any}","page":"Advanced API","title":"PETSC.MatZeroEntries","text":"PETSC.MatZeroEntries(mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PCFactorGetMatrix-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.PCFactorGetMatrix","text":"PETSC.PCFactorGetMatrix(ksp, mat)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PCFactorSetMatSolverType-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.PCFactorSetMatSolverType","text":"PETSC.PCFactorSetMatSolverType(pc, typ)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PCFactorSetUpMatSolverType-Tuple{Any}","page":"Advanced API","title":"PETSC.PCFactorSetUpMatSolverType","text":"PETSC.PCFactorSetUpMatSolverType(pc)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PCSetType-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.PCSetType","text":"PETSC.PCSetType(pc, typ)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PCView-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.PCView","text":"PETSC.PCView(pc, viewer)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PETSC_VIEWER_DRAW_-Tuple{Any}","page":"Advanced API","title":"PETSC.PETSC_VIEWER_DRAW_","text":"PETSC.PETSC_VIEWER_DRAW_(comm)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PETSC_VIEWER_STDOUT_-Tuple{Any}","page":"Advanced API","title":"PETSC.PETSC_VIEWER_STDOUT_","text":"PETSC.PETSC_VIEWER_STDOUT_(comm)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PetscDataTypeFromString-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.PetscDataTypeFromString","text":"PetscDataTypeFromString(name,ptype,found)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PetscDataTypeGetSize-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.PetscDataTypeGetSize","text":"PetscDataTypeGetSize(ptype,size)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PetscFinalize-Tuple{}","page":"Advanced API","title":"PETSC.PetscFinalize","text":"PETSC.PetscFinalize()\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PetscFinalized-Tuple{Any}","page":"Advanced API","title":"PETSC.PetscFinalized","text":"PETSC.PetscFinalized(flag)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PetscInitializeNoArguments-Tuple{}","page":"Advanced API","title":"PETSC.PetscInitializeNoArguments","text":"PETSC.PetscInitializeNoArguments()\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PetscInitializeNoPointers-NTuple{4, Any}","page":"Advanced API","title":"PETSC.PetscInitializeNoPointers","text":"PETSC.PetscInitializeNoPointers(argc, args, filename, help)\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PetscInitialized-Tuple{Any}","page":"Advanced API","title":"PETSC.PetscInitialized","text":"PETSC.PetscInitialized(flag)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PetscObjectRegisterDestroy-Tuple{Any}","page":"Advanced API","title":"PETSC.PetscObjectRegisterDestroy","text":"PETSC.PetscObjectRegisterDestroy(obj)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.PetscObjectRegisterDestroyAll-Tuple{}","page":"Advanced API","title":"PETSC.PetscObjectRegisterDestroyAll","text":"PETSC.PetscObjectRegisterDestroyAll()\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESCreate-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.SNESCreate","text":"PETSC.SNESCreate(comm, snes)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESDestroy-Tuple{Any}","page":"Advanced API","title":"PETSC.SNESDestroy","text":"PETSC.SNESDestroy(snes)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESGetIterationNumber-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.SNESGetIterationNumber","text":"PETSC.SNESGetIterationNumber(snes, iter)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESGetKSP-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.SNESGetKSP","text":"PETSC.SNESGetKSP(snes, ksp)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESGetLinearSolveFailures-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.SNESGetLinearSolveFailures","text":"PETSC.SNESGetLinearSolveFailures(snes, nfails)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESGetLinearSolveIterations-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.SNESGetLinearSolveIterations","text":"PETSC.SNESGetLinearSolveIterations(snes, iter)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESGetNumberFunctionEvals-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.SNESGetNumberFunctionEvals","text":"PETSC.SNESGetNumberFunctionEvals(snes, nfuncs)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESSetCountersReset-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.SNESSetCountersReset","text":"PETSC.SNESSetCountersReset(snes, reset)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESSetFromOptions-Tuple{Any}","page":"Advanced API","title":"PETSC.SNESSetFromOptions","text":"PETSC.SNESSetFromOptions(snes)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESSetFunction-NTuple{4, Any}","page":"Advanced API","title":"PETSC.SNESSetFunction","text":"PETSC.SNESSetFunction(snes, vec, fptr, ctx)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESSetJacobian-NTuple{5, Any}","page":"Advanced API","title":"PETSC.SNESSetJacobian","text":"PETSC.SNESSetJacobian(snes, A, P, jacptr, ctx)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESSetType-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.SNESSetType","text":"PETSC.SNESSetType(snes, type)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESSolve-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.SNESSolve","text":"PETSC.SNESSolve(snes, b, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.SNESView-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.SNESView","text":"PETSC.SNESView(snes, viewer)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecAXPBY-NTuple{4, Any}","page":"Advanced API","title":"PETSC.VecAXPBY","text":"PETSC.VecAXPBY(y, alpha, beta, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecAXPY-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.VecAXPY","text":"PETSC.VecAXPY(y, alpha, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecAYPX-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.VecAYPX","text":"PETSC.VecAYPX(y, beta, x)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecAssemblyBegin-Tuple{Any}","page":"Advanced API","title":"PETSC.VecAssemblyBegin","text":"PETSC.VecAssemblyBegin(vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecAssemblyEnd-Tuple{Any}","page":"Advanced API","title":"PETSC.VecAssemblyEnd","text":"PETSC.VecAssemblyEnd(vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecCopy-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecCopy","text":"PETSC.VecCopy(x, y)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecCreateGhost-NTuple{6, Any}","page":"Advanced API","title":"PETSC.VecCreateGhost","text":"PETSC.VecCreateGhost(comm, n, N, nghost, ghosts, vv)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecCreateGhostWithArray-NTuple{7, Any}","page":"Advanced API","title":"PETSC.VecCreateGhostWithArray","text":"PETSC.VecCreateGhostWithArray(comm, n, N, nghost, ghosts, array, vv)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecCreateMPI-NTuple{4, Any}","page":"Advanced API","title":"PETSC.VecCreateMPI","text":"PETSC.VecCreateMPI(comm, n, N, v)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecCreateMPIWithArray-NTuple{6, Any}","page":"Advanced API","title":"PETSC.VecCreateMPIWithArray","text":"PETSC.VecCreateMPIWithArray(comm, bs, n, N, array, vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecCreateSeq-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.VecCreateSeq","text":"PETSC.VecCreateSeq(comm, n, vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecCreateSeqWithArray-NTuple{5, Any}","page":"Advanced API","title":"PETSC.VecCreateSeqWithArray","text":"PETSC.VecCreateSeqWithArray(comm, bs, n, array, vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecDestroy-Tuple{Any}","page":"Advanced API","title":"PETSC.VecDestroy","text":"PETSC.VecDestroy(vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecDuplicate-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecDuplicate","text":"PETSC.VecDuplicate(v, newv)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecGetArray-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecGetArray","text":"PETSC.VecGetArray(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecGetArrayRead-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecGetArrayRead","text":"PETSC.VecGetArrayRead(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecGetArrayWrite-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecGetArrayWrite","text":"PETSC.VecGetArrayWrite(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecGetLocalSize-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecGetLocalSize","text":"PETSC.VecGetLocalSize(vec, n)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecGetSize-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecGetSize","text":"PETSC.VecGetSize(vec, n)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecGetValues-NTuple{4, Any}","page":"Advanced API","title":"PETSC.VecGetValues","text":"PETSC.VecGetValues(x, ni, ix, y)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecGhostGetLocalForm-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecGhostGetLocalForm","text":"PETSC.VecGhostGetLocalForm(g, l)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecGhostRestoreLocalForm-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecGhostRestoreLocalForm","text":"PETSC.VecGhostRestoreLocalForm(g, l)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecNorm-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.VecNorm","text":"PETSC.VecNorm(x, typ, val)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecPlaceArray-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecPlaceArray","text":"PETSC.VecPlaceArray(vec, array)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecResetArray-Tuple{Any}","page":"Advanced API","title":"PETSC.VecResetArray","text":"PETSC.VecResetArray(vec)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecRestoreArray-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecRestoreArray","text":"PETSC.VecRestoreArray(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecRestoreArrayRead-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecRestoreArrayRead","text":"PETSC.VecRestoreArrayRead(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecRestoreArrayWrite-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecRestoreArrayWrite","text":"PETSC.VecRestoreArrayWrite(x, a)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecScale-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecScale","text":"PETSC.VecScale(x, alpha)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecSet-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecSet","text":"PETSC.VecSet(x, alpha)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecSetOption-Tuple{Any, Any, Any}","page":"Advanced API","title":"PETSC.VecSetOption","text":"PETSC.VecSetOption(x, op, flg)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecSetValues-NTuple{5, Any}","page":"Advanced API","title":"PETSC.VecSetValues","text":"PETSC.VecSetValues(x, ni, ix, y, iora)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.VecView-Tuple{Any, Any}","page":"Advanced API","title":"PETSC.VecView","text":"PETSC.VecView(vec, viewer)\n\nSee PETSc manual.\n\n\n\n\n\n","category":"method"},{"location":"advanced/#PETSC.@PETSC_VIEWER_DRAW_SELF-Tuple{}","page":"Advanced API","title":"PETSC.@PETSC_VIEWER_DRAW_SELF","text":"@PETSC_VIEWER_DRAW_SELF\n\nSee PETSc manual.\n\n\n\n\n\n","category":"macro"},{"location":"advanced/#PETSC.@PETSC_VIEWER_DRAW_WORLD-Tuple{}","page":"Advanced API","title":"PETSC.@PETSC_VIEWER_DRAW_WORLD","text":"@PETSC_VIEWER_DRAW_WORLD\n\nSee PETSc manual.\n\n\n\n\n\n","category":"macro"},{"location":"advanced/#PETSC.@PETSC_VIEWER_STDOUT_SELF-Tuple{}","page":"Advanced API","title":"PETSC.@PETSC_VIEWER_STDOUT_SELF","text":"@PETSC_VIEWER_STDOUT_SELF\n\nSee PETSc manual.\n\n\n\n\n\n","category":"macro"},{"location":"advanced/#PETSC.@PETSC_VIEWER_STDOUT_WORLD-Tuple{}","page":"Advanced API","title":"PETSC.@PETSC_VIEWER_STDOUT_WORLD","text":"@PETSC_VIEWER_STDOUT_WORLD\n\nSee PETSc manual.\n\n\n\n\n\n","category":"macro"},{"location":"advanced/#PETSC.@check_error_code-Tuple{Any}","page":"Advanced API","title":"PETSC.@check_error_code","text":"@check_error_code expr\n\nCheck if expr returns an error code equal to zero(PetscErrorCode). If not, throw an instance of PetscError.\n\n\n\n\n\n","category":"macro"},{"location":"reference/#Reference","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/#Configuration","page":"Reference","title":"Configuration","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [PETSC]\nPages = [\"preferences_tail.jl\"]","category":"page"},{"location":"reference/#PETSC.use_petsc_jll-Tuple{}","page":"Reference","title":"PETSC.use_petsc_jll","text":"PETSC.use_petsc_jll(;kwargs...)\n\nDocument me!\n\n\n\n\n\n","category":"method"},{"location":"reference/#PETSC.use_system_petsc-Tuple{}","page":"Reference","title":"PETSC.use_system_petsc","text":"PETSC.use_system_petsc(;kwargs...)\n\nDocument me!\n\n\n\n\n\n","category":"method"},{"location":"reference/#Environment","page":"Reference","title":"Environment","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [PETSC]\nPages = [\"environment.jl\"]","category":"page"},{"location":"reference/#PETSC.finalize-Tuple{}","page":"Reference","title":"PETSC.finalize","text":"PETSC.finalize()\n\nDocument me!\n\n\n\n\n\n","category":"method"},{"location":"reference/#PETSC.init-Tuple{}","page":"Reference","title":"PETSC.init","text":"PETSC.init(:kwargs...)\n\nDocument me!\n\n\n\n\n\n","category":"method"},{"location":"reference/#PETSC.initialized-Tuple{}","page":"Reference","title":"PETSC.initialized","text":"PETSC.initialized()\n\nDocument me!\n\n\n\n\n\n","category":"method"},{"location":"reference/#PETSC.with-Tuple{Any}","page":"Reference","title":"PETSC.with","text":"PETSC.with(f;kwargs...)\n\nDocument me!\n\n\n\n\n\n","category":"method"},{"location":"reference/#Linear-solvers-(KSP)","page":"Reference","title":"Linear solvers (KSP)","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Modules = [PETSC]\nPages = [\"ksp.jl\"]","category":"page"},{"location":"reference/#PETSC.ksp_finalize!","page":"Reference","title":"PETSC.ksp_finalize!","text":"PETSC.ksp_finalize!(setup)\n\nDocument me!\n\n\n\n\n\n","category":"function"},{"location":"reference/#PETSC.ksp_setup","page":"Reference","title":"PETSC.ksp_setup","text":"PETSC.ksp_setup(x,A,b;kwargs...)\n\nDocument me!\n\n\n\n\n\n","category":"function"},{"location":"reference/#PETSC.ksp_setup!","page":"Reference","title":"PETSC.ksp_setup!","text":"PETSC.ksp_setup!(setup,A)\n\nDocument me!\n\n\n\n\n\n","category":"function"},{"location":"reference/#PETSC.ksp_solve!","page":"Reference","title":"PETSC.ksp_solve!","text":"PETSC.ksp_solve!(x,setup,b)\n\nDocument me!\n\n\n\n\n\n","category":"function"},{"location":"reference/#Nonlinear-solvers-(SNES)","page":"Reference","title":"Nonlinear solvers (SNES)","text":"","category":"section"},{"location":"reference/","page":"Reference","title":"Reference","text":"Work in progress (help wanted).","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = PETSC","category":"page"},{"location":"#PETSC","page":"Home","title":"PETSC","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for PETSC.","category":"page"},{"location":"#What","page":"Home","title":"What","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The goal of this package is to provide a high-level Julia interface to solvers from the PETSc library. At this moment it wraps linear solvers from the KSP module in PETSc, but the goal is to also provide nonlinear solvers from the SNES module in PETSc. The package also provides a low-level interface with functions that are almost 1-to-1 to the corresponding C functions for advanced users. The low level API is mostly taken from GridapPETSc.jl.","category":"page"},{"location":"#Why","page":"Home","title":"Why","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The main difference of this package with respect to other Julia bindings to PETSc (e.g., PETSc.jl), is that our high-level interface is based on pure Julia types. I.e., the high-level interface only provides new functions, and the inputs and outputs of the such functions are pure Julia types. For instance, the functions to solve systems of linear equations take standard Julia (sparse) matrices and vectors. For parallel computations, one can use the pure Julia parallel sparse matrices and vectors implemented in PartitionedArrays.jl.","category":"page"},{"location":"#Related-packages","page":"Home","title":"Related packages","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"https://github.com/JuliaParallel/PETSc.jl\nhttps://github.com/gridap/GridapPETSc.jl","category":"page"},{"location":"config/#Configuration","page":"Configuration","title":"Configuration","text":"","category":"section"},{"location":"config/","page":"Configuration","title":"Configuration","text":"This packages uses Preferences.jl for its configuration.","category":"page"},{"location":"config/#Use-jll-binary","page":"Configuration","title":"Use jll binary","text":"","category":"section"},{"location":"config/","page":"Configuration","title":"Configuration","text":"To use the petsc installation provided by BinaryBuilder.jl run these commands:","category":"page"},{"location":"config/","page":"Configuration","title":"Configuration","text":"using PETSC\nPETSC.use_petsc_jll()\n# Restart Julia now","category":"page"},{"location":"config/","page":"Configuration","title":"Configuration","text":"You can also specify the integer and scalar types to use:","category":"page"},{"location":"config/","page":"Configuration","title":"Configuration","text":"using PETSC\nPETSC.use_petsc_jll(PetscInt=Int32,PetscScalar=Float32)\n# Restart Julia now","category":"page"},{"location":"config/#Use-a-system-binary","page":"Configuration","title":"Use a system binary","text":"","category":"section"},{"location":"config/","page":"Configuration","title":"Configuration","text":"To use PETSc installed in your system:","category":"page"},{"location":"config/","page":"Configuration","title":"Configuration","text":"using PETSC\nPETSC.use_system_petsc()\n# Restart Julia now","category":"page"},{"location":"config/","page":"Configuration","title":"Configuration","text":"This will look in LD_LIBRARY_PATH for a file called libpetsc.so.","category":"page"},{"location":"config/","page":"Configuration","title":"Configuration","text":"You can also provide the full path to libpetsc.so.","category":"page"},{"location":"config/","page":"Configuration","title":"Configuration","text":"using PETSC\nPETSC.use_system_petsc(;libpetsc_path=path/to/libpetsc.so)\n# Restart Julia now","category":"page"}]
}

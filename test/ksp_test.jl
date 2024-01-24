module KSPTests

using SparseArrays
using Test
using PetscCall
using LinearAlgebra

# Create a spare matrix and a vector in Julia
I = [1,1,2,2,2,3,3,3,4,4]
J = [1,2,1,2,3,2,3,4,3,4]
V = [4,-2,-1,6,-2,-1,6,-2,-1,4]
m = 4
n = 4
A = sparse(I,J,V,m,n)
x = ones(m)
b = A*x

# PetscCall options
options = "-ksp_type gmres -ksp_monitor -pc_type ilu"
PetscCall.init(args=split(options))

# Say, we want to solve A*x=b
x2 = similar(x); x2 .= 0
setup = PetscCall.ksp_setup(x2,A,b)
results = PetscCall.ksp_solve!(x2,setup,b)
@test x ≈ x2

# Info about the solution process
@show results

# Now with the same matrix, but a different rhs
b = 2*b
results = PetscCall.ksp_solve!(x2,setup,b)
@test 2*x ≈ x2

# Now with a different matrix, but reusing as much as possible
# from the previous solve.
A = 2*A
PetscCall.ksp_setup!(setup,A)
results = PetscCall.ksp_solve!(x2,setup,b)
@test x ≈ x2

# The user needs to explicitly destroy
# the setup object. This cannot be hidden in
# Julia finalizers since destructors in petsc are
# collective operations (in parallel runs).
# Julia finalizers do not guarantee this.
PetscCall.ksp_finalize!(setup)

# The setup object cannot be used anymore.
# This now would be provably a code dump:
# PetscCall.ksp_solve!(x2,setup,b)

end

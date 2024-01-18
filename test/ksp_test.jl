module KSPTests

using SparseArrays
using Test
using PETSC

options = "-ksp_type gmres -ksp_monitor -pc_type ilu"
PETSC.init(args=split(options))

I = [1,1,2,2,2,3,3,3,4,4]
J = [1,2,1,2,3,2,3,4,3,4]
V = [4,-2,-1,6,-2,-1,6,-2,-1,4]
m = 4
n = 4
A = sparse(I,J,V,m,n)

x = ones(m)
b = A*x

handles = PETSC.ksp_create_handles()

x2 = similar(x)
fill!(x2,0)

setup = PETSC.ksp_setup(x2,A,b,handles)
results = PETSC.ksp_solve!(x2,b,setup)
@test x ≈ x2
@show results

b = 2*b
results = PETSC.ksp_solve!(x2,b,setup)
@test 2*x ≈ x2

PETSC.ksp_destroy_handles!(handles)





end

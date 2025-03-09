#%%
from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import ConvexTerm, QuadraticTerm
from dolfinx_optim.cones import SDP
from dolfinx_optim.utils import to_vect

# Define the box mesh
L, W, H = (1.2, 2.0, 1.0)
Nx, Ny, Nz = (10, 3, 20)
domain = mesh.create_box(MPI.COMM_WORLD, [(0, 0, 0), (L, W, H)], [Nx, Ny, Nz])

# Define the conic representation of the Mohr-Coulomb support function
c = fem.Constant(domain, 1.0)
phi = fem.Constant(domain, np.pi / 6.0)

class MohrCoulomb(ConvexTerm):
    """SDP implementation of Mohr-Coulomb criterion."""

    def conic_repr(self, X):
        Y1 = self.add_var((3,3), cone=SDP(3))
        Y2 = self.add_var((3,3), cone=SDP(3))
        a = (1 - ufl.sin(phi)) / (1 + ufl.sin(phi))
        self.add_eq_constraint(X - to_vect(Y1) + to_vect(Y2))
        self.add_eq_constraint(ufl.tr(Y2) - a * ufl.tr(Y1))
        self.add_linear_term(2 * c * ufl.cos(phi) / (1 + ufl.sin(phi)) * ufl.tr(Y1))

# Set up the loading, function spaces and boundary conditions
gamma = 10.0
f = fem.Constant(domain, (0, 0, -gamma))

def border(x):
    return np.isclose(x[0], L) | np.isclose(x[2], 0)

gdim = 3
deg_quad = 4


#%%

#V = fem.functionspace(domain, ("CR", 1, (gdim,)))
V = fem.functionspace(domain, ("CG", 2, (gdim,)))
bc_dofs = fem.locate_dofs_geometrical(V, border)
bcs = [fem.dirichletbc(np.zeros((gdim,)), bc_dofs, V)]

# Initiate the MosekProblem object and add the linear equality constraint
prob = MosekProblem(domain, "3D limit analysis")
u = prob.add_var(V, bc=bcs, name="Collapse mechanism")

# Add CR stabilization 
#h = ufl.CellDiameter(domain)
#h_avg = (h("+") + h("-")) / 2.0
#n = ufl.FacetNormal(domain)
#jump_u =  1/h * ufl.jump(u)
#stabilization = QuadraticTerm(jump_u, 4, measure_type="ds")
#prob.add_convex_term(stabilization)

#%%
prob.add_eq_constraint(ufl.dot(f, u) * ufl.dx, b=1.0)

# Add the convex term corresponding to the support function
crit = MohrCoulomb(ufl.sym(ufl.grad(u)), 2)
prob.add_convex_term(crit)

# Solve the problem and export results to Paraview
pobj, dobj = prob.optimize()
# %%
V_plot = fem.functionspace(domain, ("CG", 2, (gdim,)))
u_plot = fem.Function(V_plot,name="u")
u_plot.interpolate(u)
with io.VTKFile(MPI.COMM_WORLD, "results.pvd", "w") as vtk:
    vtk.write_function(u_plot)

# Check the solution compared with the exact solution
print("2D factor [Chen] (for phi=30°):", 6.69)
print("Computed factor:", pobj * float(gamma * H / c))
# %%

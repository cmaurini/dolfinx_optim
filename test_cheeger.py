#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test case on a square Cheeger problem.

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""
import numpy as np
from mpi4py import MPI
from dolfinx import fem, mesh, io
import ufl
from ufl import dot, grad
from dolfinx_optim.mosek_io import MosekProblem
from dolfinx_optim.convex_function import (
    L2Norm,
    L1Norm,
    LinfNorm,
    AbsValue,
    LpNorm,
    Epigraph,
    Perspective,
    InfConvolution,
)


N = 50
domain = mesh.create_unit_square(
    MPI.COMM_WORLD,
    N,
    N,
    diagonal=mesh.DiagonalType.crossed,
    cell_type=mesh.CellType.quadrilateral,
)


def border(x):  # noqa
    return np.logical_or(np.isclose(x[1], 0), np.isclose(x[0], 0))


f = fem.Constant(domain, 1.0)

deg_u = 2
deg_quad = 2
V = fem.FunctionSpace(domain, ("CG", deg_u))
dofs = fem.locate_dofs_geometrical(V, border)
bc = fem.dirichletbc(0.0, dofs, V)

prob = MosekProblem(domain, "Test")
# u = prob.add_var(V, bc=bc)

V0 = fem.FunctionSpace(domain, ("DG", 0))
u, t = prob.add_var([V, V0], bc=[bc, None], name=["u", "t"])

dx = ufl.Measure("dx", domain=domain)

Pext = dot(f, u) * dx

prob.add_eq_constraint(A=Pext, b=1.0)

pi = L1Norm(grad(u), deg_quad)

d = ufl.as_vector([u.dx(0), u.dx(1), 0])
pi2 = L2Norm(d, deg_quad)
# prob.add_convex_term(pi)

# epi = Epigraph(t, pi)
# prob.add_convex_term(epi)
# prob.add_obj_func(t * dx)

infc = InfConvolution(pi2, pi, indices=(0, 1))
prob.add_convex_term(infc)

prob.optimize()

prob.get_solution_info()

with io.XDMFFile(MPI.COMM_WORLD, "limit_analysis/u.xdmf", "w") as file:
    file.write_mesh(u.function_space.mesh)
    file.write_function(u)

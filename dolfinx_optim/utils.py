#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some useful utility functions.

@author: Jeremy Bleyer, Ecole des Ponts ParisTech,
Laboratoire Navier (ENPC, Univ Gustave Eiffel, CNRS, UMR 8205)
@email: jeremy.bleyer@enpc.fr
"""
import numpy as np
from ufl import shape, as_matrix, as_vector, outer, cross, sqrt, dot, inner, avg
import ufl
from dolfinx import fem


def get_shape(expr):
    expr_shape = shape(expr)
    if len(expr_shape) == 0:  # scalar constraint
        dim = 0
    elif len(expr_shape) == 1:
        dim = expr_shape[0]
    else:
        dim = expr_shape
    return dim


def to_list(a, n=1):
    """Transform `a` to list of length `n`."""
    if not isinstance(a, list):
        return [a] * n
    else:
        return a


def subk_list(d):
    """Generate list of subk indices for block triplet format for dimension d."""
    return [i for k in range(d) for i in range(k, d)]


def subl_list(d):
    """Generate list of subl indices for block triplet format for dimension d."""
    return [i - k for k in range(d) for i in range(k, d)]


def half_vect2subk(index_list, d):
    """Return the subk index for a d-dimensional SDP variable half vector."""
    subk = subk_list(d)
    return [subk[i] for i in index_list]


def half_vect2subl(index_list, d):
    """Return the subl index for a d-dimensional SDP variable half vector."""
    subl = subl_list(d)
    return [subl[i] for i in index_list]


def to_vect(X, symmetric=False):
    if symmetric:
        raise NotImplementedError
    else:
        d1, d2 = get_shape(X)
        return as_vector([X[i, j] for i in range(d1) for j in range(d2)])


def to_mat(X, symmetric=False, shape=None):
    if symmetric:
        raise NotImplementedError
    else:
        if shape is None:  # assumes a square matrix
            d1 = int(np.sqrt(len(X)))
            d2 = d1
        else:
            d1, d2 = shape
            assert (
                len(X) == d1 * d2
            ), f"Vector of size {len(X)} cannot be written as {d1}x{d2}."
        return as_matrix([[X[j + d2 * i] for j in range(d2)] for i in range(d1)])


# def to_vect(X, symmetric=True):
#     """
#     Transform a tensor into vector by spanning diagonals.

#     Parameters
#     ----------
#     symmetric: bool
#         indicates if the tensor X must be considered as symmetric or not

#     Returns
#     -------
#     UFL vector expression
#         a d*(d+1)/2 vector if the d x d tensor is symmetric, a d**2 vector otherwise
#     """
#     s = shape(X)
#     if len(s) == 2 and s[0] == s[1]:
#         d = s[0]
#         if symmetric:
#             return as_vector([X[i - k, i] for k in range(d) for i in range(k, d)])
#         else:
#             components = [X[i, i] for i in range(d)]
#             lower_diags = [X[i, i - k] for k in range(1, d) for i in range(k, d)]
#             upper_diags = [X[i - k, i] for k in range(1, d) for i in range(k, d)]
#             diags = (upper_diags, lower_diags)
#             components += [diag[i] for i in range(len(upper_diags)) for diag in diags]
#             return as_vector(components)
#     else:
#         raise ValueError("Variable must be a square tensor")


def concatenate(vectors):
    """Concatenate vectors."""
    concat = []
    for v in vectors:
        if shape(v) == ():
            concat += [v]
        else:
            concat += [v[i] for i in range(len(v))]
    return as_vector(concat)


def vstack(arrays):
    """Vertical stack of vectors/matrix."""
    if all([len(a.ufl_shape) <= 1 for a in arrays]):
        return concatenate(arrays)

    shapes = [shape(a)[0] if len(shape(a)) == 1 else shape(a)[1] for a in arrays]
    assert len(set(shapes)) == 1, "Arrays must have matching dimensions."
    final_array = []
    for arr in arrays:
        if len(shape(arr)) == 2:
            final_array += [
                [arr[i, j] for j in range(shape(arr)[1])] for i in range(shape(arr)[0])
            ]
        else:
            final_array += [[arr[i] for i in range(len(arr))]]
    return as_matrix(final_array)


def hstack(arrays):
    """Vertical stack of vectors/matrix."""
    shapes = [a.ufl_shape for a in arrays]
    if all([len(s) <= 1 for s in shapes]):
        return as_matrix([[a[i] for a in arrays] for i in range(shapes[0][0])])

    shapes = [shape(a)[0] for a in arrays]
    assert len(set(shapes)) == 1, "Arrays must have matching dimensions."
    final_array = []
    for arr in arrays:
        if len(shape(arr)) == 2:
            final_array += [
                [arr[j, i] for j in range(shape(arr)[0])] for i in range(shape(arr)[1])
            ]
        else:
            final_array += [[arr[i] for i in range(len(arr))]]
    final_array = list(map(list, zip(*final_array)))
    return as_matrix(final_array)


def block_matrix(M):
    """Horizontal stack of vectors."""
    return vstack([hstack(m) for m in M])


def get_slice(x, start=0, end=None, step=None):
    """Get a slice x[start:end:step]."""
    dim = shape(x)[0]
    if end is None:
        end = dim
    elif end < 0:
        end = dim + end
    elif end == 0:
        end = 1
    if step is None:
        step = 1
    return as_vector([x[i] for i in range(start, end, step)])


def add_zeros(x, npad: int, pos: int = 0):
    """Vector padding with zeros.

    Parameters
    ----------
    x : UFL vector
        the intial vector
    npad : int
        number of padding zeros
    pos : int, optional
        position at which zeros are added, by default 0 (start), -1 to add at the end

    Returns
    -------
    UFL vector
        the new vector with zeros
    """
    assert npad > 0
    dim_x = shape(x)[0]
    x_list = [x[i] for i in range(dim_x)]
    zeros = [0] * npad
    if pos == 0:
        return as_vector(zeros + x_list)
    elif pos == -1:
        return as_vector(x_list + zeros)


def tail(x):
    """Get the tail x[1:] of a vector."""
    return get_slice(x, start=1)


def local_frame(n):
    """Compute projector on facet local frame (n, t1, t2)."""
    dim = shape(n)[0]
    if dim == 2:
        t = as_vector([-n[1], n[0]])
        e1 = as_vector([1.0, 0.0])
        e2 = as_vector([0.0, 1.0])
        return outer(e1, n) + outer(e2, t)
    else:
        ei = as_vector(np.random.rand(3))
        t1 = cross(n, ei)
        t1 /= sqrt(dot(t1, t1))
        t2 = cross(n, t1)
        e1 = as_vector([1.0, 0.0, 0.0])
        e2 = as_vector([0.0, 1.0, 0.0])
        e3 = as_vector([0.0, 0.0, 1.0])
        return outer(e1, n) + outer(e2, t1) + outer(e3, t2)


def split_affine_expression(expr, operand, variables):
    if expr == 0:
        return 0, 0, 0
    else:
        e = ufl.variable(operand)
        new_expr = ufl.replace(expr, {operand: e})
        new_expr = ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(
            new_expr
        )
        linear_op = ufl.algorithms.apply_derivatives.apply_derivatives(
            ufl.diff(new_expr, e)
        )
        linear_var = [
            ufl.algorithms.apply_derivatives.apply_derivatives(
                ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(
                    ufl.diff(expr, v)
                )
            )
            for v in variables
            if not isinstance(v, fem.Constant)
        ]
        d = get_shape(operand)
        if d == 0:
            z = 0
        else:
            z = ufl.as_vector([0] * d)
        constant = ufl.replace(
            ufl.replace(expr, {operand: z}), {v: 0 * v for v in variables}
        )

        return (
            ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(linear_op),
            linear_var,
            ufl.algorithms.apply_algebra_lowering.apply_algebra_lowering(constant),
        )


def reshape(x, d):
    if get_shape(x) != d:
        return ufl.as_vector([1] * d) * x
    else:
        return x

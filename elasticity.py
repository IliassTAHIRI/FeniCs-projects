#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  5 13:23:43 2018

@author: imaneguetni
"""

"""
FEniCS tutorial demo program: Linear elastic problem.
  -div(sigma(u)) = f
The model is used to simulate an elastic beam clamped at
its left end and deformed under its own weight.
"""

#from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Scaled variables
L = 1; W = 0.2
mu = 1
rho = 1
delta = W/L
gamma = 0.4*delta**2
beta = 1.25
lambda_ = beta
g = gamma

# Create mesh and define function space
mesh = BoxMesh(Point(0, 0, 0), Point(L, W, W), 10, 1, 1)
V = VectorFunctionSpace(mesh, 'P', 1)

# Define boundary condition
tol = 1E-14

def clamped_boundary(x, on_boundary):
    return on_boundary and near(x[0],0,tol)

def clamped_boundary2(x, on_boundary):
    return on_boundary and near(x[0],1,tol)

bc1 = DirichletBC(V, Constant((0, 0, 0)), clamped_boundary)
bc2 = DirichletBC(V, Constant((0, 20, 0)), clamped_boundary2)
bc=[bc1, bc2]

# Define strain and stress

def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)
    #return sym(nabla_grad(u))

def sigma(u):
    return lambda_*nabla_div(u)*Identity(d) + 2*mu*epsilon(u)

# Define variational problem
u = TrialFunction(V)
d = u.geometric_dimension()  # space dimension
v = TestFunction(V)
f = Constant((0, 0, -rho*g))
T = Constant((0, 0, 0))
a = inner(sigma(u), epsilon(v))*dx
L = dot(f, v)*dx + dot(T, v)*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot solution
plot(u, title='Displacement', mode='displacement')

# Plot stress
s = sigma(u) - (1./3)*tr(sigma(u))*Identity(d)  # deviatoric stress
von_Mises = sqrt(3./2*inner(s, s))
V = FunctionSpace(mesh, 'P', 1)
von_Mises = project(von_Mises, V)
plot(von_Mises, title='Stress intensity')

# Compute magnitude of displacement
u_magnitude = sqrt(dot(u, u))
u_magnitude = project(u_magnitude, V)
plot(u_magnitude, 'Displacement magnitude')





print('min/max u:',
      u_magnitude.vector().array().min(),
      u_magnitude.vector().array().max())

# Save solution to file in VTK format
File('elasticity/displacement2.pvd') << u
File('elasticity/von_mises2.pvd') << von_Mises
File('elasticity/magnitude2.pvd') << u_magnitude

# Hold plot
interactive()

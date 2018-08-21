#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 06:32:40 2018

@author: imaneguetni
"""


from __future__ import print_function
from fenics import *
import matplotlib

import matplotlib.pyplot as plt
from dolfin import *



#################################################
##############      Domain   ####################
#################################################

L1= 1 # Largeur
L2= 1 # Hauteur

G=1.e9
lamda=1

mesh = UnitSquareMesh(L1,L2)
n = FacetNormal(mesh)
#################################################
##########    Conditions initiales   ############
#################################################

u0=Constant(0,0,0)
p0=Constant(0,0,0)
us0=Constant(0,0,0)


#time step
dt=0.01

#################################################
##########    Conditions limites     ############
#################################################
tol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < tol
def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < tol
def up_boundary(x, on_boundary):
    return on_boundary and abs(x[1]) < tol
def down_boundary(x, on_boundary):
    return on_boundary and abs(x[1]-1) < tol


#################################################
##########    Fonctions     ############
#################################################
def epsilon(us):
    return 0.5*(nabla_grad(us) + nabla_grad(us).T)
    #return sym(nabla_grad(u))

def sigma(u):
    return lamda*nabla_div(u)*Identity(d) + 2*G*epsilon(u)
#################################################
##########    Fonctions  Spaces    ############
#################################################
order = 1
BDM = FunctionSpace(mesh, "Brezzi-Douglas-Marini", order)
DG = FunctionSpace(mesh, "Discontinuous Lagrange", order - 1)
CG1 = FunctionSpace(mesh, "CG", 1)
mixed_space = MixedFunctionSpace([BDM, DG, DG])


V  = TestFunction(mixed_space)
dU  = TrialFunction(mixed_space)
U=Function(mixed_space)
U0=Function(mixed_space)

#test functions
q, v, w = split(V)
us, p, s = split(U)
us0, p0, s0 = split(U)


#################################################
##########    Probleme variationel     ##########
#################################################
F1= 



















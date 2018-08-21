from fenics import *
import numpy as np

# Parameters
c = 1.25
l = 5.0 
P_surf = 0.0 
sigma_0 = 3.0 
gamma = 0.65 
P0 = sigma_0*gamma 
Kv = 10.0 
cm = 5.0 
# Derivative of pressure at the bottom surface.
der_p_bottom = 0.0
# Derivative of displacement at the top surface.
der_w_top = cm*gamma*sigma_0

# The initial time.
t = 0.0
# The final time
T = 2.0
# The number of time steps to take.
num_steps = 40
# The step size
dt = (T-t)/num_steps
# Width of the porous medium (y-axis length).
width = 5.0
# Thickness of the porous medium (z-axis length).
thickness = 1.0

# Create mesh and define function space.
# Mesh for the 2-D case.
mesh = RectangleMesh(Point(0.0,0.0),Point(l,width),100,10)
# Use a scalar-valued element for pressure.
P1 = FiniteElement('P', triangle, 1)
# Use a vector-valued element for displacement.
P2 = VectorElement('P', triangle, 1)
# Create a mixed element to solve the system using the fully-implicit method.
element = MixedElement(P1, P2)
# Create the function space corresponding to this mixed element.
V = FunctionSpace(mesh,element)

# Define boundary conditions for pressure.
upper_boundary = 'on_boundary && near(x[0],0)'

# An expression for the derivative of P at the bottom surface.
der_p_D = Expression('near(x[0],l) ? der_p_bottom : 0.0',degree = 1, \
    der_p_bottom=der_p_bottom, l=l)

# Set the pressure at the porous medium's top to 0 (first subsystem of V).
bc_p = DirichletBC(V.sub(0),Constant(P_surf),upper_boundary)

# Define boundary conditions for displacements.
bottom_boundary = 'on_boundary && near(x[0],%s)'%(l)

der_w_D = Expression(('near(x[0],0.0) ? der_w_top : 0.0','0.0'),degree=0, \
    der_w_top=der_w_top)

# Set the displacement at the porous medium's bottom to 0 (first dimension of
# the second subsystem of the FunctionSpace V).
bc_w = DirichletBC(V.sub(1).sub(0),Constant(0.0),bottom_boundary)

# Combine the Dirichlet boundary conditions.
bc = [bc_p, bc_w]

# Create the test functions.
v = TestFunction(V)
(v_1, v_2) = split(v)

# Define the initial conditions and previous time step values.
# The previous step's solution.
u_n = Function(V)
# Split the mixed function.
(p_n, w_n) = split(u_n)
# Create and assign the initial conditions.
u_init = Constant((P0,0.0,0.0))
u_n.interpolate(u_init)
# Create the current solution.
u_ = Function(V)
(p_, w_) = split(u_)

# Define the variational problem.
# Fluid Diffusion Equation
Eq1 = dot(p_,v_1)*dx  - dot(p_n,v_1)*dx \
    +c*dt*inner(grad(p_),grad(v_1))*dx \
    -c*dt*der_p_D*v_1*ds

# Force Equilibrium Equation
Eq2 = inner(grad(w_),grad(v_2))*dx \
    -dot(der_w_D,v_2)*ds \
    +cm*dot(grad(p_),v_2)*dx

# Add the equations to make a monolithic scheme.
F = Eq1 + Eq2

# Create the progress bar.
progress = Progress('Time-stepping')
set_log_level(PROGRESS)

# Provide the Jacobian form.
J = derivative(F,u_)
# Create the Newton solver.
problem = NonlinearVariationalProblem(F, u_, bc, J)
solver = NonlinearVariationalSolver(problem)
prm = solver.parameters.newton_solver
prm.error_on_nonconvergence = False

# Form and solve the linear system.
for n in range(num_steps):

    # Update current time.
    t += dt

    # Solve the variational problem.
    solver.solve() # for use with the NonlinearVariationalProblem structure.

    # Update the previous solution.
    u_n.assign(u_)

    # Update the progress bar.
    progress.update(t/T)
    
vtkfile = File('results.pvd')
vtkfile << u

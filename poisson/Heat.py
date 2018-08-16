from fenics import *
import time

T = 2.0            # final time
num_steps = 50    # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
nx = ny = 30
mesh = RectangleMesh(Point(-2, -2), Point(4, 4), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary


# Define initial value
u_0 = Constant(1.0)

tol = 1E-14
def left_boundary(x, on_boundary):
    return on_boundary and abs(x[0]) < tol

def right_boundary(x, on_boundary):
    return on_boundary and abs(x[0]-1) < tol


tol = 1E-14
def up_boundary(x, on_boundary):
    return on_boundary and abs(x[1]) < tol

def down_boundary(x, on_boundary):
    return on_boundary and abs(x[1]-1) < tol


Gamma_0 = DirichletBC(V, Constant(100.0), left_boundary)
Gamma_1 = DirichletBC(V, Constant(10000.0), right_boundary)
Gamma_3 = DirichletBC(V, Constant(100.0), down_boundary)
Gamma_4 = DirichletBC(V, Constant(10000.0), up_boundary)
bc = [Gamma_0, Gamma_1,Gamma_3, Gamma_4]

u_n = interpolate(u_0, V)

def q(u):
    return 1/(u**2+1)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(10.)

#F = q(u)*u*v*dx - dt*u*dot(grad(u), grad(v))*dx - q(u_n)*u_n*v*dx

F = u*v*dx - dt*dot(grad(u), grad(v))*dx - q(u_n)*v*dx
#a, L = lhs(F), rhs(F)

u_ = Function(V)      # the most recently computed solution
F  = inner(q(u)*nabla_grad(u), nabla_grad(v))*dx
F  = action(F, u_)

J  = derivative(F, u_, u)   # Gateaux derivative in dir. of u# Create VTK file for saving solution
vtkfile = File('results.pvd')


# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    #solve(a == L, u, bc)
    problem = NonlinearVariationalProblem(F, u_, bc, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.solve()
    # Save to file and plot solution
    vtkfile << (u, t)
    vtkfile << (u_, t)

    plot(u)
    plot(u_)
    # Update previous solution
    u_n.assign(u)

# Hold plot
interactive()
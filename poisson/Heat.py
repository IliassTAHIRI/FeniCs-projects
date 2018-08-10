from fenics import *
import time

T = 2.0            # final time
num_steps = 50     # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
nx = ny = 30
mesh = RectangleMesh(Point(-2, -2), Point(2, 2), nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), boundary)

# Define initial value
u_0 = Expression('exp(-a)',
                 degree=2, a=5)
u_n = interpolate(u_0, V)

def q(u):
    return u

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(10)

F = q(u)*u*v*dx - dt*u*dot(grad(u), grad(v))*dx - q(u_n)*u_n*v*dx
#a, L = lhs(F), rhs(F)

u_ = Function(V)      # the most recently computed solution
F  = inner(q(u)*nabla_grad(u), nabla_grad(v))*dx
F  = action(F, u_)

J  = derivative(F, u_, u)   # Gateaux derivative in dir. of u# Create VTK file for saving solution
vtkfile = File('heat_gaussian/s.pvd')
vtkfile2 = File('heat_gaussian/solution2.pvd')

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

    # Update previous solution
    u_n.assign(u)

# Hold plot
interactive()
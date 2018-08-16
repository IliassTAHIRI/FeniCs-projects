from fenics import *
import time
import mshr
import ufl
import os
from math import floor, ceil
get_ipython().magic('matplotlib inline')


# In[2]:


dt = 0.005                                     # Time Step
k = Constant(dt)                               # Time Step
t_end = 3.0                                    # Total simulation time
theta = Constant(0.5)                                    # Interpolation schema
g = Constant((0.0,-0.98))                      # Gravity
rho1 = 1000.0                                  # Surround density
rho2 = 100.0                                   # Bubble density
nu1 = 10.0                                     # Surround viscosity
nu2 = 1.0                                      # Bubble viscosity
sigma = 24.5


# In[3]:


Dx = 100
Dy = 2*Dx
mesh = RectangleMesh(Point(0.0,0.0), Point(1.0,2.0), Dx, Dy,'crossed')


# In[4]:


beta=0.4
epsilon = (beta*((1.0/Dx)**(0.9)))


# In[5]:


center = Point(0.5,0.5)
radius = 0.25
phi = Expression('1.0/(1.0+exp((sqrt((x[0]-A)*(x[0]-A) + (x[1]-B)*(x[1]-B))-r)/(eps)))',degree=2, eps=epsilon, A=center[0], B=center[1],r=radius)


# In[6]:


V = VectorElement("Lagrange", mesh.ufl_cell(), 2) # Velocity vector field
P = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # Pressure field
L = FiniteElement("Lagrange", mesh.ufl_cell(), 2) # Levelset field
N = VectorFunctionSpace(mesh, "CG", 1, dim=2)     # Normal vector field
VP = MixedElement([V,P])
W = FunctionSpace(mesh,VP)


# In[7]:


I = Identity(V.cell().geometric_dimension())   # Identity tensor
n = FacetNormal(mesh)
h = CellSize(mesh)                             # Mesh size


# In[8]:


bcs = list()
bcs.append(DirichletBC(W.sub(0),Constant((0.0,0.0)),"near(x[1],0.0)||near(x[1],2.0)"))
bcs.append(DirichletBC(W.sub(0).sub(0),Constant(0.0),"near(x[0],0.0)||near(x[0],1.0)"))


# In[9]:


w = Function(W); w0 = Function(W)
v,p = split(w); v0,p0 = split(w0)


# In[10]:


SL = FunctionSpace(mesh,L);
l = Function(SL); l0 = Function(SL)
l.assign(interpolate(phi,SL)); l0.assign(interpolate(phi,SL))


# In[11]:


def delta(l):
    grad_phi = project(grad(l),N)
    return(sqrt(dot(grad_phi,grad_phi)))

def rho(l):
    return(rho1+(rho2-rho1)*l)

def nu(l):
    return(nu1+(nu2-nu1)*l)


# In[12]:



def NS(v,p,l,v_):
 grad_phi = project(grad(l),N)
 nls = grad_phi/sqrt(dot(grad_phi,grad_phi))
 Ts = delta(l)*sigma*(I-outer(nls,nls))   
 T = -p*I + nu(l)*(grad(v)+grad(v).T)                                             
 return(inner(T,grad(v_))*dx + rho(l)*inner(grad(v)*v,v_)*dx - rho(l)*inner(g,v_)*dx + inner(Ts,grad(v_))*dx)
#def NS(v,p,l,v_):
#    grad_phi = project(grad(l),N)
#    mgrad=sqrt(dot(grad_phi,grad_phi))
#    nls = grad_phi/mgrad
#    ft = sigma*div(nls)*grad_phi
#    Ts = delta(l)*sigma*(I-outer(nls,nls))   
#    T = -p*I + nu(l)*(grad(v)+grad(v).T)                                             
#    return(inner(T,grad(v_))*dx + rho(l)*inner(grad(v)*v,v_)*dx - rho(l)*inner(g,v_)*dx + inner(ft,v_)*dx)


# In[13]:


def navier_stokes():
   
    v,p = split(w); v0,p0 = split(w0)   
    v_,p_ = TestFunctions(W)
   
    F = inner((rho(l)*v-rho(l0)*v0)/k,v_)*dx + theta*NS(v,p,l,v_) + (1.0-theta)*NS(v0,p,l0,v_) + div(v)*p_*dx
   
    begin("navier_stokes")
    J = derivative(F,w)
    problem=NonlinearVariationalProblem(F,w,bcs,J)
    solver=NonlinearVariationalSolver(problem)
    solver.solve()
    end()
   
    return(w)


# In[14]:


alpha=Constant(1.0)
def IP(l,l_):
    h_avg = (h('+') + h('-'))/2.0  
    r = alpha('+')*h_avg*h_avg*inner(jump(grad(l),n), jump(grad(l_),n))*dS
    return (r)
       
def LS(l,v,l_):
    return(inner(v,grad(l))*l_*dx)


# In[15]:


def level_set():
   
    v,p = split(w); v0,p0 = split(w0)   
    l_ = TestFunction(SL)
   
    F = inner((l-l0)/k,l_)*dx + theta*LS(l,v,l_) + (1.0-theta)*LS(l0,v0,l_) + IP(l,l_)
    bc = []
   
    begin("level_set")
    J = derivative(F,l)
    problem=NonlinearVariationalProblem(F,l,bc,J)
    solver=NonlinearVariationalSolver(problem)
    solver.solve()
    end()
   
    return(l)


# In[16]:


def reinit(l,epsilon,beta,Dx,mesh): 
   
    # time-step
    dtau = Constant(1.0/(beta*((1.0/Dx)**(1.1))))
   
    # space definition
    V = VectorFunctionSpace(mesh, "CG", 1, dim=2)
    FE = FunctionSpace(mesh, "CG", 2)
   
    # functions setup
    phi = Function(FE); phi0 = Function(FE)
    w = TestFunction(FE)
   
    # intial value
    phi0.assign(interpolate(l,FE))
   
    # Unit normal vector (does not change during this process)
    grad_n = project(grad(l),V)
    n = grad_n/(sqrt(dot(grad_n,grad_n)))
 
    # FEM linearization
    F = dtau*(phi-phi0)*w*dx - 0.5*(phi+phi0)*dot(grad(w),n)*dx + phi*phi0*dot(grad(w),n)*dx +         epsilon*0.5*dot(grad(phi+phi0),n)*dot(grad(w),n)*dx
   
    # Newton-Raphson parameters
    bc = []
   
    E = 1e10; E_old = 1e10
    cont = 0; num_steps = 10   
    for n in range(num_steps):
   
        begin("Reinitialization")
        solve(F == 0, phi, bc)
        end()
        error = (((phi - phi0)*dtau)**2)*dx
        E = sqrt(abs(assemble(error)))
        fail = 0
        if (E_old < E ):
            fail = 1
            print('fail',"at:", cont) 
            break
       
        phi0.assign(phi)
       
        cont +=1
        E_old = E
       
    #print("Error:", E, "nincre", cont)
    return phi


# In[17]:


vfile = File("P1_IP_S1_1/velocity.pvd")
pfile = File("P1_IP_S1_1/pressure.pvd")
lfile = File("P1_IP_S1_1/level.pvd")


# In[18]:


Vc = VectorFunctionSpace(mesh,"CG",2)
R = VectorFunctionSpace(mesh,"R",0,dim=2)

position = Function(Vc)
position.assign(Expression(["x[0]","x[1]"], element=Vc.ufl_element()))
c = TestFunction(R)


# In[19]:


out_dt = 0.1; count = 0
t = dt
if __name__ == "__main__":
    print("P1_IP_S1_1")
    print("IP", "alpha",alpha, "inR",10, "beta", beta)
    while t < t_end:
        
        for problem in [navier_stokes, level_set]:
            if (problem == navier_stokes):
                w = problem()
            else:
                l1 = problem()
           
        l2 = reinit(l1,epsilon,beta,Dx,mesh)
        l.assign(interpolate(l2,SL))
       
        # Extract solutions
        w0.assign(w); l0.assign(l)
        v,p = w.split()
        # Mass
        V=assemble(conditional(gt(l,0.5),1.0,0.0)*dx) 
            
        # Center of mass
        volume = assemble(conditional(ge(l,0.5),Constant(1.0),0.0)*dx)
        centroid = assemble(conditional(ge(l,0.5),dot(c,position),0.0)*dx)
        centroid /= volume
        xc = centroid[0][0]; yc = centroid[1][0]
            
        # Velocity
        u_c = assemble(conditional(ge(l,0.5),dot(c,v),0.0)*dx)
        u_c /= volume
        vxc = u_c[0][0]; vyc = u_c[1][0]
        #myfile3.write('%e %e' '\n'  % (vyc, t))
        print("%e %e %e %e" %(V, yc, vyc, t))

        # Save solution   
        if (t >= float(count)*out_dt):
            count+=1
            v.rename("velocity", "velocity")
            p.rename("pressure", "pressure")
            l.rename("level", "level")
            vfile << v
            pfile << p
            lfile << l
       
        t += dt

    v.rename("velocity", "velocity")
    p.rename("pressure", "pressure")
    l.rename("level", "level_set")
    vfile << v
    pfile << p
    lfile << l
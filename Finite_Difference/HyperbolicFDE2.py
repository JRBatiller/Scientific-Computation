# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 21:59:07 2017

@author: Joby
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 19:24:15 2017

@author: Joby
"""
import numpy as np
import pylab as pl
import scipy as sp
from numpy import pi
from scipy.sparse import spdiags

"""
Initialization Functions
"""
#Modify these as needed
def u_exact(x, c, L, T, n=0):
    #I put the solutions on a list to make it easy to switch around
    ans=[None]*4
    ans[0] = np.cos(np.pi*c*T/L)*np.sin(np.pi*x/L)
    ans[1] = (L/(pi*c))*np.sin(pi*c*T/L)*np.sin(pi*x/L)
    ans[2] = np.cos(pi*c*T/L)*np.cos(pi*x/L)
    ans[3] = (L/(pi*c))*np.sin(pi*c*T/L)*np.cos(pi*x/L)
    return ans[n]

def u_I(x, c, L, T):
    # initial wave distribution
    #y = np.array([np.sin(pi*x/L), 0*x])
    #y = np.array([0*x, np.sin(pi*x/L)])
    #y=np.array([np.cos(pi*x/L), 0*x])
    y=np.array([0*x,np.cos(pi*x/L)])
    return y

def gauss(x,c,L,T):
    alpha=0.5
    sigma=1
    y=np.array([alpha*np.exp(-(x-4)**2/sigma**2),-2*x*alpha*np.exp(-(x-4)**2/sigma**2)/sigma**2])
    return y

def zero_bound(u,t):
    #simplest zero boundary
    u[0]=0  #left neumann bound
    u[-1]=0 #right neumann bound
    return u

def boundary_condition(u,t):
    #takes a vector and adds boundary condition
    u[0]=0 #left neumann bound
    u[-1]=0 #right neumann bound
    return u

def no_source(x,t):
    #wave source is some function f(x,t)
    f=x*0
    return f

def q_init(x):
    alpha=0.5
    sigma=1
    q=1-alpha*np.exp(-(x-15)**2/sigma**2)
    return q

"""
Matrix Generating Functions
"""

def tridiag(v,n):
    #returns sparse tridiagonal array of size n given vector v
    v=np.tile(v,(n,1)).transpose()
    diags=np.array([-1,0,1])
    #A=spdiags(v,diags, n,n).toarray()
    A=spdiags(v,diags, n,n)
    return A

def aew(ld, n):
    v=np.array([ld**2, 2-2*ld**2, ld**2])
    A=tridiag(v,n)
    return A

def aiw(ld, n):
    v=np.array([-0.5*ld**2,1+ld**2, -0.5*ld**2 ])
    A=tridiag(v,n)
    return A

def biw(ld,n):
    v=np.array([0.5*ld**2,-1-ld**2, 0.5*ld**2])
    A=tridiag(v,n)
    return A

def q_matrix(ld, n, x, dx, dt, Q=q_init):
    A=tridiag([ld**2,-ld**2,0],n).toarray()
    B=tridiag([0,-ld**2,ld**2],n).toarray()
    Q_less=Q(x-dx/2)
    Q_plus=Q(x+dx/2)
    A=A*Q_less
    B=B*Q_plus
    C=tridiag([0,2,0],n)
    C=A+B+C
    C=sp.sparse.dia_matrix(C)
    return dt**2*C/dx**2
    

"""
Finite Diff schemes
"""
def explicit(ld, n, *args):
    A=sp.sparse.identity(n)
    B=aew(ld, n)
    return A, B/2 
    #yes B/2 is a slight cheat to make everything work

def q_explicit(ld,n,x,dx, dt):
    A=sp.sparse.identity(n)
    B=q_matrix(1,n,x,dx, dt)
    return A, B/2

def implicit(ld,n, *args):
    A=aiw(ld,n)
    B=sp.sparse.identity(n)
    #C=biw(ld,n)
    return A, B

"""
Finite Diff Solvers
"""

def fde_hyperbolic(c,L,T,xinit,mx, mt, bdry, scheme=explicit, neumann=0, w_source=no_source):
    #the main solver
    xs=np.linspace(0, L, mx+1)
    ts=np.linspace(0,T,mt+1)
    dx = xs[1] - xs[0]            # gridspacing in x
    dt = ts[1] - ts[0]            # gridspacing in t
    ld = c*dt/dx                  # courant number
    print("lambda=",ld)
    u0=xinit(xs, c, L, T) #initialize u0
    matrix=scheme(ld,mx+1, x, dx, dt) #this gives you the left and right matrixes 
    if neumann:
        u=neumann_solve(u0,xs,ts,ld,matrix,bdry, w_source)
    else:
        u=dirichlet_solve(u0,xs,ts,ld,matrix,bdry, w_source)
    return u

def dirichlet_solve(u0,xs,ts,ld,mtrx,bdry, F):
    dt=ts[1]-ts[0]
    dx=xs[1]-xs[0]
    A,B=mtrx
    A=truncate_matrix(A) #make matrix a two sizes smaller
    B=truncate_matrix(B)
    C=-A
    u, v=u0              #seperate displacement and velocity
    addends=np.zeros(A.shape[0])
    u=np.tile(u, (3,1))  #past[0] , present[1], future[2]   
    explicit=is_identity(A)
    #FIRST STEP
    if(explicit):
        addend=ld**2*bdry(addends,ts[1])+dt**2*F(xs[1:-1],ts[1])
    else:
        addend=0.5*ld**2*(bdry(addends,ts[1]+dt)+bdry(addends,ts[1]-dt))
        addend=addend+0.5*dt**2*(F(xs[1:-1],ts[1]+dt)+F(xs[1:-1],ts[1]-dt))
    rhs=B.dot(u[0][1:-1])-dt*C.dot(v[1:-1])+addend
    u[1][1:-1]=sp.sparse.linalg.spsolve(A,rhs)
    u[1]=bdry(u[1],ts[1])    
    for t in ts[2:]:
        if(explicit):
            addend=ld**2*bdry(addends,t)+dt**2*F(xs[1:-1],t)
        else:
            addend=0.5*ld**2*(bdry(addends,t+dt)+bdry(addends,t-dt))
            addend=addend+0.5*dt**2*(F(xs[1:-1],t+dt)+F(xs[1:-1],t-dt))
        rhs=2*B.dot(u[1][1:-1])+C.dot(u[0][1:-1])+addend
        u[2][1:-1]=sp.sparse.linalg.spsolve(A,rhs)
        u[2]=bdry(u[2], t)
        u[:2]=u[1:] #copy array down
    return u[2]

def neumann_solve(u0, xs, ts, ld, mtrx, nbdry, F):
    dt=ts[1]-ts[0]
    dx=xs[1]-xs[0] 
    A,B=mtrx
    A=modify_evo(A) #multiply proper elements by 2
    B=modify_evo(B)
    C=-A
    u, v=u0
    addends=np.zeros(A.shape[0])
    u=np.tile(u, (3,1)) #past[0] , present[1], future[2]   
    explicit=is_identity(A)
    #FIRST STEP
    if(explicit):
        addend=2*dx*ld**2*modify_bound(nbdry(addends,ts[1]))+dt**2*F(xs,ts[1])
    else:
        addend=dx*ld**2*(modify_bound(nbdry(addends,ts[1]+dt))+modify_bound(nbdry(addends,ts[1]-dt)))
        addend=addend+0.5*dt**2*(F(xs,ts[1]+dt)+F(xs,ts[1]-dt))
    rhs=B.dot(u[0])-dt*C.dot(v)+addend
    u[1]=sp.sparse.linalg.spsolve(A,rhs)
    for t in ts[2:]:
        if(explicit):
            addend=2*dx*ld**2*modify_bound(nbdry(addends,t))+dt**2*F(xs,t)
        else:
            addend=dx*ld**2*(modify_bound(nbdry(addends,t+dt))+modify_bound(nbdry(addends,t-dt)))
            addend=addend+0.5*dt**2*(F(xs,t+dt)+F(xs,t-dt))
        rhs=2*B.dot(u[1])+C.dot(u[0])+addend
        u[2]=sp.sparse.linalg.spsolve(A,rhs)
        u[:2]=u[1:] #copy array down
    return u[2]

def modify_bound(brdr):
    #this changes the boundary to negative
    brdr[0]=-brdr[0]
    return brdr

def truncate_matrix(A):
    #this gets the inner matrix for dirichlet
    A=A.toarray()
    A=A[1:-1,1:-1]
    A=sp.sparse.dia_matrix(A)
    return A

def modify_evo(A):
    #this multiplies the appropriate elements by 2
    A=A.toarray() #so it can handle sparse matrixes
    A[0,1]=2*A[0,1]
    A[-1,-2]=2*A[-1,-2]
    A=sp.sparse.dia_matrix(A)
    return A

def is_identity(A):
    #to find out if matrix is an identity matrix
    B=np.identity(A.shape[0])
    ans=(A.toarray()==B) #can't get not equals to work properly
    return ans.all()   

if __name__ == "__main__":
    #initialize values here
    c=1         # diffusion constant
    L=20         # length of spatial domain
    T=0.1   # total time to solve for
    mx = 100     # number of gridpoints in space
    mt = 1000   # number of gridpoints in time
    x=np.linspace(0, L, mx+1)
    t=np.linspace(0, T, mt+1)
    u=fde_hyperbolic(c,L,T,u_I, mx, mt, boundary_condition, explicit, 1)
    #u=fde_hyperbolic(c,L,T,u_I, mx, mt, boundary_condition, implicit,1)
    #u=fde_hyperbolic(c,L,T,gauss, mx, mt, boundary_condition, q_explicit, 0)
    #exact=u_exact(x,c,L,T, 3)
        
    
    fig1=pl.figure(figsize=(10,5))
    ax1=fig1.add_subplot(1,1,1)
    #ax1.plot(x,u, 'r-', x, exact,'go') #to plot u and exact
    ax1.plot(x,u)
    #ax1.plot(x,0-q_init(x))
    ax1.set_xlabel("x")
    ax1.set_ylabel("u")
    ax1.grid(1)
    fig1.savefig('variable wave.svg')
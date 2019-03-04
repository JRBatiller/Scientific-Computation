# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:06:57 2017

@author: Joby
"""

"""
Overall aim
Build a software package that performs numerical continuation on a given ODE
with respect to a given parameter.
"""
"""
Steps to get there
This is an exercise in defining appropriate interfaces and abstractions. Here you
will need to join together the different pieces of code that you have produced
over the last two weeks. In principle you could simply cut and paste all the code
together as one big function and use if statements, however, this would be poor
program design. Ideally, your program should have a modular structure with
the ability to use different discretisations.

An example top-level interface could be
results = continuation(myode, # the ODE to use
                       x0, # the initial state
                       par0, # the initial parameters
                       vary_par=0, # the parameter to vary
                       step_size=0.1, # the size of the steps to take
                       max_steps=100, # the number of steps to take
                       discretisation=shooting, # the discretisation to use
                       solver=scipy.optimize.fsolve) # the solver to use

Make (lots of) notes in your code about the decisions you are making! This is
particularly important this week!
For each of the exercises, consider in turn the following problems.



• The Duffing equation x¨ + 2ξx˙ + kx + βx3 = Γ sin(πt) with ξ = 0.05, β = 1,
and Γ = 0.5 and vary k between 0.1 and 20.
1. Write a code that performs natural parameter continuation, i.e., it simply
increments the a parameter by a set amount and attempts to find the
solution for the new parameter value using the last found solution as an
initial guess.
Write a code that performs pseudo-arclength continuation, i.e., the pseudoarclength
equation is added to the system of equations to solve for.

• Use your code to investigate the behaviour of the Duffing equation
for different values of Γ, e.g., 0.1, 0.2, 0.4, 0.8, etc.
Extensions

At certain points your code might fail because the step size is too large, but
if the step size is set to be too small the computations will take a long time.
Adaptive step sizes are a way to deal with this. A simple strategy is as follows.
If the root finder fails to converge, try again with half the step size; repeat
until the root finder converges and you can carry on as normal, or until you hit
a prespecified minimum step size (e.g. 0.000001) at which point end with an
error. If the root finder converges, increase the stepsize by 20% until you hit a
prespecified maximum step size
"""

import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fsolve

"""
ODE LIST
"""

"""
• The algebraic cubic equation 
x3 − x + c = 0. In this case a discretisation
(shooting or collocation) is not needed (in the interface above you might 
have an option discretisation=lambda x: x, i.e., the equations are just
passed straight through to the solver). Vary c between -2 and 2.
"""
def simple_cubic(x, t, pars, *args):
    k=pars[0]
    y=(x[0]**3)-x[0]+k
    return y

"""
• The mass-spring-damper equation x¨ + 2ξx˙ + kx = sin(πt) (note that the
period is always 2 and can be defined on the domain [0, 2] or [−1, 1]
equivalently). Use ξ = 0.05 and vary k between 0.1 and 20.
"""
def mass_spring(u,t,pars, *args):
    k, epsilon, gamma=pars
    omega=np.pi #for now
    du=[u[1],gamma*np.sin(omega*t)-2*epsilon*u[1]-k*u[0]]
    return du

"""
SHOOTING FUNCTIONS
"""
def shooting(ode, x0, time, pars, solver):
    data =ode, time, pars
    x, infodict, ier, mesg=solver(start_end_diff, x0, args=data, full_output=1 )
    if(ier==1):
        print(mesg)
        print("The roots are {}".format(x))
        return x # This returns array of same shape as x0 This will be a problem
    else:
        print(mesg)
        return nan

def start_end_diff(u,f, time, pars, *args):
    #this shall be our function to zero
    us=odeint(f,u,time, args=(pars,))
    f=u-us[-1]
    return f

"""
CONTINUATION Functions
"""

def dot_prod(z, oldz, ds, *args):
    #NO MORE ELEGANCE I WILL MAKE THIS WORK
    k,x = z #THIS IS JUST A GUESS
    k0, x0 =oldz[0]
    k1, x1 =oldz[1]
    
    estk,estx=predictor(oldz,ds)
    v1=np.array([k-estk, x-estx])
    v2=np.array([k1-k0, x1-x0])
    y=np.dot(v1,v2)
    return y

def F(z, time, pars,index, oldz,ds, f,disc, solver, *args):
    k,*x=z
    pars[index]=k
    ans=np.zeros(2)
    #ans[0]=f(x,pars)
    ans[0]=G(x, time, pars, f, disc, solver)
    ans[1]=dot_prod(z,oldz, ds)
    return ans

def G(x, time, pars, f, disc, solver):
    ans=disc(f(x,time, pars, solver))
    #ans=disc(f, x,time, pars, solver)
    return ans

def predictor(z, ds, *args):
    k0, *x0=z[0]
    k1, *x1=z[1]
    x2=[]
    #x2=x1+(x1-x0)*ds
    k2=k1+(k1-k0)*ds
    for a in range(len(x0)):
        x2.append(x1[a]+(x1[a]-x0[a])*ds)
    
    z2=np.append(k2, x2)
    return z2
    
def corrector(f, z, time, pars, index, oldz, ds,disc, solver,*args):
    data= time, pars,index, oldz, ds, f, disc, solver
    znew=solver(F, z, args=data)
    return znew

def initialize(f,z0, time, pars, index, step_size, disc, solver):
    #makes first 2 steps
    z=np.zeros((2, len(z0)))
    k, *x0=z0
    for a in range(2):
        pars[index]=k
        data=time, pars, f, disc, solver
        xnew, info, ier, mesg = solver(f,x0,args=data, full_output=1)
        #xnew=G(x0, time, pars, f, disc, solver)
        z[a]=[k,*xnew]
        k+=step_size    
    return z

def continuation(myode,x0,par0,vary_par=0,step_size=0.1,max_steps=100,disc=shooting,solver=sp.optimize.fsolve):
    time, end = par0.pop(-1) #get end point
    z0=np.array([par0[vary_par],*x0])
    #Generate initial 2 points
    z=initialize(myode, z0, time, par0, vary_par, step_size, disc, solver)
    
    ds=1 # this used to be called step size but now its just ds
    runs=2
    while z[-1][0]<end and runs<=max_steps:
        prevz=z[[-2,-1]] #gets last two points
        predz=predictor(prevz,ds) # predicts next point
        nextz=corrector(myode, predz, time, par0,vary_par,prevz,ds,disc,solver) #finds where it intersects graph perpendicularly
        z=np.vstack((z,nextz)) #add it to z
        runs+=1
    print("It took {} steps".format(runs))
    return z

    
if __name__ == "__main__":

    #Initialize here
    start, end=-2, 2 #for parameter to vary
    tstart = -1 #TIME DOMAIN 
    T=2 # period of the function
    x0=np.array([1]) #initial guess for x
    k=start
    par0=[k] #Initialize parameters here
    timepar=[[tstart,tstart+T], end]
    par0.append(timepar) #lets pack period in here cause lazy
    vary_par=0  # index of varied parameter
    step_size=0.1
    
    results = continuation(simple_cubic,x0,par0,vary_par,step_size,100,lambda x: x,fsolve)
    #NOTE! This produces a matrix in the form [k,x]
    ks=results[:,0]
    xs=results[:,1]
    #plt.plot(ks,xs)
    #FUCK YEAH IT WORKS

    fig1=plt.figure(figsize=(10,5))
    ax1=fig1.add_subplot(1,1,1)
    ax1.set_xlabel("k")
    ax1.set_ylabel("||x||")
    #plt.title("Duffing")
   
    ax1.scatter(ks,xs)
    #fig1.savefig('simple cubic x vs k.svg')
    
        

    
    """
    k=0.1
    end=20
    par0=[k, 0.05, 1]
    timepar=[[tstart,tstart+T], end]
    x0=np.array([1,0])
    par0.append(timepar) #lets pack period in here cause lazy
    results = continuation(mass_spring,x0,par0,vary_par,step_size,100,shooting,fsolve)
    plt.plot(*results.transpose())
    """


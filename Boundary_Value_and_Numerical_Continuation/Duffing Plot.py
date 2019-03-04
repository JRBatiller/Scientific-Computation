# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 22:58:55 2017

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
• The Duffing equation x¨ + 2ξx˙ + kx + βx3 = Γ sin(πt) with ξ = 0.05, β = 1,
and Γ = 0.5 and vary k between 0.1 and 20.
"""
def duffy(u,t, pars, *args):
    k, epsilon, gamma, beta=pars
    omega=np.pi
    du=[u[1], gamma*np.sin(omega*t)-2*epsilon*u[1]-k*u[0]-beta*u[0]**3]
    return du    
"""
• Use your code to investigate the behaviour of the Duffing equation
for different values of Γ, e.g., 0.1, 0.2, 0.4, 0.8, etc.
Extensions
"""

"""
SHOOTING FUNCTIONS
"""
"""
def shooting(ode, x0, time, pars):
    data =ode, time, pars
    x, infodict, ier, mesg=fsolve(start_end_diff, x0, args=data, full_output=1 )
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
    ans=u-us[-1]
    return ans
"""

def shooting(ode, z0, pars, index, oldz=[], ds= 1, mode="NPC", *args):
    data =ode, pars, index,  oldz, ds, mode
    z, infodict, ier, mesg=fsolve(start_end_diff, z0, args=data, full_output=1 )
    if(ier==1):
        #print(mesg)
        #print("The roots are {}".format(x))
        return z # This returns array of same shape as x0 This will be a problem
    else:
        #print(mesg)
        return np.tile(float('Nan'),len(z0))

def start_end_diff(z,f, pars, index, oldz, ds, mode="NPC", *args):
    #this shall be our function to zero
    k, t, *x = z
    #pars[index]=k
    
    if mode=="NPC":
        xs=odeint(f,x,[t,t+2], args=(pars,))
        diff=xs[0]-xs[-1]
        ans=np.append(diff,x[1] )
        return np.append(ans,k-pars[index])
    else:
        pars[index]=k
        xs=odeint(f,x,[t,t+2], args=(pars,))
        diff=xs[0]-xs[-1]
        ans=np.append(diff,x[1])
        ans=np.append(ans, is_tan(z,oldz,ds))
        return ans
        

"""
CONTINUATION Functions
"""
def islambda(v):
    LAMBDA = lambda:0
    return isinstance(v, type(LAMBDA)) and v.__name__ == LAMBDA.__name__

def integrate(f, x0, time, pars):
    t, T= time
    ts=np.linspace(t, T, 10000)
    xs=odeint(f,x0,ts, args=(pars,))        
    return xs     
    
def predictor(z,ds):
    z0=z[0,:]
    z1=z[1,:]
    z2=z1+(z1-z0)*ds
    return z2
 
def is_tan(z, oldz, ds):
    k,_,x,*_=z
    k0,_,x0,*_ = oldz[0,:]
    k1,_,x1,*_ = oldz[1,:]
    k2,_,x2,*_ = predictor(oldz,ds)
    v1 = np.array([k1-k0,x1-x0])
    v2 = np.array([k-k2,x-x2])
    return np.dot(v1,v2)
    
    """
    z0=oldz[0,:]
    z1=oldz[1,:]
    z2=predictor(oldz,ds)
    return np.dot(z-z2,z1-z0)
    """
def continuation(myode,x0,par0,vary_par=0,step_size=0.1,max_steps=100,disc=shooting,solver=sp.optimize.fsolve):
    end =par0.pop(-1)
    
    ds=1.1
    t0=0
    k=par0[vary_par]
    z0=[k, t0]
    z0=np.append(z0,x0)
    z=np.zeros((2,len(x0)+2))    
    for a in range(2):
        par0[vary_par]=k
        z[a] =initialize(myode, z0, par0, vary_par,  step_size, disc)
        z[a]=abs(z[a])
        k+=step_size
    runs=2
    
    k=z[-1,0]
    while k<=end and runs<=max_steps:
        old_z=z[[-2,-1]]
        par0[vary_par]=k
        pred_z=predictor(old_z,ds)
        next_z=corrector(myode,pred_z, par0, vary_par,old_z, ds, disc)
        if np.isfinite(sum(next_z)):
            z=np.vstack((z, abs(next_z)))
            k=z[-1,0]
            ds=1.1
        else:
            #print("I triggered")
            ds=ds/2
            if ds<1e-6:
                ds=1e-6
            
        runs+=1
        if runs%100==0: 
            print(runs)
    
    
    
    return z

def initialize(f, z0, pars, index, ss, disc):
    
    z=disc(f,z0, pars, index)
    
    return z

def corrector(f,z, pars, index, oldz, ds, disc):
    z= disc(f,z,pars,index,oldz,ds, mode="PAL")
    return z
            
    

if __name__ == "__main__":
    
    """
    Initialize here
    """
    
    
    T=2
    k=0.1
    end=20
    par0=[k, 0.05, 1]
    vary_par=0
    step_size=0.01
    disc=shooting
    x0=np.array([1,0])
    
    
    
    
    par0.append(end)    
    results = continuation(mass_spring,x0,par0,vary_par,step_size,100000,disc,fsolve)
    
    #NOTE! This produces a matrix in the form [k,x]
    ks=results[:,0]
    xs=results[:,2]
    
    fig1=plt.figure(figsize=(6,3))
    ax1=fig1.add_subplot(1,1,1)
    ax1.set_xlabel("lambda")
    ax1.set_ylabel("||x||")
    #plt.title("Duffing")
   
    ax1.scatter(ks,xs)
    #fig1.savefig('Mass Spring.svg')
    
    
    
    
    
    par0=[k, 0.05, 0.8, 1]
    par0.append(end)    
    step_size=0.001
    results2 = continuation(duffy,x0,par0,vary_par,step_size,1000,disc,fsolve)
    
    ks=results2[:,0]
    xs=results2[:,2]
    
    fig2=plt.figure(figsize=(6,3))
    ax2=fig2.add_subplot(1,1,1)
    ax2.set_xlabel("lambda")
    ax2.set_ylabel("||x||")
    #plt.title("Duffing")
    ax2.scatter(ks,xs)
    #fig2.savefig('Duffy parameter neg Beta.svg')
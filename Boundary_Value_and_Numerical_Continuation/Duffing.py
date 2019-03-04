# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 11:03:47 2017

@author: Joby
"""

import numpy as np
import scipy as sp

import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import odeint

"""
• The mass-spring-damper equation x¨ + 2ξx˙ + kx = sin(πt) (note that the
period is always 2 and can be defined on the domain [0, 2] or [−1, 1]
equivalently). Use ξ = 0.05 and vary k between 0.1 and 20.
"""
def mass_spring(u0,t,pars):
    epsilon, gamma, omega=pars
    u=u0
    du=[u[1],gamma*np.sin(omega*t)-2*epsilon*u[1]-u[0]]
    return du


"""
Simulate the nonlinear Duffing equation u¨+2ξ u˙+u+u^3 = Γsin(ωt) 
for ξ = 0.05, Γ = 0.2 and 0.5 ≤ ω ≤ 1.5.
• What behaviour do you see in the long-time limit?
• Isolate a periodic orbit. What are its starting conditions? What is its period?
1
• This will provide testing data for your numerical methods.
– Always test your code against known results or you will come unstuck!
"""
#def dudt(u,t): this is a bad name
#used to have seperate dudt functions combined into one
def duffy(u0,t, pars):
    #takes a vector,returns a vector
    epsilon, gamma, omega =pars
    #yes I can use elements of a list but I like labels
    u=u0
    du=[u[1], gamma*np.sin(omega*t)-2*epsilon*u[1]-u[0]-u[0]**3]
    return du    

def sin1(x, t, par):
      
    return np.sin(np.pi*t) - x

def start_end_diff(u, time, f, pars, *args):
    #this shall be our function to zero
    uinitial=u
    us=odeint(f,uinitial,time, args=(pars,))
    ufinal=us[-1] # this is the last element
    f=uinitial-ufinal
    #df=np.array([f[1],du_dt(uinitial,t,gamma,epsilon)[1]-du_dt(ufinal,t+T,gamma,epsilon)[1]])
    #Ans=np.array([f, df])
    return f

def plot_stuff(f, u, t, pars):
    Us=odeint(f,u,t, args=(pars,))
    us=Us[:,0]    
    dus=Us[:,1]

    fig1=plt.figure(figsize=(5,2.5))
    ax1=fig1.add_subplot(1,1,1)
    ax1.set_xlabel("t")
    ax1.set_ylabel("u")
    #plt.title("Duffing")
   
    ax1.plot(ts,us)
    ax1.plot(ts,dus)
    #fig1.savefig('Duffing Plot 2.svg')
    
    fig2=plt.figure(figsize=(5,2.5))
    ax2=fig2.add_subplot(1,1,1)
    ax2.plot(us,dus)
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Velocity")
    #fig2.savefig('Phase Plot 2.svg')
 
def shooting(ode, x0, T, pars):
    data = T, ode, pars
    x, infodict, ier, mesg=sp.optimize.fsolve(start_end_diff, x0, args=data, full_output=1 )
    if(ier==1):
        print(mesg)
        print("The roots are {}".format(x))
        return x
    else:
        print(mesg)
        return float('Nan')

if __name__ == "__main__":
    epsilon=0.05
    gamma=0.2
    omega=1.2 #Do i just set any value of omega
    
    #Do I just set initial conditions
    U0=np.array([-1, 0]) # max amplitude 0 velocity seem like good starting points
    t0=0
    pars= [epsilon, gamma, omega]
    #WELL SHIT
    Us=odeint(duffy,U0,[0,1000], args=(pars,))
    T=2*np.pi/omega #set period
    
    ts=np.linspace(0, 1000, 1000000)
    # the time spacings have to be very small too. I think the sin screws things up a lot
    #changed time scale to not depend on pi its easier this way
    #ts=np.array([np.pi*z/1000 for z in range(1000000)])
    plot_stuff(duffy, U0, ts, pars)
   
    u=U0
   
    t=t0
    time=[t,t+T]
    x=shooting(duffy,u,time,pars)
    ts=np.linspace(t, t+T, 1000)
    plot_stuff(duffy,x,ts,pars)
    

    x=shooting(mass_spring,u,time,pars)
    ts=np.linspace(t, t+T, 1000)
    plot_stuff(mass_spring,x,ts,pars)
    
    n=20
    T=2
    t=-1
    time=[t,t+T]
    pars= [epsilon, 1, np.pi]
    z=shooting(mass_spring, u, time, pars)
    ts=np.cos(np.pi*np.arange(0, n+1)/n)
    plot_stuff(mass_spring,z,ts,pars)
    

    
    """   
    
    #Isolate a period T
    periods=[]
    for index in range(len(Us)):
        A=Us[index]-Us[750000]
        if(np.sqrt(np.dot(A,A))/2<=0.00005):
            periods.append(index)
    
    print(periods)
    
    for tm in periods:
        print(Us[tm])
    last=0
    for tm in periods[:]:
        print(ts[tm]-last)
        last=ts[tm]
    #bad but okay way to check period
    #it does change to 2pi/omega when I changed omega YES IT WORKS
    
    fig3=plt.figure(figsize=(10,5))
    ax3=fig3.add_subplot(1,1,1)
    ax3.plot(ts[742000:758000],us[742000:758000])
    ax3.plot(ts[742000:758000],dus[742000:758000])
    ax3.set_xlabel("t")
    ax3.set_ylabel("u")    
    """
    
    """
    For ω = 1.2 multiple periodic orbits exist. Find them.
    • Note that one of the periodic orbits is unstable and so cannot be found via simulation as done in 1.

    
    Extensions
    Investigate how your code might be extended to an autonomous ordinary differential equation (i.e., no explicit
time dependence). In this case the period T is not known a priori; instead an extra condition is added (a phase
condition) is added and the period becomes another variable to solve for.
The phase condition can be simple, e.g., fixing the value of a particular variable at a particular time (arbitrary
choices for both) to break the time invariance, or more complicated, e.g., an integral condition that minimises
the square distance to a reference solution.
Example equations of this form are the Lotka-Volterra preditor-prey equations and the Van der Pol equation —
many models are inherently time independent like this
    """
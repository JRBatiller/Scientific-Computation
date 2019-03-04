# Scientific-Computation
Scientific Computation class codes

Boundary Value and Numerical Continuation

Dealt with shooting algorithm and continuation methods to deal with tricky nonlinear equations like the duffing equation or damped pendulums. Started with easy cubic equations first then built up on previous code.

Duffy.py - shooting alg
collocation.py - boundary value solver using chevysheb differentiation matrix
continuation.py - can plot for simple cubic and other functions of x 
Duffy_plot.py - can plot for mass spring damper and duffing equation

Finite Differential Equations

Dealt with solving 2nd order partial differential equations numerically. Comes in three types Parabolic, Hyperbolic and Elliptical. Used sparse matrix to hasten computation and lessen load on laptop

Parabolic - ParabolicFDE.py - The 2D parabolic finite difference solver is primarily contained inside the method fde_parabolic, which takes the inputs, the diffusion coefficient kappa, the size of the domain L, the computation time T, the initial temperature distribution function, the number of x grid points mx, the number of time grid points mt, and the boundary condition function. There are also three additional optional inputs, the scheme to be used, whether the boundary is Neumann or not, and whether there is a heat source present. The method will then return a vector which should be the heat distribution at time T.
There are three schemes that can be used, f_euler, b_euler, and crank_nich, with f_euler set as the default if no scheme is specified. The scheme methods themselves only generate two sparse tridiagonal matrixes, A and B.
There are two separate solvers, one that handles Neumann boundary conditions and another for Dirichlet boundary conditions. Because of this, as well as how boundary conditions are handled, the code can, unfortunately, only handle a pair of Dirichlet, or a pair of Neumann BCs. 


Hyperbolic - HyperbolicFDE2.py - The main method is fde_hyperbolic and it takes the inputs, the constant c, the size of the domain L, the computation time T, the initial disturbance function, the number of x grid points mx, the number of time grid points mt, and the boundary condition function. There are again three optional parameters, the scheme function, whether the boundary conditions are Neumann or not, and a wave source function. It returns the wave at time T. 
There are two main schemes that can be used, explicit, or implicit. Both these methods will generate two matrixes, A and B. There is also a separate q_explicit method to handle variable wave speed problems that works similarly.
  There are two separate solvers, one that handles Neumann boundary conditions and another for Dirichlet boundary conditions. Because of this, as well as how boundary conditions are handled, the code can, unfortunately, only handle a pair of Dirichlet, or a pair of Neumann BCs. 
  
Elliptical - EllipticalFDE.py - unlike previous two this exercise was mostly about imrpoving existing code. 

from __future__ import print_function
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from Problems import OrszagTangVortex
from RKDG import Mesh, AdvectionOperator, SlopeLimiter, RungeKutta, LaxFriedrichs

if __name__ == '__main__':

    nx, ny = (10,10)  #numbers of interval
    x0 = 0
    y0 = 0
    xL = 2 * np.pi
    yL = 2 * np.pi #domain
    mesh = Mesh(nx, ny, x0, xL, y0, yL)
    
    PolyDegree = 1

    pb = OrszagTangVortex(mesh, PolyDegree)
    
    gamma = 5.0 / 3.0
    
    ux0 = lambda x,y: -np.sin(y)
    rho0 = lambda x,y: gamma ** 2+ 0*x
    uy0 = lambda x,y: np.sin(x)
    p0 = lambda x,y: gamma + 0*x
    Bx0 = lambda x,y: -np.sin(y)
    By0 = lambda x,y: np.sin(2*x)

        
    u01 = lambda x,y: rho0(x, y)
    u02 = lambda x,y: rho0(x, y) * ux0(x, y) 
    u03 = lambda x,y: rho0(x, y) * uy0(x, y)
    u04 = lambda x,y: Bx0(x, y)
    u05 = lambda x,y: By0(x, y)
    u06 = lambda x,y: p0(x, y) / (gamma -1) + 0.5 * rho0(x, y) * (ux0(x, y) ** 2 + uy0(x, y) ** 2) \
                     + 0.5 /(gamma -1) * ( Bx0(x, y) ** 2 + By0(x, y) ** 2)  

    u0 = [ u01, u02, u03, u04, u05, u06 ]
    

    pb.set_initial_condition(u0)
    
    #time parameters
    t0 = 0
    t_end = 1.0
    dt = 0.005
    
    ntimesteps = int(round((t_end - t0)/ dt))
       
    L = AdvectionOperator(pb, LaxFriedrichs)

    # options rk2 or rk3 
    
    slope = SlopeLimiter(50)
    rk = RungeKutta(1, slope)
    
    coeff = pb.coeff
    L.plot(coeff)

    t_curr = t0
    for i in range(2):
        t_curr = t_curr + dt
        print('time =', t_curr)
        coeff = rk.step(dt, L, coeff)
        L.plot(coeff)
        #plt.pause(1)
    plt.show()
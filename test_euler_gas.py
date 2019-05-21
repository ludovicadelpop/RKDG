from __future__ import print_function, division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from Problems import *
from RKDG import *

if __name__ == '__main__':

    nx, ny = (40,40)  #numbers of interval
    x0 = 0
    y0 = 0
    xL = 1
    yL = 1
    mesh = Mesh(nx, ny, x0, xL, y0, yL)
    
    dx, dy = (xL - x0) / nx, (yL- y0) / ny    
    PolyDegree = 2

    pb = EulerGasDynamics(mesh, PolyDegree)
    
    gamma = 1.4
    
    ux0  = lambda x,y: 0 * x + 1.206 * ((x <= 0.5)) 
    rho0 = lambda x,y: 0 * x + 0.138 * ((x <= 0.5) & (y <= 0.5)) + 0.5323 * ((x > 0.5) & (y <= 0.5)) + \
                               0.5323 * ((x <= 0.5) & (y > 0.5)) + 1.5 * ((x > 0.5) & (y > 0.5)) 
    uy0  = lambda x,y: 0 * x + 1.206 * ((y <= 0.5))
    p0   = lambda x,y: 0 * x + 0.029 * ((x <= 0.5) & (y <= 0.5)) + 0.3* ((x > 0.5) & (y <= 0.5)) + \
                               0.3 * ((x <= 0.5) & (y > 0.5)) + 1.5 * ((x > 0.5) & (y > 0.5)) 


        
    u01 = lambda x,y: rho0(x, y)
    u02 = lambda x,y: rho0(x, y) * ux0(x, y) 
    u03 = lambda x,y: rho0(x, y) * uy0(x, y)
    u04 = lambda x,y: p0(x, y) / (gamma -1) + 0.5 * rho0(x, y) * (ux0(x, y) ** 2 + uy0(x, y) ** 2) 
    
    u0 = [ u01, u02, u03, u04]
    

    pb.set_initial_condition(u0)
    
    #time parameters
    t0 = 0
    t_end = 0.25
    dt = 1e-3
    
    ntimesteps = int(round((t_end - t0)/ dt))
       
    L = AdvectionOperator(pb, LaxFriedrichs)

    # options rk2 or rk3 
    
    slope = SlopeLimiter(50)
    rk = RungeKutta(3, slope)
    
    coeff = pb.coeff
    L.plot(coeff, 0, 1)
    #plt.pause(0.05)

    j = 0
    t_curr = t0
    for i in range(ntimesteps):
        j = j+1
        t_curr = t_curr + dt
        print('time =', t_curr)
        
        vx, vy = L.velocity(coeff) 
        cfl = np.sqrt(vx * vx + vy * vy) * dt  / min(dx, dy)
        print('CFL = {}'.format(cfl))
        
        coeff = rk.step(dt, L, coeff)
        L.plot(coeff, j, 1)
        plt.pause(0.05)
    #plt.show()
    
    
    
    t0 = 0.25
    t_end = 0.3
    dt = 0.5e-3
    
    ntimesteps = int(round((t_end - t0)/ dt))
       


    t_curr = t0
    for i in range(ntimesteps):
        j=j+1
        t_curr = t_curr + dt
        print('time =', t_curr)
        
        vx, vy = L.velocity(coeff) 
        cfl = np.sqrt(vx * vx + vy * vy) * dt  / min(dx, dy)
        print('CFL = {}'.format(cfl))
        
        coeff = rk.step(dt, L, coeff)
        L.plot(coeff, j+1, 1)
        plt.pause(0.05)
    #plt.show()
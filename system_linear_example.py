from __future__ import print_function, division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from Problems import *
from RKDG import *

if __name__ == '__main__':

    nx, ny = (20,20)  #numbers of interval
    x0 = 0
    y0 = 0
    xL = 1
    yL = 1 #domain
    
    
    dx, dy = (xL - x0) / nx, (yL- y0) / ny
    
    mesh = Mesh(nx, ny, x0, xL, y0, yL)
    
    PolyDegree = 2

    pb = LinearSystem(mesh, PolyDegree)
    
    u01 = lambda x,y: np.sin( 2*np.pi*(x))*np.sin( 2*np.pi*(y))
    u02 = lambda x,y: np.sin( 2*np.pi*(x))*np.sin( 2*np.pi*(y))
    
    u0 = [u01,u02]

    pb.set_initial_condition(u0)
    
    #time parameters
    t0 = 0
    t_end = 1.0
    dt = 0.005
    
    ntimesteps = int(round((t_end - t0)/ dt))
       
    flux_type = LaxFriedrichs   
    L = AdvectionOperator(pb, flux_type)

    # options rk2 or rk3 
    
    slope = IdentityLimiter(50)
    #slope = SlopeLimiter(50)
    rk = RungeKutta(1, slope)
    
    coeff = pb.coeff
    L.plot(coeff, 0)
    plt.pause(0.05)

    t_curr = t0
    for i in range(ntimesteps):
        t_curr = t_curr + dt
        print('time =', t_curr)
        
        vx, vy = L.velocity(coeff) 
        cfl = np.sqrt(vx * vx + vy * vy) * dt  / min(dx, dy)
        print('CFL = {}'.format(cfl))
        
        coeff = rk.step(dt, L, coeff)

        L.plot(coeff, 0)
        plt.pause(0.05)
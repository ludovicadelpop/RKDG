from __future__ import print_function, division
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from Problems import *
from RKDG import *

if __name__ == '__main__':

    nx, ny = (16,16)  #numbers of interval
    x0 = 0
    y0 = 0
    xL = 1
    yL = 1 #domain
    
    #x0 = -1
    #y0 = -1
    #xL =  1
    #yL =  1 #domain
    
    mesh = Mesh(nx, ny, x0, xL, y0, yL)
    
    PolyDegree = 2
    dx, dy = (xL - x0) / nx, (yL- y0) / ny
    
    pb = Burger(mesh, PolyDegree)
    # pb = Linear(mesh, PolyDegree)
    # pb = ConvectiveCone(mesh, PolyDegree)
    
    # u0 = [lambda x,y: np.sin( 2*np.pi*(x))*np.sin( 2*np.pi*(y))]
    u0 = [lambda x,y:0 *x +  0.9 * ( (x>= 0.1) & (y>= 0.1) & (x <= 0.6) & (y<= 0.6)) + 0.1]
    # v0 = lambda r2: np.piecewise(r2, [r2 < 0.0625 ], [lambda x: np.cos(2 * np.pi * x) ** 2, 0])
    # u0 = [lambda x,y: v0((x + 0.5) ** 2 + y*y)]

    pb.set_initial_condition(u0)
    
    #time parameters
    t0 = 0
    t_end = 2.0
    dt = 0.0025
    
    ntimesteps = int(round((t_end - t0)/ dt))
       
    flux_type = LaxFriedrichs   
    L = AdvectionOperator(pb, flux_type)

    
    #slope = IdentityLimiter(50)
    slope = SlopeLimiter(50)
    rk = RungeKutta(3, slope)
    
    coeff = pb.coeff
    L.plot(coeff, 0, 0)
    plt.pause(0.05)

    t_curr = t0
    for i in range(ntimesteps):
        t_curr = t_curr + dt
        print('time =', t_curr)
        
        vx, vy = L.velocity(coeff) 
        cfl = np.sqrt(vx * vx + vy * vy) * dt  / min(dx, dy)
        print('CFL = {}'.format(cfl))
        
        coeff = rk.step(dt, L, coeff)
        L.plot(coeff, i+1, 0)
        plt.pause(0.05)
    #plt.show()

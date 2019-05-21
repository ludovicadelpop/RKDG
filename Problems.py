#!/usr/bin/env python

from __future__ import print_function
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time
from RKDG import Mesh, Legendre, MultiVector, SlopeLimiter, RungeKutta, AdvectionOperator

class Problem(object):
    """
    Abstract base class for the problem 
        
    Args:
        mesh: contains the mesh
        degree: degrees of the Legendre polynomials 
        
    Attributes:
        mesh
        space: contains the Legendre polynomials
        coeff: contains the solution modes
        nvariables: number of variables in the system (scalar problem = 1)
    """
    def __init__(self, mesh, degree):
        self._mesh = mesh
        self._space = Legendre(degree)
        self._coeff = None
        
        self._nvariables = 1
        
    def set_initial_condition(self, u0):
        """
        Sets the initial condition and modifies the attribute coeff  with the corresponding
        initial modes of the problem
        
        Args:
            u0: initial datum
        """
        assert self._coeff is None, "Already initialized problem"
        self._coeff = MultiVector(self._mesh, self._space, self.nvariables)
        for s, f in enumerate(u0):
            self._coeff.project(s, f)
            
    def fluxF(self, s, x, y, *u):
        """
        Method that has to be  overriden in the derived class
        to define the flux on the x-direction.
        
        Args:
            s: the s-th variable of the system
            x: the x-meshgrid values used to evaluate the flux
            y: the y-meshgrid values used to evaluate the flux
            *u: values of the variables used to evaluate the flux
        
        Returns:
            F (lambda function) : s-th lambda function
        """
        raise NotImplementedError('Missing flux definition')

    def fluxG(self, s, x, y, *u):
        """
        Method that has to be  overriden in the derived class
        to define the flux on the y-direction.
        
        Args:
            s: the s-th variable of the system
            x: the x-meshgrid values used to evaluate the flux
            y: the y-meshgrid values used to evaluate the flux
            *u: values of the variables used to evaluate the flux
        
        Returns:
            G (lambda function) : s-th lambda function
        """
        raise NotImplementedError('Missing flux definition')

    def jacF(self, x, y, *u):
        """
        Method that has to be  overriden in the derived class
        to define the jacobian on the x-direction.
        
        Args:
            x: the x-meshgrid values used to evaluate the flux
            y: the y-meshgrid values used to evaluate the flux
            *u: values of the variables used to evaluate the flux
        
        Returns:
            jacF: contains the evalution of the Jacobian on the x-direction
        """
        
        raise NotImplementedError('Missing Jacobian definition')

    def jacG(self, x, y, *u):
        """
        Method that has to be  overriden in the derived class
        to define the jacobian on the y-direction.
        
        Args:
            x: the x-meshgrid values used to evaluate the flux
            y: the y-meshgrid values used to evaluate the flux
            *u: values of the variables used to evaluate the flux
        
        Returns:
            jacG: contains the evalution of the Jacobian on the y-direction
        """
        
        raise NotImplementedError('Missing Jacobian definition')
    
    @property
    def mesh(self):
        return self._mesh
    
    @property
    def space(self):
        return self._space
    
    @property
    def nvariables(self):
        return self._nvariables
    
    @property
    def coeff(self):
        return self._coeff


class Linear(Problem):
    """
    Sets the linear advection problem
    """
    def __init__(self, mesh, degree):
        super(Linear, self).__init__(mesh, degree)
        self._vx, self._vy = 1, 1
        
    def fluxF(self, s, x, y, u):
        return self._vx*u
    
    def fluxG(self, s, x, y, u):
        return self._vy*u
    
    def jacF(self, x, y, u):
        
        f = np.ones_like(u) * self._vx
        return f
    
    def jacG(self, x, y, u):

        g = np.ones_like(u) * self._vy
        return g   
    
class ConvectiveCone(Problem):
    """
    There is implemented the convective cone test,
    a scalar problem where the initial datum is usually a cone.
    """

    def fluxF(self, s, x, y, u):
        
        x,y = np.meshgrid(x, y)
        return -2*np.pi*y*u
    
    def fluxG(self, s, x, y, u):
        
        x,y = np.meshgrid(x, y)
        return 2*np.pi*x*u
    
    def jacF(self, x, y, u):
        
        f = np.ones_like(u) * (-2*np.pi*y)
        return f
    
    def jacG(self, x, y, u):

        g = np.ones_like(u) * (2*np.pi*x*u)
        return g  

class ScalarExample(Problem):
    """
    Scalar example implemented in CentPack
    https://www.cscamm.umd.edu/centpack/examples/scalar2d.htm
    """
    def fluxF(self, s, x, y, u):
        return np.sin(u)

    def fluxG(self, s, x, y, u):
        return 1/3 * u ** 3
    
    def jacF(self, x, y, u):
        return np.cos(u)

    def jacG(self, x, y, u):
        return u ** 2

    
class Burger(Problem):
    """
    Burger problem
    """
    def fluxF(self, s, x, y, u):
        return 0.5 * u * u

    def fluxG(self, s, x, y, u):
        return 0.5 * u * u
    
    def jacF(self, x, y, u):
        return u

    def jacG(self, x, y, u):
        return u
    
class LinearSystem(Problem):
    """
    Sets the linear advection problem
    """
    def __init__(self, mesh, degree):
        super(LinearSystem, self).__init__(mesh, degree)
        self._vx, self._vy = 1, 1
        self._nvariables = 2
        
        f1 = lambda u1, u2: self._vx*u1
        f2 = lambda u1, u2: self._vx*u2
        
        g1 = lambda u1, u2: self._vy*u1
        g2 = lambda u1, u2: self._vy*u2
        
        f11 = lambda u1, u2: np.ones_like(u1) * self._vx
        f12 = lambda u1, u2: 0 * u1
        f21 = lambda u1, u2: 0 * u1
        f22 = lambda u1, u2: np.ones_like(u2) * self._vx
        
        g11 = lambda u1, u2: np.ones_like(u1) * self._vy
        g12 = lambda u1, u2: 0 * u1
        g21 = lambda u1, u2: 0 * u2
        g22 = lambda u1, u2: np.ones_like(u2) * self._vy
        
        self._fluxF = [f1, f2]
        self._fluxG = [g1, g2]
        
        self._jacF= lambda *u: [[f11(*u), f12(*u)],
                               [f21(*u), f22(*u)]]
        
        self._jacG = lambda *u: [[g11(*u), g12(*u)],
                                [g21(*u), g22(*u)]]
        
    def fluxF(self, s, x, y, *u):
        f = self._fluxF[s]
        return f(*u)
    
    def fluxG(self, s, x, y, *u):
        g = self._fluxG[s]
        return g(*u)

    def jacF(self, x, y, *u):
        return self._jacF(*u)

    def jacG(self, x, y, *u):
        return self._jacG(*u)
    

class SystemExample(Problem):
    def __init__(self, mesh, degree):
        super(SystemExample, self).__init__(mesh, degree)
        self._nvariables = 2
        
        f1 = lambda u1, u2: np.sin(u1)
        f2 = lambda u1, u2: np.sin(u2)
        
        g1 = lambda u1, u2: 1/3 * u1 ** 3
        g2 = lambda u1, u2: 1/3 * u2 ** 3
        
        f11 = lambda u1, u2: np.cos(u1)
        f12 = lambda u1, u2: 0 * u1
        f21 = lambda u1, u2: 0 * u1
        f22 = lambda u1, u2: np.cos(u2)
        
        g11 = lambda u1, u2: u1 ** 2
        g12 = lambda u1, u2: 0 * u1
        g21 = lambda u1, u2: 0 * u2
        g22 = lambda u1, u2: u2 ** 2
        
        self._fluxF = [f1, f2]
        self._fluxG = [g1, g2]
        
        self._jacF= lambda *u: [[f11(*u), f12(*u)],
                               [f21(*u), f22(*u)]]
        
        self._jacG = lambda *u: [[g11(*u), g12(*u)],
                                [g21(*u), g22(*u)]]
        
    def fluxF(self, s, x, y, *u):
        f = self._fluxF[s]
        return f(*u)
    
    def fluxG(self, s, x, y, *u):
        g = self._fluxG[s]
        return g(*u)

    def jacF(self, x, y, *u):
        return self._jacF(*u)

    def jacG(self, x, y, *u):
        return self._jacG(*u)  
        
    
    
class EulerGasDynamics(Problem):
    def __init__(self, mesh, degree):
        super(EulerGasDynamics, self).__init__(mesh, degree)
        self._nvariables = 4
        
        gamma = 1.4
        
        f1 = lambda u1, u2, u3, u4: u2
        f2 = lambda u1, u2, u3, u4: u2 ** 2 / u1 + (gamma - 1) * \
                                            (u4 - 0.5 * u1 * ( (u2 / u1) ** 2 + (u3 / u1) ** 2 )) 
        f3 = lambda u1, u2, u3, u4: u2 * u3 / u1 
        f4 = lambda u1, u2, u3, u4: (u4 + (gamma - 1) * \
                                            (u4 - 0.5 * u1 * ( (u2 / u1) ** 2 + (u3 / u1) ** 2 ))) * u2 / u1
                                        
        g1 = lambda u1, u2, u3, u4: u3
        g2 = lambda u1, u2, u3, u4: u2 * u3 / u1
        g3 = lambda u1, u2, u3, u4: u3 ** 2 / u1 + (gamma - 1) * \
                                            (u4 - 0.5 * u1 * ( (u2 / u1) ** 2 + (u3 / u1) ** 2 )) 
        g4 = lambda u1, u2, u3, u4: (u4 + (gamma - 1) * \
                                            (u4 - 0.5 * u1 * ( (u2 / u1) ** 2 + (u3 / u1) ** 2 ))) * u3 / u1
                                             
                                             
        f11 = f13 = f14 = lambda u1, u2, u3, u4: 0 * u1 
        f12 = lambda u1, u2, u3, u4: 0 * u1 + 1
 
        f21 = lambda u1, u2, u3, u4: -(u2 / u1) ** 2 + (gamma -1) * ( + 0.5 * (u2 / u1) ** 2 + 0.5 * (u3 / u1) ** 2)
        f22 = lambda u1, u2, u3, u4: 2 * u2 / u1 - (gamma - 1) * (u2 / u1) 
        f23 = lambda u1, u2, u3, u4: - (gamma - 1) * (u3 / u1) 
        f24 = lambda u1, u2, u3, u4: 0 * u2 + (gamma - 1)
        
        f31 = lambda u1, u2, u3, u4: - u2 * u3 / ( u1 ** 2 )
        f32 = lambda u1, u2, u3, u4: u3 / u1
        f33 = lambda u1, u2, u3, u4: u2 / u1
        f34 = lambda u1, u2, u3, u4: 0 * u3
        
        f41 = lambda u1, u2, u3, u4: - (u4 + (gamma - 1) * \
                                            (u4 - 0.5 * u1 * ( (u2 / u1) ** 2 + (u3 / u1) ** 2 ))) * u2 / (u1 ** 2) + \
                                             u2 / u1 * ((gamma - 1) / 2 * ((u2 / u1) ** 2 + (u3 / u1) ** 2))     
        f42 = lambda u1, u2, u3, u4: (u4 + (gamma - 1) * \
                                            (u4 - 0.5 * u1 * ( (u2 / u1) ** 2 + (u3 / u1) ** 2 ))) / u1 + \
                                             - (gamma - 1) * (u2 / u1) ** 2   
        f43 = lambda u1, u2, u3, u4: - (gamma - 1) * u3 / u1 * u2 / u1
        f44 = lambda u1, u2, u3, u4: u2 / u1 + (gamma - 1) * u2 / u1

        g11 = g12 = g14 = lambda u1, u2, u3, u4: 0 * u1 
        g13 = lambda u1, u2, u3, u4: 0 * u1 + 1
 
        g21 = lambda u1, u2, u3, u4: - u2 * u3 / ( u1 ** 2 ) 
        g22 = lambda u1, u2, u3, u4: u3 / u1
        g23 = lambda u1, u2, u3, u4: u2 / u1
        g24 = lambda u1, u2, u3, u4: 0 * u2 

        g31 = lambda u1, u2, u3, u4: - (u3 / u1) ** 2 + (gamma - 1) * 0.5 * ((u2 / u1) ** 2 + (u3 / u1) ** 2) 
        g32 = lambda u1, u2, u3, u4: - (gamma - 1) * u2 / u1
        g33 = lambda u1, u2, u3, u4: 2 * u3 / u1 + (gamma - 1) * u3 / u1
        g34 = lambda u1, u2, u3, u4: 0 * u3 + (gamma - 1)      

        g41 = lambda u1, u2, u3, u4:  - (u4 + (gamma - 1) * \
                                            (u4 - 0.5 * u1 * ( (u2 / u1) ** 2 + (u3 / u1) ** 2 ))) * u3 / (u1 ** 2) + \
                                             u3 / u1 * ((gamma - 1) / 2 * ((u2 / u1) ** 2 + (u3 / u1) ** 2))
        g42 = lambda u1, u2, u3, u4: - (gamma -1) * (u3 / u1) ** 2                                                
        g43 = lambda u1, u2, u3, u4: (u4 + (gamma - 1) * \
                                            (u4 - 0.5 * u1 * ( (u2 / u1) ** 2 + (u3 / u1) ** 2 ))) / u1 + \
                                            u3 / u1 + (gamma - 1) * u3 / u1
        g44 = lambda u1, u2, u3, u4: u3 / u1 + (gamma -1) * u3 / u1
        
        self._fluxF = [f1, f2, f3, f4]
        self._fluxG = [g1, g2, g3, g4]
        
        self._jacF= lambda *u: [[f11(*u), f12(*u), f13(*u), f14(*u)],
                               [f21(*u), f22(*u), f23(*u), f24(*u)],
                               [f31(*u), f32(*u), f33(*u), f34(*u)],
                               [f41(*u), f42(*u), f43(*u), f44(*u)]]
        
        self._jacG = lambda *u: [[g11(*u), g12(*u), g13(*u), g14(*u)],
                                [g21(*u), g22(*u), g23(*u), g24(*u)],
                                [g31(*u), g32(*u), g33(*u), g34(*u)],
                                [g41(*u), g42(*u), g43(*u), g44(*u)]]
        
    def fluxF(self, s, x, y, *u):
        f = self._fluxF[s]
        return f(*u)
    
    def fluxG(self, s, x, y, *u):
        g = self._fluxG[s]
        return g(*u)

    def jacF(self, x, y, *u):
        return self._jacF(*u)

    def jacG(self, x, y, *u):
        return self._jacG(*u)    


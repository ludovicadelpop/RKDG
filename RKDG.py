#!/usr/bin/env python

from __future__ import print_function
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK
from collections import namedtuple
import warnings


class Cell(namedtuple('Cell', ['dx', 'dy', 'x0', 'y0', 'index', 'id'])):
    """
    Class used to define a cell of the mesh
    
    Args:
        Cell: the namedtuple Cell needs the size of the cell in x and y (dx, dy)   
              the initial point (x0,y0) the index of the cell and the index on the x and y direction id 
    """
    def transform(self, x, y):
        """
        Transform the nodes x, y from the reference interval to the current cell
        
        Args:
            x: array contains the x-nodes on the reference interval
            y: array contains the y-nodes on the reference interval
            
        Returns:
            x: array contains the trasformed x-nodes 
            y: array contains the trasformed y-nodes 
        """
        x, y = np.array(x), np.array(y)
        x = self.x0 + 0.5 * self.dx * (x + 1)
        y = self.y0 + 0.5 * self.dy * (y + 1)

        if (np.logical_or(x < self.x0, x > (self.x0 + self.dx)).any() or
            np.logical_or(y < self.y0, y > (self.y0 + self.dy)).any()):
            warnings.warn('Nodes out of interval to current cell')
        
        return x, y
    
    def linspace(self, nx, ny):
        """
        Create an array inside the cell with nx and ny equidistant nodes
        
        Args:
            nx: number of element on the x-axis
            ny: number of element on the y-axis
            
        Returns:
            x: the nx equidistant nodes on the x-direction
            y: the ny equidistant nodes on the y-direction
        """
        x = np.linspace(-1, 1, nx)
        y = np.linspace(-1, 1, ny)
        return self.transform(x, y)
    
    @property
    def x1(self):
        return (self.x0 + self.dx)
    
    @property
    def y1(self):
        return (self.y0 + self.dy)
    

class Mesh(object):
    """
    Class used to define the problem mesh
    
    Args:
        nintx: number of element on the x-direction
        ninty: number of element on the y-direction
        x0: initial point in x
        xL: final point in x
        y0: initial point in y
        yL: final point in y
    
    Attributes:
        nintx, ninty, x0, y0, xL, yL
        dx: contains the x-size of each element of the domain
        dy: contains the y-size of each element of the domain
        x: the x of the mesh obtained by the meshgrid command
        y: the y of the mesh obtained by the meshgrid command
        ncells: number of cells inside the mesh
        
    """
    def __init__(self, nintx, ninty, x0, xL, y0, yL):
        self._nintx = nintx
        self._ninty = ninty

        self._x0 = x0
        self._y0 = y0
        
        self._xL = xL
        self._yL = yL

        x = np.linspace(x0, xL, nintx + 1)
        y = np.linspace(y0, yL, ninty + 1)

        xx, yy = np.meshgrid(x, y)

        self._dx = xx[:, 1:] - xx[:, :-1]
        self._dy = yy[1:, :] - yy[:-1, :]

        self._x, self._y = np.meshgrid(x[:-1], y[:-1])
        
        self._ncells = nintx * ninty
         
    def cells(self):
        """
        Generator for the cells inside the mesh
        
        """
        dx, dy = np.ravel(self._dx), np.ravel(self._dy)
        x, y = np.ravel(self._x), np.ravel(self._y)
        
        for i in range(self.ncells):
            ix, iy = i % self.nx, i // self.nx
            yield Cell(dx[i], dy[i], x[i], y[i], i, (ix, iy))
    
    def neighbors(self, cell):
        """
        Given a cell returns its neighbors (the problem has periodic boundary condition
        so the neighbors are toroidal)
        
        Args:
            cell: a cell
        
        Returns:
            cellR: the cell at the right position
            cellL: the cell at the left position
            cellU: the cell at the upper position
            cellD: the cell at the down position
            
        """
        dx, dy = np.ravel(self._dx), np.ravel(self._dy)
        x, y = np.ravel(self._x), np.ravel(self._y)

        nx, ny = self.nx, self.ny
        ix, iy = cell.id

        iL = (iy * nx) + (ix - 1 + nx) % nx
        iR = (iy * nx) + (ix + 1) % nx
        iD = ((iy - 1 + ny) % ny) * nx + ix 
        iU = ((iy + 1) % ny) * nx + ix 

        return (Cell(dx[iR], dy[iR], x[iR], y[iR], iR, None),
                Cell(dx[iL], dy[iL], x[iL], y[iL], iL, None),
                Cell(dx[iU], dy[iU], x[iU], y[iU], iU, None),
                Cell(dx[iD], dy[iD], x[iD], y[iD], iD, None))

    @property
    def ncells(self):
        return self._ncells

    @property
    def nx(self):
        return self._nintx

    @property
    def ny(self):
        return self._ninty

    @property
    def size(self):
        return self.nx, self.ny

    @property
    def x0(self):
        return self._x0

    @property
    def y0(self):
        return self._y0

    @property
    def xL(self):
        return self._xL

    @property
    def yL(self):
        return self._yL


class Legendre(object):
    """
    Class used to create a Legendre polynomials
    
    Args:
        degree: the degree of the polynomials
        dtype: optional the type of the values
        
    Attributes:
        degree, dtype
    """
    def __init__(self, degree, dtype=None):
        # Default arguments
        if dtype is None:
            dtype = np.float

        # Set member values
        self._degree = degree
        self._dtype = dtype
        
    @property
    def degree(self):
        return self._degree
    
    @property
    def dim(self):
        return self.degree + 1
    
    @property
    def dtype(self):
        return self._dtype

    def __call__(self, nodes):
        """
        Computes the values ofthe polynomials and of the derivaties of the polynomials in
        the nodes.
        
        Args: 
            nodes: where the function has to evaluate the polynomials
        
        Returns:
            values: contains the evaluation of the polynomials in the nodes 
            derivs: contains the evaluation of the polynomials derivatives in the nodes
        """
        nodes = np.array(nodes)
        npolys = 1 + self.degree
        nnodes = len(nodes)

        values = np.empty((npolys, nnodes), dtype=self.dtype)
        derivs = np.empty((npolys, nnodes), dtype=self.dtype)

        # Degree 0
        values[0, :] = 1
        derivs[0, :] = 0

        if npolys > 1:
            # Degree 1
            values[1, :] = nodes
            derivs[1, :] = 1

            for i in range(2, npolys):
                values[i, :] = (2 - 1 / i) * nodes * values[i-1, :] - \
                               (1 - 1 / i) * values[i-2, :]
                derivs[i, :] = (2 * i - 1) * values[i-1, :] + derivs[i-2, :]

        return values, derivs


def qgauss(nnodes, dtype=None):
    """
    Function that returns the weights and the nodes 
    for the Gaussian quadrature rule
    
    Args:
        nnodes: number of nodes
        dtype: optional, the type of the nodes value 
        
    Returns:
        weights: weights for the quadrature rule     
        nodes: nodes for the quadrature rule
    """
    if dtype is None:
        dtype = np.float
    
    if nnodes == 1:
        nodes = np.array([0], dtype=dtype)
        weights = np.array([2], dtype=dtype)
        return weights, nodes 
    
    N = nnodes // 2
    nodes = np.empty(nnodes, dtype=dtype)
    weights = np.empty(nnodes, dtype=dtype)
    legendre = Legendre(nnodes)

    # Initial guess
    nodes[:N] = -np.cos((2 * np.arange(N, dtype=dtype) + 1) * np.pi /
                        (2 * nnodes))

    # Newton iterations
    while True:
        values, derivs = legendre(nodes[:N])
        delta = values[-1, :] / derivs[-1, :]
        nodes[:N] -= delta

        if np.linalg.norm(delta, np.inf) < np.finfo(dtype).eps:
            break

    # Symmetric nodes
    if nnodes % 2:
        nodes[N] = 0
    nodes[-N:] = -nodes[:N][::-1]
    # Weights
    values, derivs = legendre(nodes)
    weights = 2 / (1 - nodes * nodes) / (derivs[-1, :] * derivs[-1, :])

    return weights, nodes


class MultiVector(object):
    """
    Class used to manipulate easily the vector of modes 
    
    Args:
        mesh: the mesh where the problem hat to be solved
        space: the object of type legendre that contains the polynomials
        nvariables: the number of variables
    
    Attributes:
        mesh, space, nvariables
        ncells: area of the cell
        nintx: number of cell in x direction
        ninty: number of cell in x direction
        degree: polynomials degree
        dofs_cell: number of degrees of freedom in each cell
        dim: number of unknown in the problem
        data: vector contains the solution (modes of the problem) 
    """
    def __init__(self, mesh, space, nvariables=1):
        self._mesh = mesh
        self._space = space
        self._nvariables = nvariables 
        self._ncells, self._nintx, self._ninty = mesh.ncells, mesh.nx, mesh.ny
        self._degree = space.degree
        
        dofs = self._degree + 1
        self._dofs_cell = dofs * dofs
        self._dim = dim = self._ncells * dofs * dofs
        self._data = np.zeros(dim * nvariables, dtype = np.float)


    @staticmethod
    def like(other):
        return MultiVector(other._mesh, other._space, other._nvariables)

    def __getitem__(self, key):
        """
        Override the method getitem ( [] ) 
        Args: 
            key: it could be made  by only the cell index or by apair (cell_index and variables)
                
        Returns:
            Calls the method get_slice
        """
        try:
            return self._get_slice(*key)
        except TypeError:
            return self._get_slice(key)

    def __setitem__(self, key, value):
        """
        Override the method setitem ( [] ) 
        and it changes the value in a specific position dectated by key
        Args: 
            key: it could be made  by only the cell index or by apair (cell_index and variables)
            value: the value that has to be changed
                
        Returns:
            Calls the method get_slice
        """
        
        if np.any(np.isnan(value)):
            warnings.warn('NaN values')
        try:
            data = self._get_slice(*key)
        except TypeError:
            data = self._get_slice(key)
        data[:] = np.ravel(value)

    def _get_slice(self, index_cell, index_eq=None):
        """
        Function used to select a specific part of the vector
        
        Args:
            index_cell: the index cell
            index_eq: othe index of the variable
        
        Return:
            data: the reference to the part of the underlined vector 
         
        """
        data = self._data.reshape((self._nvariables, -1))
        dofs_cell = self._dofs_cell
        i0 = index_cell * dofs_cell
        if index_eq is None:
            data = data[:, i0:i0+dofs_cell]
        else:
            data = data[index_eq, i0:i0+dofs_cell]
        return data

    def copy(self):
        """
        Makes a cpy of the object
        """
        new = MultiVector.like(self)
        new._data[:] = self.data
        return new

    def project(self, s, f):
        """
        Project in the L^2 space the function f
        
        Args:
            s: index for the variable 
            f: function to project
        
        """
        npolys = self._space.dim
        weights, nodes = qgauss(npolys)
        weights = np.outer(weights, weights)
        values, derivs = self._space(nodes)
        
        for cell in self._mesh.cells():
                x, y = cell.transform(nodes, nodes)
                X, Y = np.meshgrid(x, y)
                
                u0i = f(X, Y)
                
                # for each variables, for each cell (slide from left to right, from down to up)
                # on the column we store che coeff. where the y polynomial are varing and in the rows
                # the x polynomials.
                
                self[cell.index, s] = [[
                    (k + 0.5) * (m + 0.5) * np.sum(weights * u0i * np.outer(values[m, :], values[k, :]))
                    for k in range(npolys)] for m in range(npolys)]

    def plot(self, time, flag=0, var=0, px=10, py=10, **kwargs):
        """
        Plots the solution and saves the figures
        
        Args:
            time: the time of the solution
            flag: 0 = plots using pcolormesh
                  1 = plots using contourf
            var: indicates the variable 
            px: number of point inside each cell along the x-axis
            py: number of point inside each cell along the y-axis
        """
        nx, ny = self._mesh.size
        uglob = np.empty((px * nx, py * ny))
        
        fig = plt.figure(var)
        for cell in self._mesh.cells():
            x, y = cell.linspace(px + 1, py + 1)
            x = 0.5 * (x[1:] + x[:-1])
            y = 0.5 * (y[1:] + y[:-1])

            ix, iy = cell.id
            ix, iy = ix * px, iy * py
            uglob[iy:iy+py, ix:ix+px] = self(cell, var, x, y)

        if flag == 1:
            x = np.linspace(self._mesh.x0, self._mesh.xL, px * nx )
            y = np.linspace(self._mesh.y0, self._mesh.yL, py * ny )
            x, y = np.meshgrid(x, y)

            p = plt.contourf(x, y, uglob, vmin=0.0, vmax=1.8, **kwargs)
        else:
            x = np.linspace(self._mesh.x0, self._mesh.xL, px * nx + 1)
            y = np.linspace(self._mesh.y0, self._mesh.yL, py * ny + 1)
        
            x, y = np.meshgrid(x, y)
            p = plt.pcolormesh(x, y, uglob, vmin=0, vmax=1, **kwargs)
        #fig.colorbar(p)
        
        #name_file = './image'+str(var)+'/u'+ str(time)+'.png'
        #plt.savefig(name_file, format='png')
        #plt.close()
        
        
    def mean(self, cell, var=None):
        """
        Returns the first mode c_00 that corresponds to the mean
        
        Args:
            cell: the cell where the mean has to be computed
            var: the variable of the system
        
        Returns:
            the c_00 mode 
            
        """
        if var is None:
            return self[cell.index][:, 0].copy()
        else:
            return self[cell.index, var][0]

    def slope(self, cell, var = None):
        """
        Returns the slope modes, c_10 and c01 
        
        Args:
            cell: the cell where the mean has to be computed
            var: the variable of the system
            
        Returns:
            (c_10, c_01): the modes 
        
        """
        dim = self._space.dim
        index = cell.index
        if var is None:
            return (self[index][:, 1].copy(), self[index][:, dim].copy())
        else:
            return (self[index, var][1], self[index, var][dim].copy())

    def set_linear(self, cell, var, umean, ux, uy):
        """
        Set the new linear modes and forces to zeros the others
        Args:
            cell: the cell where the modes have to be updated
            var: variable of the system where the modes have to be updated
            umean: the mode c_00
            ux: the mode c_10
            uy: the mode c_10
        """
        dim = self._space.dim
        index = cell.index

        data = self._get_slice(index, var)
        data[:] = 0
        data[0] = umean
        data[1] = ux
        data[dim] = uy
        
    def __call__(self, cell, var, x, y):
        """
        From modal coefficient to nodal solution,
        evaluates the legendre polynomials in the cell:
        it recives the nodes in the original cell 
        
        Args:
            cell: the cell where the solution hat to be computed
            var: variable of the system where the solution has to be computed
            x: x nodes in the cell
            y: y nodes in the cell
        
        Returns:
            u: the solution in the nodes of the current cell
        """
        x, y = np.array(x), np.array(y)
        x = 2*(x - cell.x0)/cell.dx - 1
        y = 2*(y - cell.y0)/cell.dy - 1

        if (np.logical_or(x < (-1 - np.finfo(float).eps), x > (1 + np.finfo(float).eps)).any() or
            np.logical_or(y < (-1 - np.finfo(float).eps), y > (1 + np.finfo(float).eps)).any()):
            warnings.warn('Nodes out of interval in the rif cell')
 
        values_x, derivs_x = self._space(x)
        values_y, derivs_y = self._space(y)
        

        data = self._get_slice(cell.index, var)
        u = np.array([[
            (np.ravel(np.outer(values_y[:, j], values_x[:, i])) * data).sum() \
            for i in range(len(x))] for j in range(len(y))
        ])
        
        if len(x) == 1 or len(y) == 1:
            return u.reshape((u.size, ))
        else:
            return u

    @property
    def mesh(self):
        return self._mesh

    @property
    def dofs_cell(self):
        return self._dofs_cell

    @property
    def dim(self):
        return len(self._data)

    @property
    def size_mesh(self):
        return self._size_mesh

    @property
    def data(self):
        return self._data
    
    @property
    def nvariables(self):
        return self._nvariables


def minmod(a, b, c):
    """
    minmod function
    """
    return np.where(np.logical_and(a * b > 0, b * c > 0),
                    np.sign(a) * np.minimum(abs(a), np.minimum(abs(b), abs(c))),
                    0)


def minmod2(a, b, c, eps):
    """
    minmod2 function
    """
    return np.where(abs(a) < eps, a, minmod(a,  b, c))

    
class Limiter(object):
    """
    Abstract class for the Limiter
    Args:
        M: parameter1
        eps: parameter2
        
    Attributes: 
        M, eps
    """
    def __init__(self, M = 50, eps=1e-16):
        self._M = M
        self._eps = eps
        
class SlopeLimiter(Limiter):
    """
    Derived class of Limiter
    """
    def __call__(self, coeff):
        """
        Computes the modified modes c_00 c_01 c_10 
        if they are different from the original
        they are the new modes, and the others are set equal to zero
        
        Args:
            coeff: the modes of the solution
        """
        nvariables = coeff.nvariables
        mesh = coeff.mesh
        M = self._M
        eps = self._eps

        for cell in mesh.cells():
            umean = coeff.mean(cell)
            ux, uy = coeff.slope(cell)
            
            cellR, cellL, cellU, cellD = mesh.neighbors(cell)
            
            uR , uL, uU, uD = coeff.mean(cellR), coeff.mean(cellL), coeff.mean(cellU), coeff.mean(cellD)
            
            vx = minmod2(ux, uR - umean, umean - uL, M * cell.dx * cell.dx)
            vy = minmod2(uy, uU - umean, umean - uD, M * cell.dy * cell.dy)

            for s in range(nvariables):
                if max(abs(vx[s] - ux[s]), abs(vy[s] - uy[s])) < eps * max(abs(ux[s]), abs(uy[s])):
                    continue
                else:
                    coeff.set_linear(cell, s, umean[s], vx[s], vy[s])

class IdentityLimiter(Limiter):
    """
    Derived class of Limiter
    """
    def __call__(self, coeff):
        """
        Computes the identity operator, useful if we don't want 
        to apply the limiter at the solution
        
        Args:
            coeff: modes of the solution
        """
        coeff.data[:] = coeff.data[:]
     
     
class NumericalFlux(object):
    """
    Abstract class to define the numerical flux at the interface
    its hat to be called two time one for F and one for G
    Args:
        f: the problem flux 
        jac: the jacobian of the flux
        nvariables: number of variables in the system
        npolys: number of polynomials used for the space
        
    Attributes:
        f, jac, nvariable, npolys
    """
    def __init__(self, f, jac, nvariables, npolys):
        self._f = f
        self._jac = jac
        self._nvariables = nvariables
        self._npolys = npolys

    def __call__(self, s, x, y, uplus, uminus):
        """
        Call operator has to be implemented in the derived class
        Args:
            s: current variable
            x: x nodes
            y: y nodes
            uplus: solution on the outside of the cell
            uminus: solution on the inside of the cell
        """
        raise NotImplementedError('call method non implemented')

class LaxFriedrichs(NumericalFlux): 
    """
    Derived class of Numerical flux.
    """
    def max_eigen(self, valPlus, valMinus):
        """
        Computes the maximum eigenvalue of the Jacobian in valuPlus and valMinus
        and returns the maximum between the the two eigenvalues
        
        Args:
            valPlus: flux evaluation with the solution on the outside of the cell
            valMinus: flux evaluation with the solution on the outside of the cell
        
        Returns:
            the max eigenvalue
        """
        nvariables = self._nvariables
        npolys = self._npolys
        
        if nvariables == 1:
            return np.maximum(np.abs(valPlus), np.abs(valMinus))
        else:
            
            lambdaplus_max = np.zeros(npolys)
            lambdaminus_max = np.zeros(npolys)
                        
            for k in range(npolys):
                matrixplus = np.array([[valPlus[i][j][k] for i in range(nvariables)]\
                                        for j in range(nvariables)]) 
                v, _ = np.linalg.eig(matrixplus)
                lambdaplus_max[k] = np.max(np.abs(v))
            
                matrixminus = np.array([[valMinus[i][j][k] for i in range(nvariables)]\
                                        for j in range(nvariables)]) 
                v, _ = np.linalg.eig(matrixminus)
                lambdaminus_max[k] = np.max(np.abs(v))
                
            return np.maximum(lambdaplus_max, lambdaminus_max)
        
    def __call__(self, var, x, y, uplus, uminus):

        '''
        Computes the numerical flux at each element 
        using the local Lax-Friedrichs numerical flux 
        '''
        
        x = np.array(x)
        y = np.array(y)
        
        fuplus = self._f(var, x, y, *uplus)
        fuminus = self._f(var, x, y, *uminus)
        
        FfirstvalPlus = self._jac(x, y, *uplus)
        FfirstvalMinus  = self._jac(x, y, *uminus)

        alpha = self.max_eigen(FfirstvalPlus, FfirstvalMinus)

        flux_face = 0.5 * ( fuplus + fuminus - alpha * (uplus[var] - uminus[var]))
        return flux_face
 

class RungeKutta(object):
    """
    Explicit Runge-Kutta method to integrate in time the sol
    we implemented the method of order 1,2 and 3
    Args:
        order: number of stages of the RK 
        slope: limiter we want to use at each stage
    
    Attributes:
        order, slope
        alpha: coefficient1
        beta: coefficient2
    """
    def __init__(self, order, slope):
        self._order = order
        self._slope = slope
        
        if order == 1:
            self.alpha = np.array([[1]])
            self.beta = np.array([[1]])
        elif order == 2:
            self.alpha = np.array([[1, 0], [.5, .5]])
            self.beta = np.array([[1, 0], [0, .5]])
        elif order == 3:
            self.alpha = np.array([[1, 0, 0], [.75, .25, 0], [1/3, 0, 2/3]])
            self.beta = np.array([[1, 0, 0], [0, .25, 0], [0, 0, 2/3]])
        else:
            raise NotImplementedError()
    
    def step(self, dt, L, c):
        """
        Computes a complete step of the RK method
        Args:
            dt: size of the step
            L: rhs operator
            c: solution at current time tn
            
        Returns:
            the new solution at time tn + dt
        """
        slope = self._slope
        a, b = self.alpha, self.beta
        Lc = L(c)
        Lsteps = []
        steps = [MultiVector.like(c) for i in range(self._order)]
        for i in range(self._order):
            Li = steps[i]
            steps[i].data[:] = a[i, 0] * c.data[:] + b[i, 0] * dt * Lc.data[:]
            for j, Lj in enumerate(Lsteps):
                steps[i].data[:] += a[i, j+1] * steps[j].data[:] + b[i, j+1] * dt * Lj.data[:]
            slope(steps[i])
            if i < self._order -1:
                Lsteps.append(L(steps[i]))
        return steps[-1]


class AdvectionOperator(object):
    """
    Class used to implement the right and side of the conservation laws system
        
    Args:
        problem : item of type problem
        flux_type: the numerical fluxed to use at the interface
    
    Attributes:
        problem
        legendre: the legendre polynomials
        npolys: number of polynomials
        nvariables: numbero of varibles of the system
        ncells: number of cell
        weights: weights for the quadrature rule
        nodes: nodes defined by the quadrature rule                
    """
    def __init__(self, problem, flux_type = None):

        self._problem = problem
        self._legendre = problem.space
        degree = problem.space.degree
        self._npolys = degree + 1

        self._nvariables = problem.nvariables
        self._ncells = problem.mesh.ncells
        
        self._mesh = self._problem.mesh
        self._weights, self._nodes = qgauss(self._npolys)
        self._values, self._derivs = self._legendre(self._nodes)
        if flux_type is None:
            flux_type = LaxFriedrichs
        self.flux_type = flux_type
        
        
    def project(self, u0):
        """
        Calls the MultiVector project method for each variable
        
        Args:
            u0: the initial condition in the mesh
        """
        c0 = MultiVector(self._mesh, self._legendre, self._nvariables)
        for s, f in enumerate(u0):
            c0.project(s, f)
        return c0

    
    def plot(self, coeff, time, flags=0):
        """
        Calls the MultiVector plot method for each variable
        Args:
            coeff: the modal values
            time: the time of the solution
        
        """
        for s in range(self._nvariables):
            coeff.plot(time, flags, s, px=20, py=20)
            
    
    def velocity(self, coeff):
        """
        Used to compute the velocity to verify the CFL condition
        Args:
            coeff: the modes
            
        Returns:
            ux: the max velocity in x-direction
            uy: the max velocity in y-direction
        """
        
        mesh = self._mesh
        nvariables = self._nvariables
        nodes = self._nodes
        problem = self._problem
        ux = 0
        uy = 0
        
        for cell in mesh.cells():
            xnodes, ynodes = cell.transform(nodes, nodes)
            u = np.array([coeff(cell, s, xnodes, ynodes) for s in range(nvariables)])
            
            Ffirst = problem.jacF(xnodes, ynodes, *u)
            Gfirst = problem.jacG(xnodes, ynodes, *u)
            
            vx = [Ffirst[s][d].max() for s in range(nvariables) for d in range(nvariables)]
            vx = max(vx)
            vy = [Gfirst[s][d].max() for s in range(nvariables) for d in range(nvariables)]
            vy = max(vy)
            
        ux = max(ux, vx)
        uy = max(uy, vy)
        return ux, uy    
       
    
    def __call__(self, coeff):
        """
        Space discretizion using the modal dg formulation of the advection operator
        
        Args: 
            coeff: modes
            
        Returns:
            coeff_new: the ew modes changed by the advection operator
        
        
        """
        coeff_new = MultiVector.like(coeff)
        
        mesh = self._mesh
        nodes = self._nodes
        
        problem = self._problem
        
        vect_dof = [(2 * dof + 1)  for dof in range(self._npolys) ]
        vect_dof = np.ravel(np.outer(vect_dof, vect_dof))

        nvariables = self._nvariables
        npolys = self._npolys
        
        values, derivs = self._values, self._derivs
        valuesB, derivsB = self._legendre([-1., 1.])
        weights = self._weights
        cell_weights = np.outer(self._weights, self._weights)

        flux = self.flux_type
        numFluxF = flux(problem.fluxF, problem.jacF, nvariables, npolys)
        numFluxG = flux(problem.fluxG, problem.jacG, nvariables, npolys)

        for cell in mesh.cells():
 
            xnodes, ynodes = cell.transform(nodes, nodes)

            # cell
            u = np.array([coeff(cell, s, xnodes, ynodes) for s in range(nvariables)])
            
            for s in range(nvariables):
                
                fui = problem.fluxF(s, xnodes, ynodes, *u) 
                gui = problem.fluxG(s, xnodes, ynodes, *u)

                coeff_new[cell.index, s] = [[
                    0.5 * cell.dy * (cell_weights * fui * np.outer(values[m, :], derivs[k, :])).sum()
                    + 0.5 * cell.dx * (cell_weights * gui * np.outer(derivs[m, :], values[k, :])).sum()
                    for k in range(npolys)] for m in range(npolys)]
            
            # boundaries
            cellR, cellL, cellU, cellD = mesh.neighbors(cell)
            
            uR_outer = np.array([coeff(cellR, s, [cellR.x0], ynodes) for s in range(nvariables)])
            uR_inner = np.array([coeff(cell, s, [cell.x1], ynodes) for s in range(nvariables)])

            
            uL_outer = np.array([coeff(cellL, s, [cellL.x1], ynodes) for s in range(nvariables)])
            uL_inner = np.array([coeff(cell, s, [cell.x0], ynodes) for s in range(nvariables)])

            uD_outer = np.array([coeff(cellD, s, xnodes, [cellD.y1]) for s in range(nvariables)])
            uD_inner = np.array([coeff(cell, s, xnodes, [cell.y0]) for s in range(nvariables)])

            uU_outer = np.array([coeff(cellU, s, xnodes, [cellU.y0]) for s in range(nvariables)])
            uU_inner = np.array([coeff(cell, s, xnodes, [cell.y1]) for s in range(nvariables)])

            for s in range(nvariables):
                # boundary flux integral for the F flux
                fR = numFluxF(s, [cell.x1], ynodes, uR_outer, uR_inner)
                fL = numFluxF(s, [cell.x0], ynodes, uL_inner, uL_outer)

                coeff_new[cell.index, s] -= np.ravel([[
                    0.5 * cell.dy * (weights * fR * values[m, :] * valuesB[k, 1]).sum()
                    - 0.5 * cell.dy * (weights * fL * values[m, :] * valuesB[k, 0]).sum()
                    for k in range(npolys)] for m in range(npolys)])

                
                # boundary flux integral for the G flux
                gU = numFluxG(s, xnodes, [cell.y1], uU_outer, uU_inner)
                gD = numFluxG(s, xnodes, [cell.y0], uD_inner, uD_outer)

                coeff_new[cell.index, s] -= np.ravel([[
                    0.5 * cell.dx * (weights * gU * values[k, :] * valuesB[m, 1]).sum()
                    - 0.5 * cell.dx * (weights * gD * values[k, :] * valuesB[m, 0]).sum()
                    for k in range(npolys)] for m in range(npolys)])

                coeff_new[cell.index, s] *= ( vect_dof ) / ( cell.dx * cell.dy ) 

        return coeff_new        

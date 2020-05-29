import numpy as np

from scipy import integrate
from scipy import sparse
from scipy import interpolate

import os
import scipy.io as sio
import scipy.optimize
import itertools
import time

import pdb

from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

def D_u(D,dx):
    
    '''
    Create the Matrix operator for (D(u)u_x)_x, where D is a vector of values of D(u),
    and dx is the spatial resolution based on methods from Kurganov and Tadmoor 2000
    (https://www.sciencedirect.com/science/article/pii/S0021999100964593?via%3Dihub)
    '''

    D_ind = np.arange(len(D))

    #first consruct interior portion of D
    #exclude first and last point and include those in boundary
    D_ind = D_ind[1:-1] 
    #Du_mat : du_j/dt = [(D_j + D_{j+1})u_{j+1}
    #                   -(D_{j-1} + 2D_j + D_{j+1})u_j
    #                   + (D_j + D_{j-1})u_{j-1}] 
    Du_mat_row = np.hstack((D_ind,D_ind,D_ind))
    Du_mat_col = np.hstack((D_ind+1,D_ind,D_ind-1))
    Du_mat_entry = (1/(2*dx**2))*np.hstack((D[D_ind+1]+D[D_ind],
                   -(D[D_ind-1]+2*D[D_ind]+D[D_ind+1]),D[D_ind-1]+D[D_ind]))
    
    #boundary points
    Du_mat_row_bd = np.array((0,0,len(D)-1,len(D)-1))
    Du_mat_col_bd = np.array((0,1,len(D)-1,len(D)-2))
    Du_mat_entry_bd = (1.0/(2*dx**2))*np.array((-2*(D[0]+D[1]),
                    2*(D[0]+D[1]),-2*(D[-2]+D[-1]),2*(D[-2]+D[-1])))
    #add in boundary points
    Du_mat_row = np.hstack((Du_mat_row,Du_mat_row_bd))
    Du_mat_col = np.hstack((Du_mat_col,Du_mat_col_bd))
    Du_mat_entry = np.hstack((Du_mat_entry,Du_mat_entry_bd))

    return sparse.coo_matrix((Du_mat_entry,(Du_mat_row,Du_mat_col)))


def PDE_RHS(t,y,x,D,f):
    
    ''' 
    Returns a RHS of the form:
    
        q[0]*(g(u)u_x)_x + q[1]*f(u)
        
    where f(u) is a two-phase model and q[2] is carrying capacity
    '''
    
    dx = x[1] - x[0]
    
    try:
        
        # density and time dependent diffusion
        Du_mat = D_u(D(y,t),dx)
        return  Du_mat.dot(y) + y*f(y,t)
    
    except:
        
        # density dependent diffusion
        Du_mat = D_u(D(y),dx)
        return  Du_mat.dot(y) + y*f(y)
    
    


def PDE_sim(RHS,IC,x,t,D,f):
    
    # grids for numerical integration
    t_sim = np.linspace(np.min(t), np.max(t), 1000)
    x_sim = np.linspace(np.min(x), np.max(x), 200)
    
    # interpolate initial condition to new grid
    f_interpolate = interpolate.interp1d(x,IC)
    y0 = f_interpolate(x_sim)
        
    # indices for integration to write to file for
    for tp in t:
        tp_ind = np.abs(tp-t_sim).argmin()
        if tp == t[0]:
            t_sim_write_ind = np.array(tp_ind)
        else:
            t_sim_write_ind = np.hstack((t_sim_write_ind,tp_ind))

    # make RHS a function of t,y
    def RHS_ty(t,y):
        return RHS(t,y,x_sim,D,f)
            
    # initialize array for solution
    y = np.zeros((len(x),len(t)))  
    
    y[:, 0] = IC
    write_count = 0
    r = integrate.ode(RHS_ty).set_integrator("dopri5")  # choice of method
    r.set_initial_value(y0, t[0])   # initial values
    for i in range(1, t_sim.size):
        
        # write to y for write indices
        if np.any(i==t_sim_write_ind):
            write_count+=1
            f_interpolate = interpolate.interp1d(x_sim,r.integrate(t_sim[i]))
            y[:,write_count] = f_interpolate(x)
        else:
            # otherwise just integrate
            r.integrate(t_sim[i]) # get one more value, add it to the array
        if not r.successful():
            print("integration failed")
            return 1e6*np.ones(y.shape)

    return y
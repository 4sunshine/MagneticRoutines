#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code prepares the data incoming from the call to be acessed and used by
dave4vm. From now on, the variables shall be used preferentially like they were 
written by Dr. Schuck.


"""
import numpy as np

from . import odiffxy5, dave4vm

    
def do_dave4vm(dt,bx_stop,bx_start,by_stop,by_start,bz_stop,bz_start,
               dx,dy,wsize):
    '''
    Here the variables to execute pydave4vm are going to be
    prepared.
    '''
    
    #taking the average change on bz over the time interval dt
    bzt = (bz_stop - bz_start)/dt
    
    
    #Taking the average value of the images
    #Those average values will be entries for the odiffxy5 function
    bx = (bx_stop + bx_start)/2
    by = (by_stop + by_start)/2
    bz = (bz_stop + bz_start)/2
    
    #Calculating the differentials
    bxx,bxy = odiffxy5.odiff(bx)
    byx,byy = odiffxy5.odiff(by)
    bzx,bzy = odiffxy5.odiff(bz)
    
    #Defining the dictionary that will take all the information necessary
    #to the calculation performed by dave4vm
    magvm = {'bzt': bzt, 'bx': bx, 'bxx': np.divide(bxx,dx), 
             'bxy': np.divide(bxy,dy), 'by': by, 'byx': np.divide(byx,dx),
             'byy': np.divide(byy,dy), 'bz': bz, 'bzx': np.divide(bzx,dx), 
             'bzy': np.divide(bzy,dy), 'dx': dx, 'dy': dy, 'dt': dt}
    
    #Call!
    vel4vm = dave4vm.calculate_dave4vm(magvm, wsize)
    
    return(magvm, vel4vm)

































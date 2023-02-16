"""

Defines functions used to simulate time difference of arrival (TDoA) signal
tracking for the RIT LASSO senior design project.

"""

import numpy as np

def maketdoa(T,R,pairs,c,noisefun,noisemag):
    """

    Creates a vector of time difference of arrivals given locations for the
    (stationary) transmitter and receivers.
    
    For simulation purposes.

    Parameters
    ----------
    T : [3 x 1] array
        Transmitter location.
    R : [3 x N] array
        Receiver locations.
    pairs : [M x 2] integer array
        Reciever pairs, see "lasso.makepairs".
    c : scalar
        Signal speed (speed of light).
    noisefun : lambda function
        Noise distribution function, should take in "M" and return a [1 x M]
        array of noise values from a unit distribution.
    noisemag : scalar
        Value with which to multiply all noise values from noisefun

    Returns
    -------
    tdoa : [1 x N] array
        Time difference of arrival for each pair.
    ddoa : [1 x N] array
        Distance difference for each pair, ddoa = c*tdoa.
    noise : [1 x N] array
        Generated noise applied to tdoa vector

    """
    
    numpairs = pairs.shape[0]
    
    noise = noisemag*noisefun(numpairs)
    
    vects = T - R
    dists = np.linalg.norm(vects, axis = 0).reshape([1,-1])
    
    toa  = dists/c
    tdoa = toa[[0], pairs[:,1]] - toa[[0],pairs[:,0]] + noise
    ddoa = tdoa*c
    
    return tdoa,ddoa,noise

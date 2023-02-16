"""

Defines functions used to create rotation matrices.

"""

import numpy as np

def axang(ax,ang):
    """
    
    Rotation matrix from axis & angle.

    Parameters
    ----------
    ax : [3 x (1)] array, list
        Rotation axis, will be normalized by the function
    ang : scalar
        Rotation angle in radians

    Returns
    -------
    R : [3 x 3] array
        Rotation matrix

    """

    # Normalize axis & flatten vector
    ax = np.reshape(ax,3)
    ax = ax / np.linalg.norm(ax)
    
    # Make skew-symmetric matrix
    A = np.array([[     0 , -ax[2],  ax[1] ],
                  [  ax[2],     0 , -ax[0] ],
                  [ -ax[1],  ax[0],     0  ]])
    
    
    # Create rotation matrix via Rodrigues' rotation formula
    R = np.identity(3) + np.sin(ang) * A + (1 - np.cos(ang)) * A @ A
    
    return R
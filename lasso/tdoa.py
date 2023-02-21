"""

Defines functions used to perform time difference of arrival (TDoA) signal
tracking for the RIT LASSO senior design project.

"""

import numpy as np

def makepairs(R,pairs):
    """
    
    Pairs up desired receivers and returns information related to those pairs.

    Parameters
    ----------
    R : [3 x N] array
        Receiver locations (3-vectors, columns).
    pairs : [M x 2] integer array
        M pairs of receivers, each row has two entries indicating a pair.

    Returns
    -------
    dists : [1 x M] array
        Distances between receivers for each pair.
    origs : [3 x M] array
        Midpoints of receiver pairs (local frame origins).
    norms : [3 x M] array
        Points from receiver 1 to receiver 2, unit length.
    T : [M x 4 x 4] array
        A collection of transformation matrices, used to convert a quadric 
        surface between reference frames.
        
    -------------------------------------------------------------------------

    Information about the frame transformations:

        The quadric equations, for u = [ x; y; z; 1 ], are of the form:
        
            (1)     (u')*Q*u = 0  . . . . (u' indicates the transpose of u)
        
        Consider the transformation T, v = T*u. Then, u = inv(T)*v. In the
        transformed space, the quadric equation is:
        
            (2)     (v')*P*v = 0 

        Substituting u = inv(T)*v into equation (1) gives:
        
            (3)     ((inv(T)*v)')*Q*(inv(T)*v) = 0
                         v'*inv(T)'*Q*inv(T)*v = 0

        Therefore, the transformed quadric matrix is inv(T)'*Q*inv(T). For a
        rotation followed by translation, the translation matrix T is:

            [ ...   ...   ...   p1 
              ...    R    ...   p2 
              ...   ...   ...   p3
               0     0     0     1 ]

        Where the upper left 3 x 3 matrix is the associated rotation matrix 
        and the ps are the components of the translation vector. The inverse 
        can be calculated quickly as follows: transpose the rotation matrix 
        portion of the matrix, then apply that transposed (inverse) RM to the 
        translation vector. The new translation vector is the negative of this 
        rotated vector.

    """

    numpairs = pairs.shape[0]
    
    vects =  R[:, pairs[:,1]] - R[:, pairs[:,0]]
    origs = (R[:, pairs[:,1]] + R[:, pairs[:,0]])/2
    dists = np.linalg.norm(vects, axis = 0).reshape([1,-1])
    norms = vects/dists
    
    # Make rotation matrices from null space, fix improper rotations
    
    rot1 = norms.T.reshape([-1,3,1])
    rot2 = np.linalg.svd(rot1.reshape([-1,1,3]))[2][:,:,1:3]
    rot = np.r_['2', rot1, rot2]
    
    improp = np.linalg.det(rot) < 0
    rot[improp,:,:] = rot[improp,:,:][:,:,[0,2,1]]
    
    rot = np.transpose(rot, axes = (0,2,1))
    
    # Construct transformation matrices
    
    T = np.r_['2', rot, -rot @ origs.T.reshape([-1,3,1])]
    
    lastrows = np.r_['2', np.zeros([numpairs,1,3]), np.ones([numpairs,1,1])]
    
    T = np.r_['1', T, lastrows]
    
    return dists, origs, norms, T
            
def makehyps(ddoa,dists,T):
    """
    
    Creates the "hyperboloid matrices" for a set of distance differences.

    Parameters
    ----------
    ddoa : [1 x N] array
        Vector of station TDoA distance differences, signed.
    dists : [1 x N] array
        Vector of station pair separation distances.
    T : [N x 4 x 4] array
        See "tdoa.makepairs" documentation
        
    Returns
    -------
    H : [N x 4 x 4] array
        Hyperboloid array. The nth sheet is the hyperboloid matrix for the 
        nth receiver hyp. This is the matrix Q such that (v')*Q*v is the 
        hyperboloid's equation, with v = [x; y; z; 1];. The hyperboloid 
        matrices are normalized by their Frobenius norms.

    """

    numhyps = ddoa.shape[1]
    H       = np.zeros([numhyps,4,4])
    
    # Hyperbola matrices in local frame
    
    vals1 = 4/(ddoa * ddoa)
    vals2 = 4/(ddoa * ddoa - dists * dists)
    
    H[:,0,0] = vals1
    H[:,1,1] = vals2
    H[:,2,2] = vals2
    H[:,3,3] = -1
    
    # Convert to global frame
    
    H = np.transpose(T, axes = (0,2,1)) @ H @ T
    
    # Normalize
    
    H = H / np.linalg.norm(H,'fro',(1,2)).reshape([-1,1,1])
        
    return H

def hypssr(points,H):
    """
    
    Sum of squared residuals for a number of quadrics at a number of points.

    Parameters
    ----------
    points : [3 x N] array
        Cartesian 3-vectors at which to evaluate SSR.
    H : [M x 4 x 4] array
        Quadrics to use (see "tdoa.makehyps").

    Returns
    -------
    ssr : [1 x N] array
        Sum of squared residuals at all points.

    """
    
    numpoints = points.shape[1]
    
    one    = np.ones([1,numpoints])
    points = np.r_[points,one]
    
    # Compute residuals
    
    resid = np.sum(points * (H @ points), axis = 1)
    
    # Compute SSR
    
    ssr = np.sum(resid * resid, axis = 0).reshape([1,-1])
    
    return ssr

def hypgrad(points,H):
    """
    
    Calculates the gradient of the sum of squared residuals for a number of 
    quadric functions at a number of points. 

    Parameters
    ----------
    points : [3 x N] array
        Cartesian 3-vectors at which to evaluate SSR.
    H : [M x 4 x 4] array
        Quadrics to use (see "tdoa.makehyps").

    Returns
    -------
    grad : [3 x N] array
        Cartesian 3-vectors, gradient at all points.
    gradmag : [1 x N] array
        Norm of the gradient at all points.
    graddir : [3 x N array]
        Normalized gradient at all points.

    """
    
    numpoints = points.shape[1]
    
    one    = np.ones([1,numpoints])
    points = np.r_[points,one]
    
    # Compute residuals & reshape
    
    resid = (points * (H @ points)).sum(axis = 1)
    
    resid = resid.reshape([-1,1,numpoints])
    
    # Compute gradient: grad(f^2) = 2*f*grad(f) = 2*f*(2*H[0:3,:]*points)
    
    grad = (4*resid*(H[:,0:3,:] @ points)).sum(axis = 0)
    
    # Compute gradient magnitude
    
    gradmag = np.linalg.norm(grad,axis = 0).reshape([1,-1])
    
    # Set infinite gradients to zero
    
    grad[:, gradmag.flatten() == float("inf")] = 0
    
    # Compute gradient directions, avoiding dividing by zero
    
    gradmag_fix = gradmag
    gradmag_fix[gradmag == 0] = float("inf")
    
    graddir = grad/gradmag_fix
    
    return grad,gradmag,graddir

def picksheets(points,origs,norms,tdoa):
    """
    
    Eliminates points based on the sign of the time differences measured at 
    receiver pairs. This effectively eliminates one of the two sheets for each 
    TDoA hyperboloid. If none of the points are valid, the returned points are 
    those that satisfy the most time differences (possibly 0!).

    Parameters
    ----------
    points : [3 x N] array
        Collection of points to evaluate as 3-vectors.
    origs : [3 x M] array
        Receiver-pair midpoints.
    norms : [3 x M] array
        Reciever-pair bisector-plane vectors. These point from the first 
        receiver to the second, so if the time difference is negative, the 
        transmitter is expected to lie on the positive side of the plane.
    tdoa : [1 x M] array
        Time/distance difference vector (only sign is considered).

    Returns
    -------
    goodpoints : [3 x K] array
        Points that satisfy the test.
    bestscore : scalar integer
        Number of hyperboloids the returned points satisfy.

    """
    
    # Points relative to each receiver-pair midpoint
    
    relpoints = points - origs.T.reshape([-1,3,1])
    
    # Dot product w/ receiver-pair bisector plane normals
    
    dotprod = (relpoints * norms.T.reshape([-1,3,1])).sum(axis = 1)
    
    # Check if each is on the correct sign of the plane
    
    checks = (np.sign(dotprod) == np.sign(-tdoa.reshape([-1,1])))
    
    # Score the points based on how many hyperoboloids they satisfy
    
    scores = checks.sum(axis = 0) 
    
    # Create returned values
    
    bestscore = np.max(scores)
    
    goodpoints = points[:,scores == bestscore]
    
    return goodpoints,bestscore

def hypdescend(points,H,howmany,maxstep,minstep,a,b,storepoints = False):
    """
    
    Performs a basic gradient descent algorithm to minimize the squared 
    residuals of a number of different quadrics. Uses a backtracking line 
    search to determine step size, and stops when that step reaches a value.

    Parameters
    ----------
    points : [3 x N] array
        Starting points as Cartesian 3-vectors.
    H : [4 x 4 x M] array
        Quadric functions to use (see tdoa.makehyps).
    howmany : integer scalar
        If this many points have fully descended, the algorithm stops.
    maxstep : scalar
        Maximum/starting step size.
    minstep : scalar
        Minimum/stopping step size.
    a : scalar, (0 < a < 1)
        Sufficent decrease constant. Should be small, maybe 1e-2 or 1e-4.
    b : scalar,(0 < a < 1)
        Step size modifier. Try values greater than 1/2.
    storepoints : boolean scalar
        If true, points are logged at each step to be returned in allpoints.

    Returns
    -------
    points : [3 x K] array
        Descended points.
    allpoints : [J x K] array
        All descended points - K points at the Jth step.

    """
    
    # Initialize
    
    numpoints = points.shape[1]
    
    step = np.tile(maxstep,[1,numpoints])
    trypoints = np.zeros([3,numpoints])
    tryssr = np.zeros([1,numpoints])
    gradmag = np.zeros([1,numpoints])
    graddir = np.zeros([3,numpoints])
    
    if storepoints:
        allpoints = points.reshape([1,3,-1])
    
    done = np.zeros(numpoints, dtype = bool)
    armijo = np.zeros(numpoints, dtype = bool)
    alldone = False
    
    # Gradient descent algorithm
    
    ssr = hypssr(points,H)  # First-time SSR
    
    while not alldone:
        
        gradmag[0,~done],graddir[:,~done] = hypgrad(points[:,~done], H)[1:3]
        
        # Take a step against the gradient and get the SSR
        
        trypoints[:,~done] = points[:,~done] - step[:,~done]*graddir[:,~done]
        tryssr[0,~done] = hypssr(trypoints[:,~done], H)
        
        # Armijo inequality
        
        armijo[~done] = tryssr[0,~done] < ssr[0,~done] - \
            a * step[0,~done] * gradmag[0,~done]
            
        stepstep = ~armijo & ~done
            
        # Prepare for next loop
        
        points[:,armijo] = trypoints[:,armijo]    # Passed Armijo      
        ssr[0,armijo] = tryssr[0,armijo]
        
        step[0,stepstep] = b * step[0,stepstep]   # Didn't pass, isn't done
        
        done[stepstep] = step.flatten()[stepstep] < minstep
    
        if done.sum() >= howmany or done.all():
            
            alldone = True
            
        # Store points for plotting, etc.
        
        if storepoints and armijo.any():
            
            allpoints = np.r_['0',allpoints,points.reshape([1,3,-1])]
            
    if storepoints: 
        return points,allpoints
    else:
        return points
            

import numpy as np
import rotmat
from lasso import tdoa
from lasso import simtools

import matplotlib.pyplot as plt


plt.close('all')

# %% STATION SETUP

c = 1
r = 1.61e-4     

R0 = np.array([[0],[0],[r/2]])
R1 = np.array([[1],[0],[0]]) * r
R2 = rotmat.axang([0,0,1],2*np.pi/3) @ R1
R3 = rotmat.axang([0,0,1],4*np.pi/3) @ R1

receivers = np.c_[R0,R1,R2,R3]

transmitter = r * np.array([[1],[2],[3]])

pairs = np.array([[0,1],[0,2],[0,3]])

numpair = pairs.shape[0]

noisefun = lambda x: np.random.randn(1,x)
noisemag = 0

# %% Initial Computations

dists,origins,normals,T = tdoa.makepairs(receivers,pairs)
tdoas,ddoas,noise = simtools.maketdoa(transmitter,receivers,pairs,c,noisefun,noisemag)
H = tdoa.makehyps(ddoas,dists,T)

# %% Tests for ssr, gradient...

here = np.array([[0],[0],[r]])

cubeside = 10*r
subdivs = 5

ruler = np.linspace(-0.5,0.5,subdivs + 2)[1:-1]

x,y,z = np.meshgrid(ruler,ruler,ruler)
points = np.r_[x.T.reshape([1,-1]), y.T.reshape([1,-1]), z.T.reshape([1,-1])]*cubeside + here

points,bestscore = tdoa.picksheets(points,origins,normals,tdoas)

ssr = tdoa.hypssr(points,H)

grad,gradmag,graddir = tdoa.hypgrad(points,H)

# %% Gradient descent

a = 1e-2;
b = 4/5;

maxstep = r/3;
minstep = r/25;

howmany = points.shape[1];

points,allpoints = tdoa.hypdescend(points,H,howmany,maxstep,minstep,a,b,True)

pointsfinal,finalscore = tdoa.picksheets(points,origins,normals,tdoas)

# %% Plot stuff...

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

xs = allpoints[:,0,:].flatten()
ys = allpoints[:,1,:].flatten()
zs = allpoints[:,2,:].flatten()

xp = points[0,:].flatten()
yp = points[1,:].flatten()
zp = points[2,:].flatten()

xf = pointsfinal[0,:].flatten()
yf = pointsfinal[1,:].flatten()
zf = pointsfinal[2,:].flatten()

xr = receivers[0,:].flatten()
yr = receivers[1,:].flatten()
zr = receivers[2,:].flatten()

xt = transmitter[0,:].flatten()
yt = transmitter[1,:].flatten()
zt = transmitter[2,:].flatten()

ax.scatter(xs, ys, zs, marker='.', c='#538ce6')
ax.scatter(xt, yt, zt, marker='H', c='#db4f2c', s=50)
ax.scatter(xr, yr, zr, marker='^', c='#000000', s=50)
ax.scatter(xp, yp, zp, marker='o', c='#f5c92a', s=50)
ax.scatter(xf, yf, zf, marker='x', c='#1fbf24', s=100)




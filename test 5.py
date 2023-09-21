import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the function to plot
def f(theta, phi, r):
    return np.sin(phi)

# Define the range of theta and phi
#theta from 0 to 2pi, picking k^2 points
k=10
k2=k*k
theta = np.linspace(-np.pi/2, np.pi/2, k2)
phi = np.linspace(0, np.pi, k2)
r=np.linspace(0, 2, k2)
e=2.718
#A is a decaying constant which I dont know how to be accrute about it, but if A has a range of (0,1), the whole intergal of r will be odd. It will be demonstrate later.
A=1.5
#r0 stands for the radius of the smallest distance between the detector and the IP
r0=-0.8
#phi0 and
phi0=np.pi/2
theta0=np.pi/2
sigmaphi=3.14159/8
sigmatheta=3.14159/16
sigmar=0.5

# Create a meshgrid from theta and phi
theta, phi = np.meshgrid(theta, phi)

phim = np.repeat(phi[:, np.newaxis], k2, axis=1)
thetam = np.repeat(theta[:, np.newaxis], k2, axis=1)
rm = np.repeat(r[:, np.newaxis], k2, axis=1)

print('the size of rm is ', rm.shape)
print('the size of thetam is ', thetam.shape)
print('the size of phim is ', phim.shape)

# Calculate the Cartesian coordinates
# There is a Problem since the Matrix is strict generated and mutiplied, so every element is mutilplied by the cprresponding element of the other
# Matrix. That is, the first element in phi only multiplied by the first element in theta. 
# Hence, if we set the theta = np.linespace(0, 2*np.pi, 100), we only get half the figure.

xm = rm*np.sin(phim) * np.cos(thetam)
ym = rm*np.sin(phim) * np.sin(thetam)
zm = rm*np.cos(phim)
print('the size of xm is ', xm.shape)
print('the size of ym is ', ym.shape)
print('the size of zm is ', zm.shape)
# Reshape Z to a 2-D array
#x = xm.reshape(-2, zm.shape[-2])
#y = ym.reshape(-2, zm.shape[-2])
z = zm.reshape(k2*k, k2*k)
x = xm.reshape(k2*k, k2*k)
y = ym.reshape(k2*k, k2*k)
print('the size of x is ', x.shape)
print('the size of y is ', y.shape)
print('the size of z is ', z.shape)
#print(x)
# the probability distribution function

pm = (1 / (2 * np.pi * sigmaphi * sigmatheta)) * np.exp((-A * (r0 + rm)) - ((thetam - theta0) ** 2 / (2 * sigmatheta ** 2)) - ((phim - phi0) ** 2 / (2 * sigmaphi ** 2)))
#or
#p = (1 / (2 * np.pi * sigmaphi * sigmatheta)*np.sqrt(2 * np.pi * sigmar)) * np.exp((-A * (r0 + r))
# -((theta - theta0) ** 2 / (2 * sigmatheta ** 2)) - ((phi - phi0) ** 2 / (2 * sigmaphi ** 2)))
#This is the pure Gaussian Distribution
print('the size of pm is ', pm.shape)
# Create the figure and the 3D axis
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

p=pm.reshape(k2*k, k2*k)
print('the size of p is ', p.shape)
# Plot the surface
ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=plt.cm.jet(p))

# Set the axis limits and labels
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=0., azim=-170)
# Show the plot
plt.savefig('3d_plot.png')
plt.show()



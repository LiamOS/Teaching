import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as math

# Here are the limits for plotting:
xMin = -2
xMax = 2
yMin = -2
yMax = 2

# Edit this funtion to return the scalar field of your choice!
def functionToPlot(x,y):
  return math.exp( -(x**2 + y**2) ) # Gaussian
  #return x**2 - y**3 # x squared minus y cubed
  #return x**2 + y**2 # r squared, harmonic oscillator potential

  
# Gridpoint spacing, smaller spacing = longer to run but finer detail. 100 should be fine.
dx = (xMax - xMin)/100.
dy = (yMax - yMin)/100.

# The rest is magic mostly.
x   = math.arange(yMin,xMax+dx,dx)
y   = math.arange(yMin,yMax+dy,dy)
X, Y = math.meshgrid(x,y) # This produces a 2D grid necessary for pcolormesh()
dat = math.zeros( (len(x),len(y)) , dtype=float , order='C')
for i in range(len(x)):
  for j in range(len(y)):
    dat[i][j] = functionToPlot(x[i], y[j])

plt.pcolormesh(X,Y,dat,cmap=cm.get_cmap("viridis"),edgecolors='face')
plt.xlim([xMin,xMax])
plt.ylim([yMin,yMax])
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
#plt.savefig("Scalar.png",format="PNG") # Uncomment to save a picture.

# Now for the vector field thing
grad_x = math.array([ [ dat[i][j]*x[i] for i in range(len(x)) ] for j in range(len(y)) ])
grad_y = math.array([ [ dat[i][j]*x[j] for i in range(len(x)) ] for j in range(len(y)) ])
grad = math.gradient(dat)
grad_x = -grad[1]
grad_y = -grad[0]

plt.streamplot(x, y, grad_x, grad_y, arrowsize=3, density=1, color=math.abs(math.sqrt(math.square(grad_x)+math.square(grad_y))), cmap=cm.get_cmap("viridis"))
plt.xlim([xMin, xMax])
plt.ylim([yMin, yMax])
plt.xlabel("x")
plt.ylabel("y")
plt.colorbar()
plt.show()
#plt.savefig("Vector.png",format="PNG") # Uncomment to save a picture.

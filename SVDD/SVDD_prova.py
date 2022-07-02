import sys
sys.path
sys.path.append('/Users/test1/Documents/TESI_Addfor_Industriale/Python_Projects_Shapelets/SVM/')
import numpy as np
from numpy.random import seed
from numpy.random import normal, uniform
from SVDD import SVDD
import matplotlib.pyplot as plt

################# SYNTHETIC DATA FOR SVDD ALGORITHM TRIALS #################

################# PART 1: center of the hypersphere is a parameter
seed(0)
X = []
# generate 3 samples of length 2 from N(0,1)
for i in range(3):
    X.append(normal(0,1,2))

# generate 40 samples of length 2 from N(10,1)
for j in range(40):
    X.append(normal(10, 1 , 2))
X= np.array(X)
X
print(X.shape)


svdd = SVDD(C=0.09)
svdd.fit(X)
svdd.sv
svdd.sv_index
svdd.decision_function(X)
svdd.center
svdd.radius
svdd.boundary_sv_index
svdd.sv_index
np.sum(svdd.total_alpha) 

############## PLOT

x_max = max(X[:,0]) + 1
x_min = min(X[:,0]) - 1

y_max = max(X[:,1]) + 1
y_min = min(X[:,1]) - 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
# Z = svdd.project(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

plt.figure()
plt.title(f"Decision boundary", fontweight="bold")

# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")
plt.scatter(X[:, 0], X[:, 1], facecolor='C0', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Normal")

plt.legend(ncol=1, loc='upper left', edgecolor='black',
                     markerscale=1.2, fancybox=True)
plt.xlim((x_min, x_max ))
plt.ylim((y_min, y_max))
plt.ylabel("shapelet 2")
plt.xlabel("shapelet 1")
plt.show()

################# PART 2: center zero 

seed(1)
X = []
# generate 40 samples of length 2 from U(0,1)
for i in range(40):
    X.append(uniform(0,1,2))

# generate 3 samples of length 2 from U(1, 2)
for j in range(3):
    X.append(uniform(1, 2 , 2))
X= np.array(X)
X
print(X.shape)


svdd = SVDD(C=0.3, zero_center=True, tol=1e-7)
svdd.fit(X)
svdd.alpha
svdd.total_alpha
svdd.sv
svdd.kernel(X[42], X[42])
svdd.sv_index
svdd.decision_function(X)
svdd.center
svdd.radius
svdd.boundary_sv_index
svdd.sv_index
np.sum(svdd.total_alpha) 


############## PLOT

x_max = max(X[:,0]) + 1
x_min = min(X[:,0]) - 1

y_max = max(X[:,1]) + 1
y_min = min(X[:,1]) - 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
# Z = svdd.project(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)

plt.figure()
plt.title(f"Decision boundary", fontweight="bold")

# plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")
plt.scatter(X[:, 0], X[:, 1], facecolor='C0', marker='o', s=100, linewidths=2,
                     edgecolor='black', zorder=2, label="Normal")

plt.legend(ncol=1, loc='upper left', edgecolor='black',
                     markerscale=1.2, fancybox=True)
plt.xlim((x_min, x_max ))
plt.ylim((y_min, y_max))
plt.ylabel("shapelet 2")
plt.xlabel("shapelet 1")
plt.show()
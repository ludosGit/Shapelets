import scipy
from scipy.signal import correlate
x = [1,1,1,1,5,5,5,1,1]
y = [1,1,1,1,1,1,1,1,1]
z = correlate(x,y)
z
max(z)


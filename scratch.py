import scipy
import numpy as np
from scipy.signal import correlate
x = np.array([1,1,1,1,5,5,5,1,1])
mu_x = np.mean(x)
sd_x = np.std(x)
x = (x - mu_x) / (sd_x*len(x))

y = np.array([1,5,5,5,1,1,1,1,1])
mu_y = np.mean(y)
sd_y = np.std(y)
y = (y - mu_y) /sd_y


z = np.correlate(x,y, mode='full')
max(z)



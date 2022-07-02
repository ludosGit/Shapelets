cimport cython
import numpy as np
cimport numpy as np

DTYPE = np.float32
ctypedef np.float32_t DTYPE_t

@cython.boundscheck(False)
def compare_bboxes(
       np.ndarray[DTYPE_t, ndim=2] boxes1,
       np.ndarray[DTYPE_t, ndim=2] boxes2):
...
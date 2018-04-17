import numpy as np
from numba import vectorize
import numpy as np
from pyculib import rand as curand

@vectorize(['float32(float32, float32)'], target='cuda')
def Add(a, b):
  return a + b
# Initialize arrays
N = 100000
A = np.ones(N, dtype=np.float32)
B = np.ones(A.shape, dtype=A.dtype)
C = np.empty_like(A, dtype=A.dtype)
# Add arrays on GPU
C = Add(A, B)

prng = curand.PRNG(rndtype=curand.PRNG.XORWOW)
rand = np.empty(100000)
prng.uniform(rand)
print(rand[:10])

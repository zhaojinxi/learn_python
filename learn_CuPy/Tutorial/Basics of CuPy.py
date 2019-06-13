import numpy as np
import cupy as cp

x_gpu = cp.array([1, 2, 3])

x_cpu = np.array([1, 2, 3])
l2_cpu = np.linalg.norm(x_cpu)

x_gpu = cp.array([1, 2, 3])
l2_gpu = cp.linalg.norm(x_gpu)

x_on_gpu0 = cp.array([1, 2, 3, 4, 5])

x_on_gpu0 = cp.array([1, 2, 3, 4, 5])
cp.cuda.Device(1).use()
x_on_gpu1 = cp.array([1, 2, 3, 4, 5])

with cp.cuda.Device(1):
    x_on_gpu1 = cp.array([1, 2, 3, 4, 5])
x_on_gpu0 = cp.array([1, 2, 3, 4, 5])

# with cp.cuda.Device(0):
#     x_on_gpu0 = cp.array([1, 2, 3, 4, 5])
# with cp.cuda.Device(1):
#     x_on_gpu0 * 2  # raises error

with cp.cuda.Device(1):
    x = cp.array([1, 2, 3, 4, 5])
x.device

x_cpu = np.array([1, 2, 3])
x_gpu = cp.asarray(x_cpu)  # move the data to the current device.

with cp.cuda.Device(0):
    x_gpu_0 = cp.ndarray([1, 2, 3])  # create an array in GPU 0
with cp.cuda.Device(1):
    x_gpu_1 = cp.asarray(x_gpu_0)  # move the array to GPU 1

x_gpu = cp.array([1, 2, 3])  # create an array in the current device
x_cpu = cp.asnumpy(x_gpu)  # move the array to the host.

x_cpu = x_gpu.get()

# Stable implementation of log(1 + exp(x))
def softplus(x):
    xp = cp.get_array_module(x)
    return xp.maximum(0, x) + xp.log1p(xp.exp(-abs(x)))

np.allclose(x_cpu, [1, 2, 3])
np.allclose(x_gpu, [1, 2, 3])
np.allclose(cp.asnumpy(x_cpu), [1, 2, 3])
np.allclose(cp.asnumpy(x_gpu), [1, 2, 3])
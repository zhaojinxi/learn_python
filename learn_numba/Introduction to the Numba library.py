import numba
import numpy

# jit decorator tells Numba to compile this function.
# The argument types will be inferred by Numba when function is called.
@numba.jit
def sum2d(arr):
    M, N = arr.shape
    result = 0.0
    for i in range(M):
        for j in range(N):
            result += arr[i,j]
    return result
a = numpy.arange(9).reshape(3,3)
print(sum2d(a))

# pure Python implementation of the ABC-Model
def abc_model_py(a, b, c, rain):
    # initialize array for the stream discharge of each time step
    outflow = numpy.zeros((rain.size), dtype=numpy.float64)
    # placeholder, in which we save the storage content of the previous and
    # current timestep
    state_in = 0
    state_out = 0
    for i in range(rain.size):
        # Update the storage
        state_out = (1 - c) * state_in + a * rain[i]
        # Calculate the stream discharge
        outflow[i] = (1 - a - b) * rain[i] + c * state_out
        # Overwrite the storage variable
        state_in = state_out
    return outflow

@numba.jit
def abc_model_numba_jit(a, b, c, rain):
    outflow = numpy.zeros((rain.size), dtype=numpy.float64)
    state_in = 0
    state_out = 0
    for i in range(rain.size):
        state_out = (1 - c) * state_in + a * rain[i]
        outflow[i] = (1 - a - b) * rain[i] + c * state_out
        state_in = state_out
    return outflow

@numba.vectorize
def abc_model_numba_vectorize(a, b, c, rain):
    outflow = numpy.zeros((rain.size), dtype=numpy.float64)
    state_in = 0
    state_out = 0
    for i in range(rain.size):
        state_out = (1 - c) * state_in + a * rain[i]
        outflow[i] = (1 - a - b) * rain[i] + c * state_out
        state_in = state_out
    return outflow

@numba.cuda
def abc_model_numba_cuda(a, b, c, rain):
    outflow = numpy.zeros((rain.size), dtype=numpy.float64)
    state_in = 0
    state_out = 0
    for i in range(rain.size):
        state_out = (1 - c) * state_in + a * rain[i]
        outflow[i] = (1 - a - b) * rain[i] + c * state_out
        state_in = state_out
    return outflow
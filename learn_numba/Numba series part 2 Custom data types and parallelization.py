import timeit
import numpy as np
from numba import jit, prange

abc_dtype = np.dtype([('a', np.float64),
                      ('b', np.float64),
                      ('c', np.float64)])

arr = np.zeros(3, dtype=abc_dtype)

arr['a'][0] = 1

# Reference implementation without Numba optimization
def abc_model_py(params, rain):
    # initialize model variables
    outflow = np.zeros((rain.size, params.size), dtype=np.float64)

    # loop over parameter sets
    for i in range(params.size):
        # unpack model parameters
        a = params['a'][i]
        b = params['b'][i]
        c = params['c'][i]

        # Reset model states
        state_in = 0
        state_out = 0

        # Actual simulation loop
        for j in range(rain.size):
            state_out = (1 - c) * state_in + a * rain[j]
            outflow[j,i] = (1 - a - b) * rain[j] + c * state_in
            state_in = state_out
    return outflow
# Jit'ed but not parallelized implementation
@jit(nopython=True)
def abc_model_jit(params, rain):
    # initialize model variables
    outflow = np.zeros((rain.size, params.size), dtype=np.float64)

    # loop over parameter sets
    for i in range(params.size):
        # unpack model parameters
        a = params['a'][i]
        b = params['b'][i]
        c = params['c'][i]

        # Reset model states
        state_in = 0
        state_out = 0

        # Actual simulation loop
        for j in range(rain.size):
            state_out = (1 - c) * state_in + a * rain[j]
            outflow[j,i] = (1 - a - b) * rain[j] + c * state_in
            state_in = state_out
    return outflow
# Implementation with implicit parallelization
@jit(nopython=True, parallel=True)
def abc_model_impl(params, rain):
    # initialize model variables
    outflow = np.zeros((rain.size, params.size), dtype=np.float64)

    # loop over parameter sets
    for i in range(params.size):
        # unpack model parameters
        a = params['a'][i]
        b = params['b'][i]
        c = params['c'][i]

        # Reset model states
        state_in = 0
        state_out = 0

        # Actual simulation loop
        for j in range(rain.size):
            state_out = (1 - c) * state_in + a * rain[j]
            outflow[j,i] = (1 - a - b) * rain[j] + c * state_in
            state_in = state_out
    return outflow
# Implementation with explicit parallelization (see prange in 1st loop)
@jit(nopython=True, parallel=True)
def abc_model_expl(params, rain):
    # initialize model variables
    outflow = np.zeros((rain.size, params.size), dtype=np.float64)

    # loop over parameter sets
    for i in prange(params.size):
        # unpack model parameters
        a = params['a'][i]
        b = params['b'][i]
        c = params['c'][i]

        # Reset model states
        state_in = 0
        state_out = 0

        # Actual simulation loop
        for j in range(rain.size):
            state_out = (1 - c) * state_in + a * rain[j]
            outflow[j,i] = (1 - a - b) * rain[j] + c * state_in
            state_in = state_out
    return outflow

params = np.random.random(8).astype(abc_dtype)
rain = np.random.random(10**6)


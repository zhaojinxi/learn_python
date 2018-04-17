import timeit
import numpy as np
from numba import jit

@jit
def abc_model_1(a, b, c, rain):
    """First implementation of the ABC-Model.

    Args:
        a, b, c: Model parameter as scalars.
        rain: Array of input rain.

    Returns:
        outflow: Simulated stream flow.

    """
    # Initialize model variables
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0

    # Actual simulation loop
    for i in range(rain.size):
        state_out = (1 - c) * state_in + a * rain[i]
        outflow[i] = (1 - a - b) * rain[i] + c * state_in
        state_in = state_out
    return outflow
@jit
def abc_model_2(params, rain):
    """Second implementation of the ABC-Model.

    Args:
        params: A dictionary, containing the three model parameters.
        rain: Array of input rain.

    Returns:
        outflow: Simulated stream flow.

    """
    # Initialize model variables
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0

    # Actual simulation loop
    for i in range(rain.size):
        state_out = (1 - params['c']) * state_in + params['a'] * rain[i]
        outflow[i] = ((1 - params['a'] - params['b']) * rain[i]
                      + params['c'] * state_in)
        state_in = state_out
    return outflow

@jit(nopython=True)
def abc_model_3(params, rain):
    """Second implementation of the ABC-Model.

    Args:
        params: A dictionary, containing the three model parameters.
        rain: Array of input rain.

    Returns:
        outflow: Simulated stream flow.

    """
    # Initialize model variables
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0

    # Actual simulation loop
    for i in range(rain.size):
        state_out = (1 - params['c']) * state_in + params['a'] * rain[i]
        outflow[i] = ((1 - params['a'] - params['b']) * rain[i]
                      + params['c'] * state_in)
        state_in = state_out
    return outflow

@jit('float64[:](float64, float64, float64, float64[:])')
def abc_model_4(a, b, c, rain):
    """First implementation of the ABC-Model.

    Args:
        a, b, c: Model parameter as scalars.
        rain: Array of input rain.

    Returns:
        outflow: Simulated stream flow.

    """
    # Initialize model variables
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0

    # Actual simulation loop
    for i in range(rain.size):
        state_out = (1 - c) * state_in + a * rain[i]
        outflow[i] = (1 - a - b) * rain[i] + c * state_in
        state_in = state_out
    return outflow

@jit('float64(float64, float64, float64, float64)', nopython=True)
def get_new_state(old_state, a, c, rain):
    return (1 - c) * old_state + a * rain
@jit('float64(float64, float64, float64, float64, float64)', nopython=True)
def get_outflow(a, b, c, rain, state):
    return (1 - a - b) * rain + c * state
@jit('float64[:](float64, float64, float64, float64[:])', nopython=True)
def abc_model_5(a, b, c, rain):
    """First implementation of the ABC-Model.

    Args:
        a, b, c: Model parameter as scalars.
        rain: Array of input rain.

    Returns:
        outflow: Simulated stream flow.

    """
    # Initialize model variables
    outflow = np.zeros((rain.size), dtype=np.float64)
    state_in = 0
    state_out = 0

    # Actual simulation loop
    for i in range(rain.size):
        state_out = get_new_state(state_in, a, c, rain[i])
        outflow[i] = get_outflow(a, b, c, rain[i], state_out)
        state_in = state_out
    return outflow
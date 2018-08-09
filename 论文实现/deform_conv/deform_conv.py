import numpy
from scipy.ndimage.interpolation import map_coordinates as sp_map_coordinates
import tensorflow


def tf_flatten(a):
    """Flatten tensor"""
    return tensorflow.reshape(a, [-1])


def tf_repeat(a, repeats, axis=0):
    """TensorFlow version of numpy.repeat for 1D"""
    # https://github.com/tensorflow/tensorflow/issues/8521
    assert len(a.get_shape()) == 1
    a = tensorflow.expand_dims(a, -1)
    a = tensorflow.tile(a, [1, repeats])
    a = tf_flatten(a)
    return a


def tf_repeat_2d(a, repeats):
    """Tensorflow version of numpy.repeat for 2D"""
    assert len(a.get_shape()) == 2
    a = tensorflow.expand_dims(a, 0)
    a = tensorflow.tile(a, [repeats, 1, 1])
    return a


def tf_map_coordinates(input, coords, order=1):
    """Tensorflow verion of scipy.ndimage.map_coordinates

    Note that coords is transposed and only 2D is supported

    Parameters
    ----------
    input : tensorflow.Tensor. shape = (s, s)
    coords : tensorflow.Tensor. shape = (n_points, 2)
    """

    assert order == 1

    coords_lt = tensorflow.cast(tensorflow.floor(coords), 'int32')
    coords_rb = tensorflow.cast(tensorflow.ceil(coords), 'int32')
    coords_lb = tensorflow.stack([coords_lt[:, 0], coords_rb[:, 1]], axis=1)
    coords_rt = tensorflow.stack([coords_rb[:, 0], coords_lt[:, 1]], axis=1)

    vals_lt = tensorflow.gather_nd(input, coords_lt)
    vals_rb = tensorflow.gather_nd(input, coords_rb)
    vals_lb = tensorflow.gather_nd(input, coords_lb)
    vals_rt = tensorflow.gather_nd(input, coords_rt)

    coords_offset_lt = coords - tensorflow.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[:, 0]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[:, 0]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[:, 1]

    return mapped_vals


def sp_batch_map_coordinates(inputs, coords):
    """Reference implementation for batch_map_coordinates"""
    coords = coords.clip(0, inputs.shape[1] - 1)
    mapped_vals = numpy.array([sp_map_coordinates(input, coord.T, mode='nearest', order=1) for input, coord in zip(inputs, coords)])
    return mapped_vals

def tf_batch_map_coordinates(input, coords, order=1):
    """Batch version of tf_map_coordinates

    Only supports 2D feature maps

    Parameters
    ----------
    input : tensorflow.Tensor. shape = (b, s, s)
    coords : tensorflow.Tensor. shape = (b, n_points, 2)
    """

    input_shape = tensorflow.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]
    n_coords = tensorflow.shape(coords)[1]

    coords = tensorflow.clip_by_value(coords, 0, tensorflow.cast(input_size, 'float32') - 1)
    coords_lt = tensorflow.cast(tensorflow.floor(coords), 'int32')
    coords_rb = tensorflow.cast(tensorflow.ceil(coords), 'int32')
    coords_rt = tensorflow.stack([coords_lt[..., 0], coords_rb[..., 1]], axis=-1)
    coords_lb = tensorflow.stack([coords_rb[..., 0], coords_lt[..., 1]], axis=-1)

    # idx = tf_repeat(tensorflow.range(batch_size), n_coords)
    a = tensorflow.expand_dims(tensorflow.range(batch_size), -1)
    a = tensorflow.tile(a, [1, n_coords])
    # idx = tf_flatten(a)
    idx=tensorflow.reshape(a, [-1])

    # def _get_vals_by_coords(input, coords):
    #     indices = tensorflow.stack([idx, tf_flatten(coords[..., 0]), tf_flatten(coords[..., 1])], axis=-1)
    #     vals = tensorflow.gather_nd(input, indices)
    #     vals = tensorflow.reshape(vals, (batch_size, n_coords))
    #     return vals

    # vals_lt = _get_vals_by_coords(input, coords_lt)
    indices = tensorflow.stack([idx, tensorflow.reshape(coords_lt[..., 0],[-1]), tensorflow.reshape(coords_lt[..., 1],[-1])], axis=-1)
    vals = tensorflow.gather_nd(input, indices)
    vals_lt = tensorflow.reshape(vals, (batch_size, n_coords))   
    # vals_rb = _get_vals_by_coords(input, coords_rb)
    indices = tensorflow.stack([idx, tensorflow.reshape(coords_rb[..., 0],[-1]), tensorflow.reshape(coords_rb[..., 1],[-1])], axis=-1)
    vals = tensorflow.gather_nd(input, indices)
    vals_rb = tensorflow.reshape(vals, (batch_size, n_coords))   
    # vals_lb = _get_vals_by_coords(input, coords_lb)
    indices = tensorflow.stack([idx, tensorflow.reshape(coords_lb[..., 0],[-1]), tensorflow.reshape(coords_lb[..., 1],[-1])], axis=-1)
    vals = tensorflow.gather_nd(input, indices)
    vals_lb = tensorflow.reshape(vals, (batch_size, n_coords))  
    # vals_rt = _get_vals_by_coords(input, coords_rt)
    indices = tensorflow.stack([idx, tensorflow.reshape(coords_rt[..., 0],[-1]), tensorflow.reshape(coords_rt[..., 1],[-1])], axis=-1)
    vals = tensorflow.gather_nd(input, indices)
    vals_rt = tensorflow.reshape(vals, (batch_size, n_coords))  

    coords_offset_lt = coords - tensorflow.cast(coords_lt, 'float32')
    vals_t = vals_lt + (vals_rt - vals_lt) * coords_offset_lt[..., 1]
    vals_b = vals_lb + (vals_rb - vals_lb) * coords_offset_lt[..., 1]
    mapped_vals = vals_t + (vals_b - vals_t) * coords_offset_lt[..., 0]

    return mapped_vals


def sp_batch_map_offsets(input, offsets):
    """Reference implementation for tf_batch_map_offsets"""

    batch_size = input.shape[0]
    input_size = input.shape[1]

    offsets = offsets.reshape(batch_size, -1, 2)
    grid = numpy.stack(numpy.mgrid[:input_size, :input_size], -1).reshape(-1, 2)
    grid = numpy.repeat([grid], batch_size, axis=0)
    coords = offsets + grid
    coords = coords.clip(0, input_size - 1)

    mapped_vals = sp_batch_map_coordinates(input, coords)
    return mapped_vals


def tf_batch_map_offsets(input, offsets, order=1):
    """Batch map offsets into input

    Parameters
    ---------
    input : tensorflow.Tensor. shape = (b, s, s)
    offsets: tensorflow.Tensor. shape = (b, s, s, 2)
    """

    input_shape = tensorflow.shape(input)
    batch_size = input_shape[0]
    input_size = input_shape[1]

    offsets = tensorflow.reshape(offsets, (batch_size, -1, 2))
    grid = tensorflow.meshgrid(tensorflow.range(input_size), tensorflow.range(input_size), indexing='ij')
    grid = tensorflow.stack(grid, axis=-1)
    grid = tensorflow.cast(grid, 'float32')
    grid = tensorflow.reshape(grid, (-1, 2))
    # grid = tf_repeat_2d(grid, batch_size)
    grid = tensorflow.expand_dims(grid, 0)
    grid = tensorflow.tile(grid, [batch_size, 1, 1])
    coords = offsets + grid

    mapped_vals = tf_batch_map_coordinates(input, coords)
    return mapped_vals

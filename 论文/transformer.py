import tensorflow

def spatial_transformer_network(input_fmap, theta, out_dims=None, **kwargs):
    B = tensorflow.shape(input_fmap)[0]
    H = tensorflow.shape(input_fmap)[1]
    W = tensorflow.shape(input_fmap)[2]

    theta = tensorflow.reshape(theta, [B, 2, 3])

    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = affine_grid_generator(out_H, out_W, theta)
    else:
        batch_grids = affine_grid_generator(H, W, theta)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

    return out_fmap

def get_pixel_value(img, x, y):
    shape = tensorflow.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tensorflow.range(0, batch_size)
    batch_idx = tensorflow.reshape(batch_idx, (batch_size, 1, 1))
    b = tensorflow.tile(batch_idx, (1, height, width))

    indices = tensorflow.stack([b, y, x], 3)

    return tensorflow.gather_nd(img, indices)

def affine_grid_generator(height, width, theta):
    num_batch = tensorflow.shape(theta)[0]

    # create normalized 2D grid
    x = tensorflow.linspace(-1.0, 1.0, width)
    y = tensorflow.linspace(-1.0, 1.0, height)
    x_t, y_t = tensorflow.meshgrid(x, y)

    # flatten
    x_t_flat = tensorflow.reshape(x_t, [-1])
    y_t_flat = tensorflow.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tensorflow.ones_like(x_t_flat)
    sampling_grid = tensorflow.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tensorflow.expand_dims(sampling_grid, axis=0)
    sampling_grid = tensorflow.tile(sampling_grid, tensorflow.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    theta = tensorflow.cast(theta, 'float32')
    sampling_grid = tensorflow.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    batch_grids = tensorflow.matmul(theta, sampling_grid)
    # batch grid has shape (num_batch, 2, H*W)

    # reshape to (num_batch, H, W, 2)
    batch_grids = tensorflow.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids

def bilinear_sampler(img, x, y):
    H = tensorflow.shape(img)[1]
    W = tensorflow.shape(img)[2]
    max_y = tensorflow.cast(H - 1, 'int32')
    max_x = tensorflow.cast(W - 1, 'int32')
    zero = tensorflow.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tensorflow.cast(x, 'float32')
    y = tensorflow.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tensorflow.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tensorflow.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tensorflow.cast(tensorflow.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tensorflow.cast(tensorflow.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tensorflow.clip_by_value(x0, zero, max_x)
    x1 = tensorflow.clip_by_value(x1, zero, max_x)
    y0 = tensorflow.clip_by_value(y0, zero, max_y)
    y1 = tensorflow.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tensorflow.cast(x0, 'float32')
    x1 = tensorflow.cast(x1, 'float32')
    y0 = tensorflow.cast(y0, 'float32')
    y1 = tensorflow.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tensorflow.expand_dims(wa, axis=3)
    wb = tensorflow.expand_dims(wb, axis=3)
    wc = tensorflow.expand_dims(wc, axis=3)
    wd = tensorflow.expand_dims(wd, axis=3)

    # compute output
    out = tensorflow.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out
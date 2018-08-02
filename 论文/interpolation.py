import numpy
from PIL import Image
import skimage.io
import skimage.transform
import math

def affine_grid_generator(height, width, M):
    x = numpy.linspace(-1, 1, width)
    y = numpy.linspace(-1, 1, height)

    x_t, y_t = numpy.meshgrid(x, y)

    ones = numpy.ones(numpy.prod(x_t.shape))
    sampling_grid = numpy.vstack([x_t.flatten(), y_t.flatten(), ones])
    sampling_grid=numpy.expand_dims(sampling_grid,0).repeat(M.shape[0],0)

    batch_grids = numpy.matmul(M, sampling_grid)

    batch_grids = batch_grids.reshape(M.shape[0], 2, height, width)
    batch_grids = numpy.moveaxis(batch_grids, 1, -1)

    x_s = (batch_grids[:, :, :, 0] + 1.)/2
    y_s = (batch_grids[:, :, :, 1] + 1.)/2

    return x_s,y_s

def bilinear_sampler(input_img, x, y):
    # rescale x and y to [0, W/H]
    x = x * (input_img.shape[1])
    y = y * (input_img.shape[1])
    # x = x * (input_img.shape[1])
    # y = y * (input_img.shape[1])    

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = numpy.floor(x).astype(numpy.int64)
    x1 = numpy.ceil(x).astype(numpy.int64)
    y0 = numpy.floor(y).astype(numpy.int64)
    y1 = numpy.ceil(y).astype(numpy.int64)

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # make sure it's inside img range [0, H] or [0, W]
    x0 = numpy.clip(x0, 0, (input_img.shape[1] - 1))
    x1 = numpy.clip(x1, 0, (input_img.shape[1] - 1))
    y0 = numpy.clip(y0, 0, (input_img.shape[2] - 1))
    y1 = numpy.clip(y1, 0, (input_img.shape[2] - 1))

    # look up pixel values at corner coords
    Ia = input_img[numpy.arange(input_img.shape[0])[:, None, None], y0, x0]
    Ib = input_img[numpy.arange(input_img.shape[0])[:, None, None], y1, x0]
    Ic = input_img[numpy.arange(input_img.shape[0])[:, None, None], y0, x1]
    Id = input_img[numpy.arange(input_img.shape[0])[:, None, None], y1, x1]

    # add dimension for addition
    wa = numpy.expand_dims(wa, axis=3)
    wb = numpy.expand_dims(wb, axis=3)
    wc = numpy.expand_dims(wc, axis=3)
    wd = numpy.expand_dims(wd, axis=3)

    # compute output
    out = wa*Ia + wb*Ib + wc*Ic + wd*Id

    return out

img1=numpy.expand_dims(skimage.transform.resize(skimage.io.imread('./data/cat1.jpg'),(500,500)),0)
img2=numpy.expand_dims(skimage.transform.resize(skimage.io.imread('./data/cat2.jpg'),(500,500)),0)
input_img = numpy.concatenate([img1, img2], axis=0)

M = numpy.array([[math.sin(math.pi/6), -math.cos(math.pi/6), -0.2], [math.cos(math.pi/6), math.sin(math.pi/6), 0.2]]).reshape(1,2,3).repeat(input_img.shape[0],0)

x_s,y_s = affine_grid_generator(input_img.shape[1], input_img.shape[2], M)

# x_s = batch_grids[:, :, :, 0]
# y_s = batch_grids[:, :, :, 1]

out = bilinear_sampler(input_img, x_s, y_s)

# skimage.io.imshow(out[0]*255*255)

x_max = numpy.max(out[-1])
if x_max != 0:
    x =out[-1]/ x_max
x *= 255
Image.fromarray(x.astype('uint8'), 'RGB').show()
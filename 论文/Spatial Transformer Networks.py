import tensorflow
import scipy.ndimage
import numpy
import matplotlib.pyplot
import cv2

def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    def _repeat(x, n_repeats):
        with tensorflow.variable_scope('_repeat'):
            rep = tensorflow.transpose(tensorflow.expand_dims(tensorflow.ones(shape=tensorflow.stack([n_repeats, ])), 1), [1, 0])
            rep = tensorflow.cast(rep, 'int32')
            x = tensorflow.matmul(tensorflow.reshape(x, (-1, 1)), rep)
            return tensorflow.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tensorflow.variable_scope('_interpolate'):
            # constants
            num_batch = tensorflow.shape(im)[0]
            height = tensorflow.shape(im)[1]
            width = tensorflow.shape(im)[2]
            channels = tensorflow.shape(im)[3]
 
            x = tensorflow.cast(x, 'float32')
            y = tensorflow.cast(y, 'float32')
            height_f = tensorflow.cast(height, 'float32')
            width_f = tensorflow.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tensorflow.zeros([], dtype='int32')
            max_y = tensorflow.cast(tensorflow.shape(im)[1] - 1, 'int32')
            max_x = tensorflow.cast(tensorflow.shape(im)[2] - 1, 'int32')
 
            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0
 
            # do sampling
            x0 = tensorflow.cast(tensorflow.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tensorflow.cast(tensorflow.floor(y), 'int32')
            y1 = y0 + 1
 
            x0 = tensorflow.clip_by_value(x0, zero, max_x)
            x1 = tensorflow.clip_by_value(x1, zero, max_x)
            y0 = tensorflow.clip_by_value(y0, zero, max_y)
            y1 = tensorflow.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tensorflow.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1
 
            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tensorflow.reshape(im, tensorflow.stack([-1, channels]))
            im_flat = tensorflow.cast(im_flat, 'float32')
            Ia = tensorflow.gather(im_flat, idx_a)
            Ib = tensorflow.gather(im_flat, idx_b)
            Ic = tensorflow.gather(im_flat, idx_c)
            Id = tensorflow.gather(im_flat, idx_d)
 
            # and finally calculate interpolated values
            x0_f = tensorflow.cast(x0, 'float32')
            x1_f = tensorflow.cast(x1, 'float32')
            y0_f = tensorflow.cast(y0, 'float32')
            y1_f = tensorflow.cast(y1, 'float32')
            wa = tensorflow.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tensorflow.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tensorflow.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tensorflow.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tensorflow.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output
 
    def _meshgrid(height, width):
        print('begin--meshgrid')
        with tensorflow.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = numpy.meshgrid(numpy.linspace(-1, 1, width), numpy.linspace(-1, 1, height))
            #  ones = numpy.ones(numpy.prod(x_t.shape))
            #  grid = numpy.vstack([x_t.flatten(), y_t.flatten(), ones])
            
            x_t = tensorflow.matmul(tensorflow.ones(shape=tensorflow.stack([height, 1])), tensorflow.transpose(tensorflow.expand_dims(tensorflow.linspace(-1.0, 1.0, width), 1), [1, 0]))
            print('meshgrid_x_t_ok')
            y_t = tensorflow.matmul(tensorflow.expand_dims(tensorflow.linspace(-1.0, 1.0, height), 1), tensorflow.ones(shape=tensorflow.stack([1, width])))
            print('meshgrid_y_t_ok')
            x_t_flat = tensorflow.reshape(x_t, (1, -1))
            y_t_flat = tensorflow.reshape(y_t, (1, -1))
            print('meshgrid_flat_t_ok')
            ones = tensorflow.ones_like(x_t_flat)
            print('meshgrid_ones_ok')
            print(x_t_flat)
            print(y_t_flat)
            print(ones)
            
            grid = tensorflow.concat( [x_t_flat, y_t_flat, ones],0)
            print ('over_meshgrid')
            return grid
 
    def _transform(theta, input_dim, out_size):
        print('_transform')
        
        with tensorflow.variable_scope('_transform'):
            num_batch = tensorflow.shape(input_dim)[0]
            height = tensorflow.shape(input_dim)[1]
            width = tensorflow.shape(input_dim)[2]
            num_channels = tensorflow.shape(input_dim)[3]
            theta = tensorflow.reshape(theta, (-1, 2, 3))
            theta = tensorflow.cast(theta, 'float32')
 
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tensorflow.cast(height, 'float32')
            width_f = tensorflow.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tensorflow.expand_dims(grid, 0)
            grid = tensorflow.reshape(grid, [-1])
            grid = tensorflow.tile(grid, tensorflow.stack([num_batch]))
            grid = tensorflow.reshape(grid, tensorflow.stack([num_batch, 3, -1]))
            #tensorflow.batch_matrix_diag
            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            print('begin--batch--matmul')
            T_g = tensorflow.matmul(theta, grid)
            x_s = tensorflow.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tensorflow.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tensorflow.reshape(x_s, [-1])
            y_s_flat = tensorflow.reshape(y_s, [-1])
 
            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
 
            output = tensorflow.reshape(input_transformed, tensorflow.stack([num_batch, out_height, out_width, num_channels]))
            print('over_transformer')
            return output
 
    with tensorflow.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output

def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
    with tensorflow.variable_scope(name):
        num_batch, num_transforms = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i]*num_transforms for i in range(num_batch)]
        input_repeated = tensorflow.gather(U, tensorflow.reshape(indices, [-1]))
        return transformer(input_repeated, thetas, out_size)

im=scipy.ndimage.imread('/home/zhao/Downloads/k.jpeg')
im=im/255.
im=im.reshape(1,1200,1600,3)
im=im.astype('float32')
out_size=(600,800)
batch=numpy.repeat(im,3,0)
num_batch=3

x=tensorflow.placeholder(tensorflow.float32,[None,1200,1600,3])
x=tensorflow.convert_to_tensor(batch)

with tensorflow.variable_scope('spatial_transformer'):
    initial=numpy.array([[0.5,0,0],[0,0.5,0]])
    initial=initial.astype('float32')
    initial=initial.flatten()
    w=tensorflow.Variable(tensorflow.zeros([1200*1600*3,6]),name='w')
    b=tensorflow.Variable(initial_value=initial,name='b')
    h=tensorflow.matmul(tensorflow.zeros([num_batch,1200*1600*3]),w)+b
    print(x,h,out_size)
    h_trans=transformer(x,h,out_size)
    
sess=tensorflow.Session()
sess.run(tensorflow.global_variables_initializer())
y=sess.run(h_trans,feed_dict={x:batch})
matplotlib.pyplot.imshow(y[0])
matplotlib.pyplot.show()
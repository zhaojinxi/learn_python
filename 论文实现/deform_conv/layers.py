import tensorflow
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import RandomNormal
from deform_conv.deform_conv import tf_batch_map_offsets

# class ConvOffset2D(Conv2D):
class ConvOffset2D(object):
    """ConvOffset2D"""

    def __init__(self, filters, init_normal_stddev=0.01, **kwargs):
        """Init"""

        self.filters = filters
        # super(ConvOffset2D, self).__init__(
        #     self.filters * 2, (3, 3), padding='same', use_bias=False,
        #     # TODO gradients are near zero if init is zeros
        #     kernel_initializer='zeros',
        #     # kernel_initializer=RandomNormal(0, init_normal_stddev),
        #     **kwargs)

    # def call(self, x):
    def __call__(self, x, name):
        w1=tensorflow.get_variable(name, [3,3,self.filters,self.filters * 2], initializer=tensorflow.zeros_initializer)
        offsets=tensorflow.nn.conv2d(x,w1,[1,1,1,1],'SAME')

        # TODO offsets probably have no nonlinearity?
        x_shape = x.get_shape()
        # offsets = super(ConvOffset2D, self).call(x)

        # offsets = self._to_bc_h_w_2(offsets, x_shape)
        # "(b, h, w, 2c) -> (b*c, h, w, 2)"
        offsets = tensorflow.transpose(offsets, [0, 3, 1, 2])
        offsets = tensorflow.reshape(offsets, (-1, int(x_shape[1]), int(x_shape[2]), 2))
        ##############################删除#####################
        offsets=tensorflow.stack([offsets[:,:,:,0]+0.1,offsets[:,:,:,0]+0.5], axis=-1)
        ##############################删除#####################
        # x = self._to_bc_h_w(x, x_shape)
        #"""(b, h, w, c) -> (b*c, h, w)"""
        x = tensorflow.transpose(x, [0, 3, 1, 2])
        x = tensorflow.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))

        x_offset = tf_batch_map_offsets(x, offsets)

        # x_offset = self._to_b_h_w_c(x_offset, x_shape)
        #"""(b*c, h, w) -> (b, h, w, c)"""
        x_offset = tensorflow.reshape(x_offset, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2])))
        x_offset = tensorflow.transpose(x_offset, [0, 2, 3, 1])

        return x_offset

    def compute_output_shape(self, input_shape):
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tensorflow.transpose(x, [0, 3, 1, 2])
        x = tensorflow.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 2))
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tensorflow.transpose(x, [0, 3, 1, 2])
        x = tensorflow.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tensorflow.reshape(x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2])))
        x = tensorflow.transpose(x, [0, 2, 3, 1])
        return x

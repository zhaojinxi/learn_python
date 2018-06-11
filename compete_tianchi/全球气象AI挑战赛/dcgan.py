import tensorflow

class Generator:
    def __init__(self, depths=[1024, 512, 256, 128], s_size=4):
        self.depths = depths + [3]
        self.s_size = s_size
        self.reuse = False

    def __call__(self, inputs, training=False):
        inputs = tensorflow.convert_to_tensor(inputs)
        with tensorflow.variable_scope('g', reuse=self.reuse):
            # reshape from inputs
            with tensorflow.variable_scope('reshape'):
                outputs = tensorflow.layers.dense(inputs, self.depths[0] * self.s_size * self.s_size)
                outputs = tensorflow.reshape(outputs, [-1, self.s_size, self.s_size, self.depths[0]])
                outputs = tensorflow.nn.relu(tensorflow.layers.batch_normalization(outputs, training=training), name='outputs')
            # deconvolution (transpose of convolution) x 4
            with tensorflow.variable_scope('deconv1'):
                outputs = tensorflow.layers.conv2d_transpose(outputs, self.depths[1], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tensorflow.nn.relu(tensorflow.layers.batch_normalization(outputs, training=training), name='outputs')
            with tensorflow.variable_scope('deconv2'):
                outputs = tensorflow.layers.conv2d_transpose(outputs, self.depths[2], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tensorflow.nn.relu(tensorflow.layers.batch_normalization(outputs, training=training), name='outputs')
            with tensorflow.variable_scope('deconv3'):
                outputs = tensorflow.layers.conv2d_transpose(outputs, self.depths[3], [5, 5], strides=(2, 2), padding='SAME')
                outputs = tensorflow.nn.relu(tensorflow.layers.batch_normalization(outputs, training=training), name='outputs')
            with tensorflow.variable_scope('deconv4'):
                outputs = tensorflow.layers.conv2d_transpose(outputs, self.depths[4], [5, 5], strides=(2, 2), padding='SAME')
            # output images
            with tensorflow.variable_scope('tanh'):
                outputs = tensorflow.tanh(outputs, name='outputs')
        self.reuse = True
        self.variables = tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES, scope='g')
        return outputs
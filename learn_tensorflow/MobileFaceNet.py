import tensorflow

bias_initializer = tensorflow.keras.initializers.glorot_uniform()
conv2d_regularizer = None
depthwise_regularizer = None
alpha_regularizer = None
dense_regularizer = None

class MobileNetV2Block(tensorflow.keras.Model):
    def __init__(self, stride, t, output_channel):
        super(MobileNetV2Block, self).__init__()
        self.stride = stride
        self.t = t
        self.output_channel = output_channel

    def build(self, input_shape):
        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=int(self.t * input_shape[3]),
            kernel_size=1,
            strides=1,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=conv2d_regularizer,
            bias_regularizer=conv2d_regularizer)
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=conv2d_regularizer,
            gamma_regularizer=conv2d_regularizer)
        self.PReLU_1 = tensorflow.keras.layers.PReLU(
            alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
            alpha_regularizer=alpha_regularizer,
            shared_axes=[1, 2])

        self.DepthwiseConv2D_2 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=self.stride,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=depthwise_regularizer,
            bias_regularizer=depthwise_regularizer)
        self.BatchNormalization_2 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=depthwise_regularizer,
            gamma_regularizer=depthwise_regularizer)
        self.PReLU_2 = tensorflow.keras.layers.PReLU(
            alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
            alpha_regularizer=alpha_regularizer,
            shared_axes=[1, 2])

        self.Conv2D_3 = tensorflow.keras.layers.Conv2D(
            filters=self.output_channel,
            kernel_size=1,
            strides=1,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=conv2d_regularizer,
            bias_regularizer=conv2d_regularizer)
        self.BatchNormalization_3 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=conv2d_regularizer,
            gamma_regularizer=conv2d_regularizer)
        super().build(input_shape)

    def call(self, x, training):
        y = self.Conv2D_1(x)
        y = self.BatchNormalization_1(y, training=training)
        y = self.PReLU_1(y)

        y = self.DepthwiseConv2D_2(y)
        y = self.BatchNormalization_2(y, training=training)
        y = self.PReLU_2(y)

        y = self.Conv2D_3(y)
        y = self.BatchNormalization_3(y, training=training)

        if x.shape.as_list() == y.shape.as_list():
            y = x + y
        return y

class MobileFaceNet(tensorflow.keras.Model):
    def __init__(self, num_classes):
        super(MobileFaceNet, self).__init__(name='MobileFaceNet')

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            strides=2,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=conv2d_regularizer,
            bias_regularizer=conv2d_regularizer)
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=conv2d_regularizer,
            gamma_regularizer=conv2d_regularizer)
        self.PReLU_1 = tensorflow.keras.layers.PReLU(
            alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
            alpha_regularizer=alpha_regularizer,
            shared_axes=[1, 2])

        self.DepthwiseConv2D_2 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=1,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=depthwise_regularizer,
            bias_regularizer=depthwise_regularizer)
        self.BatchNormalization_2 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=depthwise_regularizer,
            gamma_regularizer=depthwise_regularizer)
        self.PReLU_2 = tensorflow.keras.layers.PReLU(
            alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
            alpha_regularizer=alpha_regularizer,
            shared_axes=[1, 2])

        self.MobileNetV2Block_3 = MobileNetV2Block(2, 2, 64)
        self.MobileNetV2Block_4 = MobileNetV2Block(1, 2, 64)
        self.MobileNetV2Block_5 = MobileNetV2Block(1, 2, 64)
        self.MobileNetV2Block_6 = MobileNetV2Block(1, 2, 64)
        self.MobileNetV2Block_7 = MobileNetV2Block(1, 2, 64)

        self.MobileNetV2Block_8 = MobileNetV2Block(2, 4, 128)

        self.MobileNetV2Block_9 = MobileNetV2Block(1, 2, 128)
        self.MobileNetV2Block_10 = MobileNetV2Block(1, 2, 128)
        self.MobileNetV2Block_11 = MobileNetV2Block(1, 2, 128)
        self.MobileNetV2Block_12 = MobileNetV2Block(1, 2, 128)
        self.MobileNetV2Block_13 = MobileNetV2Block(1, 2, 128)
        self.MobileNetV2Block_14 = MobileNetV2Block(1, 2, 128)

        self.MobileNetV2Block_15 = MobileNetV2Block(2, 4, 128)

        self.MobileNetV2Block_16 = MobileNetV2Block(1, 2, 128)
        self.MobileNetV2Block_17 = MobileNetV2Block(1, 2, 128)

        self.Conv2D_18 = tensorflow.keras.layers.Conv2D(
            filters=512,
            kernel_size=1,
            strides=1,
            padding='same',
            bias_initializer=bias_initializer,
            kernel_regularizer=conv2d_regularizer,
            bias_regularizer=conv2d_regularizer)
        self.BatchNormalization_18 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=conv2d_regularizer,
            gamma_regularizer=conv2d_regularizer)
        self.PReLU_18 = tensorflow.keras.layers.PReLU(
            alpha_initializer=tensorflow.keras.initializers.Constant(0.25),
            alpha_regularizer=alpha_regularizer,
            shared_axes=[1, 2])

        self.DepthwiseConv2D_19 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=7,
            strides=1,
            bias_initializer=bias_initializer,
            kernel_regularizer=depthwise_regularizer,
            bias_regularizer=depthwise_regularizer)
        self.BatchNormalization_19 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=depthwise_regularizer,
            gamma_regularizer=depthwise_regularizer)

        self.Conv2D_20 = tensorflow.keras.layers.Conv2D(
            filters=128,
            kernel_size=1,
            strides=1,
            bias_initializer=bias_initializer,
            kernel_regularizer=conv2d_regularizer,
            bias_regularizer=conv2d_regularizer)
        self.BatchNormalization_20 = tensorflow.keras.layers.BatchNormalization(
            beta_regularizer=conv2d_regularizer,
            gamma_regularizer=conv2d_regularizer)
        self.Flatten_20 = tensorflow.keras.layers.Flatten()

        self.Dense_21 = tensorflow.keras.layers.Dense(
            units=num_classes,
            use_bias=False,
            name='last_layer',
            kernel_regularizer=dense_regularizer)

    def call(self, x, training):
        x = self.Conv2D_1(x)
        x = self.BatchNormalization_1(x, training=training)
        x = self.PReLU_1(x)

        x = self.DepthwiseConv2D_2(x)
        x = self.BatchNormalization_2(x, training=training)
        x = self.PReLU_2(x)

        x = self.MobileNetV2Block_3(x, training=training)
        x = self.MobileNetV2Block_4(x, training=training)
        x = self.MobileNetV2Block_5(x, training=training)
        x = self.MobileNetV2Block_6(x, training=training)
        x = self.MobileNetV2Block_7(x, training=training)

        x = self.MobileNetV2Block_8(x, training=training)

        x = self.MobileNetV2Block_9(x, training=training)
        x = self.MobileNetV2Block_10(x, training=training)
        x = self.MobileNetV2Block_11(x, training=training)
        x = self.MobileNetV2Block_12(x, training=training)
        x = self.MobileNetV2Block_13(x, training=training)
        x = self.MobileNetV2Block_14(x, training=training)

        x = self.MobileNetV2Block_15(x, training=training)

        x = self.MobileNetV2Block_16(x, training=training)
        x = self.MobileNetV2Block_17(x, training=training)

        x = self.Conv2D_18(x)
        x = self.BatchNormalization_18(x, training=training)
        x = self.PReLU_18(x)

        x = self.DepthwiseConv2D_19(x)
        x = self.BatchNormalization_19(x, training=training)

        x = self.Conv2D_20(x)
        x = self.BatchNormalization_20(x, training=training)
        embed = self.Flatten_20(x)

        predict = self.Dense_21(embed)
        return embed, predict

if __name__ == '__main__':
    tensorflow.enable_eager_execution(config=tensorflow.ConfigProto(allow_soft_placement=True, gpu_options=tensorflow.GPUOptions(allow_growth=True)))
    data = tensorflow.random.uniform([4, 112, 112, 3])
    model = MobileFaceNet(85742)
    embed, predict = model(data, True)
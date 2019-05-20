import tensorflow

tensorflow.enable_eager_execution()

def swish(x):
    y = x * tensorflow.nn.sigmoid(x)
    return y

def hard_swish(x):
    y = x * (tensorflow.nn.relu6(x + 3)) / 6
    return y

class MobileNetV3Block(tensorflow.keras.layers.Layer):
    def __init__(self, input_shape, kernel_size, exp_size, output_channle, use_se, activation, stride, **kwargs):
        super(MobileNetV3Block, self).__init__(**kwargs)
        self.input_shape = input_shape
        self.stride = stride
        self.kernel_size = kernel_size
        self.exp_size = exp_size
        self.output_channle = output_channle
        if activation == 're':
            self.active = tensorflow.nn.relu6
        elif activation == 'hs':
            self.active = hard_swish

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=self.exp_size,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()

        self.DepthwiseConv2D_2 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=self.stride,
            padding='same')
        self.BatchNormalization_2 = tensorflow.keras.layers.BatchNormalization()

        if use_se == True:
            self.AveragePooling2D_3 = tensorflow.keras.layers.AveragePooling2D(
                pool_size=self.input_shape[:2],
                strides=1)

            self.Conv2D_4 = tensorflow.keras.layers.Conv2D(
                filters=self.exp_size / 4,
                kernel_size=1,
                strides=1,
                activation=tensorflow.nn.relu)

            self.Conv2D_5 = tensorflow.keras.layers.Conv2D(
                filters=self.output_channle,
                kernel_size=1,
                strides=1,
                activation=hard_swish)

        self.Conv2D_6 = tensorflow.keras.layers.Conv2D(
            filters=self.output_channle,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_6 = tensorflow.keras.layers.BatchNormalization()

    def call(self, x):
        y = self.Conv2D_1(x)
        y = self.BatchNormalization_1(y)
        y = self.active(y)

        y = self.DepthwiseConv2D_2(y)
        y = self.BatchNormalization_2(y)
        y = self.active(y)

        if use_se == True:
            z = self.AveragePooling2D_3(y)

            z = self.Conv2D_4(z)

            z = self.Conv2D_5(z)

            y = tensorflow.multiply(y, z)

        y = self.Conv2D_6(y)
        y = self.BatchNormalization_6(y)

        if self.input_shape[2] == self.output_channle:
            y = y + x
        return x

class MobileNetV3Large(tensorflow.keras.Model):
    def __init__(self, num_classes=1000):
        super(MobileNetV3Large, self).__init__(name='MobileNetV3Large')

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=16,
            kernel_size=3,
            strides=2,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()
        self.hard_swish_1 = hard_swish

        self.BR2 = MobileNetV3Block([112, 112, 16], 3, 16, 16, False, 're', 1)

        self.BR3 = MobileNetV3Block([112, 112, 16], 3, 64, 24, False, 're', 2)

        self.BR4 = MobileNetV3Block([56, 56, 24], 3, 72, 24, False, 're', 1)

        self.BR5 = MobileNetV3Block([56, 56, 24], 5, 72, 40, True, 're', 2)

        self.BR6 = MobileNetV3Block([28, 28, 40], 5, 120, 40, True, 're', 1)

        self.BR7 = MobileNetV3Block([28, 28, 40], 5, 120, 40, True, 're', 1)

        self.BR8 = MobileNetV3Block([28, 28, 40], 3, 240, 80, False, 'hs', 2)

        self.BR9 = MobileNetV3Block([14, 14, 80], 3, 200, 80, False, 'hs', 1)

        self.BR10 = MobileNetV3Block([14, 14, 80], 3, 184, 80, False, 'hs', 1)

        self.BR11 = MobileNetV3Block([14, 14, 80], 3, 184, 80, False, 'hs', 1)

        self.BR12 = MobileNetV3Block([14, 14, 80], 3, 480, 112, True, 'hs', 1)

        self.BR13 = MobileNetV3Block([14, 14, 112], 3, 672, 112, True, 'hs', 1)

        self.BR14 = MobileNetV3Block([14, 14, 112], 5, 672, 160, True, 'hs', 1)

        self.BR15 = MobileNetV3Block([14, 14, 160], 5, 672, 160, True, 'hs', 2)

        self.BR16 = MobileNetV3Block([7, 7, 160], 5, 960, 160, True, 'hs', 1)

        self.Conv2D_17 = tensorflow.keras.layers.Conv2D(
            filters=960,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_17 = tensorflow.keras.layers.BatchNormalization()
        self.hard_swish_17 = hard_swish

        self.AveragePooling2D_18 = tensorflow.keras.layers.AveragePooling2D(
            pool_size=7,
            strides=1)

        self.Conv2D_19 = tensorflow.keras.layers.Conv2D(
            filters=1280,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=hard_swish)

        self.Conv2D_20 = tensorflow.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            strides=1,
            padding='same')

    def call(self, x):
        x = self.C1(x)
        x = self.BN1(x)

        x = self.BR2(x)

        x = self.BR3(x)
        x = self.BR4(x)

        x = self.BR5(x)
        x = self.BR6(x)
        x = self.BR7(x)

        x = self.BR8(x)
        x = self.BR9(x)
        x = self.BR10(x)
        x = self.BR11(x)

        x = self.BR12(x)
        x = self.BR13(x)
        x = self.BR14(x)

        x = self.BR15(x)
        x = self.BR16(x)
        x = self.BR17(x)

        x = self.BR18(x)

        x = self.C19(x)
        x = self.BN19(x)

        x = self.AP20(x)

        x = self.C21(x)
        return x

if __name__ == '__main__':
    data = tensorflow.random.uniform([100, 224, 224, 3])
    model = MobileNetV3Large()
    y = model(data)
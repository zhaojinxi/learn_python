import tensorflow

tensorflow.enable_eager_execution()

class MobileNetV2Block(tensorflow.keras.layers.Layer):
    def __init__(self, stride, t, input_channle, output_channle, **kwargs):
        super(MobileNetV2Block, self).__init__(**kwargs)
        self.stride = stride
        self.t = t
        self.input_channle = input_channle
        self.output_channle = output_channle

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=self.t * self.input_channle,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()
        self.ReLU_1 = tensorflow.keras.layers.ReLU(max_value=6)

        self.DepthwiseConv2D_2 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=self.stride,
            padding='same',
            activation=tensorflow.keras.layers.ReLU(max_value=6))
        self.BatchNormalization_2 = tensorflow.keras.layers.BatchNormalization()
        self.ReLU_2 = tensorflow.keras.layers.ReLU(max_value=6)

        self.Conv2D_3 = tensorflow.keras.layers.Conv2D(
            filters=self.output_channle,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_3 = tensorflow.keras.layers.BatchNormalization()

    def call(self, x):
        y = self.Conv2D_1(x)
        y = self.BatchNormalization_1(y)
        y = self.ReLU_1(y)

        y = self.DepthwiseConv2D_2(y)
        y = self.BatchNormalization_2(y)
        y = self.ReLU_2(y)

        y = self.Conv2D_3(y)
        y = self.BatchNormalization_3(y)

        if self.input_channle == self.output_channle:
            y = x + y
        return y

class MobileNetV2(tensorflow.keras.Model):
    def __init__(self, num_classes=1000):
        super(MobileNetV2, self).__init__(name='MobileNetV2')
        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()
        self.ReLU_1 = tensorflow.keras.layers.ReLU(max_value=6)

        self.MobileNetV2Block_2 = MobileNetV2Block(1, 1, 32, 16)

        self.MobileNetV2Block_3 = MobileNetV2Block(2, 6, 16, 24)
        self.MobileNetV2Block_4 = MobileNetV2Block(1, 6, 24, 24)

        self.MobileNetV2Block_5 = MobileNetV2Block(2, 6, 24, 32)
        self.MobileNetV2Block_6 = MobileNetV2Block(1, 6, 32, 32)
        self.MobileNetV2Block_7 = MobileNetV2Block(1, 6, 32, 32)

        self.MobileNetV2Block_8 = MobileNetV2Block(2, 6, 32, 64)
        self.MobileNetV2Block_9 = MobileNetV2Block(1, 6, 64, 64)
        self.MobileNetV2Block_10 = MobileNetV2Block(1, 6, 64, 64)
        self.MobileNetV2Block_11 = MobileNetV2Block(1, 6, 64, 64)

        self.MobileNetV2Block_12 = MobileNetV2Block(1, 6, 64, 96)
        self.MobileNetV2Block_13 = MobileNetV2Block(1, 6, 96, 96)
        self.MobileNetV2Block_14 = MobileNetV2Block(1, 6, 96, 96)

        self.MobileNetV2Block_15 = MobileNetV2Block(2, 6, 96, 160)
        self.MobileNetV2Block_16 = MobileNetV2Block(1, 6, 160, 160)
        self.MobileNetV2Block_17 = MobileNetV2Block(1, 6, 160, 160)

        self.MobileNetV2Block_18 = MobileNetV2Block(1, 6, 160, 320)

        self.Conv2D_19 = tensorflow.keras.layers.Conv2D(
            filters=1280,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_19 = tensorflow.keras.layers.BatchNormalization()
        self.ReLU_19 = tensorflow.keras.layers.ReLU(max_value=6)

        self.AveragePooling2D_20 = tensorflow.keras.layers.AveragePooling2D(
            pool_size=7,
            strides=1)

        self.Conv2D_21 = tensorflow.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            strides=1,
            padding='same')

    def call(self, x):
        x = self.Conv2D_1(x)
        x = self.BatchNormalization_1(x)
        x = self.ReLU_1(x)

        x = self.MobileNetV2Block_2(x)

        x = self.MobileNetV2Block_3(x)
        x = self.MobileNetV2Block_4(x)

        x = self.MobileNetV2Block_5(x)
        x = self.MobileNetV2Block_6(x)
        x = self.MobileNetV2Block_7(x)

        x = self.MobileNetV2Block_8(x)
        x = self.MobileNetV2Block_9(x)
        x = self.MobileNetV2Block_10(x)
        x = self.MobileNetV2Block_11(x)

        x = self.MobileNetV2Block_12(x)
        x = self.MobileNetV2Block_13(x)
        x = self.MobileNetV2Block_14(x)

        x = self.MobileNetV2Block_15(x)
        x = self.MobileNetV2Block_16(x)
        x = self.MobileNetV2Block_17(x)

        x = self.MobileNetV2Block_18(x)

        x = self.Conv2D_19(x)
        x = self.BatchNormalization_19(x)
        x = self.ReLU_19(x)

        x = self.AveragePooling2D_20(x)

        x = self.Conv2D_21(x)
        return x

if __name__ == '__main__':
    data = tensorflow.random.uniform([100, 224, 224, 3])
    model = MobileNetV2()
    y = model(data)
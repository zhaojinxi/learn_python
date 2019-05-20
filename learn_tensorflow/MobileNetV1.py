import tensorflow

tensorflow.enable_eager_execution()

class MobileNetV1Block(tensorflow.keras.layers.Layer):
    def __init__(self, stride, output_channle, **kwargs):
        super(MobileNetV1Block, self).__init__(**kwargs)
        self.stride = stride
        self.output_channle = output_channle

        self.DepthwiseConv2D_1 = tensorflow.keras.layers.DepthwiseConv2D(
            kernel_size=3,
            strides=self.stride,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()
        self.ReLU_1 = tensorflow.keras.layers.ReLU()

        self.Conv2D_2 = tensorflow.keras.layers.Conv2D(
            filters=self.output_channle,
            kernel_size=1,
            strides=1,
            padding='same')
        self.BatchNormalization_2 = tensorflow.keras.layers.BatchNormalization()
        self.ReLU_2 = tensorflow.keras.layers.ReLU()

    def call(self, x):
        y = self.DepthwiseConv2D_1(x)
        y = self.BatchNormalization_1(y)
        y = self.ReLU_1(y)

        y = self.Conv2D_2(y)
        y = self.BatchNormalization_2(y)
        y = self.ReLU_2(y)
        return y

class MobileNetV1(tensorflow.keras.Model):
    def __init__(self, num_classes=1000):
        super(MobileNetV1, self).__init__(name='MobileNetV1')

        self.Conv2D_1 = tensorflow.keras.layers.Conv2D(
            filters=32,
            kernel_size=3,
            strides=2,
            padding='same')
        self.BatchNormalization_1 = tensorflow.keras.layers.BatchNormalization()
        self.ReLU_1 = tensorflow.keras.layers.ReLU()

        self.MobileNetV1Block_2 = MobileNetV1Block(1, 64)

        self.MobileNetV1Block_3 = MobileNetV1Block(2, 128)

        self.MobileNetV1Block_4 = MobileNetV1Block(1, 128)

        self.MobileNetV1Block_5 = MobileNetV1Block(2, 256)

        self.MobileNetV1Block_6 = MobileNetV1Block(1, 256)

        self.MobileNetV1Block_7 = MobileNetV1Block(2, 512)

        self.MobileNetV1Block_8 = MobileNetV1Block(1, 512)
        self.MobileNetV1Block_9 = MobileNetV1Block(1, 512)
        self.MobileNetV1Block_10 = MobileNetV1Block(1, 512)
        self.MobileNetV1Block_11 = MobileNetV1Block(1, 512)
        self.MobileNetV1Block_12 = MobileNetV1Block(1, 512)

        self.MobileNetV1Block_13 = MobileNetV1Block(2, 1024)

        self.MobileNetV1Block_14 = MobileNetV1Block(1, 1024)

        self.AveragePooling2D_15 = tensorflow.keras.layers.AveragePooling2D(
            pool_size=7,
            strides=1)

        self.Conv2D_16 = tensorflow.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            strides=1,
            padding='same')

    def call(self, x):
        x = self.Conv2D_1(x)
        x = self.BatchNormalization_1(x)
        x = self.ReLU_1(x)

        x = self.MobileNetV1Block_2(x)

        x = self.MobileNetV1Block_3(x)

        x = self.MobileNetV1Block_4(x)

        x = self.MobileNetV1Block_5(x)

        x = self.MobileNetV1Block_6(x)

        x = self.MobileNetV1Block_7(x)

        x = self.MobileNetV1Block_8(x)
        x = self.MobileNetV1Block_9(x)
        x = self.MobileNetV1Block_10(x)
        x = self.MobileNetV1Block_11(x)
        x = self.MobileNetV1Block_12(x)

        x = self.MobileNetV1Block_13(x)

        x = self.MobileNetV1Block_14(x)

        x = self.AveragePooling2D_15(x)

        x = self.Conv2D_16(x)
        return x

if __name__ == '__main__':
    data = tensorflow.random.uniform([100, 224, 224, 3])
    model = MobileNetV1()
    y = model(data)
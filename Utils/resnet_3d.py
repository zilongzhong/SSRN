from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    merge,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Convolution3D,
    MaxPooling3D,
    AveragePooling3D,
    ZeroPadding3D
)
from keras.layers.normalization import BatchNormalization
from keras import backend as K


# Helper to build a conv -> BN -> relu block
def _conv_bn_relu(nb_filter, conv_dim1, conv_dim2, conv_dim3, subsample=(1, 1, 1)):
    def f(input):
        conv = Convolution3D(nb_filter=nb_filter, kernel_dim1=conv_dim1, kernel_dim2=conv_dim2, kernel_dim3=conv_dim3, subsample=subsample, init="he_normal")(input)
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(conv)
        return Activation("relu")(norm)

    return f


# Helper to build a BN -> relu -> conv block
# This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
def _bn_relu_conv(nb_filter, conv_dim1, conv_dim2, conv_dim3, subsample=(1, 1, 1)):
    def f(input):
        norm = BatchNormalization(mode=0, axis=CHANNEL_AXIS)(input)
        activation = Activation("relu")(norm)
        return Convolution3D(nb_filter=nb_filter, kernel_dim1=conv_dim1, kernel_dim2=conv_dim2, kernel_dim3=conv_dim3, subsample=subsample, init="he_normal", border_mode="same")(activation)

    return f


# Adds a shortcut between input and residual block and merges them with "sum"
def _shortcut(input, residual):
    # Expand channels of shortcut to match residual.
    # Stride appropriately to match residual (width, height)
    # Should be int if network architecture is correctly configured.
    print("input shape:", input._keras_shape)
    print("res shape:", residual._keras_shape)


    stride_dim1 = input._keras_shape[CONV_DIM1] // residual._keras_shape[CONV_DIM1]
    stride_dim2 = input._keras_shape[CONV_DIM2] // residual._keras_shape[CONV_DIM2]
    stride_dim3 = input._keras_shape[CONV_DIM3] // residual._keras_shape[CONV_DIM3]
    equal_channels = residual._keras_shape[CHANNEL_AXIS] == input._keras_shape[CHANNEL_AXIS]

    shortcut = input
    # 1 X 1 conv if shape is different. Else identity.
    if stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1 or not equal_channels:
        shortcut = Convolution3D(nb_filter=residual._keras_shape[CHANNEL_AXIS],
                                 kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                 subsample=(stride_dim1, stride_dim2, stride_dim3),
                                 init="he_normal", border_mode="valid")(input)

    print("three strides are %d, %d, %d" % (stride_dim1, stride_dim2, stride_dim3))
    print("equal_channels is:", equal_channels)
    print("input shape:", input._keras_shape)
    print("shortcut shape:", shortcut._keras_shape)
    print("res shape:", residual._keras_shape)

    return merge([shortcut, residual], mode="sum")


# Builds a residual block with repeating bottleneck blocks.
def _residual_block(block_function, nb_filters, repetitions, is_first_layer=False):
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2, 1)
            input = block_function(nb_filters=nb_filters, init_subsample=init_subsample)(input)
        return input

    return f


# Basic 3 X 3 convolution blocks.
# Use for resnet with layers <= 34
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
def basic_block(nb_filters, init_subsample=(1, 1, 1)):
    def f(input):
        conv1 = _bn_relu_conv(nb_filters, 3, 3, 1, subsample=init_subsample)(input)
        print("conv shape:", conv1._keras_shape)
        residual = _bn_relu_conv(nb_filters, 3, 3, 1)(conv1)
        print("res shape:", residual._keras_shape)
        #residual_pad = ZeroPadding3D(padding=(1, 1, 0), dim_ordering='tf')(residual)
        return _shortcut(input, residual)

    return f


# Bottleneck architecture for > 34 layer resnet.
# Follows improved proposed scheme in http://arxiv.org/pdf/1603.05027v2.pdf
# Returns a final conv layer of nb_filters * 4
def bottleneck(nb_filters, init_subsample=(1, 1, 1)):
    def f(input):
        conv_1_1 = _bn_relu_conv(nb_filters, 1, 1, 32, subsample=init_subsample)(input)
        conv_3_3 = _bn_relu_conv(nb_filters, 3, 3, 32)(conv_1_1)
        residual = _bn_relu_conv(nb_filters * 4, 1, 1, 32)(conv_3_3)
        return _shortcut(input, residual)

    return f


def handle_dim_ordering():
    global CONV_DIM1
    global CONV_DIM2
    global CONV_DIM3
    global CHANNEL_AXIS
    if K.image_dim_ordering() == 'tf':
        CONV_DIM1 = 1
        CONV_DIM2 = 2
        CONV_DIM3 = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1 = 2
        CONV_DIM2 = 3
        CONV_DIM3 = 4


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """
        Builds a custom ResNet like architecture.
        :param input_shape: The input shape in the form (nb_channels, nb_rows, nb_cols)

        :param num_outputs: The number of outputs at final softmax layer

        :param block_fn: The block function to use. This is either :func:`basic_block` or :func:`bottleneck`.
        The original paper used basic_block for layers < 50

        :param repetitions: Number of repetitions of various block units.
        At each block unit, the number of filters are doubled and the input size is halved

        :return: The keras model.
        """
        handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, conv_dim1, conv_dim2, conv_dim3)")

        # Permute dimension order if necessary

        if K.image_dim_ordering() == 'tf':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])

        input = Input(shape=input_shape)
        print("input shape:", input._keras_shape)
        # zeropad_input = ZeroPadding3D(padding=(0, 0, 0), dim_ordering='tf')(input)
        #
        # print("zeropad1_input shape:", zeropad_input._keras_shape)
        conv1 = _conv_bn_relu(nb_filter=32, conv_dim1=4, conv_dim2=4, conv_dim3=200)(input)
        #conv1 = Convolution3D(nb_filter=32, kernel_dim1=3, kernel_dim2=3, kernel_dim3=200)(input)
        print("conv1 shape:", conv1._keras_shape)

        #pool1 = MaxPooling3D(pool_size=(1, 1, 3), strides=(1, 1, 2), border_mode="same")(conv1)
        #print("pool1 shape:", pool1._keras_shape)

        block = conv1
        nb_filters = 64
        for i, r in enumerate(repetitions):
            block = _residual_block(block_fn, nb_filters=nb_filters, repetitions=r, is_first_layer=i == 0)(block)
            print("residual block shape:", block._keras_shape)
            print("i = ", i)
            nb_filters *= 2

        # Classifier block
        pool2 = AveragePooling3D(pool_size=(block._keras_shape[CONV_DIM1],
                                            block._keras_shape[CONV_DIM2],
                                            block._keras_shape[CONV_DIM3]),
                                 strides=(1, 1, 1))(block)
        flatten1 = Flatten()(pool2)
        dense = Dense(output_dim=num_outputs, init="he_normal", activation="softmax")(flatten1)

        model = Model(input=input, output=dense)
        return model

    @staticmethod
    def build_resnet_6(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [1, 1])      #[2, 2, 2, 2]

    @staticmethod
    def build_resnet_10(input_shape, num_outputs):
        return ResnetBuilder.build(input_shape, num_outputs, basic_block, [1, 1, 1])

def main():
    model = ResnetBuilder.build_resnet_6((1, 7, 7, 200), 16)          #model = ResnetBuilder.build_resnet_18((3, 224, 224), 1000)
#    model = ResnetBuilder.build_resnet_34((1, 27, 27, 103), 9)
    model.compile(loss="categorical_crossentropy", optimizer="sgd")
    model.summary()

if __name__ == '__main__':
    main()


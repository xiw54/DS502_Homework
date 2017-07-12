import mxnet as mx


def mlp_layer(input_layer, n_hidden, activation=None, BN=False):

    """
    A MLP layer with activation layer and BN
    :param input_layer: input sym
    :param n_hidden: # of hidden neurons
    :param activation: the activation function
    :return: the symbol as output
    """

    # get a FC layer
    l = mx.sym.FullyConnected(data=input_layer, num_hidden=n_hidden)
    # get activation, it can be relu, sigmoid, tanh, softrelu or none
    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    return l


def get_mlp_sym():

    """
    :return: the mlp symbol
    """

    data = mx.sym.Variable("data")
    # Flatten the data from 4-D shape into 2-D (batch_size, num_channel*width*height)
    data_f = mx.sym.flatten(data=data)

    # Your Design
    l = mlp_layer(input_layer=data_f, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)
    l = mlp_layer(input_layer=l, n_hidden=100, activation="relu", BN=True)

    # MNIST has 10 classes
    l = mx.sym.FullyConnected(data=l, num_hidden=10)
    # Softmax with cross entropy loss
    mlp = mx.sym.SoftmaxOutput(data=l, name='softmax')
    return mlp


def conv_layer(input_layer, kernel_dim, filter, activation=None, BN=False, pool=None, p_kernel=None, p_stride=None):
    """
    A conv layer with activation layer and BN
    :param input_layer: input sym
    :param kernel_dim: a tuple consists of two int (5, 5), size of kernel
    :param filter: number of filters
    :param activation: str, the activation function
    :param pool: str, pooling layer
    :param BN: T/F, batch normalization
    :return: a single convolution layer symbol
    """
    # todo: Design the simplest convolution layer
    # Find the doc of mx.sym.Convolution by help command
    # Do you need BatchNorm?
    # Do you need pooling?
    # What is the expected output shape?

    # conv layer
    l = mx.sym.Convolution(data=input_layer, kernel=kernel_dim, num_filter=filter)

    if activation is not None:
        l = mx.sym.Activation(data=l, act_type=activation)
    if BN:
        l = mx.sym.BatchNorm(l)
    if pool is not None:
        l = mx.sym.Pooling(data=l, pool_type=pool, kernel=p_kernel, stride=p_stride)
    return l

# Optional
def inception_layer():
    """
    Implement the inception layer in week3 class
    :return: the symbol of a inception layer
    """
    pass


def get_conv_sym():

    """
    :return: symbol of a convolutional neural network
    """
    data = mx.sym.Variable("data")
    # todo: design the CNN architecture
    # How deep the network do you want? like 4 or 5
    # How wide the network do you want? like 32/64/128 kernels per layer
    # How is the convolution like? Normal CNN? Inception Module? VGG like?

    l = conv_layer(data, (5,5), 20, activation="tanh", BN=False, pool="max", p_kernel=(2,2), p_stride=(2,2))
    l = conv_layer(data, (5,5), 50, activation="tanh", BN=False, pool="max", p_kernel=(2,2), p_stride=(2,2))
    data_f = mx.sym.flatten(data=l)
    # fullc
    fc1 = mx.sym.FullyConnected(data=data_f, num_hidden=10)
    tanh1 = mx.sym.Activation(data=fc1, act_type="tanh")
    fc2 = mx.sym.FullyConnected(data=tanh1, num_hidden=10)
    # softmax loss
    lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')

    return lenet
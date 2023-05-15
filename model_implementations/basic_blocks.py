
from tensorflow.keras import layers

def get_activation(name):
    if name == 'leakyrelu':
        return layers.ReLU(negative_slope=.1)
    return layers.Activation(name)

# we want batch-norm between conv and activation, so define function
def conv1d(input_layer, filters, kernel_size, strides=1, padding='same',
           activation='relu', batch_norm=False):
    x = layers.Conv1D(filters, kernel_size, strides=strides, padding=padding, 
                      )(input_layer)
    if batch_norm:
        x = layers.BatchNormalization()(x)
    if activation:
        x = get_activation(activation)(x)
    return x

def get_dense_stack(flatten_layer, dense_neurons, dropout, activation=None):
    x = flatten_layer
    # Hidden Layers
    dense_neurons = dense_neurons if type(dense_neurons) is list else [dense_neurons]
    for i, neurons in enumerate(dense_neurons):
        if dropout:
            x = layers.Dropout(dropout)(x)
        x = layers.Dense(neurons, activation=activation,
                         name=f'dense{i}')(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x

def residual_block(input_layer, filters=100, kernel_size=5, 
                   block_type='identity', batch_norm=True):
    if block_type == 'conv':
        # for convolutional blocks: input datapoints (and shortcut) get reduced
        first_stride = 2
        shortcut = conv1d(input_layer, filters, 1, strides=2, activation=None,
                          batch_norm=batch_norm)
    else:
        # identity block, keeps dim constant
        first_stride = 1
        shortcut = input_layer
    x = conv1d(input_layer, filters, kernel_size, first_stride, 
               batch_norm=batch_norm)
    x = conv1d(x, filters, kernel_size, 1, 
               activation=None, batch_norm=batch_norm)
    add = layers.Add()([x, shortcut])
    out = layers.Activation('relu')(add)
    return out
        
def inception_block_v1(input_layer, filter_list):
    inc1 = conv1d(input_layer, filter_list[0], 1)
    inc2 = conv1d(input_layer, filter_list[1], 1)
    inc2 = conv1d(inc2, filter_list[2], 3)
    inc3 = conv1d(input_layer, filter_list[3], 1)
    inc3 = conv1d(inc3, filter_list[4], 5)
    inc4 = layers.AveragePooling1D(3, 1, padding='same')(input_layer)
    inc4 = conv1d(inc4, filter_list[5], 1)
    stack = layers.concatenate([inc1, inc2, inc3, inc4], axis=2)
    return stack


from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
from .basic_blocks import get_dense_stack, inception_block_v1

def cnn_2(input_size=5000, dropout=.3, dense_neurons=[2000, 500],
          classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    x = layers.Conv1D(64, 50, strides=2, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(3, strides=2, 
                         name='maxpool1')(x)
    x = layers.Conv1D(64, 25, strides=3, padding='same',
                      activation='relu', 
                      name='conv2')(x)
    x = layers.MaxPool1D(2, strides=3, 
                         name='maxpool2')(x)
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model

def cnn_3(input_size=5000, dropout=.3, dense_neurons=[2500, 1000], 
          classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1), 
                               name="input")
    x = layers.Conv1D(64, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(3, strides=3, 
                         name='maxpool1')(x)
    x = layers.Conv1D(64, 15, strides=1, padding='same',
                      activation='relu', 
                      name='conv2')(x)
    x = layers.MaxPool1D(2, strides=3, 
                         name='maxpool2')(x)
    x = layers.Conv1D(64, 10, strides=2, padding='same',
                      activation='relu', 
                      name='conv3')(x)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool3')(x)
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model

def inception3(input_size=5000, dropout=.3, dense_neurons=[3700, 740], 
               classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1), 
                               name="input")
    x = layers.Conv1D(64, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(3, strides=3, 
                         name='maxpool1')(x)
    x = layers.Conv1D(64, 15, strides=1, padding='same',
                      activation='relu', 
                      name='conv2')(x)
    x = layers.MaxPool1D(2, strides=3, 
                         name='maxpool2')(x)
    x = layers.Conv1D(64, 10, strides=2, padding='same',
                      activation='relu', 
                      name='conv3')(x)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool3')(x)
    # inception blocks
    x = inception_block_v1(x, [22, 32, 42, 6, 12, 10])
    x = inception_block_v1(x, [28, 43, 56, 7, 14, 14])
    x = inception_block_v1(x, [37, 56, 73, 9, 18, 19])
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model

def inception6(input_size=5000, dropout=.3, dense_neurons=[4000, 400], 
               classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1), 
                               name="input")
    x = layers.Conv1D(64, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(3, strides=3, 
                         name='maxpool1')(x)
    x = layers.Conv1D(64, 15, strides=1, padding='same',
                      activation='relu', 
                      name='conv2')(x)
    x = layers.MaxPool1D(2, strides=3, 
                         name='maxpool2')(x)
    x = layers.Conv1D(64, 10, strides=2, padding='same',
                      activation='relu', 
                      name='conv3')(x)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool3')(x)
    # inception blocks
    x = inception_block_v1(x, [22, 32, 42, 6, 12, 10])
    x = inception_block_v1(x, [28, 43, 56, 7, 14, 14])
    x = inception_block_v1(x, [37, 56, 73, 9, 18, 19])
    x = inception_block_v1(x, [49, 74, 96, 12, 24, 25])
    x = inception_block_v1(x, [65, 97, 126, 16, 32, 32])
    x = inception_block_v1(x, [85, 128, 166, 19, 38, 43])
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model






from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
import model_implementations.basic_blocks as bb

def cnn6_bn(input_size=5000, dropout=.7, dense_neurons=[3100, 1200], 
            classes=500, lr=3e-4, bn=True):
    input_layer = layers.Input(shape=(input_size, 1), 
                               name="input")
    x = bb.conv1d(input_layer, 64, 35, batch_norm=bn)
    x = layers.MaxPool1D(3, strides=2, 
                         name='maxpool1')(x)
    x = bb.conv1d(x, 64, 30, batch_norm=bn)
    x = layers.MaxPool1D(3, strides=2, 
                         name='maxpool2')(x)
    x = bb.conv1d(x, 64, 25, batch_norm=bn)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool3')(x)
    x = bb.conv1d(x, 64, 20, batch_norm=bn)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool4')(x)
    x = bb.conv1d(x, 64, 15, batch_norm=bn)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool5')(x)
    x = bb.conv1d(x, 64, 10, batch_norm=bn)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool6')(x)
    x = layers.Flatten(name='flat')(x)
    x = bb.get_dense_stack(x, dense_neurons, dropout, activation='relu')
    opt = optimizers.Adam(learning_rate=lr)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model

def cnnbn_mod(input_size=5000, dropout=.5, dense_neurons=[2048], 
              bn=False, hidden_act='relu', classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    # Batch Norm between conv and activation so we use custom conv1d func
    # instead of default keras Conv1D layer
    x = bb.conv1d(input_layer, 16, 21, activation='leakyrelu', batch_norm=bn)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool1')(x)
    x = bb.conv1d(x, 32, 11, activation='leakyrelu', batch_norm=bn)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool2')(x)
    x = bb.conv1d(x, 64, 5, activation='leakyrelu', batch_norm=bn)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool3')(x)
    x = layers.Flatten(name='flat')(x)
    x = bb.get_dense_stack(x, dense_neurons, dropout, activation=hidden_act)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model

def cnn6_convdo(input_size=5000, dropout=0., dense_neurons=[3100, 1200], 
                dropout_conv=0.2, classes=500, lr=3e-4, bn=False):
    input_layer = layers.Input(shape=(input_size, 1), 
                               name="input")
    x = bb.conv1d(input_layer, 64, 35, batch_norm=bn)
    x = layers.MaxPool1D(3, strides=2, 
                         name='maxpool1')(x)
    x = layers.Dropout(dropout_conv, name='dropout1')(x)
    x = bb.conv1d(x, 64, 30, batch_norm=bn)
    x = layers.MaxPool1D(3, strides=2, 
                         name='maxpool2')(x)
    x = layers.Dropout(dropout_conv, name='dropout2')(x)
    x = bb.conv1d(x, 64, 25, batch_norm=bn)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool3')(x)
    x = layers.Dropout(dropout_conv, name='dropout3')(x)
    x = bb.conv1d(x, 64, 20, batch_norm=bn)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool4')(x)
    x = layers.Dropout(dropout_conv, name='dropout4')(x)
    x = bb.conv1d(x, 64, 15, batch_norm=bn)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool5')(x)
    x = layers.Dropout(dropout_conv, name='dropout5')(x)
    x = bb.conv1d(x, 64, 10, batch_norm=bn)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool6')(x)
    x = layers.Dropout(dropout_conv, name='dropout6')(x)
    x = layers.Flatten(name='flat')(x)
    x = bb.get_dense_stack(x, dense_neurons, dropout, activation='relu')
    opt = optimizers.Adam(learning_rate=lr)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model

def vgg_mod(input_size=5000, dropout_conv=0., dense_neurons=[2000, 500], 
            hidden_act='relu', dropout=0.3, classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    x = layers.Conv1D(6, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool1')(x)
    x = layers.Conv1D(16, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv2_1')(x)
    x = layers.Conv1D(16, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv2_2')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool2')(x)
    x = layers.Conv1D(32, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv3_1')(x)
    x = layers.Conv1D(32, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv3_2')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool3')(x)
    x = layers.Conv1D(64, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv4_1')(x)
    x = layers.Conv1D(64, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv4_2')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool4')(x)
    x = layers.Dropout(dropout_conv, name='dropout4')(x)
    x = layers.Flatten(name='flat')(x)
    x = bb.get_dense_stack(x, dense_neurons, dropout, activation=hidden_act)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model

def resnet_mod(input_size=5000, filters=100, layer_num=6, blocks_per_layer=2, 
               batch_norm=False, dense_neurons=[3100, 1200], hidden_act='relu',
               dropout=.5, classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    # first regular conv
    x = layers.Conv1D(64, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv1'
                      )(input_layer) # No activation (linear)
    for l in range(layer_num):
        for b in range(blocks_per_layer):
            block_type = 'conv' if b == 0 else 'identity'
            x = bb.residual_block(x, block_type=block_type, batch_norm=batch_norm)
    x = layers.Flatten(name='flat')(x)
    x = bb.get_dense_stack(x, dense_neurons, dropout, activation=hidden_act)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model
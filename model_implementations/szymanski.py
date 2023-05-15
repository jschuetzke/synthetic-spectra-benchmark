
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
from .basic_blocks import get_dense_stack

def cnn(input_size=5000, dropout=.7, dense_neurons=[3100, 1200], 
        classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1), 
                               name="input")
    x = layers.Conv1D(64, 35, strides=1, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(3, strides=2, 
                         name='maxpool1')(x)
    x = layers.Conv1D(64, 30, strides=1, padding='same',
                      activation='relu', 
                      name='conv2')(x)
    x = layers.MaxPool1D(3, strides=2, 
                         name='maxpool2')(x)
    x = layers.Conv1D(64, 25, strides=1, padding='same',
                      activation='relu', 
                      name='conv3')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool3')(x)
    x = layers.Conv1D(64, 20, strides=1, padding='same',
                      activation='relu', 
                      name='conv4')(x)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool4')(x)
    x = layers.Conv1D(64, 15, strides=1, padding='same',
                      activation='relu', 
                      name='conv5')(x)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool5')(x)
    x = layers.Conv1D(64, 10, strides=1, padding='same',
                      activation='relu', 
                      name='conv6')(x)
    x = layers.MaxPool1D(1, strides=2, 
                         name='maxpool6')(x)
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout, activation='relu')
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model

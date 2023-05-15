
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
from .basic_blocks import get_dense_stack

def vgg(input_size=5000, dropout_conv=0.2, dense_neurons=[120, 84, 186], 
        dropout=0., classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    x = layers.Conv1D(6, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv1')(input_layer)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool1')(x)
    x = layers.Dropout(dropout_conv, name='dropout1')(x)
    x = layers.Conv1D(16, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv2_1')(x)
    x = layers.Conv1D(16, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv2_2')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool2')(x)
    x = layers.Dropout(dropout_conv, name='dropout2')(x)
    x = layers.Conv1D(32, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv3_1')(x)
    x = layers.Conv1D(32, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv3_2')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool3')(x)
    x = layers.Dropout(dropout_conv, name='dropout3')(x)
    x = layers.Conv1D(64, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv4_1')(x)
    x = layers.Conv1D(64, 5, strides=1, padding='same',
                      activation='relu', 
                      name='conv4_2')(x)
    x = layers.MaxPool1D(2, strides=2, 
                         name='maxpool4')(x)
    x = layers.Dropout(dropout_conv, name='dropout4')(x)
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model

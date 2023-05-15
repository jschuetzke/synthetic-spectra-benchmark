
from tensorflow.keras import layers, optimizers, metrics
from tensorflow.keras.models import Model
from .basic_blocks import residual_block, get_dense_stack
        
def resnet(input_size=5000, filters=100, layer_num=6, blocks_per_layer=2, 
           batch_norm=True, dense_neurons=[], 
           dropout=.5, classes=500, lr=3e-4):
    input_layer = layers.Input(shape=(input_size, 1),
                               name="input")
    # first regular conv
    x = layers.Conv1D(64, 5, strides=1, padding='same', 
                      name='conv1')(input_layer) # No activation (linear)
    x = layers.BatchNormalization()(x)
    for l in range(layer_num):
        for b in range(blocks_per_layer):
            block_type = 'conv' if b == 0 else 'identity'
            x = residual_block(x, block_type=block_type, batch_norm=batch_norm)
    x = layers.Flatten(name='flat')(x)
    x = get_dense_stack(x, dense_neurons, dropout)
    out = layers.Dense(classes, activation='softmax', 
                       name='output')(x)
    model = Model(input_layer, out)
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='categorical_crossentropy',
                  metrics=[metrics.CategoricalAccuracy(name='accuracy')])
    return model
            



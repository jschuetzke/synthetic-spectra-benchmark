import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import model_implementations.modifications as models
import wandb
from wandb.keras import WandbCallback

tf.config.experimental.enable_op_determinism()

if tf.config.list_physical_devices('GPU'):
    # enable memory growth instead of blocking whole VRAM
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import data
xt = np.load('x_train.npy')
xt /= np.max(xt, axis=1, keepdims=True)
xv = np.load('x_val.npy')
xv /= np.max(xv, axis=1, keepdims=True)
xtest = np.load('x_test.npy')
xtest /= np.max(xtest, axis=1, keepdims=True)

yt = np.load('y_train.npy')
yv = np.load('y_val.npy')
ytest = np.load('y_test.npy')

classes = np.unique(ytest).size
yt = tf.one_hot(yt, classes)
yv = tf.one_hot(yv, classes)
ytest_oh = tf.one_hot(ytest, classes)
# using ytest sparse encoding during test data eval 

batch_size=128

model_list = ["cnnbn_mod","cnn6_convdo","cnn6_bn","vgg_mod","resnet_mod"]
model_types = ["cnn_bn","cnn6","cnn6","vgg","resnet"]

for i,model_name in enumerate(model_list):
    model_type = model_types[i]
    for seed in range(5):
        model_spec = f'{model_type}-mod'
        wandb.init(project="synthetic-benchmark", reinit=True, 
                    name=f'{model_name}-{seed}')
        wandb.config.update({'model_type' : model_type, 'seed' : seed,
                            'batch_size' : batch_size, 'activation' : 'relu',
                            'evaluation' : 'modifications'}, 
                            allow_val_change=True)
        callbacks = [EarlyStopping(patience=25, verbose=1,
                                restore_best_weights=True, min_delta=0.0001),
                    ReduceLROnPlateau(patience=10, verbose=1),
                    WandbCallback(save_model=False, save_graph=False)
                    ]
        tf.keras.utils.set_random_seed(seed)
        model = getattr(models, model_name)(classes=classes)
        # make sure order of data is identical during training
        tf.keras.utils.set_random_seed(seed)
        model.fit(xt, yt, batch_size=batch_size, epochs=500, verbose=2, 
                callbacks=callbacks, validation_data=(xv, yv), shuffle=True)
        #tf.keras.models.save_model(model, f'/model_weights/{model_name}-{act}-{seed}.h5',
        #                        include_optimizer=True, save_format='h5')
        exp = model.evaluate(xtest, ytest_oh)
        pred = np.argmax(model.predict(xtest), axis=1)
        misclassifications = np.where(pred != ytest)[0].size
        wandb.log({'wrong_class':misclassifications,
                    'test_accuracy': exp[1]})
        
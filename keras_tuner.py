'''
Tutorial for keras_tuner with MNIST data.
It is written by referring the below websites.
* Reference: https://keras.io/guides/keras_tuner/getting_started/ , https://www.tensorflow.org/tutorials/keras/keras_tuner?hl=ko
* Version
  - Tensorflow: 2.10
  - Python: 3.8
'''

# installation: pip install keras-tuner

#0. import libraries
import keras_tuner
import tensorflow as tf
import numpy as np

#1. prepare the MNIST dataset
(x, y), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x[:-10000]
x_val = x[-10000:]
y_train = y[:-10000]
y_val = y[-10000:]

# normalization
x_train = np.expand_dims(x_train, -1).astype("float32") / 255.0
x_val = np.expand_dims(x_val, -1).astype("float32") / 255.0
x_test = np.expand_dims(x_test, -1).astype("float32") / 255.0
# one-hot encoding
num_classses = 10
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

#2. build a model
def build_model(hp):
  # model architecture
  inputs = tf.keras.layers.Input(shape=(28,28))
  x = tf.keras.layers.Flatten()(inputs)
  # tune the number of dense layers
  for i in range(hp.Int("num_layers"), 1, 3)):
    # tune the number of units of each layer
    units = hp.Int(f"units_{i}", min_value=32, max_value=256, step=16)
    x = tf.keras.layers.Dense(units, activation='relu')(x)
  if hp.Boolean("dropout"):
    x = tf.keras.layers.Dropout(0.2)(x)
  outputs = Dense(10, activation='softmax')
  model = tf.keras.Model(inputs, outputs)

  # compile the model
  lr = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling='log')
  model.compile(optimizers=tf.keras.optimizers.Adam(lr=lr),
               loss='categorical_crossentropy',
               metrics=['accuracy'])
  return model

#3. tuner search
hp = keras_tuner.HyperParameters()
tuner = keras_tuner.RandomSearch(
  hypermodel=build_model,
  objective="val_accuracy",
  max_trials=3,
  excutions_per_trial=2,
  overwrite=True,
  directory="my_dir",  # path where the model is saved
  project_name="mnist" # filename
)

tuner.search_space_summary()
tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val))   # same kwargs with model.fit() 

# 4. query the results
tuner.results_summary()

models = tuner.get_best_models(num_models=2) # get the top 2 models evaluated on the validation data
best_model = models[0]
best_model.summary()

#5. retrain the model
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)
model.fit(x_train, y_train, epochs=100, validation_data=(x_val, y_val))

#6. prediction
pred = model.predict(x_test)





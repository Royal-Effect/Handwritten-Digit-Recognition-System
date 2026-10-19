import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # This hides the startup warnings

import tensorflow as tf
from tensorflow.keras import layers, models

# Load and normalize the dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((60000, 28, 28, 1)).astype('float32') / 255
x_test = x_test.reshape((10000, 28, 28, 1)).astype('float32') / 255

# Build the Neural Network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# The final layer calculates the 10 probabilities
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
print("Training model... please wait.")
model.fit(x_train, y_train, epochs=15, batch_size=64, validation_split=0.1)

# Save the model
model.save('my_digit_model.keras')
print("Model saved successfully as my_digit_model.keras!")
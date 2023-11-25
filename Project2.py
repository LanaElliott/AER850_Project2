#Import libraries
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.set_random_seed(2019)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(600, activation="relu"),
    tf.keras.layers.Dropout(0.1, seed=2019),
    tf.keras.layers.Dense(400, activation="relu"),
    tf.keras.layers.Dropout(0.3, seed=2019),
    tf.keras.layers.Dense(300, activation="relu"),
    tf.keras.layers.Dropout(0.4, seed=2019),
    tf.keras.layers.Dense(200, activation="relu"),
    tf.keras.layers.Dropout(0.2, seed=2019),
    tf.keras.layers.Dense(4, activation="softmax")
])

bs = 32
train_dir = "C:/Users/rmbar/OneDrive/Documents/GitHub/AER850_Project2/Data/Train/"
validation_dir = "C:/Users/rmbar/OneDrive/Documents/GitHub/AER850_Project2/Data/Test/"
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1.0 / 255.)
test_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=bs, class_mode='categorical', target_size=(180, 180))

validation_generator = test_datagen.flow_from_directory(validation_dir,
                                                        batch_size=bs,
                                                        class_mode='categorical',
                                                        target_size=(180, 180))
# Compile the model:
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=150 // bs,
                    epochs=40,
                    validation_steps=50 // bs,
                    verbose=2)

history_second_fit = model.fit(train_generator, validation_data=validation_generator, epochs=30)

# Plotting part 4
plt.plot(history_second_fit.history['accuracy'])
plt.plot(history_second_fit.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train','Validation'], loc='upper left')


plt.plot(history_second_fit.history['loss'])
plt.plot(history_second_fit.history['val_loss'])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
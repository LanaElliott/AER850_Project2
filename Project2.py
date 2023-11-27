#Import libraries
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
tf.compat.v1.set_random_seed(2019)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(100, 100, 3)),
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

train_datagen = ImageDataGenerator(rescale=1.0 / 255., shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)

train_generator = train_datagen.flow_from_directory(train_dir, batch_size=bs, class_mode='categorical', target_size=(100, 100))

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                        batch_size=bs,
                                                        class_mode='categorical',
                                                        target_size=(100, 100))
# Compile the model:
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Creating Model 2:

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1), activation='LeakyReLU', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1), activation='LeakyReLU'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1), activation='LeakyReLU'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='LeakyReLU'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')  
])

#Compiling the model2:
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Define the PrintShapeCallback class
class PrintShapeCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_true, y_pred = self.validation_data[1], self.model.predict(self.validation_data[0])
        print(f"Shapes: y_true={y_true.shape}, y_pred={y_pred.shape}")

# Training the models
history1 = model.fit(train_generator,
                     validation_data=validation_generator,
                     steps_per_epoch=150 // bs,
                     epochs=40,
                     validation_steps=50 // bs,
                     verbose=2)

# Use the PrintShapeCallback for Model 2
history2 = model2.fit(train_generator,
                      validation_data=validation_generator,
                      steps_per_epoch=150 // bs,
                      epochs=30,
                      validation_steps=50 // bs,
                      verbose=2)

# Plotting part for Model 1
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history1.history['accuracy'])
plt.plot(history1.history['val_accuracy'])
plt.title('Model 1 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history1.history['loss'])
plt.plot(history1.history['val_loss'])
plt.title('Model 1 Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Plotting part for Model 2
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.title('Model 2 Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.title('Model 2 Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()

# Saving the models to be accessible for part 5
model.save('Model_1.h5')
model2.save('Model_2.h5')
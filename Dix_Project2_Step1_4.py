#Author: Dix Paulson

from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LeakyReLU
import matplotlib.pyplot as plt
import numpy as np

#Step 1

# Define input image dimensions
im_size = (100, 100, 3)

train_dir = '/Data/train'
validation_dir = '/Data/valid'

# Data augmentation for the training set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2
)

# Rescaling validation set
validation_datagen = ImageDataGenerator(rescale=1./255)

# Generate training data
train_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=im_size[:2],
    batch_size=32,
    class_mode='categorical'
)

# Generate validation data
validation_set = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=im_size[:2],
    batch_size=32,
    class_mode='categorical'
)

#Step 2: Neural Network Architecture Design/
 
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), input_shape=(100, 100, 3)))
model.add(LeakyReLU(alpha=0.1))  
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1)))
model.add(LeakyReLU(alpha=0.1))  
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(units=128, activation='elu')) 
model.add(Dropout(0.5))

#Step 3: Hyperparameter Analysis
model.add(Dense(units=3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Step 4

history = model.fit(
    train_set,
    epochs=12,
    validation_data=validation_set
)


# Plot Training and Validation Accuracy
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1) 
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training Accuracy vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Training and Validation Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training Loss vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

model.save('DixP2.h5')

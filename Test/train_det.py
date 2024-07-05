import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt


# Function to load images and labels
def load_data(image_dir, label_file):
    images = []
    labels = []
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg"):
            img = tf.keras.preprocessing.image.load_img(os.path.join(image_dir, filename), target_size=(224, 224))
            img = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            images.append(img)

    labels = np.load(label_file)  # Assuming labels are stored in a numpy array format
    return np.array(images), np.array(labels)


# Define paths
image_dir = '/mnt/data/data/images/train'
label_file = '/mnt/data/data/labels/train_labels.npy'  # Adjust the label file path

# Load data
images, labels = load_data(image_dir, label_file)

# Split data into training and validation sets
split_index = int(0.8 * len(images))
train_images, val_images = images[:split_index], images[split_index:]
train_labels, val_labels = labels[:split_index], labels[split_index:]

# Build the detection model
input_layer = Input(shape=(224, 224, 3))

x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Output layer for bounding box coordinates (x, y, width, height)
output_layer = Dense(4)(x)

model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluate the model
val_loss, val_acc = model.evaluate(val_images, val_labels)
print(f'Validation loss: {val_loss}, Validation accuracy: {val_acc}')
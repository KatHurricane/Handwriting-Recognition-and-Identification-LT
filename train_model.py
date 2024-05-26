import os
import cv2
import numpy as np
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Conv2D, MaxPooling2D, Reshape, Dense, GRU
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ModelCheckpoint
from keras._tf_keras.keras.backend import ctc_batch_cost
from keras._tf_keras.keras.utils import Sequence

# Define the character set to include Lithuanian letters
alphabet = "abcdefghijklmnopqrstuvwxyząčęėįšųūž"

# Load individual letter images from multiple directories
def load_letter_images(directories):
    images = []
    labels = []
    for directory in directories:
        for file in os.listdir(directory):
            if file.endswith(".png") or file.endswith(".jpg"):
                letter = file.split('.')[0]
                image_path = os.path.join(directory, file)
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                images.append(image)
                labels.append(letter)
    return images, labels

# Define CTC loss function
def ctc_loss_function(y_true, y_pred):
    input_length = np.ones((y_pred.shape[0], 1)) * y_pred.shape[1]
    label_length = np.ones((y_true.shape[0], 1)) * y_true.shape[1]
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)

# Define model
input_shape = (32, 32, 1)  # Image dimensions for individual letters
num_classes = len(alphabet) + 1  # Including CTC blank token

inputs = Input(shape=input_shape)
x = Conv2D(32, (3,3), activation='relu', padding='same')(inputs)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Reshape((-1, x.shape[2] * x.shape[3]))(x)
x = GRU(128, return_sequences=True)(x)
x = GRU(128, return_sequences=True)(x)
outputs = Dense(num_classes, activation='softmax')(x)

model = Model(inputs, outputs)
model.compile(optimizer=Adam(), loss=ctc_loss_function)

model.summary()

# Define data generator
class DataGenerator(Sequence):
    def __init__(self, images, labels, batch_size, alphabet, input_shape):
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.alphabet = alphabet
        self.input_shape = input_shape
        self.indices = np.arange(len(self.images))
    
    def __len__(self):
        return len(self.images) // self.batch_size
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_images = [self.images[i] for i in batch_indices]
        batch_labels = [self.labels[i] for i in batch_indices]
        
        X = np.zeros((self.batch_size, *self.input_shape), dtype=np.float32)
        Y = np.zeros((self.batch_size, len(max(batch_labels, key=len))), dtype=np.int32)
        
        for i in range(self.batch_size):
            img = cv2.resize(batch_images[i], (self.input_shape[1], self.input_shape[0]))
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=-1)
            X[i] = img
            
            label_str = batch_labels[i]
            label_encoded = [self.alphabet.index(char) for char in label_str]
            Y[i, :len(label_encoded)] = label_encoded
        
        return X, Y

# Load individual letter images from specified directories
directories = ["input_synthetic_data", "augmented_images", "output_letters"]
images, labels = load_letter_images(directories)

# Split data into training and validation sets
train_size = int(0.8 * len(images))
train_images, val_images = images[:train_size], images[train_size:]
train_labels, val_labels = labels[:train_size], labels[train_size:]

# Training
batch_size = 32

train_generator = DataGenerator(train_images, train_labels, batch_size, alphabet, input_shape)
val_generator = DataGenerator(val_images, val_labels, batch_size, alphabet, input_shape)

checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', save_best_only=True, save_weights_only=False)

history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=50,
                    callbacks=[checkpoint])
# -*- coding: utf-8 -*-
"""Homework_1_dataset_augmentation

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1jFXkH7SfCe4GQp70gcvLCKIXLqlsvAFy

## 🌐 Connect Colab to Google Drive
"""

# Commented out IPython magic to ensure Python compatibility.
from google.colab import drive

drive.mount('/gdrive')
# %cd /gdrive/My Drive/Colab Notebooks/Homework 1_lst

"""## ⚙️ Import Libraries"""

# Commented out IPython magic to ensure Python compatibility.
# Set seed for reproducibility
seed = 42

# Import necessary libraries
import os

!pip install keras-cv
import keras_cv

# Set environment variables before importing modules
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ['MPLCONFIGDIR'] = os.getcwd() + '/configs/'

# Suppress warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)

# Import necessary modules
import logging
import random
import numpy as np

# Set seeds for random number generators in NumPy and Python
np.random.seed(seed)
random.seed(seed)

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow.keras import layers as tfkl

# Set seed for TensorFlow
tf.random.set_seed(seed)
tf.compat.v1.set_random_seed(seed)

# Reduce TensorFlow verbosity
tf.autograph.set_verbosity(0)
tf.get_logger().setLevel(logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Print TensorFlow version
print(tf.__version__)

# Import other libraries
import requests
from io import BytesIO
import cv2
from PIL import Image
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Configure plot display settings
sns.set(font_scale=1.4)
sns.set_style('white')
plt.rc('font', size=14)

# %matplotlib inline

"""## ⏳ Load the Data"""

data = np.load('training_set.npz')
X = data['images']
y = data['labels']
print(X.shape, y.shape)
print(type(X))

mask = np.array([i < 11959 for i in range(X.shape[0])])
X = X[mask]
y = y[mask]
print(X.shape, y.shape)

"""## ⛏ Splitting the Data"""

# One-hot encoding
y = tfk.utils.to_categorical(y, len(np.unique(y)))

# Split data into train_val and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y,
                  random_state = seed, test_size = 0.15,
                  stratify = np.argmax(y,axis=1))

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,
                  random_state=seed, test_size=len(X_test),
                  stratify=np.argmax(y_train_val,axis=1))

# Print shapes of the datasets
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

"""## ⚾ First augmentations"""

from tqdm import tqdm

# Custom augmentation layer for each transformation
class CustomAugmentationLayer(tf.keras.layers.Layer):
    def __init__(self, transformation):
        super(CustomAugmentationLayer, self).__init__()
        self.transformation = transformation

    def call(self, inputs):
        return self.transformation(inputs)

# Function to apply a single transformation to a random portion of X_train
def augment_with_transformation(X, y, transformation, portion=25, batch_size=128):
    # Sample a random quarter of the data
    indices = np.random.choice(len(X), len(X) // portion, replace=False)
    X_subset = tf.gather(X, indices)
    y_subset = tf.gather(y, indices)

    # Apply transformation in batches and store results
    augmented_batches = []
    num_batches = (len(X_subset) + batch_size - 1) // batch_size
    for i in tqdm(range(0, len(X_subset), batch_size), desc="Applying Transformation", total=num_batches):
        batch = X_subset[i:i + batch_size]
        augmented_batch = CustomAugmentationLayer(transformation)(batch)
        augmented_batches.append(augmented_batch)

    X_augmented = tf.concat(augmented_batches, axis=0)
    y_augmented = tf.gather(y, indices)

    X = tf.concat([X, X_augmented], axis=0)
    y = tf.concat([y, y_augmented], axis=0)
    return X, y

rand_augment = [keras_cv.layers.RandAugment(value_range=[0, 255], augmentations_per_image=1,
                                           magnitude = 0.5),
                keras_cv.layers.RandAugment(value_range=[0, 255], augmentations_per_image=1,
                                           magnitude = 0.5)]
X_train_augmented = X_train
y_train_augmented = y_train
del X_train
del y_train
for transformation in rand_augment:
  X_train_augmented, y_train_augmented = augment_with_transformation(
       X_train_augmented, y_train_augmented, transformation = transformation,
       portion = 1, batch_size=128
  )

# List of transformations from keras_cv
transformations = [
    tfkl.RandomFlip("horizontal_vertical"),
    tfkl.RandomTranslation(0.2, 0.2),
    tfkl.RandomZoom(0.2),
    tfkl.RandomRotation(0.2),
    keras_cv.layers.AutoContrast(value_range=[0, 255]),
    keras_cv.layers.RandomHue(factor=1.0, value_range=[0, 255]),
    keras_cv.layers.RandomSaturation(factor=1.0),
    keras_cv.layers.RandomCutout(height_factor=0.5, width_factor=0.4),
    keras_cv.layers.RandomSharpness(factor=0.6, value_range=[0, 255]),
    keras_cv.layers.RandomBrightness(factor=0.7, value_range=[0, 255]),
    keras_cv.layers.RandomContrast(factor=0.8, value_range=[0, 255]),
    keras_cv.layers.RandomGaussianBlur(kernel_size = 3, factor = 0.2),
    keras_cv.layers.RandomCropAndResize(target_size=(96, 96),
        crop_area_factor=(0.5, 1.0),
        aspect_ratio_factor=(0.75, 1.33)),
    keras_cv.layers.RandomChannelShift(factor = 0.2, value_range = [0, 255]),
    keras_cv.layers.RandomColorDegeneration(factor = 0.2),
    keras_cv.layers.Solarization(value_range = [0, 255])
]

for transformation in transformations:
    X_train_augmented, y_train_augmented = augment_with_transformation(
        X_train_augmented, y_train_augmented, transformation, batch_size=128
    )

X_train_augmented = tf.cast(X_train_augmented, tf.uint8)
print(X_train_augmented.shape)
print(y_train_augmented.shape)

"""## ⏰ Second augmentations: with labels (MixUp)"""

# List of transformations with labels
transformations_with_labels = [
    keras_cv.layers.CutMix(alpha=0.3),
    keras_cv.layers.MixUp(alpha=0.3)
]

# Randomly select a portion of the dataset
subset_indices = np.random.choice(len(X_train_augmented), len(X_train_augmented) // 10, replace=False)
X_subset = tf.gather(X_train_augmented, subset_indices)
y_subset = tf.gather(y_train_augmented, subset_indices)
X_subset = tf.cast(X_subset, dtype=tf.float32)
y_subset = tf.cast(y_subset, dtype=tf.float32)

# Function to apply a transformation to the entire subset
def augment_with_transformation_with_labels(X, y, transformation, batch_size=128):
    augmented_images = []
    augmented_labels = []
    num_batches = (len(X) + batch_size - 1) // batch_size

    for i in tqdm(range(0, len(X), batch_size), desc="Applying Transformation", total=num_batches):
        batch = {"images": X[i:i + batch_size], "labels": y[i:i + batch_size]}

        # Apply transformation that takes both images and labels
        augmented_batch = transformation(batch)

        # Store augmented images and labels
        augmented_images.append(augmented_batch["images"])
        augmented_labels.append(augmented_batch["labels"])

    # Concatenate augmented images and labels
    X_augmented = tf.concat(augmented_images, axis=0)
    y_augmented = tf.concat(augmented_labels, axis=0)

    return X_augmented, y_augmented

# Initialize augmented data lists
augmented_images_list = []
augmented_labels_list = []

# Apply each transformation to the selected subset and store the results
for transformation in transformations_with_labels:
    X_augmented, y_augmented = augment_with_transformation_with_labels(
        X_subset, y_subset,
        transformation,
        batch_size=128
    )

    # Store each augmented subset separately
    augmented_images_list.append(X_augmented)
    augmented_labels_list.append(y_augmented)

# Concatenate all augmented subsets together
X_subset_augmented = tf.concat(augmented_images_list, axis=0)
y_subset_augmented = tf.concat(augmented_labels_list, axis=0)

# Convert the augmented data to the desired data types if needed
X_subset_augmented = tf.cast(X_subset_augmented, dtype=tf.uint8)
y_subset_augmented = tf.cast(y_subset_augmented, dtype=tf.float64)

# Append the augmented data to the main dataset
X_train_augmented = tf.concat([X_train_augmented, X_subset_augmented], axis=0)
y_train_augmented = tf.concat([y_train_augmented, y_subset_augmented], axis=0)

# Output the shapes of the augmented dataset
print(X_train_augmented.shape)
print(y_train_augmented.shape)

np.savez("training_augmented_big.npz", X_train=X_train_augmented, y_train=y_train_augmented)
del X_train_augmented
del y_train_augmented

"""## 🪗 Little augmentation for validation robustness"""

# Apply data augmentation for validation robustness
augmentation = tf.keras.Sequential([
    tfkl.RandomFlip("horizontal_vertical"),
    tfkl.RandomTranslation(0.15, 0.15),
    tfkl.RandomZoom(0.1),
    tfkl.RandomRotation(0.1),
    tfkl.RandomContrast(0.1),
    tfkl.RandomBrightness(0.1),
    tfkl.RandomContrast(0.1),
    tfkl.RandomFlip("horizontal")
], name='preprocessing')

X_val_1 = augmentation(X_val)
X_val2 = tf.concat([X_val, X_val_1], axis=0)
y_val2 = tf.concat([y_val, y_val], axis=0)

np.savez("validation_test.npz", X_test=X_test, y_test=y_test,
         X_val=X_val2, y_val=y_val2)

np.load("training_augmented_big.npz")["X_train"].shape

"""## ‼ Plot examples"""

X_train = np.load("training_augmented_big.npz")["X_train"]

# Set the number of samples to plot
num_samples = 10

# Select 10 random indices
random_indices = np.random.choice(X_train.shape[0], num_samples, replace=False)

# Create a figure for the plots
plt.figure(figsize=(15, 8))

# Plot the 10 random samples
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[idx, :, :, :])

plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from pathlib import Path
import matplotlib.pyplot as plt
ROOT = Path(__file__).parent.parent #Root directory is the folder this file is placed in

#Directories of datasets
train_dir = os.path.join(ROOT,'Datasets/cartoon_set/img')
labels_train = os.path.join(ROOT,'Datasets/cartoon_set/labels.csv')
test_dir = os.path.join(ROOT,'Datasets/cartoon_set_test/img')
labels_test = os.path.join(ROOT, 'Datasets/cartoon_set_test/labels.csv')

#Import image paths and labels from .csv
def importData(img_dir, labels_dir):
    labels_file = pd.read_csv(os.path.join(ROOT, labels_dir), sep = '\t', engine = 'python', header = 0)
    face_labels = labels_file['eye_color'].values
    img_paths = labels_file['file_name'].values
    for i in range(0, len(img_paths)):
        img_paths[i] = os.path.join(img_dir, img_paths[i])
    print("Data imported")

    return img_paths, face_labels

#Decodes the image from a png to an array understandable by the network
def decode_img(img):
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.resize(img, [100, 100])
    return img

#Recieves the image path and returns an image understandable by the network with its associated label 
def process_path(img_id, label):
    img = tf.io.read_file(img_id)
    img = decode_img(img)
    return img, label

#Initialisze CNN model. This contains 3 convolutional layers with a maxPooling layer after each, then into 3 fully connected layers
def CNNmodel():
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation="relu", input_shape = (100,100,3)))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(32, (3,3), 1, activation="relu"))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), 1, activation="relu"))
    model.add(MaxPooling2D((2,2)))
    

    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(64, activation = 'relu'))
    model.add(Dense(5, activation = 'softmax'))

    return model


### MAIN PROGRAM ###

#Get training and testing data
train_img_paths, train_labels = importData(train_dir, labels_train)
test_img_paths, test_labels = importData(test_dir, labels_test)

#Create tensorflow datasets from data
dataset_train = tf.data.Dataset.from_tensor_slices((train_img_paths, train_labels))
dataset_test = tf.data.Dataset.from_tensor_slices((test_img_paths, test_labels))

#Split training dataset into 80% training and 20% validation data
ds_size = len(dataset_train)
train_size = int(ds_size * 0.8)
Train = dataset_train.take(train_size)
Val = dataset_train.skip(train_size)

AUTOTUNE = tf.data.AUTOTUNE #Tunes values dynamically at runtime

#Converts the image paths stored in the dataset to the arrays associated with the images
train_ds = Train.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = Val.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = dataset_test.map(process_path, num_parallel_calls=AUTOTUNE)

#Creates batches of 32 in each dataset
train_ds = train_ds.batch(32)
val_ds = val_ds.batch(32)
test_ds = test_ds.batch(32)

"""
#Visualise the data and associated labels
image_batch, label_batch = next(iter(test_ds))

plt.figure(figsize=(10, 10))
for i in range(9):
  ax = plt.subplot(3, 3, i + 1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  label = label_batch[i]
  plt.title(str(label_batch[i].numpy()))
  plt.axis("off")
"""

#Create callback function to stop training early if network converges (to prevent overfitting)
early_stop = EarlyStopping(monitor = "val_loss", restore_best_weights=True, patience=5, verbose=1)
callback = [early_stop]

#Create CNN model and show summary of network
model = CNNmodel()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

#Fit the training data to the model
history = model.fit(train_ds, epochs=10, validation_data=val_ds, callbacks=callback)

plt.plot(history.history['accuracy'], label = "Train")
plt.plot(history.history['val_accuracy'], label = "Valid")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("B2Acc.png")
plt.show()

#Test the model on the test dataset
results = model.evaluate(test_ds)
print("Accuracy of CNN:", results[1])
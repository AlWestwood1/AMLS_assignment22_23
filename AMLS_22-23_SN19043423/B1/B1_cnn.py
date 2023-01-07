import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, BatchNormalization
from keras.callbacks import EarlyStopping
from keras_preprocessing import image
from sklearn.model_selection import train_test_split
from pathlib import Path
ROOT = Path(__file__).parent.parent #Root directory is the folder this file is placed in

train_dir = os.path.join(ROOT,'Datasets/cartoon_set/img')
labels_train = os.path.join(ROOT,'Datasets/cartoon_set/labels.csv')
test_dir = os.path.join(ROOT,'Datasets/cartoon_set_test/img')
labels_test = os.path.join(ROOT, 'Datasets/cartoon_set_test/labels.csv')


def importData(img_dir, labels_dir):
    image_paths = np.array([os.path.join(img_dir, l) for l in os.listdir(img_dir)])
    labels_file = pd.read_csv(os.path.join(ROOT, labels_dir), sep = '\t', engine = 'python', header = 0)
    face_labels_df = labels_file['face_shape']
    face_labels = face_labels_df.values
    face_labels = tf.one_hot(face_labels, 5)
    face_imgs = []
    for i in image_paths:
        img = tf.io.read_file(i)
        img = tf.image.decode_png(img, channels= 3, dtype=tf.float32)
        img = tf.image.resize(img, [256, 256, 3])
        face_imgs.append(img)
    X = np.array(face_imgs)
    print("Data imported")

    return X, face_labels

def preprocessing(train_imgs, test_imgs, train_labels, test_labels):
    xTrain, xVal, yTrain, yVal = train_test_split(train_imgs, train_labels, test_size = 0.2)

    xTest = test_imgs
    yTrain = tf.keras.utils.to_categorical(yTrain)
    yVal = tf.keras.utils.to_categorical(yVal)
    yTest = tf.keras.utils.to_categorical(test_labels)

    return xTrain, xVal, xTest, yTrain, yVal, yTest

def CNNmodel():
    model = Sequential()
    model.add(Conv2D(16, (3,3), 1, activation="relu", input_shape = (256,256,1)))
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


train_imgs, train_labels = importData(train_dir, labels_train)
test_imgs, test_labels = importData(test_dir, labels_test)



dataset_train = tf.data.Dataset.from_tensor_slices((train_imgs, train_labels))
dataset_train = dataset_train.batch(32)
dataset_train = dataset_train.map(lambda x, y: (x/255, y))

ds_size = len(dataset_train)
train_size = int(ds_size * 0.8)

Train = dataset_train.take(train_size)
Val = dataset_train.skip(train_size)


early_stop = EarlyStopping(monitor = "val_loss", restore_best_weights=True, patience=5, verbose=1)
callback = [early_stop]

model = CNNmodel()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(Train, epochs=10, validation_data=Val, callbacks=callback)

"""
cnn_loss, cnn_acc = model.evaluate(xTest, yTest, verbose=2)
print("Test accuracy:", cnn_acc)
"""
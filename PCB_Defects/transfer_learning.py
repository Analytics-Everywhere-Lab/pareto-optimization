import numpy as np
import tensorflow as tf
from tensorflow import keras
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
from monitor import custom_train

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'  #Prevents libiomp5md.dll error


train_dir = 'dataset/train'
validation_dir = 'dataset/val'
test_dir = 'dataset/test'

BATCH_SIZE = 16
IMG_SIZE = (224, 224)


train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    shuffle=True,
    batch_size=BATCH_SIZE
)

#Buffering images from disk to prevent I/O blocking
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


#Build new model
inputs = keras.Input(shape=(224, 224, 3))
x = inputs 
x = keras.layers.Rescaling(scale=1/127.5, offset=-1)(inputs)

#Uncomment
base_model = keras.applications.MobileNetV2(
    weights = 'imagenet',
    input_shape=(224, 224, 3),
    include_top = False  #Do no include ImageNet classifier at the top
)

base_model.trainable = False

x = base_model(x, training=False)

x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  #Regularize the dropout
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model = keras.Model(inputs, outputs)

model.summary()


#Train the model
model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [keras.metrics.BinaryAccuracy()]
)

epochs = 60
history = custom_train(train_dataset, validation_dataset, model, epochs)

history.to_csv('History/Monitoring/mobilenetTL.csv', encoding='utf-8', index=False) 

#model.save_weights('Models/Transfer_Learning/EfficientNet/EfficientNet') #uncomment
#model.save('Models/MobileNetV2-FT')  #comment                                            

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
import os
from monitor import custom_train

os.environ['KMP_DUPLICATE_LIB_OK']='True'  #Prevents libiomp5md.dll error


train_dir = 'dataset/train'
val_dir = 'dataset/val'

BATCH_SIZE = 16
IMG_SIZE = (224, 224)

train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

#Buffering images from disk to prevent I/O blocking
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

augmentation = keras.Sequential(
	[
		layers.RandomRotation(factor=0.15),
		layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
		layers.RandomFlip(),
		layers.RandomContrast(factor=0.1),
	],
	name="augmentation",
)

#Build new model
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = augmentation(inputs)
x = keras.layers.Rescaling(scale=1/127.5, offset=-1)(x)
x = keras.applications.EfficientNetV2B1(
	weights = None, 
	include_top = False
)(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  #Regularize the dropout
outputs = keras.layers.Dense(101, activation='softmax')(x)

model = keras.Model(inputs, outputs)


model.summary()


#Train the model
model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
)

epochs = 60

history = custom_train(train_dataset, validation_dataset, model, epochs)

history.to_csv('history/efficientnet.csv', encoding='utf-8', index=False)  
                      
model.save('models/efficientnet.h5')
                                          

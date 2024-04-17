import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd
from monitor import custom_train	

import os
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

print('Dataset loaded')

augmentation = keras.Sequential(
	[
		layers.RandomRotation(factor=0.15),
		layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
		layers.RandomFlip(),
		layers.RandomContrast(factor=0.1),
	],
	name="augmentation",
)


def entry_flow(inputs):
  # Entry block
  x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)

  x = layers.Conv2D(64, 3, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)

  previous_block_activation = x  # Set aside residual
  
  # Blocks 1, 2, 3 are identical apart from the feature depth.
  for size in [128, 256, 512]:
    #expand 1x1
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(size, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    #expand 3x3
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(size, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Project residual
    residual = layers.Conv2D(
        size, 1, strides=2, padding='same')(previous_block_activation)
    x = layers.add([x, residual])  # Add back residual
    previous_block_activation = x  # Set aside next residual

  return x


def middle_flow(x, num_blocks=8):
  
  previous_block_activation = x

  for _ in range(num_blocks):
    #squeeze 1x1
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(256, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    #expand 1x1
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(512, 1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    #expand 3x3
    x = layers.Activation('relu')(x)
    x = layers.SeparableConv2D(512, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)

    x = layers.add([x, previous_block_activation])  # Add back residual
    previous_block_activation = x  # Set aside next residual
    
  return x
  
def exit_flow(x):

  previous_block_activation = x

  x = layers.Activation('relu')(x)
  x = layers.SeparableConv2D(728, 1, padding='same')(x)
  x = layers.BatchNormalization()(x)

  x = layers.Activation('relu')(x)
  x = layers.SeparableConv2D(1024, 3, padding='same')(x)
  x = layers.BatchNormalization()(x)
  
  x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

  # Project residual
  residual = layers.Conv2D(1024, 3, strides=2, padding='same')(previous_block_activation)
  x = layers.add([x, residual])  # Add back residual
  
  x = layers.SeparableConv2D(1536, 3, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  
  x = layers.SeparableConv2D(2048, 3, padding='same')(x)
  x = layers.BatchNormalization()(x)
  x = layers.Activation('relu')(x)
  
  x = layers.GlobalAveragePooling2D()(x)
  x = keras.layers.Dropout(0.2)(x)  #Regularize the dropout
  return layers.Dense(101, activation='softmax')(x)

#Build Xception architecture
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
inputs = keras.Input(shape=input_shape)
x = augmentation(inputs)
x = keras.layers.Rescaling(scale=1/127.5, offset=-1)(x)
outputs = exit_flow(middle_flow(entry_flow(x)))
model = keras.Model(inputs, outputs)
model.summary()


model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics = [keras.metrics.SparseCategoricalAccuracy()]
)

epochs = 60

## The steps below ensure resources are cleared at the end of training
def train(model, train_dataset, validation_dataset, epochs):
	
	history = custom_train(train_dataset, validation_dataset, model, epochs)

	history.to_csv('history/xception-lite.csv', encoding='utf-8', index=False) 

	#Save model
	model.save('models/xception-lite.h5')

	
import multiprocessing
p = multiprocessing.Process(target=train(model, train_dataset, validation_dataset, epochs))
p.start()
p.join()

#input()



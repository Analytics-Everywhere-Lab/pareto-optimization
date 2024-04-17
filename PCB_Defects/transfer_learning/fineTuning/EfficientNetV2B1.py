import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import pandas as pd
import os
from monitor import custom_train
from memory_profiler import profile
os.environ['KMP_DUPLICATE_LIB_OK']='True'  #Prevents libiomp5md.dll error


train_dir = '../../dataset/train'
val_dir = '../../dataset/val'

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


#Load pre-trained model
model = keras.models.load_model('../../models/pretrained/efficientnet.h5')
model.summary()

# Freeze all layers up to the second-to-last layer
for layer in model.layers:
    layer.trainable = True


#Train the model
model.compile(
    optimizer = keras.optimizers.Adam(),
    loss = keras.losses.BinaryCrossentropy(),
    metrics = [keras.metrics.BinaryAccuracy()]
)

epochs = 20

## The steps below ensure resources are cleared at the end of training
def train(model, train_dataset, validation_dataset, epochs):
	history = custom_train(train_dataset, validation_dataset, model, epochs)

	history.to_csv('../../history/fineTuned/efficientnet.csv', encoding='utf-8', index=False) 

	#Save model
	model.save('../../models/fineTuned/efficientnet.h5')

	
import multiprocessing
p = multiprocessing.Process(target=train(model, train_dataset, validation_dataset, epochs))
p.start()
p.join()
                                          

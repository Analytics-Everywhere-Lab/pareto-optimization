from jtop import jtop
import numpy as np
import tensorflow as tf
from tensorflow import keras
# import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
import os
import time

from memory_profiler import profile
os.environ['KMP_DUPLICATE_LIB_OK']='True'  #Prevents libiomp5md.dll error




data_dict = {'Epoch': [], 'Loss' : [], 'Acc': [], 'Val_acc': [], 'Mem_Use': [], 'GPU_Use': [], 'Pow_Cons': [], 'Time': []}

optimizer = keras.optimizers.Adam()
loss = keras.losses.BinaryCrossentropy()
train_acc_metric = keras.metrics.BinaryAccuracy()
val_acc_metric = keras.metrics.BinaryAccuracy()


#### Custom training loop ###
@tf.function
def train_step(model, x, y):
	with tf.GradientTape() as tape:
		#Forward pass
		logits = model(x, training=True)			
		loss_value = loss(y, logits)
		
	grads = tape.gradient(loss_value, model.trainable_weights)
	optimizer.apply_gradients(zip(grads, model.trainable_weights))
		
	train_acc_metric.update_state(y, logits)
	return loss_value
	
@tf.function
def test_step(model, x, y):
	val_logits = model(x, training=False)
	val_acc_metric.update_state(y, val_logits)	

def custom_train(train_dataset, validation_dataset, model, epochs):
	jetson = jtop()
	jetson.start()

	for epoch in range(epochs):
		print("Start of epoch %d" %(epoch,))
		start_time = time.time()

		#Iterate over dataset
		for step, (x_batch_train, y_batch_train) in enumerate (train_dataset):
			loss_value = train_step(model, x_batch_train, y_batch_train)
			
			#Log every 200 batches
			#if step % 200 == 0:
				#print("Training loss (for one batch) at step %d: %.4f" % (step, float(loss_value)))
				
			train_acc = train_acc_metric.result()
			print("step: %d - loss: %.4f - acc: %.4f" % (step, float(loss_value), float(train_acc)))
		
		#Read the stats during the epoch
		ram_stop = jetson.stats['RAM']
		gpu_stop = jetson.stats['GPU']
		pow_stop = jetson.stats['Power TOT']
		
		# Reset training metrics at the end of each epoch
		train_acc_metric.reset_states()
		
		#Validation
		for x_batch_val, y_batch_val in validation_dataset:
			test_step(model, x_batch_val, y_batch_val)

		val_acc = val_acc_metric.result()
		val_acc_metric.reset_states()
		
		

		print("Validation acc: %.4f" % (float(val_acc),))	
		
		end_time = time.time()
		time_taken = end_time - start_time
		
		print("Time taken: %.2fs" % (time_taken))
		
		

		ram_use = ram_stop 
		gpu_use = gpu_stop 
		pow_use = pow_stop 	
		
		print("Memory Use: %.4f" % (ram_use))
		# Append values to the dictionary
		data_dict['Epoch'].append(epoch)
		data_dict['Loss'].append(float(loss_value))
		data_dict['Acc'].append(float(train_acc))
		data_dict['Val_acc'].append(float(val_acc))
		data_dict['Mem_Use'].append(ram_use)
		data_dict['GPU_Use'].append(gpu_use)
		data_dict['Pow_Cons'].append(pow_use)
		data_dict['Time'].append(float(time_taken))

	jetson.close()     
	df = pd.DataFrame.from_dict(data_dict)
	return df
	
                         

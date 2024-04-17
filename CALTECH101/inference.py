from sklearn.model_selection import KFold
import tensorflow as tf
import time
import numpy as np


test_dir = 'dataset/test'

BATCH_SIZE = 16
IMG_SIZE = (224, 224)
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

model = tf.keras.models.load_model('models/xception.h5')

model.summary()

test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Acc: {test_acc}")

for input_sample, _ in test_dataset.take(5):
	model.predict(input_sample)
	
inference_latencies = []

for input_sample, _ in test_dataset:
	start_time = time.time()
	prediction = model.predict(input_sample)
	end_time = time.time()
	
	inference_latency = end_time - start_time
	inference_latencies.append(inference_latency)
	
average_latency = sum(inference_latencies) / len(inference_latencies)
print(f"Avg inf latency: {average_latency} s")

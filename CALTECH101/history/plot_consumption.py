import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


SMALL_SIZE = 8
MEDIUM_SIZE = 17
BIGGER_SIZE = 22

# plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


xceptionLite = pd.read_csv('xception-lite.csv')
effnet = pd.read_csv('efficientnet.csv')
mobnet = pd.read_csv('mobilenet.csv')
xception = pd.read_csv('xception.csv')
#squeezenet = pd.read_csv('squeezenet.csv')

xceptionLite_usage = xceptionLite['Mem_Use']
effnet_usage = effnet['Mem_Use']
mobnet_usage = mobnet['Mem_Use']
xception_usage = xception['Mem_Use']
#squeezenet_usage = squeezenet['Mem_Use']

print("Xception-Lite: ", xceptionLite_usage.mean(axis=0))
print("EfficientNet: ", effnet_usage.mean(axis=0))
print("MobileNet: ", mobnet_usage.mean(axis=0))
print("Xception: ", xception_usage.mean(axis=0))

#epochs_range = range(60)
epochs_range = range(0, len(xceptionLite_usage), 1)


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, xceptionLite_usage[::1], label='Xception-Lite')
plt.plot(epochs_range, effnet_usage[::1], label='EfficientNet')
plt.plot(epochs_range, mobnet_usage[::1], label='MobileNet')
plt.plot(epochs_range, xception_usage[::1], label='Xception')
#plt.plot(epochs_range, squeezenet_usage[::1], label='Squeezenet')
plt.legend(loc='best')
plt.title('Memory Consumption')
plt.savefig('memory.png')
plt.show()

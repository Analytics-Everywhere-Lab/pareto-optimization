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


xception_lite = pd.read_csv('xception-lite.csv')
effnet = pd.read_csv('efficientnet.csv')
#mobnet = pd.read_csv('mobilenet.csv')
#mobilevit = pd.read_csv('mobilevit.csv')


xception_lite_usage = xception_lite['Pow_Cons']
effnet_usage = effnet['Pow_Cons']
#mobnet_usage = mobnet['Pow_Cons']
#mobilevit_usage = mobilevit['Pow_Cons']

#epochs_range = range(60)
epochs_range = range(0, len(xception_lite_usage), 1)


plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, xception_lite_usage[::1], label='Xception-Lite')
plt.plot(epochs_range, effnet_usage[::1], label='EfficientNet')
#plt.plot(epochs_range, mobnet_usage[::1], label='MobileNet')
#plt.plot(epochs_range, mobilevit_usage[::1], label='MobileViT')
plt.legend(loc='best')
plt.title('Power Consumption')
plt.show()

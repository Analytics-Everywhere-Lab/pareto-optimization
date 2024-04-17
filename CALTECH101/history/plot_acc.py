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
#effnet = pd.read_csv('efficientnet.csv')
mobnet = pd.read_csv('mobilenet.csv')
xception = pd.read_csv('xception.csv')


xceptionLite_acc = xceptionLite['Val_acc']
#effnet_acc = effnet['Acc']
mobnet_acc = mobnet['Val_acc']
xception_acc = xception['Val_acc']

epochs_range = range(60)

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, xceptionLite_acc, label='Xception-Lite')
#plt.plot(epochs_range, effnet_acc, label='EfficientNet')
plt.plot(epochs_range, mobnet_acc, label='MobileNet')
plt.plot(epochs_range, xception_acc, label='Xception')
plt.legend(loc='best')
plt.title('Validation Accuracy')
plt.savefig('val_accuracy.png')
plt.show()

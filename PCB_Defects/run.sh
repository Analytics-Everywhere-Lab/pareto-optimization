#!/bin/bash

sudo -v

#Train the models from scratch
# MobileNetV2
python3 MobileNetV2.py
wait
sudo systemctl restart nvargus-daemon
echo 'MobileNet training complete'

#EfficientNetV2B1
python3 EfficientNetV2B1.py
wait
sudo systemctl restart nvargus-daemon
echo 'EffNet training complete'

#Xception-Lite
python3 Xception-Lite.py
wait
sudo systemctl restart nvargus-daemon
echo 'Xception-Lite training complete'

# Xception
python3 Xception.py
wait
sudo systemctl restart nvargus-daemon
echo 'Xception training complete'


# Train the pre-trained models
cd transfer_learning

# MobileNet
python3 MobileNetV2.py
wait
sudo systemctl restart nvargus-daemon
echo 'MobileNet training complete'
sleep 120

#EfficientNet
python3 EfficientNetV2B1.py
wait
sudo systemctl restart nvargus-daemon
echo 'EfficientNet training complete'
sleep 120

#Xception-Lite
python3 Xception-Lite.py
wait
sudo systemctl restart nvargus-daemon
echo 'Xception-Lite training complete'
sleep 120

#Xception
python3 Xception.py
wait
sudo systemctl restart nvargus-daemon
echo 'Xception PCB training complete'

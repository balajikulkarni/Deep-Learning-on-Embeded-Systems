# Deep-Learning-on-Embeded-Systems
Running inference on Edge devices for pre-trained Deep learning models (offline).

Setup:

Follow the installation instructions as mentioned in this page:
https://medium.com/@paroskwan/layman-installation-guide-for-keras-and-tensorflow-on-rpi-3-38b84f3e59dc

Laying down the same from above page:

Tensorflow installation:
wget https://github.com/samjabrahams/tensorflow-on-raspberry-pi/releases/download/v1.1.0/tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
sudo pip3 install tensorflow-1.1.0-cp34-cp34m-linux_armv7l.whl
sudo pip3 uninstall mock
sudo pip3 install mock

Keras installation:
sudo apt-get install libblas-dev
sudo apt-get install liblapack-dev
sudo apt-get install python3-dev 
sudo apt-get install libatlas-base-dev
sudo apt-get install gfortran
sudo apt-get install python3-setuptools
sudo apt-get install python3-scipy
sudo apt-get update
sudo apt-get install python3-h5py
sudo pip3 install keras 
sudo apt-get install python3-skimage


Test installation(s) as below:

Tensorflow:
python -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 2
python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3

Keras:
python -c 'import keras; print(keras.__version__)'# python 2
python3 -c 'import keras; print(keras.__version__)'  # for Python 3


Contents:
1. MNIST Inference using pre-trained weights on raspberry pi

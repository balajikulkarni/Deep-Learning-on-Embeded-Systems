import numpy as np
import tensorflow as tf

#Import needed modules from keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Add
from keras.layers import Convolution2D, MaxPooling2D,AveragePooling2D,SeparableConv2D
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist

#Image read, display and maniplulate
from skimage.io import imread, imshow, show
from skimage.transform import resize
from skimage.color import rgb2gray

#For measuring time
from time import process_time

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

# Convert 1-dimensional class arrays to 10-dimensional class matrices
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

def build_model():
    model = Sequential() 

    model.add(Convolution2D(filters = 64,kernel_size =(3,3),input_shape=(28,28,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(filters = 16, kernel_size=(3,3)))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Convolution2D(filters = 16,kernel_size =(3,1)))
    model.add(Activation('relu'))
    model.add(Convolution2D(filters = 16,kernel_size =(1,3)))
    model.add(Activation('relu'))

    model.add(Convolution2D(filters = 16,kernel_size =(3,3)))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(filters = 16,kernel_size =(3,3)))
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2,2)))

    model.add(Convolution2D(filters = 10, kernel_size=(3,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Activation(tf.nn.softmax))

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

    return model


model = build_model()
model.summary()

model.load_weights('MNIST-TSAI-3A.h5')

#Running Inference
img = imread("img_1.jpg")

# resize to 28 x 28
imresize = resize(img,(28,28), mode='constant')

# turn the image from color to gray
im_gray = rgb2gray(imresize)

# the color of the original set are inverted,so we invert it here
#im_gray_invert = 255 - im_gray*255

#treat color under threshold as black
#im_gray_invert[im_gray_invert<=90] = 0

im_final = im_gray.reshape(1,28,28,1)

start = process_time()
pred = model.predict(im_final)
print("Prediction :",pred)
end = process_time()

# choose the digit with greatest possibility as predicted dight
ans = pred[0].tolist().index(max(pred[0].tolist()))
print("The predicted digit is: ",ans)

#Time taken for prediction
print("Time for prediction:", end-start)

print("Running prediction for below image")
imshow(img)
show()

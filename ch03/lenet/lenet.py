from keras import backend as k
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.datasets import mnist
from keras.utils import np_utils
from keras.optimizers import SGD, RMSprop, Adam
import numpy as np
import matplotlib.pyplot as plt

#define the convnet
class LeNet:
	@staticmethod
	def build(input_shape, classes):
		model = Sequential()
		# CONV => ReLu => POOL
		model.add(Conv2D(20, kernel_size=5, padding="same",
		                 input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# CONV => ReLu => POOL
		model.add(Conv2D(50, kernel_size=5, padding="same",
		                 input_shape=input_shape))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# FLATTEN => ReLu layers
		model.add(Flatten())
		model.add(Dense(500))
		model.add(Activation("relu"))

		# a softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))
		return model
		
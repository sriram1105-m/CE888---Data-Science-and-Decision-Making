#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing all the required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random
import os
import cv2
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import PIL
from PIL import Image

import keras


# In[2]:


# Importing the datasets

train_dir = "C:\\Users\\SRIRAM\\Desktop\\Training"
test_dir = "C:\\Users\\SRIRAM\\Desktop\\Test"


# # There are two classes present in the dataset. The exact counts of the data are shown using the command below.

# In[3]:


# Getting the count of training images

cat_counts = {}
for cat in os.listdir(train_dir):
    counts = len(os.listdir(os.path.join(train_dir, cat)))
    cat_counts[cat] =counts
print(cat_counts)


# In[4]:


# Getting the count of test images

cat_counts = {}
for cat in os.listdir(test_dir):
    counts = len(os.listdir(os.path.join(test_dir, cat)))
    cat_counts[cat] =counts
print(cat_counts)


# # Data Pre-processing steps

# In[5]:


# Data Preprocessing
#Image Data Generator
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,   # Stretches and slant the image to a particular angle
                                   rotation_range = 40, # randomly rotates the image so that the model become inavariant to object orientation
                                   zoom_range = 0.2,    #randomly zoom-in or zoom-out the image
                                   brightness_range = [0.8, 1.2], #range changes the brightness of the image
                                   horizontal_flip = True,     #randomly flips the image horizontally
                                   fill_mode = 'nearest',
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2
                                   ) 

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('C:\\Users\\SRIRAM\\Desktop\\Training',
                                                  target_size = (224, 224),
                                                  batch_size = 32,
                                                  class_mode = 'binary', shuffle = True)

test_set = test_datagen.flow_from_directory('C:\\Users\\SRIRAM\\Desktop\\Test',
                                             target_size = (224, 224),
                                             batch_size = 32,
                                             class_mode = 'binary')


# In[6]:


#Finding the class Index
training_set.class_indices


# # Class_weight method is used to balance the data. The dataset is slightly imbalanced. In order to balance the dataset, this method is being adopted.

# In[7]:


# Utilizing Class_weight to balance the training data class labels

from sklearn.utils import class_weight
import numpy as np

class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(training_set.classes), 
                training_set.classes) 

class_weights = dict(enumerate(class_weights))


# In[8]:


# Printing the evaluated class_weights

class_weights


# In[9]:


#Importing required libraries
from keras.layers import Dropout
from keras import models, regularizers, layers, optimizers, losses, metrics

from keras.layers import GlobalAveragePooling2D
from keras.layers import GaussianNoise

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
from keras.utils import np_utils


# # Downloading the VGG19 model for the model

# In[10]:


# Importing the VGG19 required libraries

from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input

vgg_base = VGG19(weights='imagenet',    
                 include_top=False,     
                 input_shape=(224, 224, 3))
print(vgg_base.summary())


# # Fine tuning involves in freezing some layers of the transfer learning method and adding our own set of layers for better results.

# In[11]:


# Fine tuning the model
trainable = False

for layer in vgg_base.layers:
    if layer.name == 'block5_conv2':
        trainable = True
        
    layer.trainable = trainable
    
print(vgg_base.summary())


# In[12]:


# Builing the model

def build_model():
    from tensorflow.keras.optimizers import Adam, RMSprop
           
    model = keras.models.Sequential([vgg_base,
                                     keras.layers.Flatten(),
                                     keras.layers.Dropout(0.50),
                                     keras.layers.Dense(1, activation='sigmoid')])
    
    
    # Compiling the model

    model.compile(optimizer=RMSprop(lr = 1e-4), 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


# In[13]:


model = build_model()
print(model.summary())


# In[14]:


# Calculating the steps per epoch parameter

BATCH_SIZE = 32
train_steps = training_set.n // BATCH_SIZE
test_steps = test_set.n // BATCH_SIZE

train_steps, test_steps


# In[15]:


# Using the Callback

callback = keras.callbacks.EarlyStopping(
    monitor = "accuracy",
    min_delta = 0,
    patience = 3,
    verbose = 1,
    mode = "auto",
    baseline = None,
    restore_best_weights = False
)


# In[16]:


# Model Training

history = model.fit_generator(
    training_set,
    steps_per_epoch=train_steps, class_weight = class_weights,
    epochs=100, callbacks = [callback]
    )


# # Testing the model using a sample test data

# In[19]:


# Testing the model with a sample image 

import numpy as np
from keras.preprocessing import image

test_image = image.load_img('C:\\Users\\SRIRAM\\Desktop\\Test\\No_Fire\\resized_test_nofire_frame27.jpg', target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image / 255.0
result = model.predict(test_image)
training_set.class_indices
if result < 0.5:
    print('Fire')
else:
    print('No Fire')


# # Plotting the model performance 

# In[21]:


plt.plot(model.history.history['accuracy'],c='red')
plt.plot(model.history.history['loss'],c='green')
plt.title('Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train','Validation'],loc='upper left', bbox_to_anchor = (1,1))


# In[22]:


# Finding the metrics of the model

model.metrics_names


# In[23]:


acc = history.history['accuracy']
loss = history.history['loss']


# In[24]:


plt.figure(figsize=(8, 8))

plt.subplot(2, 1, 1)

plt.plot(acc, label='Training Accuracy')

plt.plot(loss, label='Loss')

plt.legend(loc='lower right')

plt.ylabel('Accuracy')

plt.ylim([min(plt.ylim()),1])

plt.title('Training and Loss')


# In[ ]:





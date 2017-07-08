
# coding: utf-8

# In[1]:

import os
import numpy as np
import scipy.misc as msc


# In[2]:

def read_data(folder):
    files = os.listdir(folder)
    return files


# In[3]:

train_folder = '../uiucTrain'
train_names = read_data(train_folder)


# In[4]:

def read_and_center(folder, names):
  X = np.stack([msc.imread(folder + '/' + f).flatten() for f in names], axis=0)
  mean = np.mean(X, axis=0)
  X_center = (X - mean).T
  return X_center, mean


# In[5]:

train_images, _ = read_and_center(train_folder, train_names)
train_images = [(image, 1 if 'Pos' in train_names[index] else -1) for index, image in enumerate(train_images)]


# In[ ]:




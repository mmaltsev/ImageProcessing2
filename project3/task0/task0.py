
# coding: utf-8

# In[9]:

import os
import numpy as np
import scipy.misc as msc


# In[10]:

def read_data(folder):
    files = os.listdir(folder)
    return files


# In[11]:

train_folder = '../uiucTrain'
train_names = read_data(train_folder)


# In[46]:

def read_and_center(folder, names):
    X = np.stack([msc.imread(folder + '/' + f) for f in names], axis=0)
    shape = X.shape[1:]
    X = [image.flatten() for image in X]
    mean = np.mean(X, axis=0)
    X_center = (X - mean)
    return X_center, mean, shape


# In[47]:

train_images, _, shape = read_and_center(train_folder, train_names)
train_images = [(image, 1 if 'Pos' in train_names[index] else -1) for index, image in enumerate(train_images)]


# In[ ]:




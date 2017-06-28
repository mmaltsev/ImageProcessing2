## project 2. computing eigenfaces.
import scipy.misc as msc
from scipy.spatial import distance as dist
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import math as mt

def draw_image(vector, name):
  plt.figure()
  plt.imshow(np.reshape(vector, (19, 19)), 'gray')
  plt.axis('off')
  plt.savefig('img/' + name, bbox_inches='tight')

def distanceComparison(distances, low_dim_distances):
  identical_num = 0
  for i in range(len(distances)):
    if distances[i].index(min(distances[i])) == \
      low_dim_distances[i].index(min(low_dim_distances[i])):
      identical_num += 1
  print 'Number of identical nearest neighbors: ' + str(identical_num) + \
      ' out of ' + str(len(distances))

def distancesPlot(distances, prefix):
  plt.figure()
  for index, distance in enumerate(distances):
    #plt.figure()
    plt.plot(distance)
    plt.savefig('img/' + str(prefix) + 'distances_combined')
    #plt.savefig('img/distance_' + str(index + 1))
    #plt.show()

def eucDistance(test_images, X_train_center):
  distances = []
  for test in test_images:
    arr = []
    for train in X_train_center.T:
      arr.append(dist.euclidean(test, train))
    arr.sort()
    arr = arr[::-1]
    distances.append(arr)
  return distances

def eigenvectorsPlot(eigenvectors, eigenvalues):
  plt.figure()
  for index, vector in enumerate(eigenvectors):
    plt.subplot(mt.ceil(len(eigenvectors)/5.0),5,index + 1)
    current_vector = np.reshape(vector, (19, 19))
    plt.imshow(current_vector, 'gray')
    #plt.title(eigenvalues[index] + 1)
    plt.axis('off')
  plt.savefig('img/vector_combined', bbox_inches='tight')

def covSpectrumPlot(eigenvalues):
  plt.plot(eigenvalues)
  plt.savefig('img/cov_spectrum')
  #plt.show()

def descSort(eigenvectors, eigenvalues, X_train_center):
  idx = np.argsort(eigenvalues)[::-1]
  return eigenvectors[idx], eigenvalues[idx], X_train_center[idx]

def covMatrixComputation(X_train, X_center):
  obs, dim = X_train.shape
  C = np.cov(X_center)
  values, vectors = np.linalg.eigh(C)
  return C, vectors.T, values

def readAndCenter(names):
  X = np.stack([msc.imread('train/' + f).flatten() for f in names], axis=0)
  mean = np.mean(X, axis=0)

  # subtract the mean (along columns)
  # print X.shape
  # print mean.shape
  X_center = (X - mean).T
  # print X_center.shape
  # print X_center.T[0][:10]
  # print mean[:10]
  # print X[:10]
  # draw_image(X_center.T[0], 'centered1')
  # draw_image(X_center.T[1], 'centered2')
  # draw_image(X_center.T[2], 'centered3')
  return X, X_center

def sampling():
  files = os.listdir('train')
  train_names = random.sample(files, int(len(files)*0.9))
  test_names = [name for name in files if name not in train_names]
  return train_names, test_names

def reconstruct_images(data_projection, important_eigenvectors, X_center):
  '''plt.figure()
  for index, vector in enumerate(data_projection[:20]):
    plt.subplot(mt.ceil(len(eigenvectors)/5.0),5,index + 1)
    current_vector = np.reshape(vector, (19, 19))
    plt.imshow(current_vector, 'gray')
    #plt.title(eigenvalues[index] + 1)
    plt.axis('off')
  plt.savefig('img/vector_combined', bbox_inches='tight')'''
  limit = 1
  images = data_projection.T[:limit].T
  reconstructed_images = np.dot(important_eigenvectors.T, images).T

  plt.figure()
  for index, image in enumerate(X_center.T):
    plt.subplot(mt.ceil(len(X_center.T)/5.0),5,index + 1)
    current_vector = np.reshape(image, (19, 19))
    plt.imshow(current_vector, 'gray')
    plt.axis('off')
  plt.savefig('img/restored_images', bbox_inches='tight')

def pca():
  # (1) - download and unzip archive with faces.
  
  # (2) - randomly select 90% for X_train and remaining for X_test.
  train_names, test_names = sampling()
  
  # (3) - read images into a matrix X_train and center the data.
  X_train, X_train_center = readAndCenter(train_names)
  
  # (4) - compute the cov. matrix with its eigenvectors and eigenvalues.
  C, eigenvectors, eigenvalues = \
    covMatrixComputation(X_train, X_train_center)

  # (5) - plot the spectrum of C (sorted eigenvalues).
  eigenvectors, eigenvalues, X_train_center = \
    descSort(eigenvectors, eigenvalues, X_train_center)
  covSpectrumPlot(eigenvalues)
  
  # (6) - determine k most important eigenvalues.
  important_eigenvalues, = \
    np.where((np.cumsum(eigenvalues) / np.sum(eigenvalues)) < 0.9)
  
  # (7) - visualize first k eigenvectors.
  important_eigenvectors = eigenvectors[:len(important_eigenvalues)]
  eigenvectorsPlot(important_eigenvectors, important_eigenvalues)
  
  # (8) - read images into a matrix X_test and center the data.
  X_test, X_test_center = readAndCenter(test_names)
  
  # (9) - sample 10 test images, compute their Euclidean distances to 
  # all training images, sort (in descending order) and plot the distances.
  test_images = random.sample(X_test_center.T, 10)
  distances = eucDistance(test_images, X_train_center)
  distancesPlot(distances, '')
  
  # (10) - project data into the subspace spanned by the k eigenvectors.
  X_center = np.c_[X_train_center, X_test_center]
  data_projection = np.dot(important_eigenvectors, X_center)
  # reconstruct_images(data_projection, important_eigenvectors, X_center.T[:20].T)
  
  # (11) - for 10 test images compute Euc. distances to all the training
  # images in the lower dim. space, sort (in desc. order) and plot them.
  test_images_projection = \
    np.dot(important_eigenvectors, np.asarray(test_images).T)
  X_train_projection = np.dot(important_eigenvectors, X_train_center)
  low_dim_distances = \
    eucDistance(test_images_projection.T, X_train_projection)
  distancesPlot(low_dim_distances, 'low_dim_')
  
  # (12) - for 10 test images compare nearest neighbors in original 
  # and lower dim. spaces.
  distanceComparison(distances, low_dim_distances)

pca()

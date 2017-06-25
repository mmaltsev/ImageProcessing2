##project 2. eigenfaces.
import scipy.misc as msc
import numpy as np
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import Axes3D
import os
im = msc.imread("train/face00001.pgm")
files = os.listdir('train')
trainNames = random.sample(files, int(len(files)*0.9))
X = np.stack([msc.imread('train/' + f).flatten() for f in trainNames[0:10]], axis=0)
meanTrain = np.mean(X, axis=0)

def pca(X, meanTrain):
    obs,dim = X.shape
    Xnorm = (X - meanTrain).T  # subtract the mean (along columns)
    M = np.dot(Xnorm,Xnorm.T)/obs
    values, vectors = np.linalg.eigh(M)
    idx= np.argsort(values)[::-1]

    values= values[ idx]# sorting eigenvalues
    vectors= vectors[:,idx]  # sorting eigenvectors

    #define only largest eigen values
    valuesBig, = np.where((np.cumsum(values) / np.sum(values)) < 0.90)
    vectorsBig = vectors[:,valuesBig]
    #eigen vector visualization
#    msc.imshow( np.reshape( vectors[1,:], (obs,dim)))
    msc.imshow(np.dot(np.diag(values),vectors))
    # projection of the data in the new space
    proj = np.dot(vectors.T, Xnorm)

    proj = np.dot(vectorsBig.T, Xnorm)

    # image reconstruction
    Xr = np.dot(vectorsBig, proj).T + meanTrain
    msc.imshow(Xr)
    #msc.imsave("task1.3/images/pcsIllumination.png", Xr)

    dist = []
    # difference in Frobenius norm
    dist.append(np.linalg.norm(X - Xr, 'euc'))

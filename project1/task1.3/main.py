#
import scipy.misc as msc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_intensity_image(filename):
    f = msc.imread(filename, flatten=True).astype('float')
    return f

def main():
    im = read_intensity_image("task1.3/images/cat.png")
    l = np.log(im + 1)
    h, w = l.shape
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    X = np.column_stack((ys.flatten() * xs.flatten(), ys.flatten(), xs.flatten(), np.ones(h*w)))
    # for i in np.arange(10) *100 :
    #     print (str(l[ X[i,0], X[i,1] ]) + "   " + str(Z[i]))


    Z = l.flatten()
    params = np.linalg.lstsq(X, Z)
    # inv = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    # params = np.dot(inv, Z)
    i = np.dot(X, params[0] )
    Ireshaped = np.reshape(i, l.shape)
    r = np.exp(l - Ireshaped)
    R = ((r - r.min())/(r.max() - r.min()))  * (im.max() - im.min())
    msc.imshow(R)
    msc.imshow(im)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xs = X[:,0],ys = X[:,1], zs = r.flatten())
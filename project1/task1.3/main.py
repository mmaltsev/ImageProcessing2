#
import scipy.misc as msc
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def read_intensity_image(filename):
    f = msc.imread(filename, flatten=True).astype('float')
    return f

def hist_equalization(image):
    width, height = image.shape
    N = float(width * height)
    hist = np.histogram(image, bins=np.arange(0, 257, 1))
    cumul = np.cumsum(hist[0])
    new_image = np.empty((width, height))
    for y in range(0, height):
        for x in range(0, width):
            new_image[x, y] = np.floor(cumul[int(image[x, y])] * 255 / N)
    return new_image
def hist_normalization(image):
    def borders(arr):
        i = 0
        while arr[i] <= 0 and i < len(arr):
            i += 1
        min = i
        i = len(arr) - 1
        while arr[i] <= 0 and i < len(arr):
            i -= 1
        max = i
        return (min, max)

    width, height = image.shape
    hist = np.histogram(image, bins=np.arange(0, 257, 1))
    min, max = borders(hist[0])

    new_image = np.empty((width, height))
    for y in range(0, height):
        for x in range(0, width):
            new_image[x, y] = (image[x, y] - min) / (max - min) * 255
    return new_image

def hist_weibullization(image, k = 2, l = 40):
    from math import pow
    width, height = image.shape
    N = float(width * height)
    hist = np.histogram(image, bins=np.arange(0, 257, 1))
    cumul = np.cumsum(hist[0])
    new_image = np.empty((width, height))
    for y in np.arange(0, height):
        for x in np.arange(0, width):
            hx = cumul[int(image[x, y])] / N
            if hx == 1.0:
                new_image[x, y] = 0
            else:
                new_image[x, y] = np.floor(pow(abs((np.log(1 - hx))), 1.0/k) * l)
    return new_image

def main():
    im = read_intensity_image("task1.3/images/portrait.png")
    l = np.log(im + 1)
    h, w = l.shape
    ####estimate parameters
    xs, ys = np.meshgrid(np.arange(w,step=10), np.arange(h,step=10))
    X = np.column_stack((ys.flatten() * xs.flatten(), ys.flatten(), xs.flatten(), np.ones(xs.flatten().__len__())))
    # for i in np.arange(10) *100 :
    #     print (str(l[ X[i,0], X[i,1] ]) + "   " + str(Z[i]))
    Z =  l[ys.flatten(),xs.flatten()]
    params = np.linalg.lstsq(X, Z)[0]


    # inv = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
    # params = np.dot(inv, Z)
    xs, ys = np.meshgrid(np.arange(w,step=1), np.arange(h,step=1))
    X = np.column_stack((ys.flatten() * xs.flatten(), ys.flatten(), xs.flatten(), np.ones(xs.flatten().__len__())))

    i = np.dot(X, params)
    rlog = l - np.reshape(i, xs.shape)

    r = np.exp(rlog)
    R = ((r - r.min())/(r.max() - r.min()))  * (im.max() - im.min())
    #R = R - np.mean(R) + np.mean(im)
    R = hist_weibullization(R)
    R = hist_normalization(R)
    #R = hist_equalization(R)

    msc.imshow(R)
    msc.imsave("task1.3/images/portraitCompWeq.png", R)
    msc.imshow(np.minimum(R * 4, 255))
    msc.imshow(im)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    from matplotlib import cm
    # cm = plt.get_cma  p("RdYlGn")
    #ax.scatter(X[:,1], X[:,2], Z.flatten(), c=Z.flatten())
    ax.scatter(X[:,1], X[:,2], rlog, c=rlog)

    plt.show()



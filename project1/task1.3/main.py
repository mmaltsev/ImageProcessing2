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

def buildModel(l ,X, nsample = None):
    h, w = l.shape
    if nsample is None:
        nsample = h*w

    ####estimate parameters
    # for pretty 3d pic
    # xs, ys = np.meshgrid(np.arange(w,step=10), np.arange(h,step=10))
#    xs, ys = np.meshgrid(np.arange(w, step=1), np.arange(h, step=1))
#    X = np.column_stack((ys.flatten() * xs.flatten(), ys.flatten(), xs.flatten(), np.ones(xs.flatten().__len__())))
    ind = np.random.choice(np.arange(X.__len__()), nsample)
    # for i in np.arange(10) *100 :
    #     print (str(l[ X[i,0], X[i,1] ]) + "   " + str(Z[i]))
    Z = l[X[ind, 1].astype(int), X[ind, 2].astype(int)]
    params = np.linalg.lstsq(X[ind,:], Z)[0]
    return params
# inv = np.dot(np.linalg.inv(np.dot(X.T, X)), X.T)
# params = np.dot(inv, Z)

def main(photo = "portrait", sSize = None):
    im = read_intensity_image("task1.3/images/"+ photo+ ".png")
    l = np.log(im + 1)
    h, w = l.shape

    xs, ys = np.meshgrid(np.arange(w,step=1), np.arange(h,step=1))
    X = np.column_stack((ys.flatten() * xs.flatten(), ys.flatten(), xs.flatten(), np.ones(xs.flatten().__len__())))

    params = buildModel(l, X,sSize)
    i = np.dot(X, params)
    rlog = l - np.reshape(i, xs.shape)

    r = np.exp(rlog)
    sSize = str(sSize)
    msc.imsave("task1.3/images/" + photo + "_"+ sSize+"Reflectance.png", r)
    R = ((r - r.min())/(r.max() - r.min()))  * (im.max() - im.min())
    msc.imsave("task1.3/images/" + photo +"_"+ sSize+ "ReflectanceMeanCentered.png", R)
    #R = R - np.mean(R) + np.mean(im)
    Rhist = hist_equalization(R)
    msc.imsave("task1.3/images/" + photo + "_"+ sSize+"HistEq.png", Rhist )
    Rw = hist_weibullization(R)
    Rw = hist_normalization(Rw)
    msc.imsave("task1.3/images/" + photo + "_"+ sSize+"CompWeq.png", Rw)
    if False:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        from matplotlib import cm
        # cm = plt.get_cma  p("RdYlGn")
        #ax.scatter(X[:,1], X[:,2], i, c=i)
        ax.scatter(X[:,1], X[:,2], r, c=r)
        plt.show()

if __name__ == '__main__':
    for pic in ['cat','portrait']:
        for s in [250, 1000,10000, None]:
            main(pic, s)

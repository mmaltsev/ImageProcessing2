# coding: utf-8

import numpy as np
import math
import common
import os
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams["savefig.directory"] = os.chdir(os.path.dirname(__file__))


def main():
    filename = "cat.png"
    img = common.read_intensity_image("images/" + filename)
    plot_hist(img, 'Initial histogram')
    plot_cumul(img, 'Initial cumulative histogram')

    equal = hist_equalization(img)
    plot_hist(equal, 'Equalized histogram')
    plot_cumul(equal, 'Equlized cumulative histogram')

    imshow(equal)

    weibull = hist_weibullization(img, 2.0, 40.0)
    plot_hist(weibull, 'Weibull distributed histogram')
    plot_cumul(weibull, 'Weibull distributed cumulative histogram')

    imshow(weibull)

    norm = hist_normalization(weibull)
    plot_hist(norm, 'Normalized weibull distributed histogram')
    plot_cumul(norm, 'Normalized weibull distributed cumulative histogram')

    imshow(weibull)

def hist(image):
    return np.histogram(image, bins=np.arange(0, 257, 1))


def imshow(image):
    plt.imshow(image, cmap='gray')
    plt.show()


def plot_hist(image, title):
    hist = np.histogram(image, bins=np.arange(0, 257, 1))
    x = range(256)
    plt.bar(x, hist[0])
    plt.title(title)
    plt.show()


def plot_cumul(image, title):
    hist = np.histogram(image, bins=np.arange(0, 257, 1))
    x = range(256)
    plt.bar(x, np.cumsum(hist[0]))
    plt.title(title)
    plt.show()


def hist_equalization(image):
    width, height = image.shape
    N = float(width * height)
    hist = np.histogram(image, bins=np.arange(0, 257, 1))
    cumul = np.cumsum(hist[0])
    new_image = np.empty((width, height))
    for y in xrange(0, height):
        for x in xrange(0, width):
            new_image[x, y] = np.floor(cumul[int(image[x, y])] * 255 / N)
    return new_image


def hist_weibullization(image, k, l):
    width, height = image.shape
    N = float(width * height)
    hist = np.histogram(image, bins=np.arange(0, 257, 1))
    cumul = np.cumsum(hist[0])
    new_image = np.empty((width, height))
    for y in xrange(0, height):
        for x in xrange(0, width):
            hx = cumul[int(image[x, y])] / N
            if hx == 1.0:
                new_image[x, y] = 0
            else:
                new_image[x, y] = np.floor(math.pow(abs((math.log(1 - hx))), 1.0/k) * l)
    return new_image


def hist_normalization(image):
    width, height = image.shape
    hist = np.histogram(image, bins=np.arange(0, 257, 1))
    min, max = borders(hist[0])

    new_image = np.empty((width, height))
    for y in xrange(0, height):
        for x in xrange(0, width):
            new_image[x, y] = (image[x, y] - min) / (max - min) * 255
    return new_image


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


if __name__ == "__main__":
    main()


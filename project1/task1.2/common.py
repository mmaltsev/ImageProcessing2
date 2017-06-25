# Viktor Lopatin. Image Processing. Common functions. 2016

import math

import scipy.misc as msc


def read_intensity_image(filename):
    f = msc.imread(filename, flatten=True).astype('float')
    return f


def write_intensity_image(f, filename):
    msc.toimage(f, cmin=0, cmax=255).save(filename)


def read_float(prompt):
    while True:
        try:
            return float(raw_input(prompt))
        except ValueError:
            print "Not a number"


def euclid_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2.0 + (y1 - y2)**2.0)


def cont():
    return raw_input('Continue?(y/n) ') != 'n'


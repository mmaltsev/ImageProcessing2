
# coding: utf-8

# In[98]:

import numpy as np
import math as mt
from PIL import Image as im
from matplotlib import pyplot as plt

def init_a_default(init_lev, L):
    a = [0]
    for i in range(L):
        a.append((i + 1) * init_lev / L)
    a.append(init_lev)
    return a

def init_a_alt(L, h):
    non_zero_indices = [i for i, v in enumerate(h) if v > 0]
    a_min = non_zero_indices[0]
    a_max = non_zero_indices[-1]
    a = [a_min]
    for i in range(L):
        a.append((i + 1) * (a_max - a_min) / L + a_min)
    a.append(a_max)
    return a


def init_a(init_lev, L, h, schema = 'default'):
    if schema == 'default':
        return init_a_default(init_lev, L)
    else:
        return init_a_alt(L, h)

def main(img_path, schema = 'default'):
    raw_image = im.open(img_path)
    image = np.array(raw_image)
    width = image[0][:].size
    height = image[:][0].size

    init_lev = 256
    h = [0] * init_lev
    for x in image:
        for pixel_intensity in x:
            h[pixel_intensity] += 1

    p = [0] * init_lev
    for i, val in enumerate(h):
        p[i] = float(val) / (width * height)


    L = 8
    a = init_a(init_lev, L, h, schema)

    b = []
    for i in range(L + 1):
        b.append(i * init_lev / L + init_lev / (2 * L))


    def E(a, b):
        sum = 0
        for nu in range(1, L + 1):
            for x in range(a[nu], a[nu + 1]):
                sum += (x - b[nu]) ** 2 * p[x]
        return sum


    T = 10
    a_new = a[:]
    b_new = b[:]
    prev_err = E(a_new, b_new)
    a_prev = a[:]
    b_prev = b[:]
    for t in range(T):   
        for nu in range(1, len(a_new) - 1):
            a_new[nu] = int(float((b_new[nu] + b_new[nu - 1])) / 2)

        for nu in range(len(b_new)):
            numerator = 0
            delimeter = 0
            for x in range(a_new[nu], a_new[nu + 1]):
                numerator += x * p[x]
                delimeter += p[x]

            if delimeter == 0:
                b_new[nu] = 0
            else:
                b_new[nu] = numerator / delimeter

        err = E(a_new, b_new)
        if prev_err - err < 0.01:
            a_new = a_prev[:]
            b_new = b_prev[:]
            break
        else:
            prev_err = err
            a_prev = a_new[:]
            b_prev = b_new[:]

    a_plot = [a_new[0]]
    for i in range(1, len(a_new) - 1):
        a_plot += [a_new[i]] * 2 
    a_plot.append(a_new[-1])

    b_plot = []
    for el in b_new:
        b_plot += [el] * 2

    a_plot[0] = 0
    a_plot[-1] = init_lev
    plt.plot(a_plot, b_plot)
    plt.show()


main('./images/bauckhage-gamma-1.png')
main('./images/bauckhage-gamma-1.png', 'alt')
main('./images/bauckhage-gamma-2.png')
main('./images/bauckhage-gamma-2.png', 'alt')
main('./images/bauckhage.jpg')
main('./images/bauckhage.jpg', 'alt')

import os,sys
import numpy as np
import cv2
import argparse
import scipy.misc
import math
import time
import pyflow
import random
import pandas as pd
from PIL import Image
from multiprocessing import Pool
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import ticker, cm


def cal_mass_center(image):
    # h, w = image.shape()
    center = ndimage.measurements.center_of_mass(image)
    return center


def get_center_otsu(image):
    # flow_mag_f[flow_mag_f < threshold] = 0
    image_h, image_w = list(image.shape)[0:2]
    blur = cv2.GaussianBlur(image,(5,5),0)
    _,flow_mag_f = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # _,flow_mag_f = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if np.max(flow_mag_f) == 0:
        x, y = int(image_w/2), int(image_h/2)
        print('{} not have silence motion'.format('otsu image'))
    else:
        center = cal_mass_center(flow_mag_f)
        y, x = center
        y, x = int(y), int(x)
    print('center is {}'.format((x, y)))
    return flow_mag_f, x, y



if __name__ == "__main__":

    flow_parent_path = '/home/katou2/github-home/UCF-101-optflow-1f/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01/'
    # flow_parent_path = '/home/katou2/github-home/UCF-101-1f/FloorGymnastics/v_FloorGymnastics_g01_c03/'
    flow_id = 'frame000033.jpg'
    flow_path = os.path.join(flow_parent_path, flow_id)
    img = cv2.imread(flow_path)
    flow_mag_f = img[:,:,2]
    # import pdb; pdb.set_trace()
    threshold=10
    flow_mag_f[flow_mag_f<threshold]=0
    # import pdb; pdb.set_trace()
    flow_mag_f0, x, y = get_center_otsu(flow_mag_f)
    # image1_bgr = image1_bgr.astype(np.uint8)
    # cv2.circle(image1_bgr, (x, y), 10, (0, 0, 255), -1) #red
    # cv2.circle(image1_bgr, (h1, w1), 10, (0, 255, 0), -1)
    # cv2.imshow('image1', image1_bgr)
    # cv2.imshow('image2', image2_bgr.astype(np.uint8))
    # cv2.imshow('error', error.astype(np.uint8))
    img_parent_path = flow_parent_path.replace('UCF-101-optflow-1f', 'UCF-101-1f')
    img_id = 'image_00' + flow_id[8:11] + '.jpg'
    img_path = os.path.join(img_parent_path, img_id)
    image1_bgr = cv2.imread(img_path)
    cv2.rectangle(image1_bgr, (x-56, y-56), (x+56,y+56), (0,0,255), 1)
    cv2.circle(image1_bgr, (x, y), 10, (255, 0, 0), -1) 
    cv2.imshow('image1', image1_bgr)
    flow_mag_f = flow_mag_f.astype(np.uint8)
    # cv2.circle(flow_mag_f, (x, y), 5, (255, 0, 0), -1) #red
    cv2.imshow('flow_mag_original', flow_mag_f)
    # cv2.imshow('flow_mag_morpho', flow_morpho)
    
    # cv2.imshow('flow_mag_cal', flow_mag_cal.astype(np.uint8))
    
    
    cv2.imshow('flow_mag_f0', flow_mag_f0)
    # cv2.imshow('flow_mag_f1', flow_mag_f1*255)
    # fig, ax = plt.subplots()
    # cs = ax.contourf(flow_mag_f, locator=ticker.LogLocator(), cmap=cm.PuBu_r)
    # cbar = fig.colorbar(cs)
    imgplot = plt.imshow(flow_mag_f, cmap=cm.Blues, aspect='equal')
    plt.xticks([])
    plt.yticks([])
    plt.plot(227,107,'ro')
    # fig, ax = plt.subplots()
    # ax.set_xticks([])
    # ax.set_yticks([])
    plt.colorbar(shrink=.80, ticks=[])
    plt.savefig('plot.png', dpi=300)

    # Alternatively, you can manually set the levels
    # and the norm:
    # lev_exp = np.arange(np.floor(np.log10(z.min())-1),
    #                    np.ceil(np.log10(z.max())+1))
    # levs = np.power(10, lev_exp)
    # cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())

    
    plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
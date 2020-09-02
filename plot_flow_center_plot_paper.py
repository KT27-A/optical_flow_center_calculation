import os,sys
import numpy as np
import cv2
import argparse
import scipy.misc
import math
import time
import random
import pandas as pd
from PIL import Image
from multiprocessing import Pool
from scipy import ndimage
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
from skimage import filters

def cal_mass_center(image):
    # h, w = image.shape()
    center = ndimage.measurements.center_of_mass(image)
    return center


def get_center_otsu(image):
    # flow_mag_f[flow_mag_f < threshold] = 0
    image_h, image_w = list(image.shape)[0:2]

    flow_mag = image
    if flow_mag.max() < 15:
        y, x = int(image_h/2), int(image_w/2)
        print('no obvious motion 1')
    else:
        threshold = filters.threshold_otsu(flow_mag)
        if threshold <= 15:
            threshold = 15
        # import pdb; pdb.set_trace()
        label = ndimage.label(flow_mag)
        # y, x = ndimage.measurements.center_of_mass(flow_mag, labels=label[0], index=threshold)
        y, x = ndimage.measurements.center_of_mass(flow_mag)
        if math.isnan(y) or math.isnan(x):
            print('no obvious motion 2')
            y, x = int(image_h/2), int(image_w/2)
        else:
            y, x = int(y), int(x)    
    print('center is {}'.format((x, y)))
    return flow_mag, x, y


def plot_series(flow_parent_path, flow_id):
    pass

if __name__ == "__main__":

    flow_parent_path = '/home/katou2/github_home/0_tool/1optical-flow/vector/'
    # flow_parent_path = '/home/katou2/github-home/UCF-101-1f/FloorGymnastics/v_FloorGymnastics_g01_c03/'
    flow_id = '0.png'
    flow_path = os.path.join(flow_parent_path, flow_id)
    img = cv2.imread(flow_path)
    flow_mag_f = img[:,:,2]
    flow_mag_f_threshold = flow_mag_f.copy()
    average = np.average(flow_mag_f)
    otsu = filters.threshold_otsu(flow_mag_f)
    print(average, otsu, otsu*2)
    threshold = 10
    flow_mag_f[flow_mag_f<threshold]=0
    _, x, y = get_center_otsu(flow_mag_f)
    # flow_mag_f_threshold[flow_mag_f_threshold<threshold]=0
    # _, x_1, y_1 = get_center_otsu(flow_mag_f_threshold)
    # if abs(x_1-x) + abs(y_1-y) < 10:
    #     x = x_1
    #     y = y_1
    #     flow_mag_f = flow_mag_f_threshold
         
    # image1_bgr = image1_bgr.astype(np.uint8)
    # cv2.circle(image1_bgr, (x, y), 10, (0, 0, 255), -1) #red
    # cv2.circle(image1_bgr, (h1, w1), 10, (0, 255, 0), -1)
    # cv2.imshow('image1', image1_bgr)
    # cv2.imshow('image2', image2_bgr.astype(np.uint8))
    # cv2.imshow('error', error.astype(np.uint8))
    img_parent_path = flow_parent_path.replace('UCF-101-optflow-1f', 'UCF-101-1f')
    img_id = 'image_000' + str(int(flow_id[9:11])+1) + '.jpg'
    img_path = os.path.join(img_parent_path, img_id)
    image1_bgr = cv2.imread(img_path)
    img_id_2 = 'image_000' + str(int(flow_id[9:11])+2) + '.jpg'

    img_path_2 = os.path.join(img_parent_path, img_id_2)
    image2_bgr = cv2.imread(img_path_2)
    
    image1_bgr_clone = np.copy(image1_bgr)
    cv2.rectangle(image1_bgr_clone, (x-56, y-56), (x+56,y+56), (0,0,255), 2)
    # cv2.circle(image1_bgr_clone, (x, y), 7, (0, 0, 255), -1) 
    # cv2.imshow('image1', image1_bgr)
    # cv2.imshow('image2', image2_bgr)
    flow_mag_f = flow_mag_f.astype(np.uint8)
    # cv2.circle(flow_mag_f, (x, y), 5, (255, 0, 0), -1) #red
    # cv2.imshow('flow_mag_original', flow_mag_f)
    # cv2.imshow('flow_mag_f0', flow_mag_f0)
    image1_gray = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2_bgr, cv2.COLOR_BGR2GRAY)
    image_error = (image1_gray-image2_gray).astype(np.int8)
    # cv2.imshow('difference', image_error)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    plt.figure('Series')
    plt.margins(tight=True)
    plt.subplots_adjust(wspace=0.05, hspace=0.0001)

    plt.subplot(361)
    plt.imshow(image1_bgr[...,::-1])
    plt.xticks([])
    plt.yticks([])

    plt.subplot(362)
    plt.imshow(image2_bgr[...,::-1])
    plt.xticks([])
    plt.yticks([])

    plt.subplot(363)
    plt.imshow(image_error, cmap='gray', vmin=0, vmax=255)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(364)
    plt.imshow(flow_mag_f, cmap=cm.Blues, aspect='equal')
    plt.xticks([])
    plt.yticks([])
    # plt.colorbar(shrink=.80, ticks=[])
    # plt.savefig('plot.png', dpi=300)

    plt.subplot(365)
    plt.imshow(flow_mag_f, cmap=cm.Blues, aspect='equal')
    plt.xticks([])
    plt.yticks([])
    plt.scatter(x,y,s=3, color='red',marker='o')

    # plt.colorbar(shrink=.80, ticks=[])
    # plt.savefig('plot.png', dpi=300)

    plt.subplot(366)
    plt.imshow(image1_bgr_clone[...,::-1])
    plt.xticks([])
    plt.yticks([])
    plt.scatter(x,y,s=3, color='red',marker='o')





    plt.savefig('plot_series_44.png', dpi=300, bbox_inches = 'tight', pad_inches=0.01)

    

    # Alternatively, you can manually set the levels
    # and the norm:
    # lev_exp = np.arange(np.floor(np.log10(z.min())-1),
    #                    np.ceil(np.log10(z.max())+1))
    # levs = np.power(10, lev_exp)
    # cs = ax.contourf(X, Y, z, levs, norm=colors.LogNorm())

    
    plt.show()
    
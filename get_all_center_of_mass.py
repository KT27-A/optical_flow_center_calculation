import os,sys
import numpy as np
import cv2
import argparse
import skvideo.io
import scipy.misc
import math
import pandas as pd
import random
import pyflow
import time
import json
from PIL import Image
from multiprocessing import Pool
from scipy import ndimage
from skimage import filters

class center_mass(object):
    
    def __init__(self, parent):
        self.parent = parent

    def __getitem__(self, file_name):
        file_path = os.path.join(self.parent, file_name)
        # return cal_center_flow(file_path)
        return cal_center_flow_no_otsu(file_path)


# def get_center_otsu(image):
    
#     image_h, image_w = list(image.shape)[0:2]
#     blur = cv2.GaussianBlur(image,(5,5),0)
#     _,flow_mag_f = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     if np.max(flow_mag_f) == 0:
#         x, y = int(image_w/2), int(image_h/2)
#         # print('{} does not have silence motion'.format('otsu image'))
#     else:
#         center = ndimage.measurements.center_of_mass(flow_mag_f)
#         y, x = center
#         y, x = int(y), int(x)
#     return x, y

# def cal_center_flow(file_path):
#     flow_mag = cv2.imread(file_path)[:,:,2]
#     threshold=10
#     flow_mag[flow_mag<threshold]=0
#     x, y = get_center_otsu(flow_mag)
#     center = (x, y)
#     return center


def cal_center_flow_no_otsu(file_path):
    flow_mag = cv2.imread(file_path)[:,:,2]
    h, w = flow_mag.shape
    if (w <= h and w == 256) or (h <= w and h == 256):
        ow = w
        oh = h
    if w < h:
        ow = 256
        oh = int(256 * h / w)
    else:
        oh = 256
        ow = int(256 * w / h)
    flow_mag = cv2.resize(flow_mag, (oh, ow))
    image_h, image_w = list(flow_mag.shape)[0:2]
    if flow_mag.max() < 15:
        y, x = int(image_h/2), int(image_w/2)
        print('no obvious motion 1')
    else:
        threshold = filters.threshold_otsu(flow_mag)
        if threshold <= 15:
            threshold = 15
        y, x = ndimage.measurements.center_of_mass(flow_mag, labels=flow_mag, index=threshold)
        if math.isnan(y) or math.isnan(x):
            print('no obvious motion 2')
            y, x = int(image_h/2), int(image_w/2)
        else:
            y, x = int(y), int(x)
    print('center is {}'.format((x, y)))

    return x, y



def main():
    start_time = time.time()
    file_path = 'train_all_flow.list'
    landmarks_frames = pd.read_csv(file_path, header=None)
    file_len = len(landmarks_frames)
    dict_center_mass = {}
    for i in range(file_len):
        line = landmarks_frames.iloc[i, 0]
        line = line.strip('\n').split()
        dir_name = line[0]
        print(dir_name)
        for parent, dirnames, imagenames in os.walk(dir_name):
            imagenames = sorted(imagenames)
            center = center_mass(parent)
            class_name = parent.split('/')[3]
            dict_center_mass[class_name] = [center[i] for i in imagenames]
    print(dict_center_mass)
    json_file = json.dumps(dict_center_mass, sort_keys=False, separators=(',', ': '))
    
    with open('ucf_center_mass_1f_otsu_filter_15_256.json', 'w') as f:
        f.write(json_file)
    print('json file saved successful')    
    end_time = time.time()
    print('running time is {} mins'.format((end_time-start_time)/60))


if __name__ == "__main__":
    main()
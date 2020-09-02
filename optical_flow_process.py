# Author: Deepak Pathak (c) 2016

import os
import cv2
import time
import pyflow
import argparse
import numpy as np
import numbers
from multiprocessing import Pool
from PIL import Image



def get_opfl_frame(image_1, image_2, save_path, frame_num):
    im1 = np.array(Image.open(image_1)).astype(float) / 255
    im2 = np.array(Image.open(image_2)).astype(float) / 255
    alpha = 0.012
    ratio = 0.75
    minWidth = 20
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))
    
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    save_path = save_path + '/' + str(frame_num) + '.png'
    cv2.imwrite(save_path, rgb)


def ToImg(raw_flow,bound):
    '''
    this function scale the input pixels to 0-255 with bi-bound

    :param raw_flow: input raw pixel value (not in 0-255)
    :param bound: upper and lower bound (-bound, bound)
    :return: pixel value scale from 0 to 255
    '''
    if type(bound) == tuple:
        import pdb; pdb.set_trace()
        flow=raw_flow
        flow[flow>bound[1]]=bound[1]
        flow[flow<bound[0]]=bound[0]
        flow -= bound[0]
        flow *= (255/float(bound[1]-bound[0]))
        import pdb; pdb.set_trace()
    else:
        flow=raw_flow
        flow[flow>bound]=bound
        flow[flow<-bound]=-bound
        flow-=-bound
        flow*=(255/float(2*bound))
    return flow.astype(np.uint8)



def cal_optical_cv2(augs):
    dir_name, bound = augs
    dir_name = dir_name.split(' ')[0]
    parent, dirnames, imagenames = next(os.walk(dir_name))
    imagenames = sorted(imagenames)
    for i in range(len(imagenames)-1): # there is no flow for the last image
        s = time.time()
        image_path = os.path.join(parent, imagenames[i])
        frame_0 = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        image_path = os.path.join(parent, imagenames[i+1])
        frame_1 = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        dtvl1=cv2.createOptFlow_DualTVL1()
        flowDTVL1=dtvl1.calc(frame_0,frame_1,None)
        e = time.time()
        print(e-s)
        u = flowDTVL1[:,:,0]
        v = flowDTVL1[:,:,1]
        mag = np.sqrt(u**2+v**2)
        u, v = [ToImg(i,bound) for i in [u, v]]
        mag = ToImg(mag, (0, bound*1.414))
        flow = np.stack((u, v, mag), axis=2)
        # flow = v
        save_path = dir_name.replace('UCF-101', 'UCF-101-optical-flow')
        save_name = os.path.join(save_path, '{:04d}.jpg'.format(i))
        print(save_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        else:
            cv2.imwrite(save_name, flow)


def cal_optflow_cpp(dirpath):
    exe_path = '/home/katou2/github-home/gpu_flow/build/compute_flow'
    cmd = exe_path + ' --start_video={} --gpuID={} --type={} --skip={} --vid_path={} --out_path={}'.format(
        1, 0, 1, 1, dirpath, dirpath.replace('hmdb-51-video', 'hmdb-51-optflow-1f')
    )
    os.system(cmd)


def optical_flow_process(frame_path, save_path):
    folder = os.path.exists(save_path)
    if not folder:
        os.makedirs(save_path)
    for parent, dirnames, filenames in os.walk(frame_path):
        filenames = sorted(filenames)
        for frame_num in range(0, len(filenames)-1):
            image_1 = str(frame_path) + '/' + str(filenames[frame_num])
            image_2 = str(frame_path) + '/' + str(filenames[frame_num+1])
            get_opfl_frame(image_1, image_2, save_path, frame_num)


if __name__ == "__main__":
    # mode = 'test'
    # bound = 20 # for gap = 4
    # train_path = 'train_all_image.list'
    # lines = open(train_path, 'r')
    # lines = list(lines)
    # pool = Pool(processes=10)
    # dir_name = '../UCF-101-1/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 0'
    # augs = (dir_name, bound)
    # if mode == 'run':
    #     # pool.map(cal_optical_cv2, list(zip(lines, [bound]*len(lines))))
    #     pool.map(cal_optflow_cpp, lines)
    # else:
    #     # [cal_optical_cv2(augs) for augs in list(zip(lines, [bound]*len(lines)))]
    #     start_time = time.time()
    #     cal_optical_cv2(augs)
    #     end_time = time.time()
    #     print('cal time is {} min'.format((end_time-start_time)/60))

    #---------------------------optflow-cpp-----------------------------------------
    mode = 'run'
    exist_list = ['']
    vid_path = '../hmdb-51-video'
    dir_paths = []
    for parent, dirnames, filenames in os.walk(vid_path):
        
        dirnames = sorted(dirnames)
        # dir_paths = [os.path.join(parent, i) for i in dirnames]
        for i in dirnames:
            if i not in exist_list:
                dir_paths.append(os.path.join(parent, i))
        break
    # pool = Pool(processes=1)
    if mode == 'run':
        for i in dir_paths:
            cal_optflow_cpp(i)
            
        # pool.map(cal_optflow_cpp, dir_paths[0:8])
    else:
        dirpath = '../UCF-101-video/ApplyEyeMakeup'
        start_time = time.time()
        cal_optflow_cpp(dirpath)
        end_time = time.time()
        print('cal time is {} min'.format((end_time-start_time)/60))
    
    print('The end time is {}'.format(time.asctime(time.localtime(time.time()))))

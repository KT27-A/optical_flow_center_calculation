import json
import cv2
import numpy as np


if __name__ == "__main__":
    with open('center_mass_1f.json') as handle:
        dict_center = json.loads(handle.read()) 
    img_path = '/home/katou2/github-home/UCF-101-1f/FloorGymnastics/v_FloorGymnastics_g01_c03/image_00003.jpg'
    img = cv2.imread(img_path)
    x, y = dict_center[img_path.split('/')[6]][2]
    print((x, y))
    cv2.circle(img, (x, y), 10, (0, 0, 255), -1) #red
    # cv2.circle(image1_bgr, (h1, w1), 10, (0, 255, 0), -1)
    cv2.imshow('image1', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
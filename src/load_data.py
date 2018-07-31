import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

N = 662
labels = np.zeros(N)
labels_dir = '../data/ChinaSet_AllFiles/ClinicalReadings/'
labels_files = [f for f in os.listdir(labels_dir)]
for f in labels_files:
    flag = 0
    with open(labels_dir+'/'+f) as f_txt:
        content = f_txt.readlines()
        content_lines = [x.strip() for x in content]
        for c in content_lines:
            if c.find('normal') != -1:
                labels[int(f[7:11])-1] = 0
                flag = 1
        if flag == 0:
            labels[int(f[7:11])-1] = 1


imgs_dir = '../data/ChinaSet_AllFiles/CXR_png/'
imgs_files = [f for f in os.listdir(imgs_dir)]
for f in imgs_files:
    flag = 0
    img = cv2.imread(imgs_dir+'/'+f)
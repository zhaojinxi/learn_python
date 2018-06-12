import tensorflow
import numpy 
import pandas
import skimage
import sklearn
import pyspark
import os
import sys
import cv2
import random

data_dir='/media/zhao/新加卷/data2'
all_file=os.listdir(data_dir)
pick_one_file=random.sample(all_file,1)[0]
one_file=os.path.join(data_dir,pick_one_file)
one_all_rad=os.listdir(one_file)
pick_one_rad=random.sample(one_all_rad,1)[0]
one_rad=os.path.join(one_file,pick_one_rad)
tensorflow.keras.applications.
def encode(x):
    h1=tensorflow.nn.conv2d(x,[3,3,1,16],[1,2,2,1],'SAME')

    return y

def decode():
    pass
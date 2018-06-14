import tensorflow
import numpy 
import pandas
import skimage.io
import sklearn
import pyspark
import os
import sys
import cv2
import random

data_dir='/media/zhao/新加卷/data2'

# def encode(x):
#     with tensorflow.variable_scope('encode'):
#         w1=tensorflow.get_variable('w1', [3,3,3,8], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
#         b1=tensorflow.get_variable('b1', 8, initializer=tensorflow.constant_initializer(0))
#         z1=tensorflow.nn.conv2d(x,w1,[1,2,2,1],'SAME')+b1
#         h1=tensorflow.nn.leaky_relu(z1)    

#         w2=tensorflow.get_variable('w2', [3,3,8,16], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
#         b2=tensorflow.get_variable('b2', 16, initializer=tensorflow.constant_initializer(0))
#         z2=tensorflow.nn.conv2d(h1,w2,[1,2,2,1],'SAME')+b2
#         h2=tensorflow.nn.leaky_relu(z2)

#         w3=tensorflow.get_variable('w3', [3,3,16,32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
#         b3=tensorflow.get_variable('b3', 32, initializer=tensorflow.constant_initializer(0))
#         z3=tensorflow.nn.conv2d(h2,w3,[1,2,2,1],'SAME')+b3
#         h3=tensorflow.nn.leaky_relu(z3)

#         w4=tensorflow.get_variable('w4', [3,3,32,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
#         b4=tensorflow.get_variable('b4', 64, initializer=tensorflow.constant_initializer(0))
#         z4=tensorflow.nn.conv2d(h3,w4,[1,2,2,1],'SAME')+b4

#     return z4

def encode(x):
    with tensorflow.variable_scope('encode'):
        h1=tensorflow.layers.conv2d(x,8,[3,3],[2,2],'same',activation=tensorflow.nn.relu,kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.1),name='c1')

        h2=tensorflow.layers.conv2d(h1,16,[3,3],[2,2],'same',activation=tensorflow.nn.relu,kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.1),name='c2')

        h3=tensorflow.layers.conv2d(h2,32,[3,3],[2,2],'same',activation=tensorflow.nn.relu,kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.1),name='c3')

        h4=tensorflow.layers.conv2d(h3,64,[3,3],[2,2],'same',activation=tensorflow.nn.relu,kernel_initializer=tensorflow.truncated_normal_initializer(stddev=0.1),name='c4')  
    
    return h4

def decode(x):
    with tensorflow.variable_scope('decode'):
        w1=tensorflow.get_variable('w1', [3,3,32,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b1=tensorflow.get_variable('b1', 32, initializer=tensorflow.constant_initializer(0))
        z1=tensorflow.nn.conv2d_transpose(x,w1,[61,63,63,32],[1,1,1,1],'SAME')+b1
        h1=tensorflow.nn.leaky_relu(z1)

        w2=tensorflow.get_variable('w2', [3,3,16,32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b2=tensorflow.get_variable('b2', 16, initializer=tensorflow.constant_initializer(0))
        z2=tensorflow.nn.conv2d_transpose(h1,w2,[61,125,125,16],[1,1,1,1],'SAME')+b2
        h2=tensorflow.nn.leaky_relu(z2)

        w3=tensorflow.get_variable('w3', [3,3,8,16], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b3=tensorflow.get_variable('b3', 8, initializer=tensorflow.constant_initializer(0))
        z3=tensorflow.nn.conv2d_transpose(h2,w3,[61,250,250,8],[1,1,1,1],'SAME')+b3
        h3=tensorflow.nn.leaky_relu(z3)

        w4=tensorflow.get_variable('w4', [3,3,3,8], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b4=tensorflow.get_variable('b4', 3, initializer=tensorflow.constant_initializer(0))
        z4=tensorflow.nn.conv2d_transpose(h3,w4,[61,501,501,3],[1,1,1,1],'SAME')+b4

    return z4

input_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,3])
input_encode=tensorflow.placeholder(tensorflow.float32,[None,32,32,64])

# encode_image=encode(input_image)
decode_image=decode(encode(input_image))

Session=tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())

for k in range(100):
    all_file=os.listdir(data_dir)
    pick_one_file=random.sample(all_file,1)[0]
    one_file=os.path.join(data_dir,pick_one_file)
    one_all_rad=os.listdir(one_file)
    pick_one_rad=random.sample(one_all_rad,1)[0]
    one_rad=os.path.join(one_file,pick_one_rad)
    all_image_dir=[os.path.join(one_rad,x) for x in os.listdir(one_rad)]
    all_image_dir.sort()
    all_image=[cv2.imread(x) for x in all_image_dir]
    all_image=numpy.array(all_image)

    Session.run(decode_image,feed_dict={input_image:all_image})


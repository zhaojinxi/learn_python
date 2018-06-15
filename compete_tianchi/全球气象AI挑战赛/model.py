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
import time

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

# def decode(x):
#     with tensorflow.variable_scope('decode'):
#         w1=tensorflow.get_variable('w1', [3,3,32,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
#         b1=tensorflow.get_variable('b1', 32, initializer=tensorflow.constant_initializer(0))
#         z1=tensorflow.nn.conv2d_transpose(x,w1,[61,63,63,32],[1,1,1,1],'SAME')+b1
#         h1=tensorflow.nn.leaky_relu(z1)

#         w2=tensorflow.get_variable('w2', [3,3,16,32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
#         b2=tensorflow.get_variable('b2', 16, initializer=tensorflow.constant_initializer(0))
#         z2=tensorflow.nn.conv2d_transpose(h1,w2,[61,125,125,16],[1,1,1,1],'SAME')+b2
#         h2=tensorflow.nn.leaky_relu(z2)

#         w3=tensorflow.get_variable('w3', [3,3,8,16], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
#         b3=tensorflow.get_variable('b3', 8, initializer=tensorflow.constant_initializer(0))
#         z3=tensorflow.nn.conv2d_transpose(h2,w3,[61,250,250,8],[1,1,1,1],'SAME')+b3
#         h3=tensorflow.nn.leaky_relu(z3)

#         w4=tensorflow.get_variable('w4', [3,3,3,8], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
#         b4=tensorflow.get_variable('b4', 3, initializer=tensorflow.constant_initializer(0))
#         z4=tensorflow.nn.conv2d_transpose(h3,w4,[61,501,501,3],[1,1,1,1],'SAME')+b4

#     return z4

def test(x):
    with tensorflow.variable_scope('encode'):
        encode_w1=tensorflow.get_variable('w1', [3,3,3,8], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b1=tensorflow.get_variable('b1', 8, initializer=tensorflow.constant_initializer(0))
        encode_z1=tensorflow.nn.conv2d(x,encode_w1,[1,2,2,1],'SAME')+encode_b1
        encode_h1=tensorflow.nn.leaky_relu(encode_z1)    

        encode_w2=tensorflow.get_variable('w2', [3,3,8,16], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b2=tensorflow.get_variable('b2', 16, initializer=tensorflow.constant_initializer(0))
        encode_z2=tensorflow.nn.conv2d(encode_h1,encode_w2,[1,2,2,1],'SAME')+encode_b2
        encode_h2=tensorflow.nn.leaky_relu(encode_z2)

        encode_w3=tensorflow.get_variable('w3', [3,3,16,32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b3=tensorflow.get_variable('b3', 32, initializer=tensorflow.constant_initializer(0))
        encode_z3=tensorflow.nn.conv2d(encode_h2,encode_w3,[1,2,2,1],'SAME')+encode_b3
        encode_h3=tensorflow.nn.leaky_relu(encode_z3)

        encode_w4=tensorflow.get_variable('w4', [3,3,32,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        encode_b4=tensorflow.get_variable('b4', 64, initializer=tensorflow.constant_initializer(0))
        encode_z4=tensorflow.nn.conv2d(encode_h3,encode_w4,[1,2,2,1],'SAME')+encode_b4

    with tensorflow.variable_scope('decode'):
        decode_w1=tensorflow.get_variable('w1', [3,3,32,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b1=tensorflow.get_variable('b1', 32, initializer=tensorflow.constant_initializer(0))
        decode_z1=tensorflow.nn.conv2d_transpose(encode_z4,decode_w1,tensorflow.shape(encode_z3),[1,2,2,1],'SAME')+decode_b1
        decode_h1=tensorflow.nn.leaky_relu(decode_z1)

        decode_w2=tensorflow.get_variable('w2', [3,3,16,32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b2=tensorflow.get_variable('b2', 16, initializer=tensorflow.constant_initializer(0))
        decode_z2=tensorflow.nn.conv2d_transpose(decode_h1,decode_w2,tensorflow.shape(encode_z2),[1,2,2,1],'SAME')+decode_b2
        decode_h2=tensorflow.nn.leaky_relu(decode_z2)

        decode_w3=tensorflow.get_variable('w3', [3,3,8,16], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b3=tensorflow.get_variable('b3', 8, initializer=tensorflow.constant_initializer(0))
        decode_z3=tensorflow.nn.conv2d_transpose(decode_h2,decode_w3,tensorflow.shape(encode_z1),[1,2,2,1],'SAME')+decode_b3
        decode_h3=tensorflow.nn.leaky_relu(decode_z3)

        decode_w4=tensorflow.get_variable('w4', [3,3,3,8], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        decode_b4=tensorflow.get_variable('b4', 3, initializer=tensorflow.constant_initializer(0))
        decode_z4=tensorflow.nn.conv2d_transpose(decode_h3,decode_w4,tensorflow.shape(x),[1,2,2,1],'SAME')+decode_b4

    return decode_z4

[x.shape for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES)]

input_image=tensorflow.placeholder(tensorflow.float32,[None,501,501,3])
input_encode=tensorflow.placeholder(tensorflow.float32,[None,32,32,64])

# encode_image=encode(input_image)
# decode_image=decode(encode(input_image))
decode_image=test(input_image)
loss=tensorflow.losses.absolute_difference(input_image,decode_image)+tensorflow.losses.mean_squared_error(input_image,decode_image)

AdamOptimizer=tensorflow.train.AdamOptimizer(0.0001).minimize(loss)

Session=tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())

for k in range(10001):
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
    all_image=(all_image-128)/128

    Session.run(AdamOptimizer,feed_dict={input_image:all_image})

    if k%100==0:
        for image in numpy.array([cv2.imread(x) for x in all_image_dir]):
            cv2.imshow('true image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.1)
        cv2.destroyAllWindows()

        for image in Session.run(decode_image,feed_dict={input_image:all_image}):
            cv2.imshow('decode image', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.1)
        cv2.destroyAllWindows()

    print(k)
import tensorflow
import numpy 
import os
import cv2
import random
import time

# data_dir='E:/SRAD2018/train'
data_dir='/media/zhao/新加卷/SRAD2018/train'
# data_dir='/home/jxzhao/tianchi/SRAD2018/train'
log_dir='log/'
model_dir='model/'

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

Session=tensorflow.Session()

Saver =tensorflow.train.import_meta_graph('model/-2000.meta')
Saver.restore(Session, 'model/-2000')
input_image = tensorflow.get_default_graph().get_tensor_by_name("input_image:0")
input_code = tensorflow.get_default_graph().get_tensor_by_name("input_code:0")
is_train = tensorflow.get_default_graph().get_tensor_by_name("is_train:0")
encode_image=tensorflow.get_default_graph().get_tensor_by_name("encode/encode_image:0")
decode_image=tensorflow.get_default_graph().get_tensor_by_name("decode/decode_image:0")

result_en=Session.run(encode_image, feed_dict={input_image:all_image[:,:,:,0:1], is_train:False})
result_de=Session.run(decode_image, feed_dict={input_code:result_en, is_train:False})
print(result_en)
print(result_de)
[x.name for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES)]
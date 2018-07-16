import tensorflow
import os
import cv2
import numpy

data_dir='/media/zhao/新加卷/SRAD2018/test/SRAD2018_Test_1'
log_dir='log/'
model_dir='model/'

Saver =tensorflow.train.import_meta_graph(tensorflow.train.latest_checkpoint(model_dir)+'.meta')
Session=tensorflow.Session()
Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))

train_image=tensorflow.get_default_graph().get_tensor_by_name('train_image:0')
cnn_decode_result=tensorflow.get_default_graph().get_tensor_by_name("cnn_decode_result/TensorArrayStack/TensorArrayGatherV3:0")

all_rad=os.listdir(data_dir)
rads=[os.path.join(data_dir,x) for x in all_rad]
all_image_dir=[]
for x in rads:
    image_dir=[os.path.join(x,y) for y in os.listdir(x)]
    image_dir.sort()
    all_image_dir.append(image_dir)
for x in all_image_dir:
    k1=[]
    for y in x:
        k1.append(cv2.imread(y))
    k1=numpy.array(k1).reshape([1,31,501,501,3])
    predict_result=Session.run(cnn_decode_result, feed_dict={train_image:k1[:,:,:,:,0:1]})
    predict_result=predict_result[0,:,:,:,:]
    predict_result=predict_result.repeat(3,axis=3)
    folder_name=os.path.split(os.path.split(y)[0])[1]
    save_root=os.path.join(os.path.split(data_dir)[0],'predict',folder_name)
    name1=os.path.split(y)[1]
    for z in range(6):
        result=predict_result[5*z+4]
        cv2.imwrite(save_root, predict_result)
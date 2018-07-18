import tensorflow
import numpy
import os
import random
import time
import cv2
import skimage.io

# data_dir='E:/SRAD2018/train'
# data_dir='/media/zhao/新加卷/SRAD2018/train'
data_dir='/home/jxzhao/tianchi/SRAD2018/train'
log_dir='log/'
model_dir='model/'
init_lr=0.001
decay_rate=0.01
batch_file=1
batch_rad=1
batch_size=batch_file*batch_rad
max_step=300000//batch_size+1

def gru_process(input_code):
    all_output_encode=[]
    init_hide=numpy.zeros([batch_size,32,32,32]).astype(numpy.float32)
    for i in range(31):
        if i==0:
            output_hide=convgru_encode(init_hide,input_code[:,i,:,:,:])
            all_output_encode.append(output_hide)
        else:
            output_hide=convgru_encode(output_hide,input_code[:,i,:,:,:])
            all_output_encode.append(tensorflow.reshape(output_hide,[batch_size,1,32,32,32]))

    all_output_decode=[]
    for i in range(30):
        output_hide=convgru_decode(output_hide)
        all_output_decode.append(output_hide)

    return all_output_encode, all_output_decode

len([x.name for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES)])

train_image=tensorflow.placeholder(tensorflow.float32,[batch_size,31,501,501,1],name='train_image')
answer_image=tensorflow.placeholder(tensorflow.float32,[batch_size,30,501,501,1],name='answer_image')
global_step = tensorflow.get_variable('global_step',initializer=0,trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step*30,decay_rate)
which_opt = tensorflow.get_variable('which_opt',initializer=0,trainable=False)

encoder_cell = tensorflow.contrib.rnn.Conv2DLSTMCell(input_shape=[batch_size,501,501],output_channels=4,kernel_shape=[3,3],use_bias=True,forget_bias=True,initializers=tensorflow.truncated_normal_initializer,name='encoder_convlstm1')
encoder_outputs, encoder_final_state = tensorflow.nn.dynamic_rnn(
        encoder_cell, train_image,
        dtype=tensorflow.float32, time_major=True,)
decoder_cell = tensorflow.contrib.rnn.Conv2DLSTMCell(input_shape=[batch_size,501,501],output_channels=4,kernel_shape=[3,3],use_bias=True,forget_bias=True,initializers=tensorflow.truncated_normal_initializer,name='decoder_convlstm1')
decoder_outputs, decoder_final_state = tensorflow.nn.dynamic_rnn(
        decoder_cell, answer_image,
        initial_state=encoder_final_state,
        dtype=tensorflow.float32, time_major=True, scope="plain_decoder",)

cnn_encode_result=tensorflow.map_fn(cnn_encode,train_image,name='cnn_encode_result')
gru_result=gru_process(cnn_encode_result)
pre_result=tensorflow.stack(gru_result[1],1)
cnn_decode_result=tensorflow.map_fn(cnn_decode,pre_result,name='cnn_decode_result')

loss=tensorflow.losses.mean_squared_error(answer_image[:,which_opt,:,:,:],cnn_decode_result[:,which_opt,:,:,:])

minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

Saver = tensorflow.train.Saver(max_to_keep=0,filename='cnn_convgru')

Session=tensorflow.Session()
if tensorflow.train.latest_checkpoint(model_dir):
    Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
else:
    Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.image('answer_images', answer_image[0,:,:,:,:], 10)
tensorflow.summary.image('output_images', cnn_decode_result[0], 10)
merge_all = tensorflow.summary.merge_all()
FileWriter = tensorflow.summary.FileWriter(log_dir, Session.graph)

for _ in range(max_step):
    all_file=os.listdir(data_dir)
    pick_files=random.sample(all_file,batch_file)
    files=[os.path.join(data_dir,x) for x in pick_files]
    all_rad=[os.listdir(x) for x in files]
    pick_rads=[random.sample(x,batch_rad) for x in all_rad]
    rads=[[os.path.join(files[x],y) for y in pick_rads[x]] for x in range(len(files))]
    all_image_dir=[]
    for x in rads:
        for y in x:
            image_dir=[os.path.join(y,z) for z in os.listdir(y)]
            image_dir.sort()
            all_image_dir.append(image_dir)
    all_image=[]
    for x in all_image_dir:
        k1=[]
        for y in x:
            k1.append(skimage.io.imread(y))
        all_image.append(k1)
    all_image=numpy.array(all_image)
    try:
        for j in range(30):
            Session.run(minimize,feed_dict={train_image:all_image[:,:31,:,:,0:1],answer_image:all_image[:,31:,:,:,0:1],which_opt:j})
        if Session.run(global_step)%3000==30:
            summary = Session.run(merge_all, feed_dict={train_image:all_image[:,:31,:,:,0:1],answer_image:all_image[:,31:,:,:,0:1],which_opt:10})
            FileWriter.add_summary(summary, Session.run(global_step))
            Saver.save(Session, model_dir, global_step)
            print(Session.run(loss,feed_dict={train_image:all_image[:,:31,:,:,0:1],answer_image:all_image[:,31:,:,:,0:1]}))
    except:
        with open('log/异常数据目录.txt','a') as f:
            f.write('异常数据:%s\n'%(rads))

    print(Session.run(global_step))
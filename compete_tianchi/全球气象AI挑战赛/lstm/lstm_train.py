import tensorflow
import numpy 
import os
import cv2
import random
import time
import encode_decode

# data_dir='E:/SRAD2018/train'
data_dir='/media/zhao/新加卷/SRAD2018/train'
# data_dir='/home/jxzhao/tianchi/SRAD2018/train'
log_dir='log/'
model_dir='model/'
init_lr=0.001
decay_rate=0.01
max_step=300001

def LSTM(x,is_train):
    # Parameters:
    # Input gate: input, previous output, and bias.
    ix=tensorflow.get_variable('ix', [100,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    im=tensorflow.get_variable('im', [64,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    ib=tensorflow.get_variable('ib', 64, initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    # Forget gate: input, previous output, and bias.
    fx=tensorflow.get_variable('fx', [100,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    fm=tensorflow.get_variable('fm', [64,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    fb=tensorflow.get_variable('fb', 64, initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    # Memory cell: input, state and bias.                             
    cx=tensorflow.get_variable('cx', [100,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    cm=tensorflow.get_variable('cm', [64,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    cb=tensorflow.get_variable('cb', 64, initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    # Output gate: input, previous output, and bias.
    ox=tensorflow.get_variable('ox', [100,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    om=tensorflow.get_variable('om', [64,64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    ob=tensorflow.get_variable('ob', 64, initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    # Variables saving state across unrollings.
    saved_output=tensorflow.get_variable('saved_output', [32,64], initializer=tensorflow.zeros_initializer(), trainable=False)
    saved_state=tensorflow.get_variable('saved_state', [32,64], initializer=tensorflow.zeros_initializer(), trainable=False)

    # Definition of the cell computation.
    def lstm_cell(i, o, state):
        input_gate = tensorflow.sigmoid(tensorflow.matmul(i, ix) + tensorflow.matmul(o, im) + ib)
        forget_gate = tensorflow.sigmoid(tensorflow.matmul(i, fx) + tensorflow.matmul(o, fm) + fb)
        update = tensorflow.matmul(i, cx) + tensorflow.matmul(o, cm) + cb
        state = forget_gate * state + input_gate * tensorflow.tanh(update)
        output_gate = tensorflow.sigmoid(tensorflow.matmul(i, ox) + tensorflow.matmul(o, om) + ob)
        return output_gate * tensorflow.tanh(state), state

input_code=tensorflow.placeholder(tensorflow.float32,[None,32,32,64],name='input_code')
is_train=tensorflow.placeholder(tensorflow.bool,name='is_train')
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step*61,decay_rate)

output_code=LSTM(input_code,is_train)

loss=tensorflow.losses.mean_squared_error(input_code,output_code)

with tensorflow.control_dependencies(tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)):
    minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

Saver = tensorflow.train.Saver(max_to_keep=0,filename='cnn')

Session=tensorflow.Session()
if tensorflow.train.latest_checkpoint(model_dir):
    Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
else:
    Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.image('input_images', input_image, 61)
tensorflow.summary.image('output_images', decode_z4, 61)
merge_all = tensorflow.summary.merge_all()
FileWriter = tensorflow.summary.FileWriter(log_dir, Session.graph)

for _ in range(max_step):
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

    encode_data=encode_decode.get_encode(all_image)
    decode_data=encode_decode.get_decode(encode_data)

    try:
        # for j in range(all_image.shape[0]):
        #     Session.run(minimize,feed_dict={input_image:all_image[j:j+1,:,:,0:1],is_train:True})
        Session.run(minimize,feed_dict={input_image:all_image[:,:,:,0:1],is_train:True})
        if Session.run(global_step)%1000==1:
            summary = Session.run(merge_all, feed_dict={input_image:all_image[:,:,:,0:1],is_train:False})
            FileWriter.add_summary(summary, Session.run(global_step))
            Saver.save(Session, model_dir, global_step)
            print(Session.run(loss,feed_dict={input_image:all_image[:,:,:,0:1],is_train:False}))
    except:
        with open('log/异常数据目录.txt','a') as f:
            f.write('异常数据:%s\n'%(one_rad))

    print(Session.run(global_step))

Session.close()
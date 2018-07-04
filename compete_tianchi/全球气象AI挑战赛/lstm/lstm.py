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
batch_file=2
batch_rad=2
code_dim=32*32*64
lstm_dim=1024

def lstm(x,is_train):
    x = tensorflow.unstack(x, 31, 1)
    lstm_cell =tensorflow.nn.rnn_cell.BasicLSTMCell(lstm_dim)
    outputs, states =tensorflow.nn.static_rnn(lstm_cell, x)

    w1=tensorflow.get_variable('w1', [lstm_dim,code_dim], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    b1=tensorflow.get_variable('b1', code_dim, initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
    z1=tensorflow.matmul(outputs[-1], w1) + b1
    z1 = tensorflow.nn.softmax(z1)

    return z1

input_code = tensorflow.placeholder(tensorflow.float32,[None,31,code_dim],name='input_code')
is_train=tensorflow.placeholder(tensorflow.bool,name='is_train')
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step*61,decay_rate)

output_code = lstm(input_code,is_train)

loss=tensorflow.losses.mean_squared_error(input_code,output_code)

with tensorflow.control_dependencies(tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)):
    minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

Saver = tensorflow.train.Saver(max_to_keep=0,filename='lstm')

Session=tensorflow.Session()
if tensorflow.train.latest_checkpoint(model_dir):
    Saver.restore(Session,tensorflow.train.latest_checkpoint(model_dir))
else:
    Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.image('true_images', input_image, 61)
tensorflow.summary.image('predict_images', decode_z4, 61)
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
            k1.append(cv2.imread(y))
        all_image.append(k1)
    all_image=numpy.array(all_image)
    encode_image=encode_decode.get_encode(all_image).reshape((batch_file*batch_rad,61,code_dim))

    batch_x = encode_image.reshape((batch_size, 31, 65536))
    sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
    if step % 1000 == 0:
        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
        print("Step " + str(step) + ", Minibatch Loss= " + "{:.4f}".format(loss) + ", Training Accuracy= " + "{:.3f}".format(acc))

print("Optimization Finished!")

# Calculate accuracy for 128 mnist test images
test_len = 128
test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
test_label = mnist.test.labels[:test_len]
print("Testing Accuracy:", \
    sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))
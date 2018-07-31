import tensorflow
import numpy
import os
import random
import skimage.io
import sklearn.preprocessing

log_dir='log/'
model_dir='model/'
batch_size=100
max_step=60000
repeat_times=10
init_lr=0.001
decay_rate=0.01
input_dim=28
hidden_dim=40
output_dim=10

(train_data,train_label),(test_data,test_label)=tensorflow.keras.datasets.mnist.load_data()
OneHotEncoder=sklearn.preprocessing.OneHotEncoder()
OneHotEncoder.fit(train_label.reshape(-1,1))
train_label=OneHotEncoder.transform(train_label.reshape(-1,1)).toarray()
test_label=OneHotEncoder.transform(test_label.reshape(-1,1)).toarray()

def RNN(x):
    cell = tensorflow.contrib.rnn.GRUCell(hidden_dim)
 
    outputs, final_state = tensorflow.nn.dynamic_rnn(cell, x, initial_state=cell.zero_state(tensorflow.shape(x)[0], dtype=tensorflow.float32), time_major=False)
 
    outputs = tensorflow.unstack(tensorflow.transpose(outputs, [1,0,2]))
    results = tensorflow.matmul(outputs[-1], tensorflow.get_variable('w',[hidden_dim,10])) + tensorflow.get_variable('b',10)
    return results

x = tensorflow.placeholder(tensorflow.float32, [None, input_dim, input_dim])
y = tensorflow.placeholder(tensorflow.float32, [None, 10])
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step,decay_rate)

pred = RNN(x)

loss = tensorflow.losses.softmax_cross_entropy(y,pred)

minimize = tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(tensorflow.nn.softmax(pred), 1), tensorflow.argmax(y, 1)), tensorflow.float32))

Session=tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.scalar('accuracy', accuracy)
merge_all = tensorflow.summary.merge_all()
FileWriter = tensorflow.summary.FileWriter(log_dir, Session.graph)

num = train_data.shape[0] // batch_size
for i in range(max_step*repeat_times//batch_size):
    temp_train = train_data[i % num * batch_size:i % num * batch_size + batch_size,:]
    temp_label = train_label[i % num * batch_size:i % num * batch_size + batch_size,:]
    Session.run(minimize, feed_dict={x:temp_train, y:temp_label})
    if Session.run(global_step) % 100 == 1:
        summary = Session.run(merge_all, feed_dict={x:test_data,y:test_label})
        FileWriter.add_summary(summary, Session.run(global_step))
    print(Session.run(global_step))
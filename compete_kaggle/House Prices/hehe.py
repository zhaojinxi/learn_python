import numpy 
import sklearn.preprocessing  
import tensorflow  
import sklearn.datasets  
import sklearn.model_selection  
import pandas
import matplotlib.pyplot

#
boston=sklearn.datasets.load_boston()  
x=boston.data  
y=boston.target  
x_3=x[:,3:6]  
x=numpy.column_stack([x,x_3])
train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = sklearn.model_selection.train_test_split(x, y, train_size=0.8, random_state=33)  
ss_x = sklearn.preprocessing.StandardScaler()  
train_x_disorder = ss_x.fit_transform(train_x_disorder)  
test_x_disorder = ss_x.transform(test_x_disorder)  
train_y_disorder = train_y_disorder.reshape(-1, 1)
test_y_disorder=test_y_disorder.reshape(-1, 1)

# define placeholder for inputs to network  
xs = tensorflow.placeholder(tensorflow.float32, [None, 16])
ys = tensorflow.placeholder(tensorflow.float32, [None, 1])
keep_prob = tensorflow.placeholder(tensorflow.float32)
x_image = tensorflow.reshape(xs, [-1, 4, 4, 1])
# conv1 layer 
W_conv1 = tensorflow.Variable(tensorflow.truncated_normal([3, 3, 1, 32], stddev=0.1))
b_conv1 = tensorflow.Variable(tensorflow.constant(0.1, shape=[32]))
h_conv1 = tensorflow.nn.leaky_relu(tensorflow.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
h_pool1 = tensorflow.nn.max_pool(h_conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# conv2 layer
W_conv2 = tensorflow.Variable(tensorflow.truncated_normal([1,1, 32, 64], stddev=0.1))
b_conv2 = tensorflow.Variable(tensorflow.constant(0.1, shape=[64]))
h_conv2 = tensorflow.nn.leaky_relu(tensorflow.nn.conv2d(h_pool1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
h_pool2 = tensorflow.nn.max_pool(h_conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
## fc1 layer
h_pool2_flat = tensorflow.reshape(h_pool2, [-1, 64])
W_fc1 = tensorflow.Variable(tensorflow.truncated_normal([64, 50], stddev=0.1))
b_fc1 = tensorflow.Variable(tensorflow.constant(0.1, shape=[50]))
h_fc1 = tensorflow.nn.leaky_relu(tensorflow.matmul(h_pool2_flat, W_fc1) + b_fc1)  
h_fc1_drop = tensorflow.nn.dropout(h_fc1, keep_prob)
## fc2 layer
W_fc2 = tensorflow.Variable(tensorflow.truncated_normal([50, 1], stddev=0.1))
b_fc2 = tensorflow.Variable(tensorflow.constant(0.1, shape=[1]))
#最后的计算结果   
prediction = tensorflow.nn.leaky_relu(tensorflow.matmul(h_fc1_drop, W_fc2) + b_fc2)

loss = tensorflow.reduce_mean(tensorflow.reduce_sum(tensorflow.square(ys - prediction), reduction_indices=[1]))

train_step = tensorflow.train.AdamOptimizer(0.01).minimize(loss)  
  
sess = tensorflow.Session()  
sess.run(tensorflow.global_variables_initializer())  
for i in range(1000):  
    sess.run(train_step, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 0.9})  
    print(i,'loss=',sess.run(loss, feed_dict={xs: train_x_disorder, ys: train_y_disorder, keep_prob: 1.0})) 

prediction_value = sess.run(prediction, feed_dict={xs: test_x_disorder, ys: test_y_disorder, keep_prob: 1.0})

###画图#######  
fig = matplotlib.pyplot.figure(figsize=(20, 3))
axes = fig.add_subplot(1, 1, 1)  
line1,=axes.plot(range(len(prediction_value)), prediction_value, 'b--',label='cnn',linewidth=2)  
line2,=axes.plot(range(len(test_y_disorder)), test_y_disorder, 'g',label='ground truth')  
axes.grid()  
fig.tight_layout()  
matplotlib.pyplot.legend(handles=[line1,  line2])  
matplotlib.pyplot.title('cnn')  
matplotlib.pyplot.show()  
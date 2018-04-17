import pandas
import numpy
import tensorflow
import argparse
import sklearn.preprocessing
import math
import pickle

FLAGS = []

def main():
    train = pandas.read_csv(FLAGS.data_dir + 'train.csv')
    label = train.loc[:,'label'].as_matrix().reshape(-1,1)
    lb = sklearn.preprocessing.LabelBinarizer()
    lb.fit(label)
    label = lb.transform(label)
    with open(FLAGS.data_dir + 'lb.pkl','wb') as f:
        pickle.dump(lb,f)
    data = train.drop(columns=['label']).as_matrix().reshape(-1,28,28,1)

    x = tensorflow.placeholder(tensorflow.float32,shape=(None,28,28,1))
    y = tensorflow.placeholder(tensorflow.float32,shape=(None,10))
    with tensorflow.name_scope('conv1'):
        w1 = tensorflow.Variable(tensorflow.truncated_normal(shape=[3,3,1,32],stddev=0.1),name='w1')
        b1 = tensorflow.Variable(tensorflow.constant(0.1,shape=[32]),name='b1')
        h1 = tensorflow.nn.relu(tensorflow.nn.conv2d(x,w1,[1,1,1,1],'SAME') + b1)
        hp1 = tensorflow.nn.max_pool(h1,[1,2,2,1],[1,2,2,1],'SAME')
    with tensorflow.name_scope('conv2'):
        w2 = tensorflow.Variable(tensorflow.truncated_normal(shape=[3,3,32,64],stddev=0.1),name='w2')
        b2 = tensorflow.Variable(tensorflow.constant(0.1,shape=[64]),name='b2')
        h2 = tensorflow.nn.relu(tensorflow.nn.conv2d(hp1,w2,[1,1,1,1],'SAME') + b2)
        hp2 = tensorflow.nn.max_pool(h2,[1,2,2,1],[1,2,2,1],'SAME')
    with tensorflow.name_scope('fc3'):
        w3 = tensorflow.Variable(tensorflow.truncated_normal(shape=[7 * 7 * 64,1024],stddev=0.1),name='w3')
        b3 = tensorflow.Variable(tensorflow.constant(0.1,shape=[1024]),name='b3')
        f3 = tensorflow.reshape(hp2,[-1,7 * 7 * 64])
        h3 = tensorflow.nn.relu(tensorflow.matmul(f3,w3) + b3)
    with tensorflow.name_scope('dropout'):
        keep_prob = tensorflow.placeholder(tensorflow.float32)
        d = tensorflow.nn.dropout(h3, keep_prob)
    with tensorflow.name_scope('fc4'):
        w4 = tensorflow.Variable(tensorflow.truncated_normal(shape=[1024,10],stddev=0.1),name='w4')
        b4 = tensorflow.Variable(tensorflow.constant(0.1,shape=[10]),name='b4')
        h4 = tensorflow.matmul(d,w4) + b4

    with tensorflow.name_scope('loss'):
        cross_entropy = tensorflow.nn.softmax_cross_entropy_with_logits(labels=y, logits=h4)
        cross_entropy = tensorflow.reduce_mean(cross_entropy)
    with tensorflow.name_scope('adam_optimizer'):
        train_step = tensorflow.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tensorflow.name_scope('accuracy'):
        correct_prediction = tensorflow.equal(tensorflow.argmax(h4, 1), tensorflow.argmax(y, 1))
        correct_prediction = tensorflow.cast(correct_prediction, tensorflow.float32)
        accuracy = tensorflow.reduce_mean(correct_prediction)

    saver = tensorflow.train.Saver(max_to_keep=1)

    with tensorflow.Session() as sess:
        sess.run(tensorflow.global_variables_initializer())
        for i in range(FLAGS.max_iteration):
            train_data = data[(i % math.ceil(data.shape[0] / FLAGS.batch_size)) * FLAGS.batch_size:(i % math.ceil(data.shape[0] / FLAGS.batch_size)) * FLAGS.batch_size + FLAGS.batch_size,:,:,:]
            train_label = label[(i % math.ceil(data.shape[0] / FLAGS.batch_size)) * FLAGS.batch_size:(i % math.ceil(data.shape[0] / FLAGS.batch_size)) * FLAGS.batch_size + FLAGS.batch_size,:]
            train_step.run(feed_dict={x: train_data, y: train_label, keep_prob: 0.5})
            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: train_data, y: train_label, keep_prob: 1.0})
                print('step %d, training accuracy %g' % (i, train_accuracy))

        k = []
        for i in range(math.ceil(data.shape[0] / FLAGS.batch_size)):
            batch_data = data[(i % math.ceil(data.shape[0] / FLAGS.batch_size)) * FLAGS.batch_size:(i % math.ceil(data.shape[0] / FLAGS.batch_size)) * FLAGS.batch_size + FLAGS.batch_size,:,:,:]
            batch_label = label[(i % math.ceil(data.shape[0] / FLAGS.batch_size)) * FLAGS.batch_size:(i % math.ceil(data.shape[0] / FLAGS.batch_size)) * FLAGS.batch_size + FLAGS.batch_size,:]
            pre = sess.run(accuracy,feed_dict={x: batch_data, y: batch_label, keep_prob: 1.0})
            k.append(pre)
            print('batch %g train accuracy %g' % (i,pre))
        final_accuracy = numpy.array(k).mean()
        print('final accuracy is %g' % final_accuracy)

        writer = tensorflow.summary.FileWriter(FLAGS.data_dir+"/train", sess.graph)

        saver.save(sess,FLAGS.data_dir + 'mnist.ckpt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',type=str,default='./data/Digit Recognizer/',help='')
    parser.add_argument('--test_ratio',type=float,default=0.1,help='')
    parser.add_argument('--max_iteration',type=int,default=20000,help='')
    parser.add_argument('--batch_size',type=int,default=50,help='')
    FLAGS,unparsed = parser.parse_known_args()
    main()
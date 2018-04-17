import tensorflow
import numpy
import datetime
import pandas

r_dim = 100
batch_size = 50
train_number=100000
g_per_round=5
(train_x, train_y), (test_x, test_y) =tensorflow.keras.datasets.mnist.load_data("mnist.npz")
train=numpy.concatenate([train_x.reshape(-1,784), train_y.reshape(-1,1)], 1)
train_x=train_x.reshape(-1,28,28,1)
num = train_x.shape[0] // batch_size
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"

def discriminator(x, is_train):
    with tensorflow.variable_scope('discriminator', reuse=tensorflow.AUTO_REUSE):
        w1 = tensorflow.get_variable(name='w1', shape=[5, 5, 1, 32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b1 = tensorflow.get_variable(name='b1', shape=[32], initializer=tensorflow.constant_initializer(0))
        d1 = tensorflow.nn.conv2d(input=x, filter=w1, strides=[1, 1, 1, 1], padding='SAME')
        d1 = tensorflow.nn.relu(d1 + b1)
        d1 = tensorflow.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # d1=tensorflow.layers.batch_normalization(d1, name='bn1', training=is_train)

        w2 = tensorflow.get_variable(name='w2', shape=[5, 5, 32, 64], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b2 = tensorflow.get_variable(name='b2', shape=[64], initializer=tensorflow.constant_initializer(0))
        d2 = tensorflow.nn.conv2d(input=d1, filter=w2, strides=[1, 1, 1, 1], padding='SAME')
        d2 = tensorflow.nn.relu(d2 + b2)
        d2 = tensorflow.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        d2 = tensorflow.reshape(d2, [-1, 7 * 7 * 64])
        # d2=tensorflow.layers.batch_normalization(d2, name='bn2', training=is_train)

        w3 = tensorflow.get_variable(name='w3', shape=[7 * 7 * 64, 1024], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b3 = tensorflow.get_variable(name='b3', shape=[1024], initializer=tensorflow.constant_initializer(0))
        d3 = tensorflow.matmul(d2, w3)
        d3 = tensorflow.nn.relu(d3 + b3)
        # d3=tensorflow.layers.batch_normalization(d3, name='bn3', training=is_train)

        w4 = tensorflow.get_variable(name='w4', shape=[1024, 2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b4 = tensorflow.get_variable(name='b4', shape=[2], initializer=tensorflow.constant_initializer(0))
        d4 =tensorflow.add(tensorflow.matmul(d3, w4), b4, name='d_output')

    return d4

def generator(x, is_train):
    with tensorflow.variable_scope('generator', reuse=tensorflow.AUTO_REUSE):
        w1 = tensorflow.get_variable('w1', shape=[r_dim, 3136], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b1 = tensorflow.get_variable('b1', shape=[3136], initializer=tensorflow.constant_initializer(0))
        g1 = tensorflow.matmul(x, w1) + b1
        g1=tensorflow.layers.batch_normalization(g1, name='bn1', training=is_train)
        g1 = tensorflow.nn.relu(g1)
        g1 = tensorflow.reshape(g1, [-1, 56, 56, 1])
        #g1=tensorflow.contrib.layers.batch_norm(g1, scale=True)
        #g1=tensorflow.contrib.layers.instance_norm(g1)
        #g1=tensorflow.contrib.layers.layer_norm(g1)

        w2 = tensorflow.get_variable('w2', shape=[3, 3, 1, r_dim//2], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b2 = tensorflow.get_variable('b2', shape=[r_dim//2], initializer=tensorflow.constant_initializer(0))
        g2 = tensorflow.nn.conv2d(g1, w2, strides=[1, 2, 2, 1], padding='SAME')
        g2=tensorflow.layers.batch_normalization(g2, name='bn2', training=is_train)
        g2 = tensorflow.nn.relu(g2 + b2)
        g2 = tensorflow.image.resize_images(g2, [56, 56])
        # g2=tensorflow.contrib.layers.batch_norm(g2, scale=True)
        
        w3 = tensorflow.get_variable('w3', shape=[3, 3, r_dim//2, r_dim//4], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b3 = tensorflow.get_variable('b3', shape=[r_dim//4], initializer=tensorflow.constant_initializer(0))
        g3 = tensorflow.nn.conv2d(g2, w3, strides=[1, 2, 2, 1], padding='SAME')
        g3=tensorflow.layers.batch_normalization(g3, name='bn3', training=is_train)
        g3 = tensorflow.nn.relu(g3 + b3)
        g3 = tensorflow.image.resize_images(g3, [56, 56])
        # g3=tensorflow.contrib.layers.batch_norm(g3, scale=True)

        w4 = tensorflow.get_variable('w4', shape=[1, 1, r_dim//4, 1], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b4 = tensorflow.get_variable('b4', shape=[1], initializer=tensorflow.constant_initializer(0))
        g4 = tensorflow.nn.conv2d(g3, w4, strides=[1, 2, 2, 1], padding='SAME')
        g4 = tensorflow.add(g4, b4, name='g_output')

    return g4

feed_noise = tensorflow.placeholder(tensorflow.float32, [None, r_dim], name='feed_noise')
real_image = tensorflow.placeholder(tensorflow.float32, [None,28,28,1], name='real_image')
is_d_train=tensorflow.placeholder(tensorflow.bool, name='is_d_train')
is_g_train=tensorflow.placeholder(tensorflow.bool, name='is_g_train')
global_step = tensorflow.Variable(0, trainable=False)   
learning_rate = tensorflow.train.exponential_decay(0.001, global_step, train_number, 0.1)  

fake_image=generator(feed_noise, is_g_train)
drf=discriminator(tensorflow.concat([real_image, fake_image], 0), is_d_train)
df=discriminator(fake_image, is_d_train)

d_label=tensorflow.reshape(tensorflow.one_hot(tensorflow.to_int32(tensorflow.concat([tensorflow.ones([batch_size, 1]), tensorflow.zeros([batch_size, 1])], 0)), depth=2), [batch_size+batch_size, 2])
d_loss = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits_v2(logits = drf, labels = d_label), name='d_loss')
g_label=tensorflow.reshape(tensorflow.one_hot(tensorflow.to_int32(tensorflow.ones([batch_size, 1])), depth=2), [batch_size, 2])
g_loss = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits_v2(logits = df, labels = g_label), name='g_loss')

accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(drf, 1), tensorflow.argmax(d_label, 1)), tensorflow.float32))

d_var=[]
g_var=[]
for x in tensorflow.get_collection(tensorflow.GraphKeys.TRAINABLE_VARIABLES):
    if 'discriminator' in x.name:
        d_var.append(x)
    elif 'generator' in x.name:
        g_var.append(x)

# d_mav=[]
# g_mav=[]
# for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES):
#     if 'discriminator' in x.name:
#         if 'moving_mean' in x.name:
#             d_mav.append(x)
#         elif 'moving_variance' in x.name:
#             d_mav.append(x)
#     elif 'generator' in x.name:
#         if 'moving_mean' in x.name:
#             g_mav.append(x)
#         elif 'moving_variance' in x.name:
#             g_mav.append(x)

update_ops = tensorflow.get_collection(tensorflow.GraphKeys.UPDATE_OPS)
with tensorflow.control_dependencies(update_ops):
    d_optimize = tensorflow.train.AdamOptimizer(learning_rate, name='d_Adam').minimize(d_loss, var_list=d_var, name='d_optimize', global_step=global_step)
    g_optimize = tensorflow.train.AdamOptimizer(learning_rate, name='g_Adam').minimize(g_loss, var_list=g_var, name='g_optimize', global_step=global_step)

# [x.name for x in tensorflow.get_collection()]
# tensorflow.trainable_variables(scope='discriminator')
# tensorflow.trainable_variables(scope='generator')
# tensorflow.trainable_variables(scope='batch_normalization')
# tensorflow.GraphKeys.TRAINABLE_VARIABLES
# tensorflow.GraphKeys.UPDATE_OPS

Saver = tensorflow.train.Saver()

sess=tensorflow.Session()
sess.run(tensorflow.global_variables_initializer())

# for x in dir(tensorflow.GraphKeys):
#     print(x.lower())
#     print([x.name for x in tensorflow.get_collection(x.lower())])

tensorflow.summary.scalar('Generator_loss', g_loss)
tensorflow.summary.scalar('Discriminator_loss', d_loss)
tensorflow.summary.image('Generated_images', fake_image, 10)
tensorflow.summary.scalar('Discriminator_accuracy', accuracy)
merged = tensorflow.summary.merge_all()
writer = tensorflow.summary.FileWriter(logdir, sess.graph)

for i in range(train_number//(g_per_round+1)):
    noise = numpy.random.normal(0, 1, size=[batch_size, r_dim])
    image = train_x[i % num * batch_size:i % num * batch_size + batch_size,:]
    sess.run(d_optimize, feed_dict={real_image:image, feed_noise:noise, is_d_train:True, is_g_train:False})

    for _ in range(g_per_round):
        noise = numpy.random.normal(0, 1, size=[batch_size, r_dim])
        sess.run(g_optimize, feed_dict={feed_noise:noise, is_d_train:False, is_g_train:True})

    if i % 100 == 0:
        noise = numpy.random.normal(0, 1, size=[batch_size, r_dim])
        image=test_x.sample(n=50).as_matrix().reshape(-1,28,28,1)
        summary = sess.run(merged, {feed_noise: noise, real_image: image, is_d_train:False, is_g_train:False})
        writer.add_summary(summary, i)
        Saver.save(sess,'model/gan', i)
        print(sess.run(accuracy, feed_dict={real_image:image, feed_noise:noise, is_d_train:False, is_g_train:False}))

    print(i)
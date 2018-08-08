import tensorflow
import sklearn.preprocessing

log_dir='log/'
batch_size=50
max_step=60000
repeat_times=30
init_lr=0.001
decay_rate=0.1

(train_data,train_label),(test_data,test_label)=tensorflow.keras.datasets.mnist.load_data()
OneHotEncoder=sklearn.preprocessing.OneHotEncoder()
OneHotEncoder.fit(train_label.reshape(-1,1))
train_label=OneHotEncoder.transform(train_label.reshape(-1,1)).toarray()
test_label=OneHotEncoder.transform(test_label.reshape(-1,1)).toarray()

##########################Spatial Transformer Networks##################################
def SpatialTransformer(input_fmap, theta, out_dims=None, **kwargs):
    theta = tensorflow.reshape(theta, [tensorflow.shape(input_fmap)[0], 2, 3])

    if out_dims:
        batch_grids = affine_grid_generator(out_dims[0], out_dims[1], theta)
    else:
        batch_grids = affine_grid_generator(tensorflow.shape(input_fmap)[1], tensorflow.shape(input_fmap)[2], theta)

    out_fmap = bilinear_sampler(input_fmap, batch_grids[:, 0, :, :], batch_grids[:, 1, :, :])

    return out_fmap

def get_pixel_value(img, x, y):
    batch_idx = tensorflow.range(0, tensorflow.shape(x)[0])
    batch_idx = tensorflow.reshape(batch_idx, (tensorflow.shape(x)[0], 1, 1))
    b = tensorflow.tile(batch_idx, (1, tensorflow.shape(x)[1], tensorflow.shape(x)[2]))

    indices = tensorflow.stack([b, y, x], 3)

    return tensorflow.gather_nd(img, indices)

def affine_grid_generator(height, width, theta):
    x = tensorflow.linspace(-1.0, 1.0, width)
    y = tensorflow.linspace(-1.0, 1.0, height)
    x_t, y_t = tensorflow.meshgrid(x, y)
    x_t = tensorflow.reshape(x_t, [-1])
    y_t = tensorflow.reshape(y_t, [-1])
    ones = tensorflow.ones_like(x_t)
    sampling_grid = tensorflow.stack([x_t, y_t, ones])

    sampling_grid = tensorflow.expand_dims(sampling_grid, axis=0)
    sampling_grid = tensorflow.tile(sampling_grid, tensorflow.stack([tensorflow.shape(theta)[0], 1, 1]))

    batch_grids = tensorflow.matmul(theta, sampling_grid)

    batch_grids = tensorflow.reshape(batch_grids, [tensorflow.shape(theta)[0], 2, height, width])

    return batch_grids

def bilinear_sampler(img, x, y):
    max_y=tensorflow.shape(img)[1] - 1
    max_x=tensorflow.shape(img)[2] - 1
    zero = tensorflow.zeros([], dtype='int32')

    x = 0.5 * ((x + 1.0) * tensorflow.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tensorflow.cast(max_y-1, 'float32'))

    x0 = tensorflow.cast(tensorflow.floor(x), 'int32')
    x1 = x0+1
    y0 = tensorflow.cast(tensorflow.floor(y), 'int32')
    y1 = y0+1

    x0 = tensorflow.clip_by_value(x0, zero, max_x)
    x1 = tensorflow.clip_by_value(x1, zero, max_x)
    y0 = tensorflow.clip_by_value(y0, zero, max_y)
    y1 = tensorflow.clip_by_value(y1, zero, max_y)

    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    x0 = tensorflow.cast(x0, 'float32')
    x1 = tensorflow.cast(x1, 'float32')
    y0 = tensorflow.cast(y0, 'float32')
    y1 = tensorflow.cast(y1, 'float32')

    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    wa = tensorflow.expand_dims(wa, axis=3)
    wb = tensorflow.expand_dims(wb, axis=3)
    wc = tensorflow.expand_dims(wc, axis=3)
    wd = tensorflow.expand_dims(wd, axis=3)

    out = tensorflow.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out
############################################################

def model(x):
    with tensorflow.variable_scope('cnn'):
        stw=tensorflow.get_variable('stw', [28,28,1,6], initializer=tensorflow.constant_initializer(0))
        stb=tensorflow.get_variable('stb', initializer=tensorflow.reshape(tensorflow.constant([[1., 0, 0], [0, 1., 0]]),[6]))
        stz=tensorflow.nn.conv2d(x,stw,[1,1,1,1],'VALID')+stb
        x=SpatialTransformer(x, tensorflow.reshape(stz,[-1,6]),[28,28])

        w1=tensorflow.get_variable('w1', [3,3,1,8], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b1=tensorflow.get_variable('b1', 8, initializer=tensorflow.constant_initializer(0))
        z1=tensorflow.nn.conv2d(x,w1,[1,2,2,1],'SAME')+b1
        z1=tensorflow.nn.selu(z1)

        w2=tensorflow.get_variable('w2', [3,3,8,16], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b2=tensorflow.get_variable('b2', 16, initializer=tensorflow.constant_initializer(0))
        z2=tensorflow.nn.conv2d(z1,w2,[1,2,2,1],'SAME')+b2
        z2=tensorflow.nn.selu(z2)

        w3=tensorflow.get_variable('w3', [3,3,16,32], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b3=tensorflow.get_variable('b3', 32, initializer=tensorflow.constant_initializer(0))
        z3=tensorflow.nn.conv2d(z2,w3,[1,2,2,1],'VALID')+b3
        z3=tensorflow.nn.selu(z3)

        w4=tensorflow.get_variable('w4', [3,3,32,10], initializer=tensorflow.truncated_normal_initializer(stddev=0.1))
        b4=tensorflow.get_variable('b4', 10, initializer=tensorflow.constant_initializer(0))
        z4=tensorflow.nn.conv2d(z3,w4,[1,1,1,1],'VALID')+b4
        z4=tensorflow.nn.selu(z4)

        z4=tensorflow.reshape(z4,[-1,10])
    return z4

input_data=tensorflow.placeholder(tensorflow.float32,[None,28,28,1],name='input_data')
input_label=tensorflow.placeholder(tensorflow.float32,[None,10],name='input_label')
global_step = tensorflow.get_variable('global_step',initializer=0, trainable=False)
learning_rate=tensorflow.train.exponential_decay(init_lr,global_step,max_step,decay_rate)

resullt=model(input_data)

loss=tensorflow.losses.softmax_cross_entropy(input_label,resullt)

minimize=tensorflow.train.AdamOptimizer(learning_rate).minimize(loss,global_step=global_step,name='minimize')

accuracy = tensorflow.reduce_mean(tensorflow.cast(tensorflow.equal(tensorflow.argmax(tensorflow.nn.softmax(resullt), 1), tensorflow.argmax(input_label, 1)), tensorflow.float32))

Session=tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())

tensorflow.summary.scalar('loss', loss)
tensorflow.summary.scalar('accuracy', accuracy)
merge_all = tensorflow.summary.merge_all()
FileWriter = tensorflow.summary.FileWriter(log_dir, Session.graph)

num = train_data.shape[0] // batch_size
for i in range(max_step*repeat_times//batch_size+1):
    temp_train = train_data[i % num * batch_size:i % num * batch_size + batch_size,:].reshape(-1,28,28,1)
    temp_label = train_label[i % num * batch_size:i % num * batch_size + batch_size,:]
    Session.run(minimize,feed_dict={input_data:temp_train/255,input_label:temp_label})
    if Session.run(global_step)%100==1:
        summary = Session.run(merge_all, feed_dict={input_data:test_data.reshape(-1,28,28,1)/255,input_label:test_label})
        FileWriter.add_summary(summary, Session.run(global_step))
    print(Session.run(global_step))
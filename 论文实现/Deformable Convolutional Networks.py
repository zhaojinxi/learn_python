import tensorflow
import sklearn.preprocessing

# tensorflow.enable_eager_execution()

log_dir='log/'
batch_size=50
max_step=60000
repeat_times=10
init_lr=0.001
decay_rate=0.1

(train_data,train_label),(test_data,test_label)=tensorflow.keras.datasets.mnist.load_data()
OneHotEncoder=sklearn.preprocessing.OneHotEncoder()
OneHotEncoder.fit(train_label.reshape(-1,1))
train_label=OneHotEncoder.transform(train_label.reshape(-1,1)).toarray()
test_label=OneHotEncoder.transform(test_label.reshape(-1,1)).toarray()

######################################################################
def DeformableConvolution(x, offset):
    batch=tensorflow.shape(x)[0]
    height=tensorflow.shape(x)[1]
    width=tensorflow.shape(x)[2]
    channel=tensorflow.shape(x)[3]

    grid=tensorflow.meshgrid(tensorflow.range(height),tensorflow.range(width))
    grid=tensorflow.stack([grid[1],grid[0]],-1)
    grid=tensorflow.expand_dims(grid,0)
    grid=tensorflow.tile(grid,[batch,1,1,channel])
    grid=tensorflow.cast(grid,tensorflow.float32)
    offset_result=grid+offset
    offset_result=tensorflow.reshape(offset_result,[batch,height,width,channel,2])
    coord_height = tensorflow.clip_by_value(offset_result[:,:,:,:,0], 0, tensorflow.cast(height,tensorflow.float32) - 1)
    coord_width=tensorflow.clip_by_value(offset_result[:,:,:,:,1], 0, tensorflow.cast(width,tensorflow.float32) - 1)
    coordinate=tensorflow.stack([coord_height,coord_width],-1)

    coord_left_top = tensorflow.cast(tensorflow.floor(coordinate), 'int32')
    coord_right_bottom = tensorflow.cast(tensorflow.ceil(coordinate), 'int32')
    coord_right_top = tensorflow.stack([coord_left_top[..., 0], coord_right_bottom[..., 1]], axis=-1)
    coord_left_bottom = tensorflow.stack([coord_right_bottom[..., 0], coord_left_top[..., 1]], axis=-1)
    
    batch_idx = tensorflow.range(0, batch)
    batch_idx = tensorflow.reshape(batch_idx, (batch, 1, 1, 1))
    batch_idx = tensorflow.tile(batch_idx, (1, height, width, channel))
    channel_idx = tensorflow.range(0,channel)
    channel_idx = tensorflow.reshape(channel_idx, (1, 1, 1, channel))
    channel_idx = tensorflow.tile(channel_idx, (batch, height, width, 1))

    idx_left_top = tensorflow.stack([batch_idx, coord_left_top[:,:,:,:,0], coord_left_top[:,:,:,:,1], channel_idx], -1)    
    val_left_top = tensorflow.gather_nd(x, idx_left_top)

    idx_right_bottom = tensorflow.stack([batch_idx, coord_right_bottom[:,:,:,:,0], coord_right_bottom[:,:,:,:,1], channel_idx], -1)    
    val_right_bottom = tensorflow.gather_nd(x, idx_right_bottom)    

    idx_right_top = tensorflow.stack([batch_idx, coord_right_top[:,:,:,:,0], coord_right_top[:,:,:,:,1], channel_idx], -1)    
    val_right_top = tensorflow.gather_nd(x, idx_right_top)    

    idx_left_bottom = tensorflow.stack([batch_idx, coord_left_bottom[:,:,:,:,0], coord_left_bottom[:,:,:,:,1], channel_idx], -1)    
    val_left_bottom = tensorflow.gather_nd(x, idx_left_bottom)     

    coord_minus_lt = offset_result - tensorflow.cast(coord_left_top, 'float32')
    val_top = val_left_top + (val_right_top - val_left_top) * coord_minus_lt[..., 1]
    val_bottom = val_left_bottom + (val_right_bottom - val_left_bottom) * coord_minus_lt[..., 1]
    value_reslut = val_top + (val_bottom - val_top) * coord_minus_lt[..., 0]

    return value_reslut
################################################################

def cnn(x):
    with tensorflow.variable_scope('cnn'):
        dcw1=tensorflow.get_variable('dcw1', [3,3,1,2], initializer=tensorflow.zeros_initializer)
        dcb1=tensorflow.get_variable('dcb1', 2, initializer=tensorflow.zeros_initializer)
        dcz1=tensorflow.nn.conv2d(x,dcw1,[1,1,1,1],'SAME')+dcb1
        z1 = DeformableConvolution(x, dcz1)

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
# input_data=tensorflow.random_normal([10,28,28,1],name='input_data',seed=1)

resullt=cnn(input_data)

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
    Session.run(minimize,feed_dict={input_data:temp_train,input_label:temp_label})
    if Session.run(global_step)%100==1:
        summary = Session.run(merge_all, feed_dict={input_data:test_data.reshape(-1,28,28,1),input_label:test_label})
        FileWriter.add_summary(summary, Session.run(global_step))
    print(Session.run(global_step))
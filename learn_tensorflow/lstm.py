import tensorflow
import numpy

input_dim=100
hidden_dim=200
output_dim=100
batch_size=10

def lstm(c_old,h_old,x):
    con=tensorflow.concat([h_old,x],1)

    fw=tensorflow.get_variable('fw',[input_dim+hidden_dim,hidden_dim])
    fb=tensorflow.get_variable('fb',hidden_dim)
    f=tensorflow.nn.sigmoid(tensorflow.matmul(con,fw)+fb)

    iw=tensorflow.get_variable('iw',[input_dim+hidden_dim,hidden_dim])
    ib=tensorflow.get_variable('ib',hidden_dim)
    i=tensorflow.nn.sigmoid(tensorflow.matmul(con,iw)+ib)

    mw=tensorflow.get_variable('mw',[input_dim+hidden_dim,hidden_dim])
    mb=tensorflow.get_variable('mb',hidden_dim)
    m=tensorflow.nn.tanh(tensorflow.matmul(con,mw)+mb)

    c_new=f*c_old+i*m

    ow=tensorflow.get_variable('ow',[input_dim+hidden_dim,hidden_dim])
    ob=tensorflow.get_variable('ob',hidden_dim)
    o=tensorflow.nn.sigmoid(tensorflow.matmul(con,ow)+ob)

    h_new=o*tensorflow.nn.tanh(c_new)
    return c_new,h_new

input_data=tensorflow.placeholder(tensorflow.float32,[None,input_dim])
hidden=tensorflow.placeholder(tensorflow.float32,[None,hidden_dim])
cell=tensorflow.placeholder(tensorflow.float32,[None,hidden_dim])

output=lstm(cell,hidden,input_data)

Session=tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())
all_output=[]
for i in range(5):
    if i==0:
        output_cell,output_hidden=Session.run(output,feed_dict={cell:numpy.zeros((batch_size,hidden_dim)),hidden:numpy.zeros((batch_size,hidden_dim)),input_data:numpy.ones((batch_size,input_dim))})
        last_cell=output_cell
        all_output.append(output_hidden)
    else:
        output_cell,output_hidden=Session.run(output,feed_dict={cell:last_cell,hidden:all_output[-1],input_data:numpy.random.random_sample((batch_size,input_dim))})
        last_cell=output_cell
        all_output.append(output_hidden)
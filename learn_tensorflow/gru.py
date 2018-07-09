import tensorflow
import numpy

input_dim=100
hidden_dim=200
output_dim=100
batch_size=10

def gru(h_old,x):
    con=tensorflow.concat([h_old,x],1)

    uw=tensorflow.get_variable('uw',[input_dim+hidden_dim,hidden_dim])
    ub=tensorflow.get_variable('ub',hidden_dim)
    u=tensorflow.nn.sigmoid(tensorflow.matmul(con,uw)+ub)

    rw=tensorflow.get_variable('rw',[input_dim+hidden_dim,hidden_dim])
    rb=tensorflow.get_variable('rb',hidden_dim)
    r=tensorflow.nn.sigmoid(tensorflow.matmul(con,rw)+rb)

    tw=tensorflow.get_variable('tw',[input_dim+hidden_dim,hidden_dim])
    tb=tensorflow.get_variable('tb',hidden_dim)    
    t=tensorflow.nn.tanh(tensorflow.matmul(tensorflow.concat([r*h_old,x],1),tw)+tb)

    h_new=(1-u)*h_old+u*t
    return h_new

input_data=tensorflow.placeholder(tensorflow.float32,[None,input_dim])
hidden=tensorflow.placeholder(tensorflow.float32,[None,hidden_dim])

output=gru(hidden,input_data)

Session=tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())
all_output=[]
for i in range(5):
    if i==0:
        output_hidden=Session.run(output,feed_dict={hidden:numpy.zeros((batch_size,hidden_dim)),input_data:numpy.ones((batch_size,input_dim))})
        last_hidden=output_hidden
        all_output.append(output_hidden)
    else:
        output_hidden=Session.run(output,feed_dict={hidden:last_hidden,input_data:numpy.random.random_sample((batch_size,input_dim))})
        last_hidden=output_hidden
        all_output.append(output_hidden)
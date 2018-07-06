import tensorflow
import numpy

input_dim=100
hidden_dim=200
output_dim=100
batch_size=10

iw=tensorflow.get_variable('iw',[input_dim,hidden_dim])
ib=tensorflow.get_variable('ib',hidden_dim)
sw=tensorflow.get_variable('sw',[hidden_dim,hidden_dim])
sb=tensorflow.get_variable('sb',hidden_dim) 
ow=tensorflow.get_variable('ow',[hidden_dim,output_dim])
ob=tensorflow.get_variable('ob',output_dim)

def rnn(s_old,x):
    i=tensorflow.matmul(x,iw)+ib  
    s=tensorflow.matmul(s_old,sw)+sb
    s_new=tensorflow.nn.relu(i+s)
    o=tensorflow.matmul(s_new,ow)+ob
    return s_new,o

input_data=tensorflow.placeholder(tensorflow.float32,[None,input_dim])
state=tensorflow.placeholder(tensorflow.float32,[None,hidden_dim])

output=rnn(state,input_data)

Session=tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())
all_output=[]
for i in range(5):
    if i==0:
        output_state,output_data=Session.run(output,feed_dict={state:numpy.zeros((batch_size,hidden_dim)),input_data:numpy.ones((batch_size,input_dim))})
        last_state=output_state
        all_output.append(output_data)
    else:
        output_state,output_data=Session.run(output,feed_dict={state:last_state,input_data:numpy.random.random_sample((batch_size,input_dim))})
        last_state=output_state
        all_output.append(output_data)
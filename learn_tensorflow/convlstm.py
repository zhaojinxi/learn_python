import tensorflow
import numpy

batch_size=10
input_shape=[batch_size,501,501,1]
hide_shape1=[batch_size,251,251,8]
hide_shape2=[batch_size,126,126,16]
hide_shape3=[batch_size,63,63,32]
hide_shape4=[batch_size,32,32,64]
output_shape=100

def convlstm(lay_num,x_shape,hide_shape,c_old,h_old,x):
    fxw=tensorflow.get_variable('fxw%s'%lay_num,[3,3,x_shape[3],hide_shape[3]])
    fhw=tensorflow.get_variable('fhw%s'%lay_num,[3,3,hide_shape[3],hide_shape[3]])
    fcw=tensorflow.get_variable('fcw%s'%lay_num,hide_shape)
    fb=tensorflow.get_variable('fb%s'%lay_num,hide_shape[3])
    fxw_r=tensorflow.nn.conv2d(x,fxw,[1,2,2,1],'SAME')
    fhw_r=tensorflow.nn.conv2d(h_old,fhw,[1,1,1,1],'SAME')
    fcw_r=c_old*fcw
    f=tensorflow.nn.sigmoid(fxw_r+fhw_r+fcw_r+fb)

    ixw=tensorflow.get_variable('ixw%s'%lay_num,[3,3,x_shape[3],hide_shape[3]])
    ihw=tensorflow.get_variable('ihw%s'%lay_num,[3,3,hide_shape[3],hide_shape[3]])
    icw=tensorflow.get_variable('icw%s'%lay_num,hide_shape)
    ib=tensorflow.get_variable('ib%s'%lay_num,hide_shape[3])
    ixw_r=tensorflow.nn.conv2d(x,ixw,[1,2,2,1],'SAME')
    ihw_r=tensorflow.nn.conv2d(h_old,ihw,[1,1,1,1],'SAME')
    icw_r=c_old*icw
    i=tensorflow.nn.sigmoid(ixw_r+ihw_r+icw_r+ib)

    txw=tensorflow.get_variable('txw%s'%lay_num,[3,3,x_shape[3],hide_shape[3]])
    thw=tensorflow.get_variable('thw%s'%lay_num,[3,3,hide_shape[3],hide_shape[3]])
    tb=tensorflow.get_variable('tb%s'%lay_num,hide_shape[3])
    txw_r=tensorflow.nn.conv2d(x,txw,[1,2,2,1],'SAME')
    thw_r=tensorflow.nn.conv2d(h_old,thw,[1,1,1,1],'SAME')
    t=tensorflow.nn.tanh(txw_r+thw_r+tb)

    c_new=f*c_old+i*t

    oxw=tensorflow.get_variable('oxw%s'%lay_num,[3,3,x_shape[3],hide_shape[3]])
    ohw=tensorflow.get_variable('ohw%s'%lay_num,[3,3,hide_shape[3],hide_shape[3]])
    ocw=tensorflow.get_variable('ocw%s'%lay_num,hide_shape)
    ob=tensorflow.get_variable('ob%s'%lay_num,hide_shape[3])
    oxw_r=tensorflow.nn.conv2d(x,oxw,[1,2,2,1],'SAME')
    ohw_r=tensorflow.nn.conv2d(h_old,ohw,[1,1,1,1],'SAME')
    ocw_r=c_old*ocw
    o=tensorflow.nn.sigmoid(oxw_r+ohw_r+ocw_r+ob)

    h_new=o*tensorflow.nn.tanh(c_new)
    
    return c_new,h_new

input_data=tensorflow.placeholder(tensorflow.float32,input_shape)
hide1=tensorflow.placeholder(tensorflow.float32,hide_shape1)
cell1=tensorflow.placeholder(tensorflow.float32,hide_shape1)
hide2=tensorflow.placeholder(tensorflow.float32,hide_shape2)
cell2=tensorflow.placeholder(tensorflow.float32,hide_shape2)
hide3=tensorflow.placeholder(tensorflow.float32,hide_shape3)
cell3=tensorflow.placeholder(tensorflow.float32,hide_shape3)
hide4=tensorflow.placeholder(tensorflow.float32,hide_shape4)
cell4=tensorflow.placeholder(tensorflow.float32,hide_shape4)

def model():
    with tensorflow.variable_scope('convlstm_encode'):
        output1=convlstm(1,input_shape,hide_shape1,cell1,hide1,input_data)

        output2=convlstm(2,hide_shape1,hide_shape2,cell2,hide2,output1[1])

        output3=convlstm(3,hide_shape2,hide_shape3,cell3,hide3,output2[1])

        output4=convlstm(4,hide_shape3,hide_shape4,cell4,hide4,output3[1])

    return output1,output2,output3,output4

output=model()

Session=tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())
all_output=[]
for i in range(31):
    if i==0:
        output1,output2,output3,output4=Session.run(output,feed_dict={input_data:numpy.ones(input_shape), cell1:numpy.zeros(hide_shape1), hide1:numpy.zeros(hide_shape1), cell2:numpy.zeros(hide_shape2), hide2:numpy.zeros(hide_shape2), cell3:numpy.zeros(hide_shape3), hide3:numpy.zeros(hide_shape3), cell4:numpy.zeros(hide_shape4), hide4:numpy.zeros(hide_shape4)})
        all_output.append(output4[1])
    else:
        output1,output2,output3,output4=Session.run(output,feed_dict={input_data:numpy.random.random_sample(input_shape), cell1:output1[0], hide1:output1[1], cell2:output2[0], hide2:output2[1], cell3:output3[0], hide3:output3[1], cell4:output4[0], hide4:output4[1]})
        all_output.append(output4[1])
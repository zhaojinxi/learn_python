import tensorflow
import numpy

batch_size=4

def convlstm_encode(c_old,h_old,x):
    with tensorflow.variable_scope('convlstm_encode', reuse=tensorflow.AUTO_REUSE):
        fxw=tensorflow.get_variable('fxw',[3,3,64,64])
        fhw=tensorflow.get_variable('fhw',[3,3,64,64])
        fcw=tensorflow.get_variable('fcw',[batch_size,32,32,64])
        fb=tensorflow.get_variable('fb',64)
        fxw_r=tensorflow.nn.conv2d(x,fxw,[1,1,1,1],'SAME')
        fhw_r=tensorflow.nn.conv2d(h_old,fhw,[1,1,1,1],'SAME')
        fcw_r=c_old*fcw
        f=tensorflow.nn.sigmoid(fxw_r+fhw_r+fcw_r+fb)

        ixw=tensorflow.get_variable('ixw',[3,3,64,64])
        ihw=tensorflow.get_variable('ihw',[3,3,64,64])
        icw=tensorflow.get_variable('icw',[batch_size,32,32,64])
        ib=tensorflow.get_variable('ib',64)
        ixw_r=tensorflow.nn.conv2d(x,ixw,[1,1,1,1],'SAME')
        ihw_r=tensorflow.nn.conv2d(h_old,ihw,[1,1,1,1],'SAME')
        icw_r=c_old*icw
        i=tensorflow.nn.sigmoid(ixw_r+ihw_r+icw_r+ib)

        txw=tensorflow.get_variable('txw',[3,3,64,64])
        thw=tensorflow.get_variable('thw',[3,3,64,64])
        tb=tensorflow.get_variable('tb',64)
        txw_r=tensorflow.nn.conv2d(x,txw,[1,1,1,1],'SAME')
        thw_r=tensorflow.nn.conv2d(h_old,thw,[1,1,1,1],'SAME')
        t=tensorflow.nn.tanh(txw_r+thw_r+tb)

        c_new=f*c_old+i*t

        oxw=tensorflow.get_variable('oxw',[3,3,64,64])
        ohw=tensorflow.get_variable('ohw',[3,3,64,64])
        ocw=tensorflow.get_variable('ocw',[batch_size,32,32,64])
        ob=tensorflow.get_variable('ob',64)
        oxw_r=tensorflow.nn.conv2d(x,oxw,[1,1,1,1],'SAME')
        ohw_r=tensorflow.nn.conv2d(h_old,ohw,[1,1,1,1],'SAME')
        ocw_r=c_old*ocw
        o=tensorflow.nn.sigmoid(oxw_r+ohw_r+ocw_r+ob)

        h_new=o*tensorflow.nn.tanh(c_new)
    
    return c_new,h_new

def convlstm_decode(c_old,h_old):
    with tensorflow.variable_scope('convlstm_decode', reuse=tensorflow.AUTO_REUSE):
        fhw=tensorflow.get_variable('fhw',[3,3,64,64])
        fcw=tensorflow.get_variable('fcw',[batch_size,32,32,64])
        fb=tensorflow.get_variable('fb',64)
        fhw_r=tensorflow.nn.conv2d(h_old,fhw,[1,1,1,1],'SAME')
        fcw_r=c_old*fcw
        f=tensorflow.nn.sigmoid(fhw_r+fcw_r+fb)

        ihw=tensorflow.get_variable('ihw',[3,3,64,64])
        icw=tensorflow.get_variable('icw',[batch_size,32,32,64])
        ib=tensorflow.get_variable('ib',64)
        ihw_r=tensorflow.nn.conv2d(h_old,ihw,[1,1,1,1],'SAME')
        icw_r=c_old*icw
        i=tensorflow.nn.sigmoid(ihw_r+icw_r+ib)

        thw=tensorflow.get_variable('thw',[3,3,64,64])
        tb=tensorflow.get_variable('tb',64)
        thw_r=tensorflow.nn.conv2d(h_old,thw,[1,1,1,1],'SAME')
        t=tensorflow.nn.tanh(thw_r+tb)

        c_new=f*c_old+i*t

        ohw=tensorflow.get_variable('ohw',[3,3,64,64])
        ocw=tensorflow.get_variable('ocw',[batch_size,32,32,64])
        ob=tensorflow.get_variable('ob',64)
        ohw_r=tensorflow.nn.conv2d(h_old,ohw,[1,1,1,1],'SAME')
        ocw_r=c_old*ocw
        o=tensorflow.nn.sigmoid(ohw_r+ocw_r+ob)

        h_new=o*tensorflow.nn.tanh(c_new)
    
    return c_new,h_new

input_encode=tensorflow.placeholder(tensorflow.float32,[batch_size,31,32,32,64])
encode_cell=tensorflow.placeholder(tensorflow.float32,[batch_size,32,32,64])
encode_hide=tensorflow.placeholder(tensorflow.float32,[batch_size,32,32,64])
input_decode=tensorflow.placeholder(tensorflow.float32,[batch_size,30,32,32,64])
decode_cell=tensorflow.placeholder(tensorflow.float32,[batch_size,32,32,64])
decode_hide=tensorflow.placeholder(tensorflow.float32,[batch_size,32,32,64])

def model():
    all_output_encode=[]
    for i in range(31):
        if i==0:
            output_encode_cell, output_encode_hide=convlstm_encode(encode_cell,encode_hide,input_encode[:,i,:,:,:])
            all_output_encode.append(output_encode_hide)
        else:
            output_encode_cell, output_encode_hide=convlstm_encode(output_encode_cell,output_encode_hide,input_encode[:,i,:,:,:])
            all_output_encode.append(output_encode_hide)

    all_output_decode=[]
    for i in range(30):
        if i==0:
            output_decode_cell, output_decode_hide=convlstm_decode(output_encode_cell,output_encode_hide)
            all_output_decode.append(output_encode_hide)
        else:
            output_decode_cell, output_decode_hide=convlstm_decode(output_decode_cell,output_decode_hide)
            all_output_decode.append(output_encode_hide)

    return output_encode_cell, all_output_encode, output_decode_cell, all_output_decode

len([x.name for x in tensorflow.get_collection(tensorflow.GraphKeys.GLOBAL_VARIABLES)])

output=model()

Session=tensorflow.Session()
Session.run(tensorflow.global_variables_initializer())
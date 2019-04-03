"""
使用alexNet实现数字分类
"""
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\Administrator\\Desktop\\detection', 'C:/Users/Administrator/Desktop/detection'])
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from AlexNet.read_CIFAR import *
import matplotlib.pyplot as plt

# mnist=input_data.read_data_sets("MNIST_DATA",one_hot=True)

tf.set_random_seed(1)
#1.定义卷积神经网络
tf_x=tf.placeholder(tf.float32,[None,227,227,3])
tf_y=tf.placeholder(tf.float32,[None,10])

conv1=tf.layers.conv2d(tf_x,filters=96,kernel_size=11,strides=4,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#55*55*96
pool1=tf.layers.max_pooling2d(conv1,3,2)#27*27*96
lrn=tf.nn.lrn(pool1,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)

conv2=tf.layers.conv2d(lrn,256,5,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#27*27*256
pool2=tf.layers.max_pooling2d(conv2,3,2)#13*13*256
lrn2=tf.nn.lrn(pool2,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)

conv3=tf.layers.conv2d(lrn2,384,3,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#13*13*384
lrn3=tf.nn.lrn(conv3,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)

conv4=tf.layers.conv2d(lrn3,384,3,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#13*13*384
lrn4=tf.nn.lrn(conv4,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)

conv5=tf.layers.conv2d(lrn4,256,3,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#13*13*256
pool5=tf.layers.max_pooling2d(conv5,3,2)#6*6*256
lrn5=tf.nn.lrn(pool5,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)

flat=tf.reshape(lrn5,[-1,9216])
fc1=tf.layers.dense(flat,4096,tf.nn.relu)
fc2=tf.layers.dense(fc1,1024,tf.nn.relu)
output=tf.layers.dense(fc2,10,tf.nn.softmax)

loss=tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)
train_op=tf.train.AdamOptimizer(1*1e-3).minimize(loss)
accury=tf.metrics.accuracy(labels=tf_y,predictions=output)[1]

sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

if __name__=='__main__':
    saver = tf.train.Saver(max_to_keep=2)
    # modelfile=r'C:\Users\Administrator\Desktop\detection\AlexNet\model\AlexNet-11600'
    # saver.restore(sess,modelfile)
    print("开始训练")
    accuries=[]
    losses=[]
    losses_test=[]
    plt.ion()
    for i in range(200001):
        bx,by=get_traindata(64)
        train_op_,loss_=sess.run([train_op,loss],{tf_x:bx,tf_y:by})
        if i%100==0:
            losses.append(loss_)
            bxt,byt=get_testdata(64)
            loss_,accury_=sess.run([loss,accury],{tf_x:bxt,tf_y:byt})
            accuries.append(accury_)
            losses_test.append(loss_)
            plt.subplot(2,1,1)
            plt.plot(range(len(losses)),losses,'r-')
            plt.plot(range(len(losses_test)), losses_test, 'g-')
            plt.subplot(2, 1, 2)
            plt.plot(range(len(accuries)),accuries,'g-')
            plt.pause(0.1)

            print("准确度：",accury_)
            
            if i%1000==0:
                saver.save(sess, r"F:\models\AlexNet", global_step=i)


    print("网络训练完成")
    plt.ioff()
    plt.show()
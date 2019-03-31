"""
经典网络lenet
原始lenet是32*32的输入这里我们为28*28
最终准确率为91.6%
"""

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

#1.加载mnist数据集
mnist=input_data.read_data_sets("MNIST_DATA",one_hot=True)

#定义网络
tf_x=tf.placeholder(tf.float32,[None,784])
tf_y=tf.placeholder(tf.float32,[None,10])
images=tf.reshape(tf_x,[-1,28,28,1])

conv1=tf.layers.conv2d(images,filters=6,kernel_size=5)#28*28=>24*24*6
pool1=tf.layers.average_pooling2d(conv1,pool_size=2,strides=2)#=>12*12*6

conv2=tf.layers.conv2d(pool1,filters=16,kernel_size=5)#8*8*16
pool2=tf.layers.average_pooling2d(conv2,2,2)#4*4*16

flat=tf.reshape(pool2,[-1,4*4*16])#256
fc1=tf.layers.dense(flat,120,activation=tf.nn.relu)#120
fc2=tf.layers.dense(fc1,84,tf.nn.relu)#80
output=tf.layers.dense(fc2,10,tf.nn.softmax)#10

loss=tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)
train_op=tf.train.AdamOptimizer(3*1e-3).minimize(loss)
accuracy=tf.metrics.accuracy(labels=tf_y,predictions=output)[1]

sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

print('网络定义结束，开始训练')

if __name__=='__main__':
    losses_train=[]
    losses_test=[]
    accuracies=[]
    accuracies_train=[]
    plt.ion()
    for i in range(20001):
        bx,by=mnist.train.next_batch(100)
        train_op_,loss_,accuracy_=sess.run([train_op,loss,accuracy],{tf_x:bx,tf_y:by})

        if i%100==0:
            losses_train.append(loss_)
            accuracies_train.append(accuracy_)
            loss_,accuracy_=sess.run([loss,accuracy],{tf_x:mnist.test.images,tf_y:mnist.test.labels})
            losses_test.append(loss_)
            accuracies.append(accuracy_)
            plt.subplot(2,1,1)
            plt.plot(range(len(losses_train)),losses_train,'r-')
            plt.plot(range(len(losses_train)),losses_test,'g-')
            plt.subplot(2,1,2)
            plt.plot(range(len(accuracies)),accuracies,'b-')
            plt.plot(range(len(accuracies_train)),accuracies_train,'r-')
            plt.pause((0.1))
    print("网络训练结束")
    plt.ioff()
    plt.show()

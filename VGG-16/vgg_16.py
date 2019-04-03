from AlexNet.read_CIFAR import *
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

"""
搭建VGG-16
"""

tf_x=tf.placeholder(tf.float32,[None,224,224,3])
tf_y=tf.placeholder(tf.float32,[None,10])

conv1=tf.layers.conv2d(tf_x,64,3,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
conv2=tf.layers.conv2d(conv1,64,3,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))

pool2=tf.layers.max_pooling2d(conv2,2,2)#112*112*64

conv3=tf.layers.conv2d(pool2,128,3,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
conv4=tf.layers.conv2d(conv3,128,3,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))

pool4=tf.layers.max_pooling2d(conv4,2,2)#56*56*128

conv5=tf.layers.conv2d(pool4,256,3,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
conv6=tf.layers.conv2d(conv5,256,3,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
conv7=tf.layers.conv2d(conv6,256,3,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))

pool7 = tf.layers.max_pooling2d(conv7, 2, 2)  #28*28*256

conv8 = tf.layers.conv2d(pool7, 512, 3, padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
conv9 = tf.layers.conv2d(conv8, 512, 3, padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
conv10 = tf.layers.conv2d(conv9, 512, 3, padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

pool10 = tf.layers.max_pooling2d(conv10, 2, 2)  # 14*14*512

conv11 = tf.layers.conv2d(pool10, 512, 3, padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
conv12 = tf.layers.conv2d(conv11, 512, 3, padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
conv13 = tf.layers.conv2d(conv12, 512, 3, padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))

pool13 = tf.layers.max_pooling2d(conv13, 2, 2)  # 7*7*512

conv14=tf.layers.conv2d(pool13,4096,7,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#用1*1卷积代替全连接
conv15=tf.layers.conv2d(conv14,1024,1,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#用1*1卷积代替全连接
conv16=tf.layers.conv2d(conv15,10,1,activation=tf.nn.softmax,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#用1*1卷积代替全连接
output=tf.reshape(conv16,[-1,10])

loss=tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)
train_op = tf.train.AdamOptimizer(1 * 1e-3).minimize(loss)
accury = tf.metrics.accuracy(labels=tf_y, predictions=output)[1]

losses=[]
losses_test=[]
accuracies=[]


def train(iters=1000):
    print('开始训练')
    plt.ion()
    sess=tf.Session(config=tf.ConfigProto(
                                device_count={"CPU":6,"GPU":1},
                                inter_op_parallelism_threads=1,
                                intra_op_parallelism_threads=1,
                                # gpu_options={"allow_growth" : True},
                                ))
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    for i in range(iters):
        with tf.device('/cpu:0'):
            bx,by=get_traindata(4,(224,224))
        train_,loss_=sess.run([train_op,loss],{tf_x:bx,tf_y:by})

        if i %100==0:
            losses.append(loss_)
            with tf.device('/cpu:0'):
                bxt,byt=get_testdata(4,(224,224))
            acc, loss_ = sess.run([accury, loss], {tf_x: bxt, tf_y: byt})

            print(i,',准确率为：',acc)
            if i>100 and acc>np.array(accuracies).max():
                save(sess,i)
            accuracies.append(acc)
            losses_test.append(loss_)
            plt.subplot(2,1,1)
            plt.plot(range(len(accuracies)),losses,'r-')
            plt.plot(range(len(accuracies)), losses_test, 'b-')
            plt.subplot(2, 1, 2)
            plt.plot(range(len(accuracies)), accuracies, 'r-')
            plt.pause(0.01)

    print('训练结束')
    plt.ioff()
    plt.show()


def save(sess,i):
    saver=tf.train.Saver(max_to_keep=2)
    saver.save(sess,r"F:\models\VGGNet", global_step=i)


if __name__=='__main__':
    train(1000)



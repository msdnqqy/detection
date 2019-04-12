"""
使用卷积-滑动窗口方法进行室内物体识别
"""

from CNN.室内物体检测.dataplus.load_data import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


SAVERPATH=r"F:\models\室内物体检测V2\model.ckpt"
# tf.set_random_seed(1)
i=0
"""
array([
{
'path':list[i][0],
'name':list[i][1],
'label':np.array(label),
}
])
"""
with tf.device('/cpu:0'):
    images_info=get_all_path_and_labels()

#创建分类网络
tf_x=tf.placeholder(tf.float32,[None,112,112,3])
tf_y=tf.placeholder(tf.float32,[None,12])


conv1=tf.layers.conv2d(tf_x,filters=32,kernel_size=5,strides=1,padding='same',activation=tf.nn.relu
                       # ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)
                       )#32*32*32
pool1=tf.layers.max_pooling2d(conv1,2,2)#64*64*32
# lrn1=tf.nn.lrn(pool1,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)

conv2=tf.layers.conv2d(pool1,filters=64,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu
                       # ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)
                       )#16*16*64
pool2=tf.layers.max_pooling2d(conv2,2,2)
lrn2=tf.nn.lrn(pool2,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)#32*32*64

#13*13
conv3=tf.layers.conv2d(lrn2,128,8,2,padding='valid',activation=tf.nn.relu
                       # ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)
                       )

conv4=tf.layers.conv2d(conv3,256,11,activation=tf.nn.relu
                       # ,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32)
                       )

conv5=tf.layers.conv2d(conv4,12,1,activation=tf.nn.softmax)
output=tf.reshape(conv5,[-1,12])

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(5* 1e-3).minimize(loss)
accury = tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1), predictions=tf.argmax(output,axis=1))[1]

losses=[]
losses_test=[]
accuracies=[]


def train(iters=1000):
    print('开始训练')
    print("开始获取",datetime.datetime.now())
    images_info=get_all_path_and_labels()
    print('获取结束',datetime.datetime.now())
    plt.ion()
    sess=tf.Session()
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    for i in range(iters):
        indexs = np.random.randint(0, images_info.shape[0], size=100)
        bx, by = read_image(images_info, indexs)
        # print("获取数据结束")
        train_,loss_=sess.run([train_op,loss],{tf_x:bx,tf_y:by})

        if i %1==0:
            losses.append(loss_)
            indexs = np.random.randint(0, images_info.shape[0], size=100)
            bxt, byt = read_image(images_info, indexs)
            acc, loss_,output_ ,conv4_= sess.run([accury, loss,output,conv5], {tf_x: bxt, tf_y: byt})
            print("conv4",conv4_)
            print(np.argmax(byt,axis=1))
            print(i,',准确率为：',acc,np.argmax(output_,axis=1),output_[:5,:])
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
    saver.save(sess,SAVERPATH)

def get_model(sessin):
    saver = tf.train.Saver(max_to_keep=2)
    saver.restore(sessin, SAVERPATH)

if __name__=='__main__':
    train(1001)



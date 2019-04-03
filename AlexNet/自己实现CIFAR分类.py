
"""
观察到AlexNet和VGG-16的训练效果并不大好，可能是图片32*32->224*224的原因
可能是网络太深，梯度消失梯度弥散
"""

from AlexNet.read_CIFAR import *
import tensorflow as tf
import matplotlib.pyplot as plt

tf.set_random_seed(1)

tf_x,tf_y=tf.placeholder(tf.float32,[None,32,32,3]),tf.placeholder(tf.float32,[None,10])

conv1=tf.layers.conv2d(tf_x,filters=32,kernel_size=5,strides=1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#32*32*32
pool1=tf.layers.max_pooling2d(conv1,2,2)#16*16*32
# lrn1=tf.nn.lrn(pool1,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)

conv2=tf.layers.conv2d(pool1,filters=64,kernel_size=3,strides=1,padding='same',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#16*16*64
pool2=tf.layers.max_pooling2d(conv2,2,2)
lrn2=tf.nn.lrn(pool2,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)#8*8*64

conv3=tf.layers.conv2d(lrn2,128,8,padding='valid',activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
# pool3=tf.layers.max_pooling2d(conv3,2,2)
# lrn3=tf.nn.lrn(pool3,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)#4*4*128

conv4=tf.layers.conv2d(conv3,10,1,activation=tf.nn.relu,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
# lrn4=tf.nn.lrn(conv4,depth_radius=5,bias=2,alpha=1e-4,beta=0.75)#1*1*256

# conv5=tf.layers.conv2d(conv4,10,1,activation=tf.nn.softmax,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#1*1*96
# conv6=tf.layers.conv2d(conv5,10,1,activation=tf.nn.softmax,kernel_initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))#1*1*10

output=tf.reshape(conv4,[-1,10])

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)
train_op = tf.train.AdamOptimizer(5* 1e-3).minimize(loss)
accury = tf.metrics.accuracy(labels=tf_y, predictions=output)[1]

losses=[]
losses_test=[]
accuracies=[]


def train(iters=1000):
    print('开始训练')
    plt.ion()
    sess=tf.Session()
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
    for i in range(iters):
        with tf.device('/cpu:0'):
            bx,by=get_traindata(100,(32,32))
        train_,loss_=sess.run([train_op,loss],{tf_x:bx,tf_y:by})

        if i %100==0:
            losses.append(loss_)
            with tf.device('/cpu:0'):
                bxt,byt=get_testdata(100,(32,32))
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
    saver.save(sess,r"F:\models\MyNet", global_step=i)


if __name__=='__main__':
    train(10000)





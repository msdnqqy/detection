"""
训练一个cnn网络用于手写数字识别
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

#数据准备
tf.set_random_seed(1)
mnist=input_data.read_data_sets("MNIST_DATA",one_hot=True)
test_x,test_y=mnist.test.images[:2000],mnist.test.labels[:2000]

print("data shape:",mnist.train.images[0].shape,test_x.shape,test_y.shape)
cv2.imshow('origin images[0]',cv2.resize((mnist.train.images[0].reshape(28,28)*255).astype(np.uint8),(300,300),interpolation=cv2.INTER_CUBIC))
cv2.waitKey(1000)
cv2.destroyAllWindows()
#构造卷积网络

tf_x=tf.placeholder(tf.float32,[None,784])
image=tf.reshape(tf_x,[-1,28,28,1])
tf_y=tf.placeholder(tf.float32,[None,10])

#卷积层1  =>14*14*16
conv1=tf.layers.conv2d(
    inputs=image,
    filters=16,
    kernel_size=7,
    strides=1,
    padding='same',
    activation=tf.nn.relu)

pooling1=tf.layers.max_pooling2d(
    conv1,
    pool_size=2,
    strides=2
)
#卷积层2 =>7*7*32
conv2=tf.layers.conv2d(
    pooling1,
    32,
    7,
    1,
    'same',
    activation=tf.nn.relu)
pooling2=tf.layers.max_pooling2d(
    conv2,
    pool_size=2,
    strides=2
)
#全连接层进行分类
flat=tf.reshape(pooling2,[-1,7*7*32])

conv3=tf.layers.conv2d(pooling2,64,7)
conv4=tf.layers.conv2d(conv3,10,1,activation=tf.nn.softmax)
output=tf.reshape(conv4,[-1,10])
# output=tf.layers.dense(flat,10,activation=tf.nn.softmax)

#定义损失函数
loss=tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)

#计算准确率
accuracy=tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1),predictions=tf.argmax(output,axis=1))[1]

#开始进行训练
sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

if __name__=="__main__":
    accuracys=[]
    plt.ion()
    for step in range(1201):
        b_x,b_y=mnist.train.next_batch(50)
        train_op_,loss_=sess.run([train_op,loss],{tf_x:b_x,tf_y:b_y})

        if step%50==0:
            print('step:',step)
            accuracy_,output_=sess.run([accuracy,output],{tf_x:test_x,tf_y:test_y})
            print("output_:",np.argmax(output_,axis=1)[:100])
            accuracys.append(accuracy_)
            plt.cla()
            plt.plot(range(len(accuracys)),accuracys,'g-')
            plt.show()

    #保存神经网络
    saver=tf.train.Saver()
    saver.save(sess,r"./mnist_model.ckpt")
    print("网络保存完成")

    #取训练结果
    # saver.restore(sess,r"./model/mnist_model.ckpt")

    plt.ioff()
    plt.show()


"""
验证位置变化是否对分类有影响
"""
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data

#获取数据
mnist=input_data.read_data_sets('MNIST_DATA',one_hot=True)
images=mnist.train.images
labels=mnist.train.labels

#2.在所有图片转化为56*56大小

def transpose_image(images,beginrow=0,begincol=0):
    images_cp=[]
    for image in images:
        image_zeros=np.zeros(shape=[56,56])
        image_zeros[0+beginrow:28+beginrow,0+begincol:28+begincol]=image.reshape([28,28])[:,:]
        image_zeros=image_zeros.reshape([56,56,1])
        images_cp.append(image_zeros)

    images_cp=np.array(images_cp)
    # print("转化成功：",images_cp.shape)
    return images_cp

def culaccuarcy(output,tf_y):
    output_=sess.run(tf.argmax(output,axis=1))
    tf_y_=sess.run(tf.argmax(tf_y,axis=1))
    c=(output_==tf_y_).astype(np.float32)
    return c.sum()/c.shape[0]

#3.放入卷积网络训练
tf_x=tf.placeholder(tf.float32,[None,56,56,1])
tf_y=tf.placeholder(tf.float32,[None,10])

#定义卷积层
conv1=tf.layers.conv2d(tf_x,32,5,1,'same',activation=tf.nn.relu)#56*56*32
pooling1=tf.layers.max_pooling2d(conv1,2,1)#最大池化55*55*32
conv2=tf.layers.conv2d(pooling1,64,5,1,'same',activation=tf.nn.relu)#51*51*64
pooling2=tf.layers.max_pooling2d(conv2,2,1)#最大池化50*50*64

conv3=tf.layers.conv2d(pooling2,128,5,1,'same',activation=tf.nn.relu)#46*46*128
pooling3=tf.layers.max_pooling2d(conv3,2,1)#最大池化45*45*128
conv4=tf.layers.conv2d(pooling3,128,5,1,'same',activation=tf.nn.relu)#41*41*128
pooling4=tf.layers.max_pooling2d(conv4,2,1)#最大池化40*40*128

flat=tf.reshape(pooling4,[-1,128*52*52])
output=tf.layers.dense(flat,10,activation=tf.nn.softmax)

accuracy=tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1),predictions=tf.argmax(output,axis=1))[1]
loss=tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output))
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)


sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

# train_images=transpose_image(images)
# test_images=transpose_image(mnist.test.images)
test_labels=mnist.test.labels

def train():
    plt.ion()
    losses = []
    accuracys = []
    for i in range(3001):
        indexs = np.random.randint(0, 50000, size=50)
        bx, by = transpose_image(images[indexs]), labels[indexs]
        # print(sess.run(pooling4,{tf_x:bx,tf_y:by}).shape)
        _, loss_ = sess.run([train_op, loss], {tf_x: bx, tf_y: by})

        if i % 100 == 0:
            indexs = np.random.randint(0, 10000, size=100)
            tx, ty = transpose_image(mnist.test.images[indexs]), test_labels[indexs]
            accuracy_ = sess.run(accuracy, {tf_x: tx, tf_y: ty})
            accuracys.append(accuracy_)
            losses.append(loss_)
            plt.cla()
            plt.subplot(2, 1, 1)
            plt.plot(range(len(losses)), losses, 'r-')
            plt.subplot(2, 1, 2)
            plt.plot(range(len(accuracys)), accuracys, 'b-')
            plt.pause(0.1)

    saver = tf.train.Saver()
    saver.save(sess, r"../网络保存/model/mnist_model_saver",global_step=i)
    print('训练结束')
    # plt.ioff()
    # plt.show()


def testing(begin=None):
    plt.figure()
    # 进行检测，测试是否由影响
    plt.ion()
    acc = []
    acc_oris = []
    for i in range(100):
        indext = np.random.randint(0, 10000, size=100)
        begin = begin if begin is not None else np.random.randint(0, 28, size=2)
        tbx, tby = transpose_image(mnist.test.images[indext], begin[0], begin[1]), mnist.test.labels[indext]
        out_ = sess.run(output, {tf_x: tbx, tf_y: tby})
        acc_=culaccuarcy(out_,tby)
        acc.append(acc_)


        tbx1, tby1 = transpose_image(mnist.test.images[indext]), mnist.test.labels[indext]
        output__ = sess.run(output, {tf_x: tbx1, tf_y: tby1})
        acc_ori=culaccuarcy(output__,tby1)
        acc_oris.append(acc_ori)
        plt.cla()
        plt.plot(range(len(acc)), acc, 'g-')
        plt.plot(range(len(acc)), acc_oris, 'r-.')
        plt.pause(0.1)

    indext = np.random.randint(0, 10000, size=100)
    begin = begin if begin is not None else np.random.randint(0, 28, size=2)
    tbx, tby = transpose_image(mnist.test.images[indext], begin[0], begin[1]), mnist.test.labels[indext]
    acc_ = sess.run(accuracy, {tf_x: tbx, tf_y: tby})
    print("当前准确率：",acc_)
    plt.ioff()
    plt.show()



if __name__ == '__main__':
    train()
    testing()



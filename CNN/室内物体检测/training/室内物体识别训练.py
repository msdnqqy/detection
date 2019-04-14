import tensorflow as tf
import numpy as np;
import cv2
from CNN.室内物体检测.dataplus.load_data import *
import matplotlib.pyplot as plt
images_info=get_all_path_and_labels()
print("image_info.shape",images_info.shape)

tf.set_random_seed(10000)
tf_x=tf.placeholder(tf.float32,[None,1024,1024,3])
tf_y=tf.placeholder(tf.float32,[None,12])

conv1=tf.layers.conv2d(tf_x,32,5,1,padding='same',activation=tf.nn.relu)
pooling1=tf.layers.max_pooling2d(conv1,2,2)
conv2=tf.layers.conv2d(pooling1,64,3,1,padding='same',activation=tf.nn.relu)
pooling2=tf.layers.max_pooling2d(conv2,2,2)#28*28*32

# flat=tf.reshape(pooling2,[-1,28*28*32])

l1=tf.layers.conv2d(pooling2,128,3,1,padding='same',activation=tf.nn.relu)
l1_pooling=tf.layers.max_pooling2d(l1,2,2)#14*14*256

l2=tf.layers.conv2d(l1_pooling,256,3,1,padding='same',activation=tf.nn.relu)
l2_pooling=tf.layers.max_pooling2d(l2,2,2)#7*7*256

conv3=tf.layers.conv2d(l2_pooling,512,7,1,padding='valid',activation=tf.nn.relu)
conv4=tf.layers.conv2d(conv3,128,1,1,padding='valid',activation=tf.nn.relu)
conv5=tf.layers.conv2d(conv4,12,1,1,padding='valid',activation=tf.nn.softmax)
output=tf.reshape(conv5,[-1,12])
# output=tf.layers.dense(l1,12,activation=tf.nn.softmax)

loss=tf.losses.softmax_cross_entropy(onehot_labels=tf_y,logits=output)
train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)

accuray=tf.metrics.accuracy(labels=tf.argmax(tf_y,axis=1),predictions=tf.argmax(output,axis=1))[1]

sess=tf.Session()
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])

SAVERPATH=r"F:\models\室内物体检测V4\model.ckpt"
def save(sess,i):
    saver=tf.train.Saver(max_to_keep=2)
    saver.save(sess,SAVERPATH)

def get_model(sessin):
    saver = tf.train.Saver(max_to_keep=2)
    saver.restore(sessin, SAVERPATH)

losses=[]
losses_test=[]
accuracies=[]

def train(iter=1000):
    for i in range(iter):
        plt.ion()
        indexs=np.random.randint(0,images_info.shape[0],size=100)
        bx,by=read_image(images_info,indexs)
        loss_,train_,output_,accuray_=sess.run([loss,train_op,output,accuray],{tf_x:bx,tf_y:by})

        if i % 100 == 0:
            losses.append(loss_)
            indexs = np.random.randint(0, images_info.shape[0], size=100)
            bxt, byt = read_image(images_info, indexs)
            acc, loss_, output_ = sess.run([accuray, loss, output], {tf_x: bxt, tf_y: byt})
            # print(i, ',准确率为：', acc, np.argmax(output_, axis=1))
            if i > 100 and acc > np.array(accuracies).max():
                save(sess, i)
            accuracies.append(acc)
            losses_test.append(loss_)
            plt.subplot(2, 1, 1)
            plt.plot(range(len(accuracies)), losses, 'r-')
            plt.plot(range(len(accuracies)), losses_test, 'b-')
            plt.subplot(2, 1, 2)
            plt.plot(range(len(accuracies)), accuracies, 'r-')
            plt.pause(0.01)

            print(i,'->accuray_:',accuray_,'loss:',loss_,'output:',np.argmax(output_,axis=1)==np.argmax(byt,axis=1))


if __name__=='__main__':
    train(10000)
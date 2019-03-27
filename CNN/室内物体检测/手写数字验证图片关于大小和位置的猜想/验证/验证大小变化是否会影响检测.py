import tensorflow as tf
import numpy as np
import cv2
from tensorflow.examples.tutorials.mnist import input_data
from CNN.室内物体检测.手写数字验证图片关于大小和位置的猜想.训练.CNN import *

# mnist=input_data.read_data_sets(r"../训练/MNIST_DATA",one_hot=True)

"""
训练网络用的图片是n*784大小，
1.如果目标变小，目标放在左上角，然后图片整体不变会不会影响检测
"""
# sess=tf.Session()
saver=tf.train.Saver()
saver.restore(sess,r"../网络保存/MNIST/mnist_model.ckpt")


#1.获取100张检测图片100*784 ，获取100张图片的labels
#2.将图片改为100*28*28格式
#3.将图片缩小为100*14*14，然后用0填充
#4.检测正确率
index=np.random.randint(0,2000,size=1)[0]
x=mnist.test.images[index:(index+100)]
y=mnist.test.labels[index:(index+100)]

accuracy_=sess.run(accuracy,{tf_x:x,tf_y:y})
print("当前的准确率为：",accuracy_,'\tindex:',index)

x_images=x.reshape([-1,28,28])
x_images_cp=[]
for i in range(100):
    x_image=cv2.resize(x_images[i]*255.0,(128,128))
    cv2.imshow('origin',x_image.astype(np.uint8))

    x_zeros=np.zeros(shape=[192,192],dtype=np.float32)*255.0
    x_zeros[32:160,32:160]=x_image[:,:]
    cv2.imshow('cp',x_zeros.astype(np.uint8))
    cv2.imshow('28*28',cv2.resize(x_zeros.astype(np.uint8),(28,28)))
    cv2.waitKey(300)

    x_images_cp.append(cv2.resize(x_zeros,(28,28)).flatten()/255.0)
x_images_cp=np.array(x_images_cp)
# print("转化后的图片shape：",x_images_cp.shape)

#对其进行检测
accuracy_1=sess.run(accuracy,{tf_x:x_images_cp,tf_y:y})
print("转化大小后的准确率为：",accuracy_1)
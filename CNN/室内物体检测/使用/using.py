"""
使用训练好的网络进行预测
这里要停止更新
"""
from CNN.室内物体检测.training.滑动窗口实现室内物体识别 import *
import tensorflow as tf
import cv2
import numpy as np

#加载model参数
tf_x=tf.placeholder(tf.float32,[None,768,768,3])
sess=tf.Session()
get_model(sess)

#冻结网络参数
# tf.stop_gradient(conv1)
# tf.stop_gradient(conv2)
# tf.stop_gradient(conv3)
# tf.stop_gradient(conv4)
# tf.stop_gradient(output)
# tf.stop_gradient(accury)


def load_test_img(test_img_path=r'C:\Users\Administrator\Desktop\datasets\test_images\1.jpg'):
    image=cv2.imread(test_img_path)#获取测试图片

    return image

#只是用一张图片来进行验证
def check(image):
    global  sess
    image_cover=image[np.newaxis,:,:,:].astype(np.float32)
    result=sess.run(conv4,{tf_x: image_cover})
    result_grid=result.max(axis=2)
    print("检测完成",result,result_grid)
    return result,result_grid

def resize(image,shape=(448,448)):
    image_resize=cv2.resize(image, shape)
    cv2.imshow("image", image_resize)
    cv2.waitKey(1000)
    return image_resize

if __name__=="__main__":
    path1=r'C:\Users\Administrator\Desktop\datasets\test_images\1.jpg'
    path2 = r'C:\Users\Administrator\Desktop\datasets\test_images\2.jpg'
    path3 = r'C:\Users\Administrator\Desktop\datasets\test_images\3.jpg'

    image=load_test_img(path1)
    result,result_grid=check(resize(image,shape=(768,768)))
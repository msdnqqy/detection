"""
从文件夹中读取所有图片
"""
import numpy as np
import tensorflow as tf
import cv2
import os
import datetime

with tf.device('/cpu:0'):
    sess=tf.Session()
    sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
Files=np.array([1,2,3,4,5,6,7,8,9,10])
ImagePath=r'C:\Users\Administrator\Desktop\datasets\images\{0}\deal'

def get_data(num=10000):
    lists=get_all_names()
    result=[]
    for i in range(num):
        item=get_data_one(lists)
        result.append(item)

    result=np.array(result)
    return np.array(list(result[:,0])),np.array(list(result[:,1]))

"""
从文件夹中读取一张图片，生成标签
"""
def get_data_one(lists):
    label_index=np.random.randint(0,Files.shape[0],size=1)[0]
    label=[0]*(Files.shape[0]+1)
    label[label_index+1]=1
    label=np.array(label)

    image_list=lists[label_index]
    image_index=np.random.randint(0,len(image_list[label_index]),size=1)[0]#定位一张图片的index
    image_path=image_list[image_index][0]#图片文件路径
    image=get_image(image_path)
    return np.array([image,label])

"""
标准化处理
"""
def get_image(path):
    with tf.device('/cpu:0'):
        image = cv2.imread(path)
        # cv2.imshow("input", image)
        std_image = tf.image.per_image_standardization(image)
        # cv2.waitKey(1000)
        # cv2.destroyAllWindows()
        return np.array(sess.run(std_image))


"""
获取所有文件
"""
def get_all_names():
    lists=[]
    for item in Files:
        path=ImagePath.format(item)
        list=getImgList(path)
        lists.append(list)

    return lists

"""
获取所有的文件列表及标记
"""
def get_all_path_and_labels():
    train_path_labels=[]
    for item in Files:
        path=ImagePath.format(item)
        list=getImgList(path)
        label=[0]*(len(Files)+1)
        label[item]=1
        for i in range(len(list)):
            img={
                'path':list[i][0],
                'name':list[i][1],
                'label':np.array(label),
            }
            train_path_labels.append(img)

    return np.array(train_path_labels)

"""
获取文件夹下的文件路径及名称
"""
def getImgList(path=None):
    list = os.listdir(path)
    subpath = []
    for sub in list:
        subpath.append([os.path.join(path, sub), sub])
    return subpath


"""
读取图片
"""
def read_image(images_info,indexs):
    train_images_info=images_info[indexs]
    images=[]
    labels=[]
    for item in train_images_info:
        image=cv2.imread(item['path'])
        image=image.astype(np.float32)/255.0
        label=item['label']
        images.append(image)
        labels.append(label)

    return np.array(images,dtype=np.float32),np.array(labels,dtype=np.float32)

if __name__=="__main__":
    print("开始获取",datetime.datetime.now())
    # for i in range(10):
    #     trainData,labels=get_data(100)
    images_info=get_all_path_and_labels()
    print('运行结束',datetime.datetime.now())

    print("开始读取图片", datetime.datetime.now())
    for i in range(100):
        indexs=np.random.randint(0,images_info.shape[0],size=100)
        train_images,train_labels=read_image(images_info,indexs)
    # cv2.imshow('蛇鼠一窝',train_images[0])
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    print('读取图片结束', datetime.datetime.now())
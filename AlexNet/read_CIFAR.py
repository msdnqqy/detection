"""
读取数据集
"""
import numpy as np
import pickle
import cv2

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def get_traindata(num=100):
    index=np.random.randint(1,6)
    data=unpickle(r"C:\Users\Administrator\Desktop\detection\AlexNet\cifar-10-python\cifar-10-batches-py\data_batch_{0}".format(index))
    indexs=np.random.randint(0,10000,num)
    labels_ori=np.array(data['labels'])[indexs]
    labels=get_labels(labels_ori)
    return resize(np.array(data['data'])[indexs]/255,(227,227)),labels

def get_testdata(num=100):
    index = np.random.randint(1, 6)
    data = unpickle(r"C:\Users\Administrator\Desktop\detection\AlexNet\cifar-10-python\cifar-10-batches-py\test_batch")
    indexs_test = np.random.randint(0, 10000, num)
    labels_ori = np.array(data['labels'])[indexs_test]
    labels_test = get_labels(labels_ori)
    return resize(np.array(data['data'])[indexs_test]/255,(227,227)),labels_test


def resize(images,shape):
    images_resize=[]
    for image in images:
        r=image[0:1024].reshape(32,32,1)
        g=image[1024:2048].reshape(32, 32,1)
        b=image[2048:3072].reshape(32,32,1)
        img=np.concatenate((r,g,b),axis=2)
        # cv2.imshow('img',img)
        image_resize=cv2.resize(img,shape,interpolation=cv2.INTER_CUBIC)
        images_resize.append(image_resize)
        # cv2.imshow('image_resize',image_resize)
        # cv2.waitKey(24)

    cv2.destroyAllWindows()
    return np.array(images_resize)

def get_labels(labels_ori):
    labels_ = []
    for i in range(len(labels_ori)):
        label = [0] * 10
        label[labels_ori[i]] = 1
        labels_.append(label)
    return np.array(labels_)


if __name__=='__main__':
    dict_data=unpickle(r"C:\Users\Administrator\Desktop\detection\AlexNet\cifar-10-python\cifar-10-batches-py\data_batch_1")

    train,labels=get_traindata()
    test,labels_test=get_testdata()

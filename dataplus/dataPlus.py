"""
进行数据增强，产生更多的训练数据
"""

import cv2
import numpy as np;
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from PIL import Image


class DataPlus(object):
    def __init__(self,path=None,outpath=None):
        self.path=0 if path is None else path
        self.outpath=r'/images' if outpath is None else outpath
        self.count=len(self.getImgList())


    def readImg(self):
        list=self.getImgList()
        path=list[0][0]
        file_content=tf.read_file(path)
        image_tensor = tf.image.decode_png(file_content,channels=3)

    """
    获取图像路径地址
    return[[路径，文件名]]
    """
    def getImgList(self):
        list=os.listdir(self.path)
        subpath=[]
        for sub in list:
            subpath.append([os.path.join(self.path,sub),sub])
        return subpath

    """
    图像上下，左右翻转
    """
    def transpose_img(self):
        list=self.getImgList()
        for subpath in list:
            img=Image.open(subpath[0])
            
            out1=img.transpose(Image.FLIP_LEFT_RIGHT)
            out2=img.transpose(Image.FLIP_TOP_BOTTOM)
            out3=img.transpose(Image.ROTATE_90)
            out4=img.transpose(Image.ROTATE_270)
            out=[out1,out2,out3,out4]

            for i in range(4):
                cv2.imwrite(r'{0}\{1}.jpg'.format(self.outpath,self.count),np.asarray(out[i]))
                cv2.imshow('transpose',np.asarray(out[i]))
                cv2.waitKey(10)
                self.count+=1



if __name__ == "__main__":
    dataplus=DataPlus(r'C:\Users\Administrator\Desktop\datasets\images\1',r'C:\Users\Administrator\Desktop\datasets\images\1')
    dataplus.transpose_img()
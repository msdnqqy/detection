"""
数据处理过程：
    1.mp4转图片
    2.图片resize=>56*56,112*112,224*224,448*448,996*996
    3.图片转化：
            1.旋转->90,180,
            2.局部切分/做缺失处理随机取上某块上变0
"""

import cv2
import os
import numpy as np
class dataDeal(object):
    def __init__(self,path,outpath):
        if path is None or outpath is None:
            print("需要路径，error")
            return
        self.path=path
        self.outpath=outpath

    """
    图片大小变换
    """
    def resize(self,img,width=64,height=64):
        img_cp=cv2.resize(img,(width,height),interpolation=cv2.INTER_CUBIC)
        return img_cp

    """
    转化所有的图片大小
    """
    def resize_all(self,width=64,height=64):
        for item in self.getImgList():
            img=cv2.imread(item[0])
            img_cp=self.resize(img,width,height)
            cv2.imwrite(self.outpath+r'\{0}x{1}_'.format(width,height)+item[1],img_cp)
        print("转化完成")


    """
    将图片切分为4等分+1中间区域
    """
    def cut(self,img):
        height,width=img.shape[0],img.shape[1]
        half_h,half_w=int(height/2),int(width/2)
        img_1=img[0:half_h,0:half_w]
        img_2=img[0:half_h,half_w:]
        img_3=img[half_h:,0:half_w]
        img_4=img[half_h:,half_w:]
        img_c=img[int(height-(height-half_h)/2):int(height-(height-half_h)/2)+half_h,int(width-(width-half_w)/2):int(width-(width-half_w)/2)+half_w]

        images=[img_1,img_2,img_3,img_4,img_c]
        for i in range(5):
            index_w=np.random.randint(0,half_w)
            index_h = np.random.randint(0, half_h)
            images.append(img[index_h:index_h+height,index_w:index_w+half_w])
        return images


    """
    保存所有切分的图片
    """
    def cut_all_and_save(self,path=None,shape=(64,64)):
        list=self.getImgList(path)
        for k in range( len(list)):
            item=list[k]
            imgs=self.cut(cv2.imread(item[0]))
            for i in range(len(imgs)):
                cv2.imwrite(self.outpath+r'\{0}_{1}'.format(i,item[1]),cv2.resize(imgs[i],shape))

    print("切分保存完成")

    """
    获取图像路径地址
    return[[路径，文件名]]
    """
    def getImgList(self,path=None):
        path=path if path is not None else self.path
        list=os.listdir(path)
        subpath=[]
        for sub in list:
            subpath.append([os.path.join(path,sub),sub])
        return subpath


if __name__=="__main__":

    for i in range(1,11):

        dataDeal_instance=dataDeal(path=r'C:\Users\Administrator\Desktop\datasets\images\{0}\origin'.format(i),
                                   outpath=r'C:\Users\Administrator\Desktop\datasets\images\{0}\deal'.format(i))
        dataDeal_instance.resize_all(width=112,height=112)
        dataDeal_instance.cut_all_and_save(path=r'C:\Users\Administrator\Desktop\datasets\images\{0}\origin'.format(i),shape=(112,112))
        print('已完成：',i)
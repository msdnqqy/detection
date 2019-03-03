"""
读取视频，并保存成图片
"""
#uncode=utf-8
import cv2;
import numpy as np;


class ReadVideoToImgs(object):
    def __init__(self,path=None,outpath=None):
        self.path=0 if path is None else path
        self.outpath=r'/images' if outpath is None else outpath
        self.count=0 #读取到的视频帧数

    """
    读取path中的视频并保存为图片
    方法会从中间剪切256*256的部分
    """
    def readvideo_saveas_imgs(self):
        cap=cv2.VideoCapture(self.path)#从path中读取视频，缺省值为从摄像头中读取视频
        while(cap.isOpened):
            ret,frame=cap.read()
            if ret==True and frame.shape[0]>200:
                #保存读取到的视频帧
                x,y=int((frame.shape[0]-200)/2),int((frame.shape[1]-200)/2)
                center_frame=frame[x:x+200,y:y+200]
                cv2.imshow('center_frame',center_frame)
                cv2.imwrite('{0}\\{1}.jpg'.format(self.outpath,self.count),center_frame)
                self.count+=1
                print('已完成：',self.count)
            cv2.imshow('Video',frame)
            cv2.waitKey(20)

        cap.release()




if __name__=='__main__':
    readVideoToImgs=ReadVideoToImgs(r'C:\Users\Administrator\Desktop\datasets\video\1\1.mp4',r'C:\Users\Administrator\Desktop\datasets\images\1')
    readVideoToImgs.readvideo_saveas_imgs()

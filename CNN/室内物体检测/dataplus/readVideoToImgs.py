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
    """
    def readvideo_saveas_imgs(self):
        cap=cv2.VideoCapture(self.path)#从path中读取视频，缺省值为从摄像头中读取视频
        while(cap.isOpened):
            ret,frame=cap.read()
            if ret==True and frame.shape[0]>200:
                #保存读取到的视频帧

                #图片缩放为224*224
                center_frame=cv2.resize(frame, (224, 224), interpolation=cv2.INTER_CUBIC)
                cv2.imshow('center_frame', center_frame)
                cv2.imwrite('{0}\\{1}.jpg'.format(self.outpath,self.count),center_frame)
                self.count += 1
                print('已完成：',self.count)
                cv2.imshow('Video',frame)
                cv2.waitKey(10)
            else:
                break
        cv2.destroyAllWindows()
        cap.release()



if __name__=='__main__':
    print("开始运行")
    for i in range(0,11):
        readVideoToImgs=ReadVideoToImgs(r'C:\Users\Administrator\Desktop\datasets\video\{0}\{0}.mp4'.format(i,i),r'C:\Users\Administrator\Desktop\datasets\images\{0}\origin'.format(i,))
        readVideoToImgs.readvideo_saveas_imgs()
        print('已完成：',i)

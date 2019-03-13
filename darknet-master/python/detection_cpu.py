"""
detect using cpu base darknet
"""
from darknet import *
import cv2

class DetectVideo(object):

    def __init__(self,path=None,cfg=None,weights=None,metafile=None):
        self.path=path if path is not None else 0

        cfg=bytes("./cfg/yolov3.cfg" if cfg is None else cfg,encoding='utf-8')
        weightsfile=bytes("./yolov3.weights" if weights is None else weights,encoding='utf-8')
        self.net = load_net(cfg,weightsfile , 0) 

        metafile=bytes("cfg/coco.data" if metafile is None else metafile,encoding='utf-8')
        self.meta = load_meta(metafile)

    #读取视频并检测
    def read_video_and_detect(self):
        cap=cv2.VideoCapture(self.path)
        ret, frame = cap.read()
        while ret:
            cv2.imshow("origin img",frame)
            cv2.waitKey(30)
            r = detect_cv2(self.net, self.meta, frame)
            print("检测结果：",r)
            self.show_result(r,frame)
            cv2.waitKey(30)

    #显示结果
    def show_result(self,r,img_cv):
        for  point in r:
            name=point[0]
            prob=point[1]
            center_x=point[2][0]
            center_y=point[2][1]
            width=point[2][2]
            height=point[2][3]
            img_cv[int(center_y):int(center_y+10),int(center_x):int(center_x+10)]=[255,0,0]
            begin=(int(center_x-1/2*width),int(center_y-1/2*height))
            end=(int(center_x+1/2*width),int(center_y+1/2*height))
            cv2.rectangle(img_cv,begin,end,(255,0,0),2)
            #在顶部显示prob+name
            cv2.putText(img_cv,'%.2f-%s'%(prob,name),begin,cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),1,cv2.LINE_AA)        

        cv2.imshow('dection result   using yolo3-darknet',img_cv)


if __name__ == "__main__":
    detectVideo=DetectVideo()
    detectVideo.read_video_and_detect()


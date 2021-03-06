from ctypes import *
import math
import random
import cv2

def sample(probs):
    s = sum(probs)
    probs = [a/s for a in probs]
    r = random.uniform(0, 1)
    for i in range(len(probs)):
        r = r - probs[i]
        if r <= 0:
            return i
    return len(probs)-1

def c_array(ctype, values):
    arr = (ctype*len(values))()
    arr[:] = values
    return arr

#bounding box的结构，我们预测输出为[box,box,classify]
class BOX(Structure):
    _fields_ = [("x", c_float),
                ("y", c_float),
                ("w", c_float),
                ("h", c_float)]

class DETECTION(Structure):
    _fields_ = [("bbox", BOX),
                ("classes", c_int),
                ("prob", POINTER(c_float)),
                ("mask", POINTER(c_float)),
                ("objectness", c_float),
                ("sort_class", c_int)]


class IMAGE(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                ("data", POINTER(c_float))]

class IMAGE_ARRAY(Structure):
    _fields_ = [("w", c_int),
                ("h", c_int),
                ("c", c_int),
                # ("array", c_float*921600)]
                 ("array", c_float*1327104)]

class METADATA(Structure):
    _fields_ = [("classes", c_int),
                ("names", POINTER(c_char_p))]

    
#引用编译好的darknet
#lib = CDLL("/home/pjreddie/documents/darknet/libdarknet.so", RTLD_GLOBAL)
lib = CDLL("./libdarknet.so", RTLD_GLOBAL)
lib.network_width.argtypes = [c_void_p]
lib.network_width.restype = c_int
lib.network_height.argtypes = [c_void_p]
lib.network_height.restype = c_int

predict = lib.network_predict
predict.argtypes = [c_void_p, POINTER(c_float)]
predict.restype = POINTER(c_float)

set_gpu = lib.cuda_set_device
set_gpu.argtypes = [c_int]

make_image = lib.make_image
make_image.argtypes = [c_int, c_int, c_int]
make_image.restype = IMAGE

get_network_boxes = lib.get_network_boxes
get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(c_int), c_int, POINTER(c_int)]
get_network_boxes.restype = POINTER(DETECTION)

make_network_boxes = lib.make_network_boxes
make_network_boxes.argtypes = [c_void_p]
make_network_boxes.restype = POINTER(DETECTION)

free_detections = lib.free_detections
free_detections.argtypes = [POINTER(DETECTION), c_int]

free_ptrs = lib.free_ptrs
free_ptrs.argtypes = [POINTER(c_void_p), c_int]

network_predict = lib.network_predict
network_predict.argtypes = [c_void_p, POINTER(c_float)]

reset_rnn = lib.reset_rnn
reset_rnn.argtypes = [c_void_p]

load_net = lib.load_network
load_net.argtypes = [c_char_p, c_char_p, c_int]
load_net.restype = c_void_p

do_nms_obj = lib.do_nms_obj
do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

do_nms_sort = lib.do_nms_sort
do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

free_image = lib.free_image
free_image.argtypes = [IMAGE]

letterbox_image = lib.letterbox_image
letterbox_image.argtypes = [IMAGE, c_int, c_int]
letterbox_image.restype = IMAGE

load_meta = lib.get_metadata
lib.get_metadata.argtypes = [c_char_p]
lib.get_metadata.restype = METADATA

load_image = lib.load_image_color
load_image.argtypes = [c_char_p, c_int, c_int]
load_image.restype = IMAGE

#为cv2设计，从array中生成image
create_image_using_array = lib.create_image_using_array
create_image_using_array.argtypes = [c_int, c_int, c_int,c_float*921600]
create_image_using_array.restype = IMAGE

rgbgr_image = lib.rgbgr_image
rgbgr_image.argtypes = [IMAGE]

predict_image = lib.network_predict_image
predict_image.argtypes = [c_void_p, IMAGE]
predict_image.restype = POINTER(c_float)

def classify(net, meta, im):
    out = predict_image(net, im)
    res = []
    for i in range(meta.classes):
        res.append((meta.names[i], out[i]))
    res = sorted(res, key=lambda x: -x[1])
    return res

#用于opencv-Python的方法
def detect_cv2_2(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    filename='image.jpg'
    cv2.imwrite(filename,image)
    im = load_image(bytes(filename,encoding="utf-8"), 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
is_first=True
def detect_cv2(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    global is_first
    if is_first:
        create_image_using_array.argtypes = [c_int, c_int, c_int,c_float*(image.shape[0]*image.shape[1]*image.shape[2])]
        create_image_using_array.restype = IMAGE
        is_first=False
    image_1=image.astype(float)
    arrayxxx=c_float*(image.shape[0]*image.shape[1]*image.shape[2])
    array=arrayxxx()
    h,w,c=image.shape[0],image.shape[1],image.shape[2]
    # count=0
    # for k in range(c):
    #     for i in range(h):
    #         for j in range(w):
    #              array[count]=float(image_1[i,j,k]/255)
    #              count+=1

    for i in range(h):
        for k in range(c):
            for j in range(w):
                array[k*w*h + i*w + j] = float(image_1[i,j,k]/255.)

    im=create_image_using_array(image.shape[0],image.shape[1],image.shape[2],array)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    # free_image(im)
    # free_detections(dets, num)
    return res

def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45):
    im = load_image(image, 0, 0)
    num = c_int(0)
    pnum = pointer(num)
    predict_image(net, im)
    dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, None, 0, pnum)
    num = pnum[0]
    if (nms): do_nms_obj(dets, num, meta.classes, nms);

    res = []
    for j in range(num):
        for i in range(meta.classes):
            if dets[j].prob[i] > 0:
                b = dets[j].bbox
                res.append((meta.names[i], dets[j].prob[i], (b.x, b.y, b.w, b.h)))
    res = sorted(res, key=lambda x: -x[1])
    free_image(im)
    free_detections(dets, num)
    return res
    
if __name__ == "__main__":
    #net = load_net("cfg/densenet201.cfg", "/home/pjreddie/trained/densenet201.weights", 0)
    #im = load_image("data/wolf.jpg", 0, 0)
    #meta = load_meta("cfg/imagenet1k.data")
    #r = classify(net, meta, im)
    #print r[:10]
    # net = load_net("cfg/tiny-yolo.cfg", "tiny-yolo.weights", 0)
    cfg=bytes("./cfg/yolov3.cfg",encoding='utf-8')
    weightsfile=bytes("./yolov3.weights",encoding='utf-8')
    net = load_net(cfg,weightsfile , 0)

    img="data/horses.jpg"
    metafile=bytes("cfg/coco.data",encoding='utf-8')
    imgfile=bytes(img,encoding='utf-8')
    meta = load_meta(metafile)

    for i in range(3):
        r = detect(net, meta, imgfile)
        print(r)
        

        #展示检测到的对象
        img_cv=cv2.imread(img)

        #画出中心点
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

        cv2.imshow('yolo3-darknet',img_cv)
        cv2.waitKey(2000)


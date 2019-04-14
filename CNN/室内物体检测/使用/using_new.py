import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\Administrator\\Desktop\\detection', 'C:/Users/Administrator/Desktop/detection'])
from CNN.室内物体检测.training.室内物体识别训练 import *
# tf_x=tf.placeholder(tf.float32,[None,768,768,3])
sess.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
#重载sess
get_model(sess)
#停止更新
tf.stop_gradient(accuray)
tf.stop_gradient(conv1)
tf.stop_gradient(conv2)
tf.stop_gradient(l1)
tf.stop_gradient(conv3)
tf.stop_gradient(conv4)
tf.stop_gradient(conv5)

# tf_x=tf.placeholder(tf.float32,[None,768,768,3])

def load_test_img(test_img_path=r'C:\Users\Administrator\Desktop\datasets\test_images\1.jpg'):
    image=cv2.imread(test_img_path)#获取测试图片

    return image

#只是用一张图片来进行验证
def check(image):
    global  sess
    image_cover=image[np.newaxis,:,:,:].astype(np.float32)
    print(image_cover.shape)
    result=sess.run(conv5,{tf_x: image_cover})
    result_grid=result.argmax(axis=3)[0]
    result_grid1 = result.max(axis=3)[0]
    print("检测完成",result[0].shape,'\n',result_grid,'\n',result_grid1)
    return result,result_grid,result_grid1

def resize(image,shape=(448,448)):
    image_resize=cv2.resize(image, shape)
    cv2.imshow("image", image_resize)
    cv2.waitKey(1000)
    return image_resize

if __name__=="__main__":
    path1=r'C:\Users\Administrator\Desktop\datasets\test_images\1.jpg'
    path2 = r'C:\Users\Administrator\Desktop\datasets\test_images\2.jpg'
    path3 = r'C:\Users\Administrator\Desktop\datasets\test_images\3.jpg'
    path4 = r'C:\Users\Administrator\Desktop\datasets\test_images\4.jpg'
    path5 = r'C:\Users\Administrator\Desktop\datasets\test_images\5.jpg'

    image=load_test_img(path4)
    image_resize=resize(image,shape=(1024,1024))/255.0
    image_resize=image_resize.astype(np.float32)
    print('image_resize',image_resize.shape)
    result,result_grid,result_grid1=check(image_resize)

    x=[]
    y=[]
    c=[]
    for i in range(result_grid.shape[0]):
        for j in range(result_grid.shape[1]):
            x.append(i)
            y.append(j)
            c.append(result_grid[i,j])

    import matplotlib.pyplot as plt
    plt.scatter(x,y,c=c)
    plt.show()


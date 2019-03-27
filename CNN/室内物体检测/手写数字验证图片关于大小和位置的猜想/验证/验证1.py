"""
验证位置变化是否会对检测结果又影响=>最后一层为全连接层,感觉可能有影响
"""
import sys; print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['C:\\Users\\Administrator\\Desktop\\detection', 'C:/Users/Administrator/Desktop/detection'])
from CNN.室内物体检测.手写数字验证图片关于大小和位置的猜想.训练.CNN1 import *

saver=tf.train.Saver()
sess1=tf.Session()
sess1.run([tf.global_variables_initializer(),tf.local_variables_initializer()])
saver.restore(sess1,r"../网络保存/model/mnist2.ckpt")

plt.ion()
acc=[]
acc_oris=[]
for i in range(100):
    indext=np.random.randint(0,10000,size=100)
    begin=np.random.randint(0,28,size=2)
    tbx,tby=transpose_image(mnist.test.images[indext],begin[0],begin[1]),mnist.test.labels[indext]
    acc_=sess1.run(accuracy,{tf_x:tbx,tf_y:tby})
    acc.append(acc_)

    tbx1, tby1 = transpose_image(mnist.test.images[indext]), mnist.test.labels[indext]
    acc_ori= sess1.run(accuracy, {tf_x: tbx1, tf_y: tby1})
    acc_oris.append(acc_ori)
    plt.cla()
    plt.plot(range(len(acc)),acc,'g-')
    plt.plot(range(len(acc)), acc_oris, 'r-.')
    plt.pause(0.1)

plt.ioff()
plt.show()
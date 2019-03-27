import tensorflow as tf
# msvcp140 = ctypes.WinDLL("msvcp140.dll")
with tf.device('/gpu:0'):
    if tf.test.is_built_with_cuda():
        print("support gpu")
    else:
        print("false")
"""
从dataset中抽取数据，标注 =（10000，文件夹编码=>one_hot编码）
"""

import numpy as np
from dataPlus import *
import pandas as pd

n_classifiers=1000
class DataGether(object):
    def __init__(self,path=None,outpath=None):
        self.DataPlus=DataPlus(path,outpath)
        self.path=0 if path is None else path
        self.outpath=r'/images' if outpath is None else outpath
        self.count=len(self.DataPlus.getImgList())

    def random_choice(self,count,target):
        #随机选取count个图片
        lists=np.array(self.DataPlus.getImgList())
        lists_map=lists[:,0]
        lists_random=np.random.choice(lists_map,size=count)
        labels=np.array([[0]*n_classifiers]*count)
        labels[target]=1
        return lists_random,labels

    def get_targetname(self,target):
        #读取classifier下面的xls，根据target拿到名字，返回名字
        df=pd.read_excel('./datasets/classifiers/classifiers.xlsx')
        name=df.iloc[target,1]
        return name


if __name__ == "__main__":
    dataGether=DataGether(r'./datasets/images/1',r'./datasets/images/1')
    # dataGether.get_targetname(0)
    dataGether.random_choice(100,0)
import pandas as pd
import cv2
import numpy as np
dataset=pd.read_csv("./fer2013/fer2013.csv")

dataset= dataset.iloc[:,0:2].values

uniq=0
for q in range(0, 35887):
    im=[]
    cnt=0
    x = [int(k) for k in dataset[q][1].split()]
    for i in range(0, 48):
        temp=[]
        for j in range(0,48):
            temp.append(x[cnt])
            cnt=cnt+1
        im.append(temp)
    arr=np.array(im)
    cv2.imwrite((str)(dataset[q][0])+'//'+(str)(uniq)+'.jpg',arr)
    uniq=uniq+1

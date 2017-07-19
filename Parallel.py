
# coding: utf-8

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import math
# #train_data = pd.read_table("E:/BDC_data/dsjtzs_txfz_training.txt/dsjtzs_txfz_training.txt" , sep=' ' , 
#                            #header=None , names=['id' , 'data' , 'target' , 'label'])
# #train_data

# import pandas as pd
# import scipy as sp
# import numpy as np
# import sklearn
# import gc
# import warnings
# from joblib import Parallel, delayed
# from sklearn.model_selection import train_test_split
# import lightgbm as lgb
# import xgboost as xgb
# import matplotlib
# import os
# train_data = pd.read_table("E:/BDC_data/dsjtzs_txfz_training.txt/dsjtzs_txfz_training.txt" , sep=' ' , 
#                            header=None , names=['id' , 'data' , 'target' , 'label'])
# 
# ##test_data = pd.read_table("E:/BDC_data/dsjtzs_txfz_test1.txt/dsjtzs_txfz_test1.txt" , sep=' ' , 
#                            #header=None , names=['id' , 'data' , 'target' , 'label'])
# 
# #print("ok")
# #train_data.apply(feature_select)
# c=pd.DataFrame([train_data.data.values+train_data.target.values],index=['data']).T
# data_need=c.data.apply(lambda x :x.split(";"))

# In[ ]:

def feature_select(train_data):
    orbit_data=[]
    x_cordinate=[]
    y_cordinate=[]
    time_sequence=[]
    oribit=train_data.split(";")[:-1]
    target=train_data.split(";")[-1]
    frame_index=[]
    for i in range(len(oribit)):
        temp_orbit=oribit[i]
        temp_float=[]
        #print(temp_orbit)
        for j in range(len(temp_orbit.split(","))): 

            temp_float.append(float(temp_orbit.split(",")[j]))

        orbit_data.append(temp_float)
        x_cordinate.append(temp_float[0])
        y_cordinate.append(temp_float[1])
        time_sequence.append(temp_float[2])

    target_cordinate=[0]*2
    target_cordinate[0]=float(target.split(",")[0])
    target_cordinate[1]=float(target.split(",")[1])

    #orbit_data


    #提取速度相关的特征
    speed_data=[]  #每段平均速率
    stop_times=[]        #每条曲线的停止时间
    speed_data_x=[]      #X方向上的速度
    speed_data_y=[]      #Y方向上的速度
    distance_total_list=[]
    time_total_list=[]
    time_delta_list=[]   #时间变化序列
    orbit_number_list=[] #轨迹的点数

    orbit_number_list.append(len(orbit_data))
    #三个额外的指标 7/10号增加
    S=[] #路程与位移之比
    TCM=[] 
    SC=[]
    
   
        

    if(len(orbit_data)<2): #如果点数小于2个，直接将速度变为0
        speed_data.append(0)
        speed_data_x.append(0)
        speed_data_y.append(0)
        stop_times.append(0)
        time_total_list.append(0)
        time_delta_list.append(0)
        distance_total_list.append(0)
        S.append(0)
        TCM.append(0)
        SC.append(0)
    elif(len(orbit_data)>=2):
        distance_total=0
        stoptime=0
        time_dot_distance=0
        time_pow_2_dot_distance=0
        time_total=time_sequence[-1]-time_sequence[0]
        weiyi=math.sqrt((y_cordinate[-1]-y_cordinate[0])**2+(x_cordinate[-1]-x_cordinate[0])**2)
        for i in range(len(orbit_data)-1):
            distance=math.sqrt((y_cordinate[i+1]-y_cordinate[i])**2+(x_cordinate[i+1]-x_cordinate[i])**2) #每段的距离
            time_dot_distance=time_dot_distance+distance*time_sequence[i+1] #每段距离与距离的目标的时间乘积
            time_pow_2_dot_distance=time_pow_2_dot_distance+distance*time_sequence[i+1]**2 #每段距离与距离的目标的时间的平方乘积
            distance_x=x_cordinate[i+1]-x_cordinate[i]
            distance_y=y_cordinate[i+1]-y_cordinate[i]
            distance_total= distance+ distance_total #每条曲线的总距离
            delta_t=time_sequence[i+1]-time_sequence[i] #时间间隔
            time_delta_list.append(delta_t) #获取时间间隔序列

            if(delta_t==0):
                stoptime=stoptime+1
                speed_data.append(0)
                speed_data_x.append(0)
                speed_data_y.append(0)
            elif(delta_t!=0):
                speed_data.append(distance/delta_t)
                speed_data_x.append(distance_x/delta_t)
                speed_data_y.append(distance_x/delta_t)
        if(distance_total!=0):
            S.append(weiyi/distance_total) #7.10新增
            TCM.append(time_dot_distance/distance_total) #7.10新增
            SC.append(time_pow_2_dot_distance/distance_total-TCM[0]**2)#7.10新增
        else:
            S.append(0)
            TCM.append(0)
            SC.append(0)
        distance_total_list.append(distance_total)
        time_total_list.append(time_total)
        stop_times.append(stoptime)


    speed_diff=pd.Series(speed_data).diff(1).dropna() #速度差的列表
    speed_diff_feature=pd.Series(speed_diff)
    
    #计算速率差各种指标
    Speed_diff_frame=pd.Series([
    speed_diff_feature.min()   #有最小平均速度
    ,speed_diff_feature.max()   #最大平均速度
    ,speed_diff_feature.std()   #平均速度的标准差
    ,speed_diff_feature.skew()  #平均速度的偏度
    ,speed_diff_feature.kurt()  #平均速度的峰度
    ,speed_diff_feature.median() #平均速度的中位数
    ,speed_diff_feature.median()#平均值
    ],index=['speed_diff__min','speed_diff__max','speed_diff__std','speed_diff__skew',
           'speed_diff__kurt','speed_diff__median','speed_diff__average'])
    Speed_diff_frame #平均速度指标

    frame_index.append(Speed_diff_frame)

    speed_feature=pd.Series(speed_data)

    #计算平均速率各种指标
    Speed_frame=pd.Series([
    speed_feature.min()   #有最小平均速度
    ,speed_feature.max()   #最大平均速度
    ,speed_feature.std()   #平均速度的标准差
    ,speed_feature.skew()  #平均速度的偏度
    ,speed_feature.kurt()  #平均速度的峰度
    ,speed_feature.median() #平均速度的中位数
    ,speed_feature.median()#平均值
    ,np.array(stop_times)
    ,np.array(distance_total_list)
    ,np.array(time_total_list)
    ,np.array(orbit_number_list)
    ,np.array(S)
    ,np.array(TCM)
    ,np.array(SC)],
    index=['speed_min','speed_max','speed_std','speed_skew',
           'speed_kurt','speed_median','speed_average','stop_times','distance_total','time_total','orbit_number',
           'S','TCM','SC'])
 #平均速度指标
    frame_index.append(Speed_frame)

    #计算x速率各种指标
    speed_data_x_feature=pd.Series(speed_data_x)
    speed_x_frame=pd.Series([
    speed_data_x_feature.min()   #有最小平均速度
    ,speed_data_x_feature.max()   #最大平均速度
    ,speed_data_x_feature.std()   #平均速度的标准差
    ,speed_data_x_feature.skew()  #平均速度的偏度
    ,speed_data_x_feature.kurt()  #平均速度的峰度
    ,speed_data_x_feature.median() #平均速度的中位数
    ,speed_data_x_feature.median()#平均值
    ],index=['x_speed_min','x_speed_max','x_speed_std','x_speed_skew','x_speed_kurt','x_speed_median','x_speed_average'])

    frame_index.append(speed_x_frame)

    #计算Y速率各种指标
    speed_data_y_feature=pd.Series(speed_data_y)
    speed_y_frame=pd.Series([
    speed_data_y_feature.min()   #有最小平均速度
    ,speed_data_y_feature.max()   #最大平均速度
    ,speed_data_y_feature.std()   #平均速度的标准差
    ,speed_data_y_feature.skew()  #平均速度的偏度
    ,speed_data_y_feature.kurt()  #平均速度的峰度
    ,speed_data_y_feature.median() #平均速度的中位数
    ,speed_data_y_feature.median()#平均值
    ],index=['y_speed_min','y_speed_max','y_speed_std','y_speed_skew','y_speed_kurt','y_speed_median','y_speed_average'])

    frame_index.append(speed_y_frame)

    #计算时间连续性各种指标
    time_delta_feature=pd.Series(time_delta_list)
    time_frame=pd.Series([
    time_delta_feature.min()   #有最小平均速度
    ,time_delta_feature.max()   #最大平均速度
    ,time_delta_feature.std()   #平均速度的标准差
    ,time_delta_feature.skew()  #平均速度的偏度
    ,time_delta_feature.kurt()  #平均速度的峰度
    ,time_delta_feature.median() #平均速度的中位数
    ,time_delta_feature.median()#平均值
    ],index=['time_delta__min','time_delta__max','time_delta__std','time_delta__skew',
    'time_delta__kurt','time_delta__median','time_delta__average'])

    frame_index.append(time_frame)
 
    #pd.concat([Speed_frame,speed_x_frame,speed_y_frame,time_frame])
    #print(time_delta_feature.diff(1).dropna())

    #提取角度相关的特征
    angle_data=[] #角度变化特征
    angle_bigger_90_count=[]
    Curvatute_rate=[]
    height_by_distance=[]
    
    if(len(orbit_data)<3):#如果轨迹坐标小于3个，角度变化为0
        angle_data.append(0)
        angle_bigger_90_count.append(0)
        Curvatute_rate.append(0)
        height_by_distance.append(0)
        k_list=pd.Series([0])
        k_list_diff=pd.Series([0])
    elif(len(orbit_data)>=3):
        angle_count=0
        for j in range(0,len(orbit_data)-2):
            delta_x_last=x_cordinate[j+2]-x_cordinate[j+1]
            delta_y_last=y_cordinate[j+2]-y_cordinate[j+1]
            delta_x_former=x_cordinate[j+1]-x_cordinate[j]
            delta_y_former=y_cordinate[j+1]-y_cordinate[j]
            delta_x_AC=x_cordinate[j+2]-x_cordinate[j]
            delta_y_AC=y_cordinate[j+2]-y_cordinate[j]
            a=np.squeeze([delta_x_former,delta_y_former]) #向量AB
            c=np.squeeze([delta_x_last,delta_y_last])#向量BC
            b=np.squeeze([delta_x_AC,delta_y_AC])#向量AC
            b_abs=np.sum(b**2)**0.5
            if(b_abs!=0):
                result=(a/b_abs)-np.dot(np.dot(a,b),b)/b_abs**3
                result_abs=np.sum(result**2)**0.5
                height_by_distance.append(result_abs)
            else:
                height_by_distance.append(0)
            distance=math.sqrt((y_cordinate[j+2]-y_cordinate[j])**2+(x_cordinate[j+2]-x_cordinate[j])**2) #每段弧度终点与起点的距离
            if(delta_x_last==0 and delta_x_last==0):
                angle_data.append(0)
                Curvatute_rate.append(0)
            elif(delta_x_former==0 and delta_y_former==0):
                angle_data.append(0)
                Curvatute_rate.append(0)
            else:
                vector=delta_x_last*delta_x_former+delta_y_last*delta_y_former 
                distance_last=math.sqrt(delta_x_last**2+delta_y_last**2)
                distance_former=math.sqrt(delta_x_former**2+delta_y_former**2)
                angle_cos=round(vector/(distance_last*distance_former),2)
                if(distance!=0):
                    Curvatute_rate.append(angle_cos/distance)
                else:
                    Curvatute_rate.append(0)
                angle_data.append(math.degrees(math.acos(angle_cos)))
                if(math.degrees(math.acos(angle_cos))>=90):
                    angle_count=angle_count+1
            
        angle_bigger_90_count.append(angle_count)
        k_list=pd.Series([np.log1p((orbit_data[i+1][1] - orbit_data[i][1])) - np.log1p((orbit_data[i+1][0] - orbit_data[i][0])) for i in range(len(orbit_data)-1)])    
        k_list_diff=k_list.diff(1).dropna() #斜率特征
    angle_feature=pd.Series(angle_data)
    Curvatute_rate_feature=pd.Series(Curvatute_rate)
    Curvatute_rate_feature_diff=Curvatute_rate_feature.diff(1).dropna()
    height_by_distance_feature=pd.Series(height_by_distance)
    height_by_distance_feature_diff=height_by_distance_feature.diff(1).dropna()
    
    angle_frame=pd.Series([   #有最小平均速度
    angle_feature.max()   #最大平均速度
    ,angle_feature.std()   #平均速度的标准差
    ,angle_feature.skew()  #平均速度的偏度
    ,angle_feature.kurt()  #平均速度的峰度
    ,angle_feature.median()  #平均速度的中位数
    ,np.array(angle_bigger_90_count)
    ,Curvatute_rate_feature.max()   #
    ,Curvatute_rate_feature.min() 
    ,Curvatute_rate_feature.mean()
    ,Curvatute_rate_feature.std()   #
    ,Curvatute_rate_feature.skew()  #
    ,Curvatute_rate_feature.kurt()  #
    ,Curvatute_rate_feature.median()  #
    ,Curvatute_rate_feature_diff.max()   #
    ,Curvatute_rate_feature_diff.min() 
    ,Curvatute_rate_feature_diff.mean()
    ,Curvatute_rate_feature_diff.std()   #
    ,Curvatute_rate_feature_diff.skew()  #
    ,Curvatute_rate_feature_diff.kurt()  #
    ,Curvatute_rate_feature_diff.median()  #
    ,height_by_distance_feature.max()   #
    ,height_by_distance_feature.min() 
    ,height_by_distance_feature.mean()
    ,height_by_distance_feature.std()   #
    ,height_by_distance_feature.skew()  #
    ,height_by_distance_feature.kurt()  #
    ,height_by_distance_feature.median()  #
    ,height_by_distance_feature_diff.max()   #
    ,height_by_distance_feature_diff.min()
    ,height_by_distance_feature_diff.mean()
    ,height_by_distance_feature_diff.std()   #
    ,height_by_distance_feature_diff.skew()  #
    ,height_by_distance_feature_diff.kurt()  #
    ,height_by_distance_feature_diff.median()  #
    ,],index=['angle_max','angle_std','angle_skew','angle_kurt','angle_median','angle_bigger_90_count',
              'Curvatute_rate_max','Curvatute_rate_min','Curvatute_rate_mean','Curvatute_rate_std','Curvatute_rate_skew','Curvatute_rate_kurt',
              'Curvatute_rate_median',
              'Curvatute_rate_diff_max','Curvatute_rate_diff_min','Curvatute_rate_diff_mean','Curvatute_rate_diff_std','Curvatute_rate_diff_skew',
              'Curvatute_rate_diff_kurt','Curvatute_rate_diff_median',
              'height_by_distance_max','height_by_distance_min','height_by_distance_mean','height_by_distance_std',
              'height_by_distance_skew','height_by_distance_kurt','height_by_distance_median',
              'height_by_distance_diff_max','height_by_distance_diff_min','height_by_distance_diff_mean','height_by_distance_diff_std',
              'height_by_distance_diff_skew','height_by_distance_diff_kurt','height_by_distance_diff_median'])

    
    
    frame_index.append(angle_frame)
    k_list_diff_feature=pd.Series(k_list_diff)   
    k_list_diff_frame=pd.Series([   #有最小平均速度
    k_list_diff_feature.max()   #最大平均速度
    ,k_list_diff_feature.std()   #平均速度的标准差
    ,k_list_diff_feature.skew()  #平均速度的偏度
    ,k_list_diff_feature.kurt()  #平均速度的峰度
    ,k_list_diff_feature.median()  #平均速度的中位数
    ],index=['k_list_diff_max','k_list_diff_std','k_list_diff_skew','k_list_diff_kurt','k_list_diff_median'])

    distance_to_moveline=[]
    cross_midline_time=[] #提取超过中线的方向变换次数
    distance_up_line=[]
    distance_fbs=[]
    distance_up_and_below=[]
    if(len(y_cordinate)<3):
        distance_to_moveline.append(0)
        cross_midline_time.append(0) #提取超过中线的方向变换次数
        distance_up_line.append(0)
        distance_fbs.append(0)
        distance_up_and_below.append(0)
    elif(len(y_cordinate)>=3):
        count=0
        y0=y_cordinate[0]
        x0=x_cordinate[0]
        y_last=y_cordinate[-1]
        x_last=x_cordinate[-1]
        A=-(y_last-y0)
        B=-(x0-x_last)
        C=-(x_last*y0-x0*y_last)
        distance=0
        for j in range(1,len(y_cordinate)-1):

            y1=y_cordinate[j] #计算穿过中线的次数
            y2=y_cordinate[j+1]
            if((y1<=y0 and y2>=y0) or (y1>=y0 and y2<=y0)):
                count=count+1
            if((A**2+B**2)!=0):
                distance=(A*x_cordinate[j]+B*y_cordinate[j]+C)/math.sqrt(A**2+B**2)
                distance_to_moveline.append(distance)
            else:
                distance_to_moveline.append(0)

        cross_midline_time.append(count)



    distance_up_and_below.append(sum(distance_to_moveline))
    if distance_to_moveline[0] !=0:
        for j in distance_to_moveline:
            temp_abs=0
            temp_up_line=0
            if(j>=0):
                temp_up_line=temp_up_line+j
                temp_abs=temp_abs+math.fabs(j)
        print(temp_up_line)
        distance_up_line.append(temp_up_line)
        distance_fbs.append(temp_abs)
    else:
        distance_up_line.append(0)
        distance_fbs.append(0)
    #distance_up_line[0]
    #distance_fbs[0]
    #print(len(distance_up_and_below))


    distance_change_feature=pd.Series(distance_to_moveline)
    distance_to_midline_frame=pd.Series([   #有最小平均速度
    distance_change_feature.max()   #最大平均速度
    ,distance_change_feature.std()
    ,distance_change_feature.mean()#平均速度的标准差
    ,distance_change_feature.median() #平均距离的中位数
    ,np.array(distance_up_line)             #在中线上的点的距离
    ,np.array(distance_fbs)                 #正负距离绝对值之和
    ,np.array(distance_up_and_below)      #正负距离之和
    ,np.array(cross_midline_time)],index=['distance_max','distance_std','distance_average',
    'distance_median','distance_up_line','distance_fbs','distance_up_and_below','cross_midline_times'])


    #坐标相关
    #与目的坐标先关
    distance_2_target=pd.Series(x_cordinate[i]-target_cordinate[0] for i in range(len(x_cordinate)))
    distance_2_target_diff=distance_2_target.diff(1).dropna()
    distance_final_2_target=[x_cordinate[-1]-target_cordinate[0]]
    #X坐标相关
    x_frame=pd.Series(x_cordinate)
    x_diff_frame=x_frame.diff(1).dropna()
    #y坐标相关
    y_frame=pd.Series(y_cordinate)
    y_diff_frame=x_frame.diff(1).dropna()
    x_back_num=[]
    y_back_num=[] 
    x_back_num.append(min( (x_diff_frame > 0).sum(),(x_diff_frame < 0).sum()))
    y_back_num.append(min( (y_diff_frame > 0).sum(),(y_diff_frame < 0).sum()))
    
    point_frame=pd.Series([
    distance_2_target.max()   #最大平均速度
    ,distance_2_target.min()
    ,distance_2_target.std()
    ,distance_2_target.mean()#平均速度的标准差
    ,distance_2_target.median()   
    ,np.array(distance_final_2_target)
    ,distance_2_target_diff.max()   #最大平均速度
    ,distance_2_target_diff.min()
    ,distance_2_target_diff.std()
    ,distance_2_target_diff.mean()#平均速度的标准差
    ,distance_2_target_diff.median() 
    ,x_frame.max()   #最大平均速度
    ,x_frame.min()
    ,x_frame.std()
    ,x_frame.mean()#平均速度的标准差
    ,x_frame.median()  
    ,x_diff_frame.max()   #最大平均速度
    ,x_diff_frame.min()
    ,x_diff_frame.std()
    ,x_diff_frame.mean()#平均速度的标准差
    ,x_diff_frame.median() 
    ,y_frame.max()   #最大平均速度
    ,y_frame.min()
    ,y_frame.std()
    ,y_frame.mean()#平均速度的标准差
    ,y_frame.median()  
    ,y_diff_frame.max()   #最大平均速度
    ,y_diff_frame.min()
    ,y_diff_frame.std()
    ,y_diff_frame.mean()#平均速度的标准差
    ,y_diff_frame.median()
    ,np.array(x_back_num)
    ,np.array(y_back_num)]
    ,index=['distance_2_target_max','distance_2_target_min','distance_2_target_std','distance_2_target_average',
    'distance_2_target_median','distance_final_2_target',
    'distance_2_target_diff_max','distance_2_target_diff_min','distance_2_target_diff_std','distance_2_target_diff_average',
    'distance_2_target_diff_median',
    'x_max','x_min','x_std','x_average',
    'x_median',
    'x_diff_max_diff','x_diff_min_diff','x_diff_std','x_diff_average',
    'x_diff_median',
    'y_max','y_min','y_std','y_average',
    'y_median',
    'y_diff_max_diff','y_diff_min_diff','y_diff_std','y_diff_average',
    'y_diff_median',
    'x_back_num',
    'y_back_num'])

    frame_index.append(point_frame)

    feature_frame=pd.concat([
    k_list_diff_frame
    ,angle_frame
    ,speed_y_frame
    ,Speed_diff_frame
    ,Speed_frame
    ,speed_x_frame
    ,time_frame
    ,distance_to_midline_frame
    ,point_frame])
    return feature_frame


# In[ ]:

import pandas as pd
import scipy as sp
import numpy as np
import sklearn

import math
from math import sqrt  
from joblib import Parallel, delayed 
import matplotlib
import os
if __name__ == '__main__': 
    train_data = pd.read_table("E:/BDC_data/dsjtzs_txfz_training.txt/dsjtzs_txfz_training.txt" , sep=' ' , 
                           header=None , names=['id' , 'data' , 'target' , 'label'])
    train_data_need=pd.DataFrame([train_data.data.values+train_data.target.values],index=['data']).T
    feature_train=Parallel(n_jobs=2)(delayed(feature_select)(train_data_need.data[i]) for i in range(3000))


# In[ ]:

Frame_train=pd.DataFrame(pd.concat(feature_train,axis=1)).T
Frame_train


# 
# import xgboost as xgb
# data=feature_train
# 
# 
# label=np.array(train_data['label'])    
# dtrain=xgb.DMatrix(data,label=label)
# #feature_train=np.array(feature_train)
# 
# 

# In[77]:

import xgboost as xgb
data=Frame_train


label=np.array(train_data['label'])    
dtrain=xgb.DMatrix(data,label=label)


# Frame_train_one=feature_train[0].fillna(0)
# Frame_train_one.index=range(0,2600)
# 
# Frame_train_zero=feature_train[2600].fillna(0)
# Frame_train_zero.index=range(2600,3000)
# 

# In[7]:

y=Frame_train_zero.sample(n=300,axis=0)
x=Frame_train_one.sample(n=600,axis=0)
Frame_train=pd.concat([x,y])

label=train_data['label'].tolist()[0:600]+train_data['label'].tolist()[2600:2900]
#label=np.array(label)
Frame_train.index=range(0,900)

len(Frame_train)
len(label)


# In[8]:

#特征筛选
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt


# In[9]:


XGB_1 = XGBClassifier(n_estimators=300)
#XGB_1 = XGBClassifier(n_estimators=600,min_child_weight=8.5,subsample=0.8)
#XGB_1=XGB_1.fit(Frame_train, train_data['label'])
XGB_1=XGB_1.fit(Frame_train, label)


selector_1 = RFE(XGB_1, step=2)
#selector_1 = selector_1.fit(Frame_train, train_data['label'])
selector_1 = selector_1.fit(Frame_train, label)




plot_importance(XGB_1)
plt.show()

#train_feature_frame_selected=selector_1.transform(train_feature_frame)
#test_feature_frame_selected=selector_2.transform(test_feature_frame)


# In[10]:

test_data = pd.read_table("E:/BDC_data/dsjtzs_txfz_test1.txt/dsjtzs_txfz_test1.txt" , sep=' ' , 
                           header=None , names=['id' , 'data' , 'target' , 'label'])
test_data_need=pd.DataFrame([test_data.data.values+test_data.target.values],index=['data']).T


# In[11]:

feature_test=test_data_need.data.apply(feature_select)
feature_test



# In[12]:

for i in range(1,100000):
    print(i)
    feature_test[0]=pd.concat([feature_test[0],feature_test[i]])


# In[13]:

len(feature_test[0])


# In[1]:

Frame_test=feature_test[0].fillna(0)
Frame_test.index=range(0,100000)

Frame_test


# In[89]:

XGB_2 = XGBClassifier(n_estimators=600,min_child_weight=8.5,subsample=0.8)
#XGB_2 = XGBClassifier(n_estimators=300,min_child_weight=21,subsample=0.78,learning_rate=0.09)
XGB_2.fit(Frame_train, label)


# In[90]:

result = XGB_2.predict(Frame_test)
result.sum()#84003 #83454 #83424 #83137 #82928 #82791 #82533 #82447  #81837 #81688 #81645 #81593


# In[91]:

zero_index=[]
result=result.tolist()
for i in range(len(result)):
    if(result[i]==0):
        zero_index.append(i+1)
fl=open('E:/BDC_data/BDC20170712.txt', 'w')
for i in zero_index:
    fl.write(str(i))
    fl.write("\n")
fl.close()


# In[ ]:




# In[ ]:




# In[ ]:




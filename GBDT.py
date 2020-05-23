
import xlrd
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn.externals import joblib
import joblib    #直接导入，已经从sklearn中独立
'''

name： GBDT 
author: Yuanlin Yang
time : 20200522

descripe: use 2020-4 to predict the value of indexs

note: to need manege these model, two variable need to be set. and firstly need to build 2 colums , it is Year and Month in every sheet
      @i: means chose different sheet whchi from dataset
      @name : means different model which have been save in local 
'''
def main():
    excel_name = "Real_estate-business_sample_data.xlsx"
    wb = xlrd.open_workbook(excel_name)
    # 获取workbook中所有的表格
    sheets = wb.sheet_names()

    #print(sheets)

    #遍历所有sheet
    df_28 = DataFrame()
    for i in range(len(sheets)):
        # skiprows=2 忽略前两行
        df = pd.read_excel(excel_name, sheet_name=i, skiprows=2, index=False, encoding='utf8')
        #print(df.head())
        
        #0为重庆sheet,1为杭州sheeet,以此类推
        if(i == 5):             
            sucessConsumInCQdf = DataFrame(pd.read_excel(excel_name, sheet_name=i, skiprows=2, index=False, encoding='utf8'))
        

    #sucessConsumInCQdf = sucessConsumInCQdf.iloc[:,:9]   #切片
    data = sucessConsumInCQdf[:].fillna(df[:].mean())  #缺省值用mean填充
    [row,col] = data.shape


    print(row,col)

    X = data.iloc[:,1:3]
    print(X.head())
    Y = data.iloc[:,3:16]
    GBDT_train(X,Y)  #模型训练
    GBDT_Predict()  #模型预测
    
def GBDT_train(X,Y):
    #print(X.head())
    #print(Y.head())
    for i in range(num_of_index):#训练16个模型，即输出值
        #print(Y.iloc[:200,i].head())
        #x_train, x_test, y_train, y_test = train_test_split(X, Y.iloc[:200,i].astype("str").values)
        x_train, x_test, y_train, y_test = train_test_split(X, Y.iloc[:,i])

        # 模型训练，使用GBDT算法   默认75%做训练 ， 25%做测试

        '''GradientBoostingRegressor参数介绍
          @n_estimators: 子模型的数量，默认为100
          @max_depth   ：最大深度 ，默认3
          @min_samples_split ：分裂最小样本数
          @learning_rate ：学习率
        '''
        gbr = GradientBoostingRegressor(n_estimators=200, max_depth=2, min_samples_split=2, learning_rate=0.1)
        gbr.fit(x_train, y_train.ravel())
        joblib.dump(gbr, name+"train_model_"+ str(i) +"_result.m")   # 保存模型

        y_gbr = gbr.predict(x_train)
        y_gbr1 = gbr.predict(x_test)
        acc_train = gbr.score(x_train, y_train)
        acc_test = gbr.score(x_test, y_test)
        print(name+"train_model_"+ str(i)  +"_result.m"+'训练准确率',acc_train)
        print(name+"train_model_"+ str(i)  +"_result.m"+'验证准确率',acc_test)

# 加载模型并预测
def GBDT_Predict( ):

    X_Pred = [2020,4]
    print("预测：2020-4")
    X_Pred = np.reshape(X_Pred, (1, -1))
    for i in range(num_of_index):
        gbr = joblib.load(name+"train_model_"+ str(i)  +"_result.m")    # 加载模型
        #test_data = pd.read_csv(r"./data_test.csv")
        test_y = gbr.predict(X_Pred)
        test_y = np.reshape(test_y, (1, -1))
        print(test_y)

  


#name = "CongQing_"
#name = "HangZhon"
#name = "LuoYang"
#name = "NanChong"
#name = "WuHu"
name = "ZhongShan"

num_of_index = 5
if __name__=="__main__":
    main()

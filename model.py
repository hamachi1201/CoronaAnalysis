#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


#モジュールの読み込み
from __future__ import print_function

import pandas as pd
from pandas import Series,DataFrame

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import numpy as np
import matplotlib.pyplot as plt

import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.optimizers import Adam


# In[2]:


#CSVファイルの読み込み
data_set = pd.read_csv("data.csv",sep=",",header=0)

#data_set.head(3)


# In[3]:


#time をone hotラベルに

import category_encoders as ce

# Eoncodeしたい列をリストで指定。もちろん複数指定可能。
list_cols = ['time']

# OneHotEncodeしたい列を指定。Nullや不明の場合の補完方法も指定。
#ce_ohe = ce.OneHotEncoder(cols=list_cols,handle_unknown='impute')
ce_oe = ce.OrdinalEncoder(cols=list_cols,handle_unknown='impute')

# pd.DataFrameをそのまま突っ込む
df_session_ce_ordinal = ce_oe.fit_transform(data_set)

#df_session_ce_ordinal.head(350)


# In[4]:


#print(df_session_ce_ordinal.columns.values)


# In[28]:



# データの分割
(train, test) = train_test_split(df_session_ce_ordinal, test_size=0.2, shuffle=True)

#x_train = train.ix[:,['time','latitude','longitude']]

data_set.columns = ['time','country','case','cure','death','longitude','latitude']
#   X = df.loc[:,'v1':'a3']
#    y = df['loss']
    
def norm(x):
    mean = np.mean(x)
    std = np.std(x)
    return (x - mean) / std

x_train = train.loc[:, ['time','longitude','latitude']]
y_train = train.loc[:, ['case','cure','death']]
#y_train = train.loc[:, ['case']]

x_train = norm(x_train)
y_train = norm(y_train)

x_test = test.loc[:, ['time','longitude','latitude']]
y_test = test.loc[:, ['case','cure','death']]
#y_test = test.loc[:, ['case']]

x_test = norm(x_test)
y_test = norm(y_test)

#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

y_train = y_train.astype(np.float)
y_test = y_test.astype(np.float)


# In[40]:


#x_train


# In[41]:


#y_train


# In[39]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers


#ニューラルネットワークの実装
model = Sequential()

model.add(Dense(600, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),input_shape=(3,)))
#model.add(Dropout(0.1))

model.add(Dense(600, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001), input_shape=(3,)))
#model.add(Dropout(0.1))

model.add(Dense(600, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001), input_shape=(3,)))
#model.add(Dropout(0.1))

model.add(Dense(3, activation='linear'))




model.summary()
print("\n")

#ニューラルネットワークの実装②
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])


early_stopping = EarlyStopping(
                        monitor='val_loss',
                        min_delta=0.0,
                        patience=50,
                )

# val_lossの改善が2エポック見られなかったら、学習率を0.5倍する。
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.7,
                        patience=20,
                        min_lr=0.001
                )

#ニューラルネットワークの学習
history = model.fit(x_train, y_train,batch_size=200,epochs=100,verbose=1,validation_data=(x_test, y_test),callbacks=[early_stopping, reduce_lr])

#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test mean_squared_error",score[1])



#学習履歴のグラフ化に関する参考資料
#http://aidiary.hatenablog.com/entry/20161109/1478696865


def plot_history(history):
    # print(history.history.keys())

    # 精度の履歴をプロット
    plt.plot(history.history['mean_squared_error'])
    plt.plot(history.history['val_mean_squared_error'])
    plt.title('model mean_squared_error')
    plt.xlabel('epoch')
    plt.ylabel('mean_squared_error')
    plt.legend(['mean_squared_error', 'val_mean_squared_error'], loc='lower right')
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    plt.show()

# 学習履歴をプロット
plot_history(history)

#モデルの概要を表示
#model.summary()






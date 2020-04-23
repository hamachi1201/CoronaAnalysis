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
data_set = pd.read_csv("Sample_Data.csv",sep=",",header=0)

data_set.head(3)


# In[ ]:





# In[ ]:





# In[3]:



# データの分割
(train, test) = train_test_split(data_set, test_size=0.2, shuffle=True)


#data_set.columns = ['time','country','case','cure','death','longitude','latitude']


x_train = train.loc[:, ['x1','x2','x3','x4']]
y_train = train.loc[:, ['S']]

x_test = test.loc[:, ['x1','x2','x3','x4']]
y_test = test.loc[:, ['S']]

#データの整形
x_train = x_train.astype(np.float)
x_test = x_test.astype(np.float)

y_train = y_train.astype(np.float)
y_test = y_test.astype(np.float)


# In[4]:


x_train


# In[6]:


x_test


# In[7]:


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import regularizers


#ニューラルネットワークの実装
model = Sequential()

model.add(Dense(20, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001), input_shape=(4,)))
#model.add(Dropout(0.1))

model.add(Dense(20, activation='relu',kernel_regularizer=keras.regularizers.l2(0.001), input_shape=(4,)))
#model.add(Dropout(0.1))

model.add(Dense(1, activation='linear'))



model.summary()
print("\n")



#ニューラルネットワークの実装②
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])


early_stopping = EarlyStopping(
                        monitor='val_loss',
                        min_delta=0.0,
                        patience=50,
                )

# val_lossの改善が20エポック見られなかったら、学習率を0.7倍する。
reduce_lr = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.7,
                        patience=20,
                        min_lr=0.001
                )

#ニューラルネットワークの学習
history = model.fit(x_train, y_train,batch_size=10,epochs=100,verbose=1,validation_data=(x_test, y_test),callbacks=[early_stopping, reduce_lr])

#ニューラルネットワークの推論
score = model.evaluate(x_test,y_test,verbose=1)
print("\n")
print("Test loss:",score[0])
print("Test mean_squared_error",score[1])


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


# In[ ]:


#新たなデータで学習したモデルを使ってみる


# In[8]:


#CSVファイルの読み込み
data_set = pd.read_csv("Predict_Data.csv",sep=",",header=0)

data_set.head(5)


# In[9]:


x_pred = data_set.loc[:, ['x1','x2','x3','x4']]
x_pred = x_pred.astype(np.float)

print(x_pred)


# In[10]:


#モデルを使って予測
predictions = model.predict(x_pred)
print(predictions)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





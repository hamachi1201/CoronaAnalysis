#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import re
import numpy as np


# In[2]:


df = pd.read_csv('./res20200416.csv', sep=',')


# In[3]:


print(df.head(3))


# In[4]:


print(df.columns.values)
rescolumn = df.columns.values


# In[5]:


data_np = np.asarray(df)
print(data_np)
print(data_np[0][1])


# In[19]:


print(len(data_np))


# In[6]:


p = re.compile(r'(.+)(-case)', re.IGNORECASE)
country = [] #国名リスト
 
for i in range(len(rescolumn)):
    if rescolumn[i].endswith("-case"):
        m = p.match(rescolumn[i])
        country.append(m.groups()[0])
        print(m.groups())
    else : 
        print("no")
    


# In[7]:


#地理データ取得
df_geo = pd.read_csv('./country_centroids_az8.csv', sep=',')
df_geo = df_geo[["name_sort","Longitude","Latitude","economy"]]


# In[8]:


print(df_geo.head(3))


# In[9]:


l_index = df_geo.reset_index().values.tolist()
print(l_index)


# In[10]:


match = [] #こちらだけ使う
mismatch = []

for i in l_index:
    info = []
    name = i[1]
    longi = i[2]
    lati = i[3]
    for j in range(len(country)):
        if name == country[j]:
            info.append(name)
            info.append(longi)
            info.append(lati)
            match.append(info)
            break
        elif j == len(country)-1:
            mismatch.append(name)
        
print(match)
print(mismatch)
            
        


# In[11]:


res_data = [] #日付、国名、case, cure, death

for i in range(len(data_np)): #日にち全てを回る。
    for j in range(len(country)): #国全てを回る。
        for one in match:
            if one[0] == country[j]:
                res_data_mini = []
                res_data_mini.append(data_np[i][0]) #日付
                res_data_mini.append(country[j]) #国名
                res_data_mini.append(data_np[i][j*3+1]) #case
                res_data_mini.append(data_np[i][j*3+2]) #cure
                res_data_mini.append(data_np[i][j*3+3]) #death
                res_data_mini.append(one[1]) #longitude
                res_data_mini.append(one[2]) #latitude
                res_data.append(res_data_mini)

print(res_data)


# In[14]:


print(len(res_data))


# In[17]:


#CSVファイル書き込み
import csv
label = ["time","country","case","cure","death","longitude","latitude"]
with open('data.csv', 'w') as f: 
    writer = csv.writer(f)
    writer.writerow(label)
    for data in res_data:
        writer.writerow(data)
        


# In[18]:


print(len(match))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





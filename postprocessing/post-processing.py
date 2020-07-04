#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import pickle
import json


# In[11]:


with open('../result.pkl', 'rb') as f:
    results = pickle.load(f)


# In[13]:


with open('../file_info.pkl', 'rb') as f:
    info = pickle.load(f)


# In[16]:


for i, image_results in enumerate(results):
    for j, class_detection in enumerate(image_results):
        for detection in class_detection:
            temp = dict()
            temp['score'] = float(detection[4])
            temp['tag'] = 1
            temp['box'] = [float(detection[0]), float(detection[1]), float(detection[2])-float(detection[0]), float(detection[3])-float(detection[1])]
            if detection[4]>0.5:
                info[i]['dtboxes'].append(temp)


# In[19]:


with open('submission.txt', 'w') as f:
    for r in info:
        f.write(json.dumps(r)+'\n')


# In[ ]:





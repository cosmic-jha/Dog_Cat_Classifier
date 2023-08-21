#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install opencv-python


# In[64]:


import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import TensorBoard


# In[12]:


import numpy as np
import os
import random
import matplotlib.pyplot as plt
import pickle


# In[20]:


File = r'C:\Users\jhaad\Downloads\archive\dogs_vs_cats\train'
subfiles = ['dogs','cats']


# In[107]:


img_size = 110
dataset = []


# In[43]:


for subfile in subfiles:
    folder = os.path.join(File, subfile)
    label = subfiles.index(subfile)
    for img in os.listdir(folder):
        img_path = os.path.join(folder, img)
        img_array = cv2.imread(img_path)
        img_array = cv2.resize(img_array, (img_size, img_size))
        dataset.append([img_array, label])


# 

# In[39]:


random.shuffle(dataset)


# In[94]:


a = [] 
b = []
for features, labels in dataset:
    a.append(features)
    b.append(labels)
a = np.array(a)
b = np.array(b)


# In[95]:


pickle.dump(a, open('a.pkl', 'wb'))
pickle.dump(b, open('b.pkl', 'wb'))


# In[105]:


c = pickle.load(open('a.pkl', 'rb'))
d = pickle.load(open('b.pkl', 'rb'))


# In[106]:


c = c/255
c


# In[60]:


model = Sequential()
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64, (3,3), activation = 'relu'))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128, input_shape = c.shape[1:], activation = 'relu'))
model.add(Dense(2, activation = 'softmax'))


# In[61]:


model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[62]:


model.fit(c, d, epochs = 5, validation_split = 0.1)


# In[68]:


model.save('3x3x64-catvsdog.model')


# In[113]:


def image(folder):
    img = cv2.imread(folder, cv2.IMREAD_COLOR)
    arr = cv2.resize(img, (img_size, img_size))
    arr = np.array(arr)
    arr = arr.reshape(-1, img_size, img_size, 3)
    return arr

model = keras.models.load_model('3x3x64-catvsdog.model')


# In[118]:


prediction = model.predict([image(r'C:\Users\jhaad\AppData\Local\Temp\Temp1_archive.zip\dogs_vs_cats\test\dogs\dog.38.jpg')])
print(subfiles[prediction.argmax()])


# In[ ]:





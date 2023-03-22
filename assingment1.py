
# coding: Ali Nur Ali

# assignment1 of mnista dataset classification


from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import math
from scipy.spatial.distance import cdist



# In[47]:


def image_griding(img):
    image_grid=np.empty((16, 7, 7))
    x_1 =0
    x_2 = 7
    i=0
    while i<16:
      image_grid[i]=img[x_1:x_2,0:7]
      i +=1
      image_grid[i]=img[x_1:x_2,7:14]
      i +=1
      image_grid[i]=img[x_1:x_2,14:21]
      i +=1
      image_grid[i]=img[x_1:x_2:,21:]
      i +=1
      x_1 +=7
      x_2 +=7
    return image_grid


# In[48]:


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train[0:10000]
y_train=y_train[0:10000]
x_test=x_test[0:1000]
y_test=y_test[0:1000]

print(x_train.shape)
print(y_test.shape)


# In[49]:


ef=image_griding(x_train[2])


# In[50]:


def display_img(mnist_index):
   
    image = mnist_index
    image = np.array(image, dtype='float')
    pixels = image.reshape((28, 28))
    plt.imshow(pixels, cmap='gray')
    plt.show()
    
display_img(x_train[0])


# In[51]:


def get_centroid(img):
    
    feature_vector = []
    d=0
    for grid in image_griding(img) :
        #print("qeyb qeyb",d)
        #d+=1
        
        Xc = 0 
        Yc = 0 
        sum = 0
        for i in range(len(grid)):
            for j in range(len(grid[i])):
              Xc +=grid[i][j]*i
              Yc +=grid[i][j]*j
              sum+=grid[i][j]
    
        if sum != 0 :
            feature_vector.append( Xc/ sum )
            feature_vector.append(Yc/ sum )
        else :
             feature_vector.append(0)
             feature_vector.append(0)
        
    return np.array(feature_vector)


# In[52]:


train_feature=[get_centroid(img) for img in x_train ]
train_feature = np.array(train_feature)
print(train_feature.shape)


# In[53]:


test_feature=[get_centroid(img) for img in x_test ]
test_feature = np.array(test_feature)
print(test_feature.shape)


# In[54]:


def Euclidean_distance(train_features,test_features):
    distances = cdist(test_features,train_features)
    return np.array(distances) 


# In[55]:


distance=np.empty((1000, 10000))
distance=Euclidean_distance(train_feature,test_feature)
distance.shape


# In[56]:


def get_min_index(distance):
    predicted_index = []
    for img in distance:
        ind = np.unravel_index(np.argmin(img, axis=None), img.shape)
        predicted_index.append(ind[0])
    return np.array(predicted_index)
        
    


# In[57]:


index=get_min_index(distance)
print(index.shape)


# In[58]:


def prediction(index,x_test):
    predicted=[]
    for item in index:
        predicted.append(x_test[item]) 
    return np.array(predicted)


# In[59]:


predicted=prediction(index,y_train)
print(predicted.shape)


# In[60]:


def Accuracy(predicted, y_test):
    total=len(predicted)
    score=0
    for i in range(len(predicted)):
        if(predicted[i]==y_test[i]):
            score+=1
    return (score/total)


# In[61]:


Accuracy=Accuracy(predicted, y_test)
Accuracy=str(round(Accuracy*100))+'%'
print("Accuracy is : ",Accuracy)


# In[62]:


for i in range(9):
    img = x_test[i]
    plt.subplot(330 + 1 + i)
    plt.imshow(img, cmap="Greys")
    plt.title(predicted[i])
plt.show()


# In[ ]:





# In[16]:





# In[17]:





# In[93]:


arr1 = np.array([[4,5,6,100,102,103,104,105],  [7,8,9,200,2001,2002,2003,2004],
                [10,11,12,300,3001,3002,3003,  3004],[13,14,15,400,4001,4002,4003,4004],
                [20,21,22,300,3001,3002,3003,  3004],[33,34,35,400,4001,4002,4003,4004]])
arr1.shape


# In[94]:


print(arr1[0:5,0:2])


# In[ ]:





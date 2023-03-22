# Ali Nur Ali
# coding: fdm dataset images classification assignment


import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import math
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


# In[5]:


# Load the FMD_dataset
data_path = "C:/fmd_dataset/image/"
categories = ["fabric", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood"]
X_train = []
X_test = []
y_train = []
y_test = []


# In[6]:


def Load_dataset(categories):
    for i, category in enumerate(categories):
        for j in range(5):
            img_path = data_path + category + "/" + category + "_moderate_00" + str(j+1) + "_new"  + ".jpg"
            img = cv2.imread(img_path, 0)
            if j < 3: 
                X_train.append(img)
                y_train.append(i)
            else: 
                X_test.append(img)
                y_test.append(i)


# In[12]:


def Coocrance_Matrix(img,angle):
    max_value = img.max()+1
    Co_matrix=[]
    matrix = np.zeros((max_value,max_value))
    for row in range(0,max_value):
        for col in range(0,max_value):
            #-------------------------------------
            counter=0
            if angle==0:
                for i in range(len(img)):
                    for j in range(len(img[i])-1):
                        a=str(img[i][j])+str(img[i][j+1])
                        b=str(row)+str(col)
                        if a==b:
                            counter+=1
                matrix[row][col]=counter
            elif angle==45:
                for i in range(len(img)-1):
                    for j in range(len(img[i])-1):
                        a=str(img[i][j])+str(img[i-1][j+1])
                        b=str(row)+str(col)
                        if a==b:
                            counter+=1
                matrix[row][col]=counter
            #-------------------------------------
        Co_matrix=matrix
    return np.array(Co_matrix)


# In[13]:


def matrix_coocurrence(img):
    max_value = 256
    matrix_coocurrence = graycomatrix(img, [1], [0], levels=max_value, normed=True, symmetric=False)
    matrix_coocurrence=matrix_coocurrence.reshape(max_value,max_value)
    return matrix_coocurrence


# In[14]:


def contrast(image):
    #-------------------------------------
    total=0
    for i in range(len(image)):
        for j in range(len(image[i])):
            total+=((i-j)**2)*(image[i][j])
        #-------------------------------------
    return total
          


# In[15]:


def Energy(image):
    #-------------------------------------
    total=0
    for i in range(len(image)):
        for j in range(len(image[i])):
            total+=(image[i][j])**2
        #-------------------------------------
    return total


# In[16]:


def Entrpy(image):
    #-------------------------------------
    total=0
    for i in range(len(image)):
        for j in range(len(image[i])):
            total+=(image[i][j])*(math.log2(5))
        #-------------------------------------
    return total


# In[17]:


def Homogeniety(image):
    #-------------------------------------
    total=0
    for i in range(len(image)):
        for j in range(len(image[i])):
            total+=(image[i][j])/(1+ abs(i-j))
        #-------------------------------------
    return total


# In[18]:


def glcm_mean_i_j(glcm):
    #-------------------------------------
    mean_i=0
    mean_j=0
    for i in range(len(glcm)):
        for j in range(len(glcm[i])):
            mean_i += (i+1)*(glcm[i][j])
            mean_j += (j+1)*(glcm[i][j])
        #-------------------------------------
    return mean_i,mean_j


# In[19]:


def glcm_segma_i_j(glcm):
    #-------------------------------------
    mean_i, mean_j=glcm_mean_i_j(glcm)
    segma_i=0
    segma_j=0
    for i in range(len(glcm)):
        for j in range(len(glcm[i])):
            segma_i += ( ((i+1)-mean_i)**2) * (glcm[i][j])
            segma_j += ( ((j+1)-mean_j)**2) *(glcm[i][j])
        #-------------------------------------
    return segma_i,segma_j


# In[20]:


def Correlation(glcm):
    #-------------------------------------
    mean_i, mean_j = glcm_mean_i_j(glcm)
    segma_i, segma_j = glcm_segma_i_j(glcm)
    Correlation=0
    for i in range(len(glcm)):
        for j in range(len(glcm[i])):
            Correlation += ( ((i+1)-mean_i)*((j+1)-mean_j)*(glcm[i][j]) )/(math.sqrt(segma_i*segma_j))
        #-------------------------------------
    return Correlation


# In[ ]:





# In[21]:


def Feature_Extraction(image):
    #-------------------------------------
    feature_vector=[]
    feature_vector.append(contrast(image))
    feature_vector.append(Energy(image))
    feature_vector.append(Entrpy(image))
    feature_vector.append(Homogeniety(image))
    feature_vector.append(Correlation(image))
    #-------------------------------------
    return np.array(feature_vector)


# In[22]:


Load_dataset(categories)
Train_Matrix=[matrix_coocurrence(img) for img in X_train]
Train_Features=[Feature_Extraction(img) for img in Train_Matrix] 
print(np.shape(Train_Features))


# In[23]:


Test_Matrix=[matrix_coocurrence(img) for img in X_test]
Test_Features=[Feature_Extraction(img) for img in Test_Matrix] 
print(np.shape(Test_Features))


# In[26]:


def Euclidean_distance(train_features,test_features):
    distances = cdist(test_features,train_features)
    return np.array(distances)


# In[27]:


distance=np.empty((18, 27))
distance=Euclidean_distance(Train_Features,Test_Features)
distance.shape


# In[28]:


def get_min_index(distance):
    predicted_index = []
    for img in distance:
        ind = np.unravel_index(np.argmin(img, axis=None), img.shape)
        predicted_index.append(ind[0])
    return np.array(predicted_index)


# In[29]:


index=get_min_index(distance)
print(index.shape)


# In[30]:


def prediction(index,y_test):
    predicted=[]
    for item in index:
        i=round(item/3)
        predicted.append(y_test[i]) 
    return np.array(predicted)


# In[31]:


predicted=prediction(index,y_test)
print(predicted.shape)


# In[32]:


def Accuracy(predicted, y_test):
    total=len(predicted)
    score=0
    for i in range(len(predicted)):
        if(predicted[i]==y_test[i]):
            score+=1
    return (score/total)


# In[33]:


Accuracy=Accuracy(predicted, y_test)
Accuracy=str(round(Accuracy*100))+'%'
print("Accuracy is : ",Accuracy)


# In[34]:


def Get_category(i):
    if i==0:
        return "fabric"
    elif i==1:
        return "glass"
    elif i==2:
        return "leather"
    elif i==3:
        return "metal"
    elif i==4:
        return "paper"
    elif i==5:
        return "plastic"
    elif i==6:
        return "stone"
    elif i==7:
        return "water"
    elif i==8:
        return "wood"


# In[ ]:





# In[35]:


for i in range(9):
    img = X_test[i]
    plt.subplot(330 + 1 + i+10 )
    plt.imshow(img, cmap="Greys")
    plt.title(Get_category(predicted[i]))
plt.show()


# In[292]:


#categories = ["fabric", "glass", "leather", "metal", "paper", "plastic", "stone", "water", "wood"]
            # [   0         1         2         3        4          5        6        7        8  ]  
print(y_train)
print(y_test)
print(index)
print(predicted)


# In[2]:


x =27/3

print(round(x))
# 3


# In[ ]:





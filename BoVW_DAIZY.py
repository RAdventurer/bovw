#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import os


# In[2]:


# Get the training classes names and store them in a list
#Here we use folder names for class names

#train_path = 'dataset/train'  # Names are Aeroplane, Bicycle, Car
train_path = 't'  # Folder Names are Parasitized and Uninfected
training_names = os.listdir(train_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0


# In[3]:


def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

#Fill the placeholder empty lists with image path, classes, and add class ID number
#
    
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1


# In[16]:


# Create feature extraction and keypoint detector objects
    #SIFT is not available anymore in openCV    
# Create List where all the descriptors will be stored
des_list = []

#BRISK is a good replacement to SIFT. ORB also works but didn;t work well for this example
brisk =  cv2.xfeatures2d.DAISY_create()

for image_path in image_paths:
    im = cv2.imread(image_path)
    fastdetector = cv2.FastFeatureDetector_create()
    kpts = fastdetector.detect(im, None)
    kpts, des = brisk.compute(im, kpts)
    des_list.append((image_path, des))   


# In[5]:


##descriptors = descriptors.astype(np.float64)


# In[6]:


# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]



# In[7]:


for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  


# In[14]:


print(type(descriptors))


# In[8]:


#kmeans works only on float, so convert integers to float
descriptors_float = descriptors.astype(float)  

# Perform k-means clustering and vector quantization
from scipy.cluster.vq import kmeans, vq

k = 200  #k means with 100 clusters gives lower accuracy for the aeroplane example
voc, variance = kmeans(descriptors_float, k, 1) 



# In[9]:


# Calculate the histogram of features and represent them as vector
#vq Assigns codes from a code book to observations.
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
#Standardize features by removing the mean and scaling to unit variance
#In a way normalization
from sklearn.preprocessing import StandardScaler
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)
#Train an algorithm to discriminate vectors corresponding to positive and negative training images
# Train the Linear SVM
from sklearn.svm import LinearSVC
clf = LinearSVC(max_iter=10000, dual=False)  #Default of 100 is not converging
clf.fit(im_features, np.array(image_classes))

#Train Random forest to compare how it does against SVM
#from sklearn.ensemble import RandomForestClassifier
#clf = RandomForestClassifier(n_estimators = 100, random_state=30)
#clf.fit(im_features, np.array(image_classes))


  
    


# In[10]:


# Save the SVM
#Joblib dumps Python object into one file

import joblib
joblib.dump((clf, training_names, stdSlr, k, voc), "bovwbrisk.pkl", compress=3)  



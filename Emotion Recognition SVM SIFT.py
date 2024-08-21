# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:48:25 2022

@author: Igor
"""
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision import datasets, transforms 
import os
import cv2
from sklearn import metrics
from skimage import feature
from torch.utils.data import DataLoader
import progressbar
from joblib import dump, load
from sklearn.svm import SVC
from os import listdir
import random
from sklearn.cluster import MiniBatchKMeans
from PIL import Image
import torch
from skimage import color, img_as_ubyte

os.environ['KMP_DUPLICATE_LIB_OK']='True'

PATH = 'C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/emotionRecognition SIFT_SVM.joblib'

widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('#'),' (',
           progressbar.ETA(), ') ',
          ]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

emotions = ['Anger','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']

# imshow function from Computer Vision Tutorial 8
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))     # Reshape: C, H, W -> H, W, C
    plt.show()

# Creates the data loaders from the testing and trainging data
trainset = datasets.ImageFolder(root="C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/train",transform = transforms.ToTensor())  
trainloader = DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)

testset = datasets.ImageFolder(root="C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/test",transform = transforms.ToTensor())  
testloader = DataLoader(testset, batch_size=4,shuffle=True, num_workers=0)


 
sift = cv2.SIFT_create()
trainModel = 'n'

# Training the model using the SIFT BOVW method from lecture 7 
if trainModel == 'y':
    # Create empty lists for feature descriptors and labels
    des_list = []
    y_train_list = []
     
    fig, ax = plt.subplots(1, 4, figsize=(10, 8), sharey=True)
     
    bar = progressbar.ProgressBar(max_value=len(trainloader), widgets=widgets).start()
    total = 0
    for data in trainloader:
        daiter = iter(trainloader)
        images, labels = dataiter.next()
        # Identify keypoints and extract descriptors with SIFT
        
        labels = labels.tolist()
        images = np.transpose(images, (0,2,3,1))
        
        counter = 0
        ta
        for image in images:
            img = img_as_ubyte(color.rgb2gray(image))
            kp, des = sift.detectAndCompute(img, None)
    
            # Show results for first 4 images
            if total<1:
                img_with_SIFT = cv2.drawKeypoints(img, kp, img)
                ax[counter].imshow(img_with_SIFT)
                ax[counter].set_axis_off()
        
            # Append list of descriptors and label to respective lists
            if des is not None:
                des_list.append(des)
                y_train_list.append(labels[counter])
            
            counter += 1
        total += 1
        bar.update(total)
        
    fig.tight_layout()
    plt.show()
     
    # Convert to array for easier handling
    des_array = np.vstack(des_list)
     
    # Number of centroids/codewords: good rule of thumb is 10*num_classes
    k = len(np.unique(y_train_list)) * 10
     
    # Use MiniBatchKMeans for faster computation and lower memory usage
    batch_size = des_array.shape[0] // 4
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=batch_size).fit(des_array)
    # Convert descriptors into histograms of codewords for each image
    hist_list = []
    idx_list = []
    
    
    bar2 = progressbar.ProgressBar(max_value=len(des_list), widgets=widgets).start()
    total = 0
    for des in des_list:
        hist = np.zeros(k)
    
        idx = kmeans.predict(des)
        idx_list.append(idx)
        for j in idx:
            hist[j] = hist[j] + (1 / len(des))
        hist_list.append(hist)
        total += 1
        bar2.update(total)
    
    hist_array = np.vstack(hist_list)
    
    # Create a classifier: a support vector classifier
    classifier = SVC(kernel='rbf', verbose = 1, max_iter = 100, tol=1e-5)
    
    # We learn the digits on the first half of the digits
    classifier.fit(hist_array, y_train_list)
    
    dump(classifier, PATH)

# Testing the method based on the code from Tutorial 7    
elif trainModel == 'n':   
    des_list = []
    hist_list = []
    y_test = []
    
    bar = progressbar.ProgressBar(max_value=len(testloader), widgets=widgets).start()
    total = 0
    for data in testloader:
        dataiter = iter(testloader)
        images, labels = dataiter.next()
       
        images = np.transpose(images, (0,2,3,1))
        
        # Applies SIFT to extract a list of keypoints from each image
        for i in range(len(images)):
            img = img_as_ubyte(color.rgb2gray(images[i]))
            kp, des = sift.detectAndCompute(img, None)
            
            # Checks for empty descriptors
            if des is not None:
                des_list.append(des)  
                y_test.append(labels[i])
                
        total += 1
        bar.update(total)
                
    des_array = np.vstack(des_list)  
    
    batch_size = des_array.shape[0] // 4
    kmeans = MiniBatchKMeans(n_clusters=70, batch_size=batch_size).fit(des_array)
    
    bar2 = progressbar.ProgressBar(max_value=len(des_list), widgets=widgets).start()
    total = 0
    for des in des_list:
        hist = np.zeros(70)
        idx = kmeans.predict(des)
        for j in idx:
            hist[j] = hist[j] + (1 / len(des))
        hist_list.append(hist)
        
        total += 1
        bar2.update(total)
        
    hist_array = np.vstack(hist_list)
    
    classifier = load(PATH) 
    y_pred = classifier.predict(hist_array).tolist()
    
    # Used for providing results, based on Computer Vision Lab 8
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # show images and print labels
    imshow(torchvision.utils.make_grid(images))
    first_labels = [emotions[labels[j]] for j in range(4)]
    print('Ground-truth:', first_labels)
    
    first_predicted = [emotions[y_pred[j]] for j in range(4)]
    print('Predicted:', first_predicted)
    
    # Estimate average accuracy
    correct = 0
    total = 0
    
    for i in range(len(y_pred)):
        total += 1
        correct += (y_pred[i] == y_test[i])
    
    print(f"Accuracy of the network on the test images: {100 * correct / total}%")
    
    # Estimate class-wise accuracy
    class_correct = list(0. for i in range(7))
    class_falsePositive = list(0. for i in range(7))
    class_falseNegative = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    for i in range(len(y_pred)):
        c = (y_pred[i] == y_test[i]).squeeze()
        label = y_test[i]
        class_correct[label] += c.item()
        if c.item() == 0:
            class_falsePositive[y_pred[i]] += 1
        class_total[label] += 1
    
    print(f"""Classification report for classifier {classifier}:\n
      {metrics.classification_report(y_test, y_pred)}""")
    for i in range(7):
        print(f"Accuracy of {emotions[i]}: {100 * class_correct[i] / class_total[i]}%")
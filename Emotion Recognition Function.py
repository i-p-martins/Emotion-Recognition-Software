# -*- coding: utf-8 -*-
"""
Created on Wed May  4 22:15:34 2022

@author: Igor
"""
# Import the necessary libraries
import cv2
import joblib
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import os
import PIL
import random
from skimage import color, img_as_ubyte, transform, feature
from sklearn import svm, metrics
from sklearn.cluster import MiniBatchKMeans
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms 
from torch.utils.data import DataLoader, TensorDataset


# Assign some important static variables
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
emotions = ['Anger','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']
sift = cv2.SIFT_create()
defaultPath = "C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/test"
defaultPath2 = "C:\\Users\\Igor\\Documents\\University Materials\\Year 3 (2021-2022)\\IN3060 - Computer Vision\\coursework\\CW_Dataset CNN\\test"

# Defines the CNN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(7744, 774)
        self.fc2 = nn.Linear(774, 77)
        self.fc3 = nn.Linear(77, 7)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 7744)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# imshow function from Computer Vision Tutorial 8, used for testing purposes
def imshow(img):
    npimg = img.numpy()
    plt.imshow(npimg)     # Reshape: C, H, W -> H, W, C
    plt.show()

# Applies the HOG identifier to all the images in the provided data loader
# returns the required data to run the CNN
def applyHog(loader):       
    hogList = []
    labelList = []
    
    for data in loader:
        dataiter = iter(loader)
        images, labels = dataiter.next()
       
        labels = labels.tolist()
        images = np.transpose(images, (0,2,3,1))

        # Create feature descriptors for each image in the batch using the  
        # HOG algorithm and convert them into a shape that can be processed
        # by the CNN.
        for image in images:
            image = cv2.resize(np.array(image), (128, 256))
            fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, channel_axis=-1)
            
            fd = np.array(fd)
            hogList.append(fd)
        
        for label in labels:
            labelList.append(label)
        
    length = len(hogList)
    HogArray = np.asarray(hogList, dtype = np.float32)
    
    labelArray = np.array(labelList)

    return HogArray, labelArray, length

# Creates the histogram used to train the SVM and returns the needed data
# Based on the Computer Vision Lab 7
def CreateHist(loader):
    des_list = []
    hist_list = []
    
    for data in loader:
        dataiter = iter(loader)
        images, labels = dataiter.next()
       
        y_test = [labels.tolist()]
        
        # Applies SIFT to extract a list of keypoints from each image
        for image in images:
            img = img_as_ubyte(color.rgb2gray(image))
            kp, des = sift.detectAndCompute(img, None)
            
            # Checks for empty descriptors
            if des is not None:
                des_list.append(des)  
                
    des_array = np.vstack(des_list)  
    
    batch_size = des_array.shape[0] // 4
    kmeans = MiniBatchKMeans(n_clusters=70, batch_size=batch_size).fit(des_array)
    
    for des in des_list:
        hist = np.zeros(70)
        idx = kmeans.predict(des)
        for j in idx:
            hist[j] = hist[j] + (1 / len(des))
        hist_list.append(hist)
    
    hist_array = np.vstack(hist_list)
            
    return hist_array, y_test

# The emotion recognition function
def EmotionRecognition(path_to_testset, model_type):
    folders = os.listdir(path_to_testset)
    imgList = []
    labelList = []
    displayList = []
    faceList = []
    
    # Randomly selects 4 images from the testing set and loads them in
    for i in range(4):
        folder = random.choice(folders)
        labelList.append(emotions.index(folder))
        path = path_to_testset + "/" + folder
        files = os.listdir(path)
        image = random.choice(files)
        imagePath = path + "/" + image
        img = PIL.Image.open(imagePath)
        img = np.asarray(img)/255
         
        # checks if the images loaded in are from the provided dest set
        # or from a personal test set
        if path_to_testset != defaultPath and path_to_testset != defaultPath2:
            displayList.append(img)
            img_gray = color.rgb2gray(img)
            img_gray = img_as_ubyte(img_gray)  
            # looks for faces in the image
            face_cascade_frontal = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade_frontal.detectMultiScale(img_gray, 1.3, 5)
            face = faces[0]
            faceList.append(face)
            # crops the image to only include the face and crops it to a 
            # size of 100x100 just like the provided set
            image = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
            img = cv2.resize(image, dsize=(100, 100), interpolation=cv2.INTER_CUBIC)
        
        # converts the images to tensors
        img = torch.Tensor(img)
        imgList.append(img)
        
    tensor_X = torch.stack(imgList)
    tensor_y = torch.tensor(labelList)
        
    # creates the data loaders ready for applying whichever identifier
    # is chosen
    testset = TensorDataset(tensor_X,tensor_y)
    testloader = DataLoader(testset, batch_size=4,shuffle=False, num_workers=0)

    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # SVM with SIFT
    if model_type == 1:
        model = joblib.load('SIFT_SVM.joblib')
        HistList, y_test = CreateHist(testloader)
        y_pred = model.predict(HistList).tolist()
        predicted = []
        for i in y_pred:
            predicted.append(emotions.index(i))
    
    # SVM with HOG
    elif model_type == 2:
        model = joblib.load('emotionRecognition HOG_SVM.joblib')
        X_test, y_test, testLength = applyHog(testloader)
        y_pred = model.predict(X_test)
        predicted = []
        for i in y_pred:
            predicted.append(emotions.index(i))
        
    # CNN with HOG        
    elif model_type == 3:
        model = Net()
        PATH = "C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/emotionRecognitionCNN.pth"
        model.load_state_dict(torch.load(PATH))
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
    else: 
        print("This model is not available")
       
    # Outputs the four images with predictions and actual labels as titles
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 5), sharex=True, sharey=True)
    ax = axes.ravel() 
    
    for i in range(4):
        ax[i].imshow(images[i])
        ax[i].set_title(f'Label: {emotions[int(y_test[0][i])]} \n Prediction: {emotions[predicted[i]]}')
        ax[i].set_axis_off()
    fig.tight_layout()
    plt.show()
    
    # if the image is from a personal dataset outputs the four images
    # with a bounding box and the prediction underneath
    if path_to_testset != defaultPath and path_to_testset != defaultPath2:
        for i in range(0,len(displayList)):
            offset = (displayList[i].shape)[0]/30
            fig, ax = plt.subplots(figsize=(18, 12))
            ax.imshow(displayList[i]), ax.set_axis_off()
            ax.add_patch(patches.Rectangle(xy=(faceList[i][0], faceList[i][1]), width=faceList[i][2], height=faceList[i][3], fill=False, color='r', linewidth=5))
            fig.tight_layout
            plt.text(faceList[i][0],faceList[i][1]+faceList[i][3]+offset,emotions[predicted[i]], c= 'red', size = offset)
            plt.show()
            
# Takes user input to decide which classifier to use     
path_to_testset = input("Path to test set: ")
print("\nSIFT identifier SVM classifier - 1")
print("SIFT identifier CNN classifier - 2")
print("HOG identifier CNN classifier - 3")
model_type = int(input("Choose model type: "))
EmotionRecognition(path_to_testset,model_type)

# C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/test
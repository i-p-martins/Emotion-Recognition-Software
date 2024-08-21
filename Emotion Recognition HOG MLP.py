# -*- coding: utf-8 -*-
"""
Created on Tue May 10 15:24:55 2022

@author: Igor
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May  4 17:08:45 2022

@author: Igor
"""
# Import the necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import time
import torchvision
from torchvision import datasets, transforms 
import os
import cv2
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from skimage import color, transform, feature, exposure
from torch.utils.data import DataLoader
import progressbar
from joblib import dump, load
import prettytable
from sklearn.svm import LinearSVC

os.environ['KMP_DUPLICATE_LIB_OK']='True'

PATH = 'C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/emotionRecognition HOG_.joblib'

widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('#'),' (',
           progressbar.ETA(), ') ',
          ]

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

emotions = ['Anger','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']

# Applies the HOG identifier to all the images in the provided data loader
# returns the required data to run the CNN
def applyHog(loader):       
    hogList = []
    labelList = []

    Hogbar = progressbar.ProgressBar(max_value=len(loader), widgets=widgets).start()
    counter=0
    for i in range(len(loader)):
        dataiter = iter(loader)
        images, labels = dataiter.next()
       
        labels = labels.tolist()
        images = np.transpose(images, (0,2,3,1))

        # Create feature descriptors for each image in the batch using the  
        # HOG algorithm and convert them into a shape that can be processed
        # by the CNN.
        fdList = []
        for image in images:
            image = cv2.resize(np.array(image), (128, 256))
            fd, hog_image = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),cells_per_block=(2, 2), visualize=True, channel_axis=-1)
            
            fd = np.array(fd)
            hogList.append(fd)
        
        for label in labels:
            labelList.append(label)
        
        counter+=1
        Hogbar.update(counter)
        
    
    # Converts the lists of feature descriptor batches into tensors and
    # reshapes them into the shape needed by the CNN
    length = len(hogList)
    HogArray = np.asarray(hogList, dtype = np.float32)
    
    labelArray = np.array(labelList)

    return HogArray, labelArray, length

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

# classifier = MLPClassifier(hidden_layer_sizes=(545,68,7), max_iter=100, alpha=1e-4,
#                     solver='sgd', verbose=True, random_state=1,
#                     learning_rate_init=.1)

classifier = LinearSVC(random_state=42, tol=1e-5, verbose=1, max_iter = 100)

trainModel = 'n'
            
if trainModel == 'y':
    X_train, y_train, trainLength = applyHog(trainloader)
    
    classifier.fit(X_train, y_train)
    dump(classifier, PATH)
    
elif trainModel == "n":   
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # show images and print labels
    imshow(torchvision.utils.make_grid(images))
    first_labels = [emotions[labels[j]] for j in range(4)]
    print('Ground-truth:', first_labels)
    
    X_test, y_test, testLength = applyHog(testloader)
    
    classifier = load(PATH) 
    y_pred = classifier.predict(X_test)
    
    # Used for providing results, based on Computer Vision Lab 8
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # show images and print labels
    imshow(torchvision.utils.make_grid(images))
    first_labels = [emotions[labels[j]] for j in range(4)]
    print('Ground-truth:', first_labels)
    
    y_pred = classifier.predict(X_test)
    first_predicted = [emotions[y_pred[j]] for j in range(4)]
    print('Predicted:', first_predicted)
    
    # Estimate average accuracy
    correct = 0
    total = 0
    
    for i in range(len(y_pred)):
        total += 1
        correct += (y_pred[i] == y_test[i]).sum().item()
    
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


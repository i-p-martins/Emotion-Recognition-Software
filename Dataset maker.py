# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 20:30:09 2022

@author: Igor
"""
# Import the necessary libraries
import shutil
from os import listdir,mkdir

# Open the path to the labels
with open('C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/labels/list_label_test.txt') as f:
    testLabels = [[s.strip()] for s in f] 
    
# Creates a list of the destination of all the images based on the last
# character in the labels text file
destinationFolder = []
for ele in testLabels:
    destinationFolder.append(int(ele[0][-1])-1)

# list containing the emotion labels
emotions = ['Surprise','Fear','Disgust','Happiness','Sadness','Anger','Neutral']

# Open the path to the images
imagesList = listdir("C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/test/")

# Open the path to the new folder for storing the sorted images
path = "C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/"

# Makes a folder for each emotion
for i in emotions:
    mkdir(path + i)

# Stores each image in the correct folder
for i in range (0, len(imagesList)):
    destination = "C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/" + emotions[destinationFolder[i]]
    original = "C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/test/" + imagesList[i]
    print(f"Destination: {destination}")
    print(f"Image: {imagesList[i]}")
    shutil.move(original, destination)
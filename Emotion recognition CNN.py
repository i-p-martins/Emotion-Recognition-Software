import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torchvision
from torchvision import datasets, transforms 
from torch.utils import data  
import progressbar
import os
import prettytable

os.environ['KMP_DUPLICATE_LIB_OK']='True'

widgets = [' [',
         progressbar.Timer(format= 'elapsed time: %(elapsed)s'),
         '] ',
           progressbar.Bar('#'),' (',
           progressbar.ETA(), ') ',
          ]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

PATH = 'C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/emotionRecognitionCNN.pth'

train = datasets.ImageFolder(root="C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/train",transform = transforms.ToTensor())  
split = int(len(train)*0.9)
trainset, validset = data.random_split(train,[split,len(train)-split])
trainloader = data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0)
validloader = data.DataLoader(validset, batch_size=4,shuffle=True, num_workers=0)

testset = datasets.ImageFolder(root="C:/Users/Igor/Documents/University Materials/Year 3 (2021-2022)/IN3060 - Computer Vision/coursework/CW_Dataset CNN/test",transform = transforms.ToTensor())  
testloader = data.DataLoader(testset, batch_size=4,shuffle=True, num_workers=0)


emotions = ['Anger','Disgust','Fear','Happiness','Neutral','Sadness','Surprise']

# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # Unnormalize: back to range [0, 1] just for showing the images
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))     # Reshape: C, H, W -> H, W, C
    plt.show()

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

trainModel = 'n'

if trainModel == 'y':
    net = Net()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9)
    
    t0 = time.time()
    
    epochs = 5
    min_valid_loss = np.inf
    
    # Based on Computer Vision Tutorial 8 and Geeks for Geeks 
    # Training Neural Networks with Validation using PyTorch
    
    for e in range(epochs):  # loop over the training set 20 times
        
        Trainbar = progressbar.ProgressBar(max_value=len(trainloader), widgets=widgets).start()
        counter=0
    
        running_loss = 0.0
        net.train()
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            # print statistics (loss.item() returns the mean loss in the mini-batch)
            running_loss += loss.item()
    
            counter += 1
            Trainbar.update(i)
        
        Validbar = progressbar.ProgressBar(max_value=len(validloader), widgets=widgets).start()
        counter=0    
        
        valid_loss = 0.0
        net.eval()
        for i, data in enumerate(validloader, 0):
            inputs, labels = data
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            valid_loss = loss.item() * inputs.size(0)
            
            counter += 1
            Validbar.update(counter)  
           
        print(f'Epoch {e+1} \t\t Training Loss: {running_loss / len(trainloader)} \t\t Validation Loss: {valid_loss / len(validloader)}')
        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss
            # Saving State Dict
            torch.save(net.state_dict(), PATH)
            
    print('Finished Training: total time in seconds =', time.time() - t0)



elif trainModel == 'n':
    net = Net()
    net.load_state_dict(torch.load(PATH))
    
    # provides results, based on Computer Vision Lab 8
    
    dataiter = iter(testloader)
    images, labels = dataiter.next()
    
    # show images and print labels
    imshow(torchvision.utils.make_grid(images))
    first_labels = [emotions[labels[j]] for j in range(4)]
    print('Ground-truth:', first_labels)
    
    outputs = net(images)
    
    _, predicted = torch.max(outputs, 1)
    first_predicted = [emotions[predicted[j]] for j in range(4)]
    print('Predicted:', first_predicted)
    
    # Estimate average accuracy
    correct = 0
    total = 0
    with torch.no_grad():             # Avoid backprop at test 
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Accuracy of the network on the test images: {100 * correct / total}%")
    
    # Estimate class-wise accuracy
    class_correct = list(0. for i in range(7))
    class_falsePositive = list(0. for i in range(7))
    class_falseNegative = list(0. for i in range(7))
    class_total = list(0. for i in range(7))
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                if c[i].item() == 0:
                    class_falsePositive[predicted[i]] += 1
                class_total[label] += 1
    
    x = prettytable.PrettyTable(["Emotion", "Accuracy", "Precision"])
    denom = class_correct[0] + class_falsePositive[0]
    for i in range(7):
        x.add_row([emotions[i],str(round(100 * class_correct[i] / class_total[i], 2)) + "%",str(round(100 * class_correct[i] / (class_correct[i]+class_falsePositive[i]),2)) + "%"])
        # print(f"{emotions[i]}:\t\tAccuracy: {round(100 * class_correct[i] / class_total[i], 2)}%\tPrecision: {round(100 * class_correct[i] / (class_correct[i]+class_falsePositive[i]),2)}%")
    print(x)
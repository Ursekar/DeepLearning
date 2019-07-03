#import sys
#sys.modules[__name__].__dict__.clear()

import argparse

import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import numpy as np

#import helper
from checkpoint import save_checkpoint, load_checkpoint, category_names

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--data_dir', action='store')
    parser.add_argument('--arch', dest='arch', default='densenet121', choices=['vgg13', 'densenet121'])
    parser.add_argument('--learning_rate', dest='learning_rate', default='0.001')
    parser.add_argument('--hidden_units', dest='hidden_units', default='512')
    parser.add_argument('--epochs', dest='epochs', default='3')
#    parser.add_argument('--gpu', action='store', default='gpu')
    parser.add_argument('--print_every', dest='print_every', default='10', help='No. of steps o/p display in each epoch')
    parser.add_argument('--save_dir', dest="save_dir", action="store", default="checkpoint.pth")
    return parser.parse_args()

def train(model, criterion, epochs, optimizer, print_every, trainloader, valloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
#    criterion = nn.NLLLoss()
    criterion = criterion
    optimizer = optimizer

#    epochs = 3
    epochs = epochs
    steps = 0
    running_loss = 0
    print_every = print_every

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                for inputs, labels in valloader:
                    optimizer.zero_grad()
                    inputs, labels = inputs.to(device), labels.to(device)
                    with torch.no_grad():
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                        val_loss += batch_loss.item()
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"val loss: {val_loss/len(valloader):.4f}.. "
                      f"val accuracy: {accuracy/len(valloader):.3f}")
                running_loss = 0
                #model.train()
def main():
    #I must know that atleast something is running
    print("Please wait while I train")
    
    args = parse_args()
    #path of data directories
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    val_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    #transformations to be applied on dataset
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])
    val_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], 
                                                                  [0.229, 0.224, 0.225])])
    
    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)
    val_datasets = datasets.ImageFolder(val_dir, transform=val_transforms)
    
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_datasets, batch_size = 64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_datasets, batch_size = 64, shuffle=True)
    
    #print(summary(trainloaders))
    #image, label = next(iter(trainloader))
    #helper.imshow(image[0,:]);
    
    #defining parameters that will be passed as default to the model under training
    
    model = getattr(models, args.arch)(pretrained=True)
    
    #choose out of two models
    if args.arch == 'vgg13':
    # TODO: Build and train your network
        model = models.vgg13(pretrained=True)
        print(model)
        for param in model.parameters():
            param.requires_grad = False
    
        classifier = nn.Sequential(nn.Linear(25088, 4096),
                               nn.Dropout(p=0.2),
                               nn.ReLU(),
                               nn.Linear(4096, 4096),
                               nn.ReLU(),
                               nn.Dropout(p=0.2),
                               nn.Linear(4096,102),
                               nn.LogSoftmax(dim=1))
        model.classifier= classifier
   
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        print(model)
        for param in model.parameters():
            param.requires_grad = False
    
        classifier = nn.Sequential(nn.Linear(1024, 512),
                               nn.Dropout(p=0.6),
                               nn.ReLU(),
                               nn.Linear(512, 256),
                               nn.ReLU(),
                               nn.Dropout(p=0.6),                               
                               nn.Linear(256,102),
                               nn.LogSoftmax(dim=1))
        model.classifier = classifier
    
    model.classifier = classifier
    criterion = nn.NLLLoss()
    epochs = int(args.epochs)
    learning_rate = float(args.learning_rate)
    print_every = int(args.print_every)
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    train(model, criterion, epochs, optimizer, print_every, trainloader, valloader)
    model.class_to_idx = train_datasets.class_to_idx        
    path = args.save_dir
    save_checkpoint(args, model, optimizer, learning_rate, epochs, path)

if __name__ == "__main__":
    main()
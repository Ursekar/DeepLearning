import argparse

import torch
from torch.autograd import Variable
from torchvision import transforms, models
import torch.nn.functional as F
import numpy as np

from PIL import Image

import json
import os
import random

from checkpoint import load_checkpoint, category_names

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default=3)
    parser.add_argument('--filepath', dest='filepath', default='flowers/test/1/image_06743.jpg') # using default filepath of primrose image 
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
#    parser.add_argument('--gpu', action='store', default='gpu')
    return parser.parse_args()

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    #adjustments = transforms.Compose(
    #    [
    #        transforms.Resize(256),
    #        transforms.CenterCrop(224),
    #        transforms.ToTensor(),
    #        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    #    ]
    #)

    #image = adjustments(image)    
    # TODO: Process a PIL image for use in a PyTorch model
    size = [0, 0]
    if image.size[0] > image.size[1]:
        size = [image.size[0], 256]
    else:
        size = [256, image.size[1]]
    
    image.thumbnail(size, Image.ANTIALIAS)    
    w, h = image.size  

    l = (256 - 224)/2
    t = (256 - 224)/2
    r = (256 + 224)/2
    b = (256 + 224)/2

    image = image.crop((l, t, r, b))
    image = np.array(image)
    image = image/255.
                       
    mean = np.array([0.485, 0.456, 0.406])
    sd = np.array([0.229, 0.224, 0.225])                       
    image = ((image - mean) / sd)    
    image = np.transpose(image, (2, 0, 1))
    
    return image

def predict(image_path, model, topk=3):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # TODO: Implement the code to predict the class from an image file
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    img_torch = Image.open(image_path) # process the image, from reviewer advice
    img_torch = process_image(img_torch)
    img_torch = torch.from_numpy(img_torch)
    
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
 
    with torch.no_grad():
        output = model.forward(img_torch.cuda()) # use cuda
        
    probability = F.softmax(output.data,dim=1) # use F
    
    probs = np.array(probability.topk(topk)[0][0])
   
    index_to_class = {val: key for key, val in model.class_to_idx.items()} # from reviewer advice
    top_classes = [np.int(index_to_class[each]) for each in np.array(probability.topk(topk)[1][0])]
    
    return probs, top_classes

def main():
    args = parse_args()
    model = load_checkpoint(args.checkpoint)
    categories = category_names(args.category_names)
    image_path = args.filepath
    model.eval()
    probs,classes = predict(image_path, model, topk = args.top_k)
    names = [categories[str(index)] for index in classes]
    print(probs)
    print(names)
    
    print('File selected: ' + image_path)
    i=0
    while i < len(names):
        print("{} with a probability of {}".format(names[i], probs[i]))
        i = i+1
        
if __name__ == "__main__":
    main()
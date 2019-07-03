import argparse

import torch
from torchvision import transforms, datasets
import torchvision
import copy
import os
import json
 
def save_checkpoint(args, model, optimizer, learning_rate, epochs, path):
    checkpoint = {'arch': args.arch,
                  'model': model,
                  'hidden_units': args.hidden_units,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                 'class_to_idx': model.class_to_idx,
                 'learning_rate': learning_rate,
                 'epochs': epochs,
                 'optimizer':  optimizer.state_dict()}
    
    torch.save(checkpoint, path)

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['classifier']
    learning_rate = checkpoint['learning_rate']
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def category_names(c):
    import json
#    with open('cat_to_name.json', 'r') as f:
    with open(c, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
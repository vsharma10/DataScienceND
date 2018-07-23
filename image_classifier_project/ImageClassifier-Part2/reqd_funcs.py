# Udacity Image Classifier Project: Part 2
# Building a command line applications: required functions
################################
# Created by Vivekanand Sharma #
# Dated July 21, 2018          #
################################

import matplotlib.pyplot as plt
import time
import json
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

# Define transforms for the training, validation, and testing sets
def trainer(arch, data_dir, save_dir, learning_rate, num_epochs, hidden_units, processor):
    model_names = models.__dict__

    if model_names.get(arch, 0) != 0:
        model = models.__dict__[arch](pretrained=True)
    else:
        print("No valid model specified...Using DEFAULT")

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'

    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset  = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    train_dataloaders = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_dataloaders = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(hidden_units, 512)),
                          ('relu', nn.ReLU()),
                          ('drop2', nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier

    #Verify availability of GPU
    use_gpu = torch.cuda.is_available()
    print("GPU available:", use_gpu)
    print("Start Training")

    #Train network-1
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    if processor == "GPU":
        device = "cuda:0"
        #model.cuda()
    else:
        device = "cpu"
        #model.cpu()

    #Train network
    epochs = int(num_epochs)
    steps = 0
    running_loss = 0
    print_every = 40

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for e in range(epochs):
        model.train()
        for ii, (images, labels) in enumerate(train_dataloaders):
            steps += 1

            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Forward and backward passes
            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, valid_dataloaders, criterion, processor)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(valid_dataloaders)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(valid_dataloaders)))

                running_loss = 0
                # Make sure training is back on
                model.train()

    print("Training complete")

    # Save the checkpoint
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
        
    model.class_to_idx = train_dataset.class_to_idx

    checkpoint_dict = {
        'arch': arch,
        'class_to_idx': model.class_to_idx,
        'state_dict': model.state_dict(),
        'epochs': epochs + 1,
        'learning_rate': learning_rate,
        'optimizer_state': optimizer.state_dict(),
        'criterion_state': criterion.state_dict()
        }

    torch.save(checkpoint_dict, 'checkpoint.pth')
    print("Checkpoint saved!")

    return model


# Implement a function for the validation pass
def validation(model, testloader, criterion, processor):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if processor == "GPU":
        device = "cuda:0"
    else:
        device = "cpu"

    test_loss = 0
    accuracy = 0
    model.to(device)
    for images, labels in testloader:

        images, labels = images.to(device), labels.to(device)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy

# Load a checkpoint and rebuilds the model
def load_checkpoint(file):
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(25088, 4096)),
                          ('relu', nn.ReLU()),
                          ('drop1', nn.Dropout(p=0.2)),
                          ('fc2', nn.Linear(4096, 512)),
                          ('relu', nn.ReLU()),
                          ('drop2', nn.Dropout(p=0.2)),
                          ('fc3', nn.Linear(512, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    chkpt = torch.load(file, map_location='cpu')
    model.load_state_dict(chkpt['state_dict'])
    #model.load_state_dict(chkpt['optimizer_state'])
    #model.load_state_dict(chkpt['criterion_state'])
    model.class_to_idx = chkpt['class_to_idx']
    
    return model

# Process image
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''  
    # Process a PIL image for use in a PyTorch model
    image = Image.open(image_path)
    image_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])
    image_processed = image_transforms(image)
    
    return image_processed


# Return top K most likely classes and probabilities
def predict(image_path, checkpoint, topk, cat_to_name, processor):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = load_checkpoint(checkpoint)
    # TODO: Implement the code to predict the class from an image file
    # Turn on Cuda if available
    #Verify availability of GPU
    use_gpu = torch.cuda.is_available()
    
    print("GPU available:", use_gpu)

    if processor == "GPU":
        device = "cuda:0"
        #model.cuda()
    else:
        device = "cpu"

    model.eval()
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Process image into inputs and run through model
    img = process_image(image_path).unsqueeze(0)
    outputs = model.forward(img)  
    
    # Get Probabilities and Classes
    probs, classes = outputs.topk(topk)
    probs = probs.exp().data.numpy()[0]
    #print(np.sort(probs))
    #print(classes)
    classes = classes.data.numpy()[0]
    class_keys = {x:y for y, x in model.class_to_idx.items()}
    classes = [class_keys[i] for i in classes]

    y = np.arange(len(classes))
    y_labels = [cat_to_name.get(i) for i in classes[::-1]]
    y_labels.reverse()
 
    return probs, classes, y_labels

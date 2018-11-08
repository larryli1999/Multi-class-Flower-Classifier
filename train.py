import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict

import argparse
import json

gpu_ava = torch.cuda.is_available()

def main():
    parser = argparse.ArgumentParser(description = 'Training Image Classifier')
    parser.add_argument('data_dir', action = 'store', type = str, default = 'flowers', help = 'Set data directory')
    parser.add_argument('--save_dir', type = str, default = 'new_checkpoint.pth', help = 'Set directory to save checkpoints')
    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Choose architecture')
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'Set learning rate')
    parser.add_argument('--hidden_units', type = int, default = 200, help = 'Set number of hidden units')
    parser.add_argument('--epochs', type = int, default = 3, help = 'Set number of epochs')
    parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU for training if avaliable')
    args = parser.parse_args()
    
    
    train_model(args)
    
def train_model(args):
    data_dir = args.data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.Resize(255),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                     [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                               [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], 
                                               [0.229, 0.224, 0.225])])
    
    image_data = dict()
    image_data['train'] = datasets.ImageFolder(train_dir,transform = train_transforms)
    image_data['valid'] = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    image_data['test'] = datasets.ImageFolder(test_dir, transform = test_transforms)
    
    dataloader = dict()
    dataloader['train'] = torch.utils.data.DataLoader(image_data['train'], batch_size=64, shuffle=True)
    dataloader['valid'] = torch.utils.data.DataLoader(image_data['valid'], batch_size=32)
    dataloader['test'] = torch.utils.data.DataLoader(image_data['test'], batch_size=32)
    
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained = True)
        featureNum = model.classifier[0].in_features
    elif args.arch == 'vgg13':
        model = models.vgg13(pretrained = True)
        featureNum = model.classifier[0].in_features
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained = True)
        featureNum = model.classifier.in_features
        
    
    for param in model.parameters():
        param.requires_grad = False
        
    
    
    classifier = nn.Sequential(OrderedDict([
                     ('fc1', nn.Linear(featureNum,512)),
                     ('relu1', nn.ReLU()),
                     ('dropout1', nn.Dropout(p = 0.5)),
                     ('fc2',nn.Linear(512,args.hidden_units)),
                     ('relu2', nn.ReLU()),
                     ('dropout2', nn.Dropout(p = 0.5)),
                     ('fc3', nn.Linear(args.hidden_units, 102)),
                     ('output', nn.LogSoftmax(dim=1))
                     ]))
    
    model.classifier = classifier
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    if args.gpu:
        if gpu_ava:
            model = model.to('cuda')
        else:
            print('GPU is not avaliable')
    epochs = args.epochs
    print_every = 40
    steps = 0

    for e in range(epochs):
        running_loss = 0
        model.train()
        for inputs, labels in dataloader['train']:
            steps+=1
            
            if args.gpu and gpu_ava:
                inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, dataloader['valid'], criterion, args)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(dataloader['valid'])),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(dataloader['valid'])))

                running_loss = 0

                model.train()
    
    model.class_to_idx = image_data['train'].class_to_idx
    model.epochs = args.epochs
    checkpoint = {
        'epoch' : model.epochs,
        'arch' : args.arch,
        'class_to_idx': model.class_to_idx,
        'model_state_dic' : model.state_dict(),
        'optimizer_state_dic' : optimizer.state_dict(),
        'classifier' : model.classifier
    }
    torch.save(checkpoint, args.save_dir)
    print('done')
    
def validation(model, valid_loader, criterion, args):
    valid_loss = 0
    accuracy = 0
    for images, labels in valid_loader:
        if args.gpu and gpu_ava:
            images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        valid_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return valid_loss, accuracy
    
    
if __name__ == '__main__':
    main()
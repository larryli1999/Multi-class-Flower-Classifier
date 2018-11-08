import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import numpy as np
from torch.autograd import Variable

import argparse
import json

gpu_ava = torch.cuda.is_available()

def main():
    parser = argparse.ArgumentParser(description = 'Predict Image')
    parser.add_argument('input', action = 'store', type = str, help = 'Set input image directory')
    parser.add_argument('checkpoint', action = 'store', type = str, default = 'new_checkpoint.pth', help = 'Set directory to load checkpoints')
    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'Use mapping of catergories to real names')
    parser.add_argument('--top_k', type = int, default = 1, help = 'Return top K most likely classes')
    parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU for prediction if avaliable')
    args = parser.parse_args()
    
    predict_image(args)
    
def predict_image(args):
    model = load_checkpoint(args.checkpoint)
    image_path = args.input
    topk = args.top_k
    image = torch.from_numpy(process_image(image_path)).float()
    image = Variable(torch.FloatTensor(image), requires_grad=True)
    image = image.unsqueeze(0)
    if args.gpu:
        if gpu_ava:
            model = model.to('cuda')
            image = image.to('cuda')
        else:
            print('GPU is not avaliable')
    model.eval()        
    output = model.forward(image)
    ps, items = torch.exp(output).topk(topk)
    probs = ps.data[0].tolist()
    mapdict = {v: k for k, v in model.class_to_idx.items()}
    classes = []
    for i in items.data[0]:
        classes.append(mapdict[i.item()])
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    class_label = [cat_to_name[x] for x in classes]
    result = {class_label[i]:probs[i] for i in range(topk)}
    print()
    print('Prediction')
    for flower,prob in result.items():
        print(flower,': ',prob)
    print()
    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    if(checkpoint['arch'] == 'vgg16'):
        model = models.vgg16(pretrained = True)
    elif(checkpoint['arch'] == 'vgg13'):
        model = models.vgg13(pretrained = True)
    elif(checkpoint['arch'] == 'densenet121'):
        model = models.densenet121(pretrained = True)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['model_state_dic'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    pil_image = Image.open(image)
    pil_image.thumbnail((256,256))
    pil_image = pil_image.resize((224,224))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = np.array(pil_image)
    np_image =  np_image / 255
    np_image = (np_image-mean)/std
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image
    
if __name__ == '__main__':
    main()
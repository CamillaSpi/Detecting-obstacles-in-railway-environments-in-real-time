import os
import cv2
import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchscan import summary #network summary
import torch.nn.functional as F
from torchvision import transforms
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader,random_split
from sklearn.model_selection import train_test_split
import time
import copy
import matplotlib.pyplot as plt
import torchvision.models as models
import argparse, csv



#class of complete model
class CompleteClassifierModel(nn.Module):
    def __init__(self, Backbone, LastClassifierModel):
        super(CompleteClassifierModel, self).__init__()
        self.Backbone = Backbone
        self.LastClassifierModel = LastClassifierModel
        
    def forward(self, image):
        featureVect = self.Backbone(image)
        output = self.LastClassifierModel(featureVect)
        return output



def createModel(checkpoint_path):
    backbone = models.resnet50(pretrained = True) #pretrained on ImageNet

    #number of features of the point at which we cut the backbone to add our layers
    num_ftrs = backbone.fc.out_features
    print("num_ftrs", num_ftrs)

    #our classification layers
    LastClassifierModel = nn.Sequential(
        nn.Linear(num_ftrs, num_ftrs),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(num_ftrs, int(num_ftrs/2)),
        nn.BatchNorm1d(num_features = int(num_ftrs/2)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(num_ftrs/2), int(num_ftrs/4)),
        nn.BatchNorm1d(int(num_ftrs/4)),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(int(num_ftrs/4), 2),
    )


    #complete model definition
    model = CompleteClassifierModel(backbone, LastClassifierModel)

    checkpoint1 = torch.load(checkpoint_path)
    checkpoint1.keys()
    model.load_state_dict(checkpoint1)

    return model


#arguments' description:
#data: the path of the csv containing the patches to be tested and the corresponding labels
#images: the base path of the folder containing the patches to be tested
#results: the path of the csv on which to write the results of the classifiers
#checkpoint: the path to the checpoint to be used as classifier
def test_images(data, images, results, checkpoint, resultant_masks_path):

    print(torch.cuda.device_count())
    CUDA_VISIBLE_DEVICES=0
    torch.cuda.set_per_process_memory_fraction(0.3, CUDA_VISIBLE_DEVICES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)  

    model = createModel(checkpoint)    
    model.to(device)
    model.eval()


    #data transformation
    resize_dim_x = 224
    resize_dim_y = 224
    data_transforms_val = transforms.Compose([
    transforms.Resize((resize_dim_x,resize_dim_y)), 
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


    # Reading CSV test file containing the patches to be tested and the corresponding labels
    with open(data, mode='r') as csv_file:
        gt = csv.reader(csv_file, delimiter=',')
        gt_num = 0
        gt_dict = {}
        for i,row in enumerate(gt):
            if i!=0:
                gt_dict.update({row[0]: row[1]})
                gt_num += 1
    print(gt_num)
    print(gt_dict)



    # Opening CSV results file
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    #open the csv on which to write the results
    resultant_csv = results + "/" + os.path.splitext(os.path.basename(checkpoint))[0] + ".csv"
    print("resultant_csv", resultant_csv)
    with open(resultant_csv, 'w', newline='') as res_file:
        writer = csv.writer(res_file)
        # Processing all the images
        for image in tqdm(gt_dict.keys()):
            img = Image.open(images+image)
            
            if img.size == 0:
                print("Error")
            img = data_transforms_val(img).to(device).unsqueeze(0)
            #print("model(img)",model(img))
            obstacle = model(img)
            prob = nn.Softmax(dim=1)(obstacle)
            preds = torch.argmax(prob,dim=1).unsqueeze(1).item()
            #print("preds",preds)
            
            
            # Writing a row in the CSV file
            writer.writerow([image, preds])

            #make also performance measurment
            if "False" in image and preds == 0:
                true_negative+=1
                color = (0, 255, 0) #green non c'è nulla per la nostra rete
            if "True" in image and preds == 1:
                true_positive+=1
                color = (0, 0, 255) #red per la nostra rete c'è un masso    
            if "False" in image and preds == 1: #non c'è ostacolo ma la rete ha detto si
                false_positive+=1
                color = (0, 0, 255) #red per la nostra rete c'è un masso
            if "True" in image and preds == 0: #c'è ostacolo ma la rete ha detto no
                false_negative+=1
                color = (0, 255, 0) #green non c'è nulla per la nostra rete

            patch_name = os.path.splitext(os.path.basename(image))[0]
            x1 = int(patch_name.split("-")[1])
            y1 = int(patch_name.split("-")[2])
            x2 = int(patch_name.split("-")[3])
            y2 = int(patch_name.split("-")[4])
            find = results + "/" + image.split("/")[-2] + ".png"
            if not os.path.exists(find):
                folder_to_search = image.split("/")[-3]
                image_to_search = image.split("/")[-2]
                find = resultant_masks_path + "/overlay_images" + "/" + folder_to_search + "/" + image_to_search + ".png"
            overlay = cv2.imread(find)
            overlay = cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            cv2.imwrite(results + "/" + image.split("/")[-2] + ".png", overlay)

    print("True Positive: ", true_positive, "True Negative:", true_negative, "False Positive:", false_positive, "False Negative:", false_negative)
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1 = (2*precision*recall)/(precision+recall)
    print("precision:", precision, "recall: ", recall, "F1-score: ", f1)

    

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
from sklearn.metrics import roc_curve, auc
import datetime


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
        nn.Linear(int(num_ftrs/4), 1),
        nn.Sigmoid(),
    )


    #complete model definition
    model = CompleteClassifierModel(backbone, LastClassifierModel)

    checkpoint1 = torch.load(checkpoint_path)
    checkpoint1.keys()
    model.load_state_dict(checkpoint1)

    for param in model.parameters(): 
        param.requires_grad = False
    return model

def create_roc(truths, predictions):
    truths = torch.tensor(truths)
    predictions = torch.tensor(predictions)
    fpr, tpr, thresholds = roc_curve(truths, predictions)
    #print("thresholds", thresholds)
    #print("ho chiamato la funzione")
    roc_auc = auc(fpr, tpr)

    # Calcola l'indice di Youden
    youden_index = tpr - fpr
    optimal_threshold = thresholds[np.argmax(youden_index)]

    # Disegna la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    # Indica l'ottimale punto di taglio sulla curva ROC
    plt.scatter(fpr[np.argmax(youden_index)], tpr[np.argmax(youden_index)], c='red', marker='o', label=f'Youden Index Threshold (threshold = {optimal_threshold:.2f})')
    plt.legend()
    plt.savefig('curva_roc.png')
    #plt.show()

    print(f'Optimal Threshold (Youden Index): {optimal_threshold:.2f}')



class dataFrameDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = df
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.df.iloc[idx, 0])
        obstacle = self.df.iloc[idx,1]
        #print("img_name", img_name)
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)
            obstacle = int(obstacle)

        sample = {'image': image, 'obstacle': obstacle, 'name':img_name}
        return sample




#arguments' description:
#data: the path of the csv containing the patches to be tested and the corresponding labels
#images: the base path of the folder containing the patches to be tested
#results: the path of the csv on which to write the results of the classifiers
#checkpoint: the path to the checpoint to be used as classifier

def test_images(data, images, results, checkpoint, resultant_masks_path, threshold):

    print(torch.cuda.device_count())
    CUDA_VISIBLE_DEVICES=0
    torch.cuda.set_per_process_memory_fraction(0.5, CUDA_VISIBLE_DEVICES)
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
    
    df = pd.read_csv(data,
    names=["image", "obstacle"],dtype={'image':'str','obstacle':'str'})
    df.head()
    df = df.iloc[1:]
    TesteDataSet = dataFrameDataset(df,images,data_transforms_val)
    batch_size = 1
    # create batches
    testSet = DataLoader(TesteDataSet, shuffle=False,batch_size=batch_size,num_workers=15)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    predictions = []
    truths = []
    processing_times = [] #time required for the inference of each sample
    #open the csv on which to write the results
    resultant_csv = results + "/" + os.path.splitext(os.path.basename(checkpoint))[0] + ".csv"
    print("resultant_csv", resultant_csv)
    with open(resultant_csv, 'w', newline='') as res_file:
        writer = csv.writer(res_file)

        for sample_batched in tqdm(testSet):
            inputs = sample_batched['image'].float().to(device)
            thruth = sample_batched['obstacle'].to(device)
            img_name = sample_batched['name']
            start = datetime.datetime.now()
            obstacles = model(inputs).to(device)
            processing_times.append((datetime.datetime.now() - start).microseconds)

            preds = obstacles.round()
            predictions += obstacles
            truths += thruth
            for index,name_of_image in enumerate(img_name): #img_name è la lista contenente tutte le immagini del batch
                pred = preds[index].item()
                writer.writerow([name_of_image, pred])
                #make also performance measurment
                if "False" in name_of_image and pred == 0:
                    true_negative+=1
                    color = (0, 255, 0) #green non c'è nulla per la nostra rete
                if "True" in name_of_image and pred == 1:
                    true_positive+=1
                    color = (0, 0, 255) #red per la nostra rete c'è un masso    
                if "False" in name_of_image and pred == 1: #non c'è ostacolo ma la rete ha detto si
                    false_positive+=1
                    color = (0, 0, 255) #red per la nostra rete c'è un masso
                if "True" in name_of_image and pred == 0: #c'è ostacolo ma la rete ha detto no
                    false_negative+=1
                    color = (0, 255, 0) #green non c'è nulla per la nostra rete
    #print("type truths", type(truths))
    #print("type predictions", type(predictions))
    create_roc(truths, predictions)
    print("True Positive: ", true_positive, "True Negative:", true_negative, "False Positive:", false_positive, "False Negative:", false_negative)
    precision = true_positive/(true_positive + false_positive)
    recall = true_positive/(true_positive + false_negative)
    f1 = (2*precision*recall)/(precision+recall)
    print("precision:", precision, "recall: ", recall, "F1-score: ", f1)
    #print("processing_time: ", processing_times)
    print("mean of processing times: ", np.mean(np.array(processing_times)), "standard deviation of processing times: ", np.std(np.array(processing_times)))
    #print("len of timing", len(processing_times))

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


#necessary setup to read data and corrispondent labels and split in training and validation
csv_train_path = "/user/cspingola/TESI/Tesi/TestSets/TestSetForThesis/test_generalization.csv"
#csv_val_path = '/user/cspingola/TESI/Tesi/Datasets/DatasetV2/valid_black_patches_corrected.csv'

df_train = pd.read_csv(csv_train_path,
    names=["image", "obstacle"],dtype={'image':'str','obstacle':'str'})
df_train.head()
df_train = df_train.iloc[1:]



#data transformation
resize_dim_x = 224
resize_dim_y = 224


data_transforms_train = transforms.Compose([
#transforms.Resize((resize_dim_x,resize_dim_y)), 
#transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
#transforms.RandomGrayscale(p=1),
#transforms.ColorJitter(brightness=(1.2), contrast=(1.2), saturation=(1.2)),
transforms.RandomHorizontalFlip(p=1),
#transforms.ToTensor(),
#transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

#path to the dataset
PATH = "../"


#batch size 
batch_size = 1

#class used for dataloaders
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
            #image_to_save = transforms.ToPILImage()(image)
            image.save("./augmentation/" + img_name.split("/")[-1])
            print("ho slavato")
            #cv2.imwrite("./augmentation/" + img_name , image)
            obstacle = int(obstacle)

        sample = {'image': transforms.ToTensor()(image), 'obstacle': obstacle}
        return sample

trainDataSet = dataFrameDataset(df_train,PATH,data_transforms_train)
train_set = DataLoader(trainDataSet,shuffle=True,batch_size=batch_size,num_workers=15)
for sample_batched in tqdm(train_set):
    print("sto andando")
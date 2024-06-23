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



#name of the trained model
PARAMETERS_AND_NAME_MODEL = 'Prova'


#gpu settings
#to_complete
print(torch.cuda.device_count())
CUDA_VISIBLE_DEVICES=0
torch.cuda.set_per_process_memory_fraction(0.5, CUDA_VISIBLE_DEVICES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)  


#necessary setup to read data and corrispondent labels and split in training and validation
csv_train_path = '../../Datasets/DatasetV2/train_black_patches_corrected.csv'
csv_val_path = '../../Datasets/DatasetV2/valid_black_patches_corrected.csv'

df_train = pd.read_csv(csv_train_path,
    names=["image", "obstacle"],dtype={'image':'str','obstacle':'str'})
df_train.head()
df_train = df_train.iloc[1:]

df_val = pd.read_csv(csv_val_path,
    names=["image", "obstacle"],dtype={'image':'str','obstacle':'str'})
df_val.head()
df_val = df_val.iloc[1:]

# X_train , X_val, y_train, y_val = train_test_split(df['image'],df['obstacle'],train_size=0.74,random_state=2022, shuffle=True)#,stratify=df['obstacle'])
# df_train = pd.DataFrame({'image':X_train,'obstacle':y_train})
# df_val = pd.DataFrame({'image':X_val,'obstacle':y_val})


#data transformation
resize_dim_x = 224
resize_dim_y = 224
#to_complete normalization values depending on the pretraining of the choosen network

# data_transforms_train = transforms.Compose([
# transforms.Resize((resize_dim_x,resize_dim_y)), 
# transforms.ToTensor(),
# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

# data_transforms_val = transforms.Compose([
# transforms.Resize((resize_dim_x,resize_dim_y)), 
# transforms.ToTensor(),
# transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
# ])

data_transforms_train = transforms.Compose([
transforms.Resize((resize_dim_x,resize_dim_y)), 
transforms.GaussianBlur(5, sigma=(0.1, 2.0)),
transforms.RandomGrayscale(p=0.2),
transforms.ColorJitter(brightness=(0.8,1.2), contrast=(0.8,1.2), saturation=(0.8,1.2)),
transforms.RandomHorizontalFlip(p=0.5),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

data_transforms_val = transforms.Compose([
transforms.Resize((resize_dim_x,resize_dim_y)), 
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


#path to the dataset
PATH = "../"


#batch size 
batch_size = 64


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
            obstacle = int(obstacle)

        sample = {'image': image, 'obstacle': obstacle}
        return sample


#dataloaders
trainDataSet = dataFrameDataset(df_train,PATH,data_transforms_train)
valnDataSet = dataFrameDataset(df_val,PATH,data_transforms_val)

train_set = DataLoader(trainDataSet,shuffle=True,batch_size=batch_size,num_workers=15)
val_set = DataLoader(valnDataSet,shuffle=True, batch_size=batch_size,num_workers=15)

dataloaders = {'train':train_set,'val':val_set}
dataset_sizes = {'train':len(trainDataSet),'val':len(valnDataSet)}
print("dataset_sizes", dataset_sizes)


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
    

#backbone
import torchvision.models as models

backbone = models.resnet50(pretrained = True) #pretrained on ImageNet

backbone.eval()
summary(backbone,(3,224,224)) #the dimension must be at least 224*224


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


# #complete model definition
model = CompleteClassifierModel(backbone, LastClassifierModel)


#trainable layers definition
#set all as not trainable
# count = 0
# for param in model.Backbone.conv1.parameters(): 
#     param.requires_grad = False
#     count+=1
# print(count)

# count = 0
# for param in model.Backbone.bn1.parameters(): 
#     param.requires_grad = False
#     count+=1
# print(count)

# count = 0
# for param in model.Backbone.relu.parameters(): 
#     param.requires_grad = False
#     count+=1
# print(count)

# count = 0
# for param in model.Backbone.maxpool.parameters(): 
#     param.requires_grad = False
#     count+=1
# print(count)

# count = 0
# for param in model.Backbone.layer1.parameters(): 
#     param.requires_grad = False
#     count+=1
# print(count)

# count = 0
# for param in model.Backbone.layer2.parameters(): 
#     param.requires_grad = False
#     count+=1
# print(count)

# count = 0
# for param in model.Backbone.layer3.parameters(): 
#     param.requires_grad = False
#     count+=1
# print(count)

# count = 0
# for param in model.Backbone.layer4.parameters(): 
#     param.requires_grad = True
#     count+=1
# print(count)

# count = 0
# for param in model.Backbone.avgpool.parameters(): 
#     param.requires_grad = True
#     count+=1
# print(count)

# count = 0
# for param in model.Backbone.fc.parameters(): 
#     param.requires_grad = True
#     count+=1
# print(count)

count = 0
for param in model.Backbone.parameters(): 
    param.requires_grad = True
    count+=1
print(count)

count = 0
for param in model.LastClassifierModel.parameters(): 
    param.requires_grad = True
    count+=1
print(count)


model = model.to(device)

model.eval()
summary(model, (3,224,224))


#classifier loss
#'to_complete'

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=1e-4)
#base_criterion = nn.CrossEntropyLoss(reduction='mean').to(device)
base_criterion = nn.BCELoss(reduction='mean').to(device)

#early stopping class definition
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


#function to train the model
def train_model(model, base_criterion, optimizer, scheduler, early_stopper,num_epochs=25,best_loss=0.0, numTrain=1, acc_best_loss=0.0):
    train_losses=[]
    val_losses=[]
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = best_loss
    acc_best_loss = acc_best_loss

    toPrint = f'Ciao , sto per allenare {PARAMETERS_AND_NAME_MODEL}'
    print(toPrint)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample_batched in tqdm(dataloaders[phase]):

                inputs = sample_batched['image'].float().to(device)
                labels = sample_batched['obstacle'].float().to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    #outputs = model(inputs, one_hot_input)
                    outputs = model(inputs)
                    labels = labels.unsqueeze(1)
                    #prob = nn.Softmax(dim=1)(outputs)
                    # Classifier EV: prediction = torch.sum(torch.mul(possibleChoise ,prob),dim=1) -> Classifier argmax: prediction = torch.argmax(prob,dim=1)
                    #prediction = torch.argmax(prob,dim=1)
                    prediction = outputs.round()
                    #print("prediction", prediction, "labels", labels)
                    loss = base_criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item()
                tmp = prediction == labels.data
                running_corrects += torch.sum(tmp)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / (dataset_sizes[phase]/batch_size)
            epoch_acc = ((running_corrects / (dataset_sizes[phase]/batch_size))/batch_size)*100
            epoch_acc = epoch_acc.item()

            toPrint = f'{PARAMETERS_AND_NAME_MODEL}: Epochs {epoch}, {phase} Loss: {epoch_loss:.15f} Accuracy: {epoch_acc:.15f}'
            print(toPrint)

            path_to_save = '../../TrainingsResults/ClassifierWeights/'
            if not os.path.exists(path_to_save):
                    os.makedirs(path_to_save)
            if phase == 'val':
                val_losses.append({'ValLoss': epoch_loss, 'Valacc': epoch_acc})
                if early_stopper.early_stop(epoch_loss) == True:
                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s.\nStopped at epoch {epoch}')
                    print(f'Best val Acc: {best_loss:4f}')
                    print(f'Best val Acc: {acc_best_loss:4f}')
                    model.load_state_dict(best_model_wts)
                    torch.save(model.state_dict(),path_to_save + PARAMETERS_AND_NAME_MODEL+'_updated.pt')
                    return model,best_loss,train_losses,val_losses
            else:    
                train_losses.append({'TrainLoss': epoch_loss, 'Valacc': epoch_acc})

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                acc_best_loss = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(),path_to_save + PARAMETERS_AND_NAME_MODEL+'_updated.pt')

        print()

    time_elapsed = time.time() - since
    
    
    toPrint = f'Training of {PARAMETERS_AND_NAME_MODEL} complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s'
    print(toPrint)
    
    toPrint = f'{PARAMETERS_AND_NAME_MODEL}: Best val loss: {best_loss:4f}, Best val Acc: {acc_best_loss:4f}'
    print(toPrint)
    

    path_to_graphs = '../../TrainingsResults/EvaluationCurves/'
    if not os.path.exists(path_to_graphs):
        os.makedirs(path_to_graphs)
    with open(path_to_graphs+PARAMETERS_AND_NAME_MODEL+'_updated'+ str(numTrain) +'training.json', 'w') as f:
        dict = {'trainData' : train_losses,'valData' : val_losses, 'num epoch': epoch}
        json.dump(dict, f)
    

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,best_loss,train_losses,val_losses
    

#creation of early stopper and training of the model
# optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, momentum=0.9)
early_stopper = EarlyStopper(patience=5, min_delta=0.12)
best_loss = 100000
num_epochs = 100
model_ft,best_loss,train_losses,val_losses = train_model(model, base_criterion, optimizer, exp_lr_scheduler,
                       num_epochs=num_epochs,best_loss=best_loss,early_stopper=early_stopper,numTrain=1)


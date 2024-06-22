import csv
import os
import random

def write(paths_of_each_image, to_use):
  for path_of_each_image in os.listdir(paths_of_each_image): #cartella contenente tutte le patch di una stessa immagine
        path_of_patches = paths_of_each_image + "/" + path_of_each_image
        print(path_of_patches)
        for nome_file in os.listdir(path_of_patches):
          if 'total' not in nome_file:
            image_file_path = os.path.join(path_of_patches,nome_file)
            #if os.path.isfile(image_file_path):
            if "True" in nome_file:
              gtruth = 1
            else:
              gtruth = 0
            to_use.writerow([image_file_path,gtruth])



def prepare_csv(images_path , csv_train_path, csv_valid_path, split_ratio = 0.75):
  
  with open(csv_train_path, 'a', newline='\n') as file_train_csv:
    writer_train = csv.writer(file_train_csv)
    writer_train.writerow(['image', 'obstacle'])

  with open(csv_valid_path, 'a', newline='\n') as file_valid_csv:
    writer_valid = csv.writer(file_valid_csv)
    writer_valid.writerow(['image', 'obstacle'])

  #the selection to decide if to use that images in train or in validation will be made
  #on the folder like rs0... in which all patches relative to all the same images with 
  #different rocks are contained
  
  #set random seed
  random.seed(18)
  #iterate over folders
  for folder_images in os.listdir(images_path):
    print(folder_images)
    paths_of_each_image = images_path + "/" + folder_images
    print(paths_of_each_image)
    if random.random() < split_ratio: #to use into the train
      with open(csv_train_path, 'a', newline='') as file_train_csv1:
        writer_train = csv.writer(file_train_csv1)
        write(paths_of_each_image,writer_train)
    else:
      with open(csv_valid_path, 'a', newline='') as file_valid_csv1:
        writer_valid = csv.writer(file_valid_csv1)
        write(paths_of_each_image,writer_valid)



def prepare_csv_test(images_path , csv_test_path):
  with open(csv_test_path, 'a', newline='\n') as file_train_csv:
    writer_train = csv.writer(file_train_csv)
    writer_train.writerow(['image', 'obstacle'])
    for folder_images in os.listdir(images_path):
      print(folder_images)
      paths_of_each_image = images_path + "/" + folder_images
      print(paths_of_each_image)
      writer_train = csv.writer(file_train_csv)
      write(paths_of_each_image,writer_train)


 
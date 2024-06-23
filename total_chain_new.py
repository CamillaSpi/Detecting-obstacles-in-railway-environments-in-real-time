from rail_marking_master.scripts.segmentation.test_one_image import main as segment
from elaboration.new_preelaboration_version import Patch_extractor
from elaboration.dataset_preparation import prepare_csv
import argparse
import os

class Segmentation():
    def __init__(self, snapshot, image_path, result_path):
        self.snapshot = snapshot
        self.image_path = image_path
        self.result_path = result_path

    def perform_segmentation(self):
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        segment(self.snapshot, self.image_path, self.result_path)

class Elaboration():
    def __init__(self, processed_images_path, resultant_masks_path, resultant_patch_path, labels_path,slice_heights,background_subtraction):
        self.processed_images_path = processed_images_path
        self.resultant_masks_path = resultant_masks_path
        self.resultant_patch_path = resultant_patch_path
        self.labels_path = labels_path
        self.slice_heights = slice_heights
        self.background_subtraction = background_subtraction

    def elaboration(self):
        if not os.path.exists(self.resultant_patch_path):
            os.mkdir(self.resultant_patch_path)
        extractor = Patch_extractor(self.processed_images_path, self.resultant_masks_path, self.resultant_patch_path , self.labels_path, self.slice_heights)
        extractor.extraction(self.background_subtraction)


class CSVCreation():
    def __init__(self, images_path, csv_train_path ,csv_valid_path, split_ratio, random_seed):
        self.images_path = images_path
        self.csv_train_path = csv_train_path
        self.csv_valid_path = csv_valid_path
        self.split_ratio = split_ratio
        self.random_seed = random_seed

    def create_csv(self):
        prepare_csv(self.images_path, self.csv_train_path, self.csv_valid_path, self.split_ratio, self.random_seed)


def get_args():
    parser = argparse.ArgumentParser()
    

    #needed for segmentation
    parser.add_argument("-snapshot", type=str, required=True) #path to semantic segmentation model
    parser.add_argument("-source_images_path", type=str, required=True) #path to source images to be used
    parser.add_argument("-segmentation_result_path", type=str, required=True) #path where to save the results of the segmentation

    #needed for elaboration
    parser.add_argument("-resultant_patch_path", type=str, required=True) #path in which to save the created patches
    parser.add_argument("-labels_path", type=str, required=True) #path in which are contained the groundtruth files of the rocks
    parser.add_argument("-slice_heights", nargs = '+', type= int, required = True) #slisce heights at which divide the image
    parser.add_argument("-background_subtraction", type=bool, default=True) #if you want to generate patches with no segmented pixels set to True

    #needed to create csv for training set and validation set
    parser.add_argument("-csv_train_path", type=str, required=True) #path to the csv containing the images
    parser.add_argument("-csv_valid_path", type=str, required=True)
    parser.add_argument("-split_ratio", type=float, default = 0.75)
    parser.add_argument("-random_seed", type=int, default = 42)

    args = parser.parse_args()

    return args



def main():
    args = get_args()
   
    #perform segmentation
    segmentator = Segmentation(args.snapshot,args.source_images_path,args.segmentation_result_path)
    segmentator.perform_segmentation()

    #perform elaboration
    elaborator = Elaboration(args.segmentation_result_path + "/processed_images", args.segmentation_result_path, args.resultant_patch_path, args.labels_path, args.slice_heights, args.background_subtraction)
    elaborator.elaboration()
        

    #create csv for training set and validation set
    #this sets will be created in order to have no correlation between images in different sets
    csv_creator = CSVCreation(args.resultant_patch_path, args.csv_train_path, args.csv_valid_path, args.split_ratio, args.random_seed)
    #the value used as random seed and as split ratio are default ones. If you want to change them, pass them. 
    #Default values are split_ratio = 0.75 and random_seed = 0.42
    csv_creator.create_csv()
    

    
        
if __name__ == "__main__":
    main()
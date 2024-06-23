from rail_marking_master.scripts.segmentation.test_one_image import main as segment
from elaboration.new_preelaboration_version import Patch_extractor
from elaboration.dataset_preparation import prepare_csv
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


snapshot = "./rail_marking_master/bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth"#weights of semantic segmentation networks
source_images_path = "../Datasets/DatasetProva/source_images" #images generated from which to start
segmentation_result_path = "../Datasets/DatasetProva/segmented_images" #path where to save the results of the segmentation
segmentator = Segmentation(snapshot,source_images_path,segmentation_result_path)
segmentator.perform_segmentation()


processed_images_path = "../Datasets/DatasetProva/segmented_images/processed_images"
resultant_masks_path = "../Datasets/DatasetProva/segmented_images"
resultant_patch_path = "../Datasets/DatasetProva/patches_images"
labels_path = "../Datasets/DatasetProva/labels_path"
slice_heights = [160,128,96,64,32]
background_subtraction = True
elaborator = Elaboration(processed_images_path, resultant_masks_path, resultant_patch_path, labels_path, slice_heights, background_subtraction)
elaborator.elaboration()

images_path = "../Datasets/DatasetProva/patches_images"
csv_train_path = "../Datasets/DatasetProva/train.csv"
csv_valid_path = "../Datasets/DatasetProva/valid.csv"
split_ratio = 0.75
random_seed = 42
csv_creator = CSVCreation(images_path, csv_train_path, csv_valid_path, split_ratio, random_seed)
csv_creator.create_csv()


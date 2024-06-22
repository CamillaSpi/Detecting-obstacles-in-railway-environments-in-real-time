from rail_marking_master.scripts.segmentation.test_one_image import main as segment
from elaboration.new_preelaboration_version import Patch_extractor
from elaboration.dataset_preparation import prepare_csv_test
from classification.test_classifier import test_images as test_two_neuron
from classification.test_classifier_really_binary_val import test_images as test_one_neuron
from classification.test_classifier_really_binary_val_ResNet101 import test_images as test_one_neuron_ResNet101
from classification.test_classifier_really_binary_val_MobileNet import test_images as test_one_neuron_MobileNetv2
import os
import argparse


class Segmentation():
    def __init__(self, snapshot, image_path, result_path, test_type):
        self.snapshot = snapshot
        self.image_path = image_path
        self.result_path = result_path
        self.test_type = test_type

    def perform_segmentation(self):
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
        segment(self.snapshot, self.image_path, self.result_path, self.test_type)


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
    def __init__(self, images_path, csv_test_path):
        self.images_path = images_path
        self.csv_test_path = csv_test_path

    def create_csv(self):
        prepare_csv_test(self.images_path, self.csv_test_path)


class ClassificationTest():
    def __init__(self, csv_test, base_path_test_images, test_result_path, classifier_weights, resultant_masks_path, threshold = 0.5):
        self.csv_test = csv_test
        self.base_path_test_images = base_path_test_images
        self.test_result_path = test_result_path
        self.classifier_weights = classifier_weights
        self.resultant_masks_path = resultant_masks_path
        self.threshold = threshold
        if not os.path.exists(self.test_result_path):
            os.mkdir(self.test_result_path)

    def test_one(self):
        test_one_neuron(self.csv_test,  self.base_path_test_images, self.test_result_path,self.classifier_weights, self.resultant_masks_path, self.threshold)

    def test_two(self):
        test_two_neuron(self.csv_test,  self.base_path_test_images, self.test_result_path,self.classifier_weights, self.resultant_masks_path, self.threshold)
    
    def test_ResNet101(self):
        test_one_neuron_ResNet101(self.csv_test,  self.base_path_test_images, self.test_result_path,self.classifier_weights, self.resultant_masks_path, self.threshold)

    def test_MobileNetv2(self):
        test_one_neuron_MobileNetv2(self.csv_test,  self.base_path_test_images, self.test_result_path,self.classifier_weights, self.resultant_masks_path, self.threshold)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-todo", type=str, required=True) # possible options: classification: if you have already a dataset with patch extracted and you need only classification.                                                                       # all: if you have only source images and you need also to do segmentation and elaboration.
    args, _ = parser.parse_known_args()
    #so we know what are required argument
    if args.todo == "classification":
        required = False
    if args.todo == "all":
        required = True
    #know we add all other argument

    #needed for segmentation
    parser.add_argument("-snapshot", type=str, required=required) #path to semantic segmentation model
    parser.add_argument("-source_images_path", type=str, required=required) #path to source images to be used
    parser.add_argument("-segmentation_result_path", type=str, required=True) #path where to save the results of the segmentation
    parser.add_argument("-test_type", type=str, default='generated', required = required) #the possible options are "generated" or "other" depending on the name of images

    #needed for elaboration
    parser.add_argument("-resultant_patch_path", type=str, required=True) #path in which to save the created patches
    parser.add_argument("-labels_path", type=str, required=required) #path in which are contained the groundtruth files of the rocks
    parser.add_argument("-slice_heights", nargs = '+', type= int, required = required) #slisce heights at which divide the image
    parser.add_argument("-background_subtraction", type=bool, default=True) #if you want to generate patches with no segmented pixels set to True

    #needed to create csv
    parser.add_argument("-csv_test_path", type=str, required=required) #path to the csv containing the images

    #needed for classification
    parser.add_argument("-classifier_weights", type=str, required=True) #path in which are contained the weight of the classifier to be used
    parser.add_argument("-test_result_path", type=str, required=True) #path in which to save the results of the test
    parser.add_argument("-threshold", type=float, default=0.5) #value to be used ad threshold
    parser.add_argument("-network_name", type=str, default="resNet50") #architecture to be used, possible options are "resNet50", "resNet101" and "mobileNetV2"

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    #if we need to do everything, so we have only source images and groundtruth of rocks
    if args.todo == "all":
        #perform segmentation
        segmentator = Segmentation(args.snapshot,args.source_images_path,args.segmentation_result_path, args.test_type)
        segmentator.perform_segmentation()
        #perform elaboration
        elaborator = Elaboration(args.segmentation_result_path + "/processed_images", args.segmentation_result_path, args.resultant_patch_path, args.labels_path, args.slice_heights, args.background_subtraction)
        elaborator.elaboration()
        
        #create csv
        csv_creator = CSVCreation(args.resultant_patch_path, args.csv_test_path)
        csv_creator.create_csv()
        
    #perform classification
    base_path_test_images = ""
    test_result_path = args.test_result_path
    
    tester_classifier = ClassificationTest(args.csv_test_path,base_path_test_images, test_result_path, args.classifier_weights, args.segmentation_result_path, args.threshold)
    network = args.network_name
    if network == "resNet50":
        tester_classifier.test_one()
    if network == "resNet101":
        tester_classifier.test_ResNet101()
    if network == "mobileNetV2":
        tester_classifier.test_MobileNetv2()
    
        
if __name__ == "__main__":
    main()
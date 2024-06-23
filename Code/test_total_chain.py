from rail_marking_master.scripts.segmentation.test_one_image import main as segment
from elaboration.new_preelaboration_version import Patch_extractor
from elaboration.dataset_preparation import prepare_csv_test
from classification.test_classifier import test_images as test_two_neuron
from classification.test_classifier_really_binary_val import test_images as test_one_neuron
from classification.test_classifier_really_binary_val_ResNet101 import test_images as test_one_neuron_ResNet101
from classification.test_classifier_really_binary_val_MobileNet import test_images as test_one_neuron_MobileNetv2
import os


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
    def __init__(self, csv_test, base_path_test_images, test_result_path, classifier_weigths, resultant_masks_path, threshold = 0.5):
        self.csv_test = csv_test
        self.base_path_test_images = base_path_test_images
        self.test_result_path = test_result_path
        self.classifier_weigths = classifier_weigths
        self.resultant_masks_path = resultant_masks_path
        self.threshold = threshold
        if not os.path.exists(self.test_result_path):
            os.mkdir(self.test_result_path)


    def test_one(self):
        test_one_neuron(self.csv_test,  self.base_path_test_images, self.test_result_path,self.classifier_weigths, self.resultant_masks_path, self.threshold)

    def test_two(self):
        test_two_neuron(self.csv_test,  self.base_path_test_images, self.test_result_path,self.classifier_weigths, self.resultant_masks_path, self.threshold)
    
    def test_ResNet101(self):
        test_one_neuron_ResNet101(self.csv_test,  self.base_path_test_images, self.test_result_path,self.classifier_weigths, self.resultant_masks_path, self.threshold)

    def test_MobileNetv2(self):
        test_one_neuron_MobileNetv2(self.csv_test,  self.base_path_test_images, self.test_result_path,self.classifier_weigths, self.resultant_masks_path, self.threshold)

snapshot = "./rail_marking_master/bisenetv2_checkpoint_BiSeNetV2_epoch_300.pth"#weights of semantic segmentation networks
source_images_path = "../Datasets/DatasetProva/source_images" #images generated from which to start
segmentation_result_path = "../Datasets/DatasetProva/segmented_images" #path where to save the results of the segmentation
test_type = "generated" #the possible options are "generated" or "internet"
segmentator = Segmentation(snapshot,source_images_path,segmentation_result_path,test_type)
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
csv_test_path = "../Datasets/DatasetProva/test.csv"
csv_creator = CSVCreation(images_path, csv_test_path)
csv_creator.create_csv()

csv_test = "../Datasets/DatasetProva/test.csv"
base_path_test_images = ""
classifier_weigths = "../TrainingsResults/ClassifierWeights/Training_03_10_Resize_DatasetV2BigBlackPatchesSiCorrected_BatchSize64_SGD_reallyBinary_AllSblocked_AugmentationTraining_updated_updated.pt"
test_result_path = "../Datasets/DatasetProva/results"
if not os.path.exists(test_result_path):
    os.mkdir(test_result_path)
threshold = 0.5
tester_classifier = ClassificationTest(csv_test,base_path_test_images, test_result_path, classifier_weigths, resultant_masks_path, threshold)
#tester_classifier.test_two()
tester_classifier.test_one() #to test ResNet50
#tester_classifier.test_ResNet101() #to test ResNet101
#tester_classifier.test_MobileNetv2() #to test MobileNetv2
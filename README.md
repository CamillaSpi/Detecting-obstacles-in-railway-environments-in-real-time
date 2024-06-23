# Detecting obstacles in railway environments in real time

## Description
This repository contains the code of my master thesis project, whose aim is the development of an Artificial Vision system for the detection, among all possible obstacles, of rocks, using a camera mounted on board a vehicle. Due to the absence of available data, it has been proceeded with the generation of a synthetic dataset using background images of railway environment. 
The importance of this work lies in the fact that, despite the steady increase in attention devoted to rail safety, obstacles such as rocks are often ignored, deven if, whether dislodged from natural formations (adverse weather conditions or geological instability) or intentionally placed by external forces, whether small pebbles or larger boulders, when encountered on the tracks, rocks could damage components located in the lower part of the vehicle, or more seriously, they could derail trains, causing catastrophic accidents, service disruptions, and potential derailments. Addressing these dangers is not merely a matter of operational efficiency or of the preservation of invaluable transport infrastructure, it is a matter of life and death.

## Repository Structure
- `Code/`:
	- `total_chain.py` and `total_chain_new.py`: are the same code, used for, starting from source images and groundtruth files, obtaining the segmentation 	masks, the patches and the csv files with the split for training and validation and the associated ground truth. The only difference between them are the parameters to be passed. 
	- `test_total_chain.py` and `test_total_chain_new.py`: are the same code, used for, specifying the architetcure and the weigths, test the classifiers. It is possible to test only the classification network or to perform also segmentation and elaboration on the images, depending on the used parameters. The only difference between them are the parameters to be passed.
	- `classification/`: contains all the scripts that are necessary to train different developed configurations. The most important are:
   		- binaryClassifierReallyBinaryAllSblock.py (for the model with resnet50 backbone, the one with best performance);
       - binaryClassifierReallyBinaryAllSblockMobileNet (for the model with mobileNetV2 backbone);
       - binaryClassifierReallyBinaryAllSblockResNet101 (for the model with resnet101 backbone)
	- `elaboration/`: contains all the scripts used for obtaining, given the segmentation mask, the patches to be classified.
	- `rail_marking/`: contains all the necessary scripts and also the weights used for semantic segmentation of rail tracks.
	- `utils/`: contains different scripts useful during analysis.

 - `Slides.pptx`: the presentation slides of the project.
 - `Thesis.pdf`: the pdf of master thesis.
 - `Demo.avi`: is a video demonstrating how the system works.

## How to run

### To create the environment
To create the environment execute:
```bash
  conda env create -f environment.yml
```
An environment called rail_marking will be created.
Then run: 
```bash
  conda activate rail_marking
```

### To test the system
Download from [here](https://drive.google.com/drive/folders/1t5afyYfvrL8GmgLzTDgFToMYDZRM3L1k?usp=sharing) the weights of the already trained classifiers (ResNet50 that is the best one, ResNet101 and MobileNetV2), and put it wherever you want.

Enter the `Code` folder.
- If you have only source images in railway environment, run the command: 
```
  python test_total_chain_new.py -todo "all" -snapshot [path to segmentation weights] -source_images_path [path to source images] -segmentation_result_path [path to save segmentation results] -test_type [type of the test] -resultant_patch_path [path to save resultant patches] -labels_path [path to groundtruth files] -slice_heights [slice heights as a list] -background_subtraction [if to set no segmented pixels to black] -csv_test_path [path to csv files with path and grountruth of the patches] -classifier_weights [path to classifier weights] -test_result_path [path to save classification results] -threshold [threshold to be used] -network_name [architecture name]
```
- Alternatively, if you already have the extracted patches and a correctly organized csv, and you want only to test the classifier, run the command:
```
    python test_total_chain_new.py -todo "classification"  -segmentation_result_path [path to save segmentation results] -resultant_patch_path [path in which are the patches] -classifier_weights [path to classifier weights] -test_result_path [path to save classification results] -threshold [threshold to be used] -network_name [architecture name]
```

Using these commands, in the folder "test_result_path" it will be saved a csv containing for each image the output given by the selected network. Also you will see performance information like precision, recall and F1-score. 
If you want also to see these results in a visual manner, and in particular the patch colored of red if network identify an obstacle, green otherwise:
- Enter the folder `utils`
- Run the command:
    ```
    python show_test_results.py -csv_file_path [path to csv with classifier results] -segmented_images_path [path to segmented images] -resultant_image_path [path in which to save resultant images]
    ```

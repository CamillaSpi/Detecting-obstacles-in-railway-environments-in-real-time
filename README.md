# Detecting obstacles in railway environments in real time

## Description
This repository contains the code of my master thesis project, whose aim is the development of an artificial vision system for the detection, among all possible obstacles, of rocks, using a camera mounted on board a vehicle.Due to the absence of available data, it has been proceeded with the generation of a synthetic dataset using background images of railway environment. 
The importance of this work lies in the fact that, despite the steady increase in attention devoted to rail safety, obstacles such as rocks are often ignored, despite these, whether dislodged from natural formations (adverse weather conditions or geological instability) or intentionally placed by external forces, whether small pebbles or larger boulders, when encountered on the tracks, could damage components located in the lower part of the vehicle, or more seriously, they could derail trains, causing catastrophic accidents, service disruptions, and potential derailments. Addressing these dangers is not merely a matter of operational efficiency or of the preservation of invaluable transport infrastructure, it is a matter of life and death.

## Repository Structure
- 'Code/':
	- 'total_chain.py' and 'total_chain_new.py': are the same code, used for, starting from source images and groundtruth files, obtaining the segmentation 	masks, the patches and the csv files with the split for training and validation and the associated ground truth. The only difference between them are the parameters to be passed. 
	- 'test_total_chain.py' and 'test_total_chain_new.py': are the same code, used for, specifying the architetcure and the weigths, test the classifiers. It is possible to test only the classification network or to perform also segmentation and elaboration on the images, depending on the used parameters. The only difference between them are the parameters to be passed.
	- 'classification/': contains all the scripts that are necessary to train different developed configurations. The most important are:
   		- binaryClassifierReallyBinaryAllSblock.py (that for the model with resnet50 backbone, the best one);
       - binaryClassifierReallyBinaryAllSblockMobileNet (that for the model with mobileNetV2 backbone);
       - binaryClassifierReallyBinaryAllSblockResNet101 (that for the model with resnet101 backbone)
	- 'elaboration/': contains all the scripts used for obtaining, given the segmentation mask, the patches to be classified.
	- 'rail_marking/': contains all the necessary scripts and also the weights used for semantic segmentation of rail tracks.
	- 'utils/': contains different scripts useful during analysis.

 - 'Slides.pptx': the presentation slides of the project.
 - 'Thesis.pdf': the pdf of master thesis.
 - 'Demo': is a video demonstrating the operation of the system






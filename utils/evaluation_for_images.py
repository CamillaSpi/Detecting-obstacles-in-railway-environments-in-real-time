import csv
results_csv = "/user/cspingola/TESI/Tesi/ResultsOnTestSets/ResultsBestModelsGenerated/Training_03_10_Resize_DatasetV2BigBlackPatchesSiCorrected_BatchSize64_SGD_reallyBinary_AllSblocked_AugmentationTraining_updated_updated.csv"

with open(results_csv, 'r') as file:
    # Create a CSV reader
    csv_reader = csv.reader(file)
    matching_alarms = 0
    non_matching_alarms = 0
    num_images = 0
    possible_true_positive = False
    possible_true_negative = False
    possible_false_positive = False
    possible_false_negative = False
    images_with_obstacle = []
    images_without_obstacle = []
    possible_true = False
    possible_false = True
    old_image_name = ""
    for i,row in enumerate(csv_reader):
        image_fold = row[0]
        total_image_name = image_fold.split("/")[-2]
        patch_name = image_fold.split("/")[-1].split(".")[0]
        if total_image_name != old_image_name and old_image_name != "":
            if possible_true == True:
                images_with_obstacle.append(old_image_name)
            else:
                images_without_obstacle.append(old_image_name)
            possible_true = False
            possible_false = True
        if "True" in patch_name:
            possible_true = True
            possible_false = False
        old_image_name = total_image_name
if possible_true == True:
    images_with_obstacle.append(total_image_name)
else:
    images_without_obstacle.append(total_image_name)
print("images_with_obstacle" , images_with_obstacle)
print("images_without_obstacle",images_without_obstacle)
#here we have obtained true images and false images
#now it is necessary to compute TP, TN, FP, FN
TP = 0 #all the images in which at least one among the true patch has been recognized as tue
TN = 0 #all the images without rock in which all the patches have been recognized as False
FP = 0 #all the images not containing boulders where at least one patch was recognized as positive, or those samples containing boulders where, however, only patches different from the genuinely positive ones were recognized as positive
FN = 0 #all the images in which there is a rock but the classifier does not assign True to no one of them

#iteriamo sui campioni originariamente positivi:
for true_image_name in images_with_obstacle:
    possible_true_positive = False
    possible_true_negative = False
    possible_false_positive = False
    possible_false_negative = False
    true_patches = {} #patches containing an obstale
    false_patches = {} #patches containing nothing
    with open(results_csv, 'r') as file:
        csv_reader = csv.reader(file)
        for i,row in enumerate(csv_reader):
            image_fold = row[0]
            network_output = row[1]
            if ("/" + true_image_name + "/") in image_fold:
                patch_name = image_fold.split("/")[-1].split(".")[0]
                if "True" in patch_name:
                    true_patches[patch_name] = network_output
                else: 
                    false_patches[patch_name] = network_output
    print(true_patches)
    print(false_patches)
    #we have obtained all true and all false patches
    for positive_patch in true_patches.keys():
        if true_patches[positive_patch] == "1.0":
            possible_true_positive = True
    if possible_true_positive != True: # no matching
        for negative_patch in false_patches.keys():
            if false_patches[negative_patch] == "1.0":
                possible_false_positive = True
    if possible_true_positive == True:
        TP += 1
    if possible_false_positive == True:
        FP += 1
    if possible_true_positive!=True and possible_false_positive != True:
        FN += 1
    
for false_image_name in images_without_obstacle: 
    possible_false_positive = False
    with open(results_csv, 'r') as file:
        csv_reader = csv.reader(file)
        for i,row in enumerate(csv_reader):
            image_fold = row[0]
            network_output = row[1]
            if ("/" + true_image_name + "/") in image_fold:
                patch_name = image_fold.split("/")[-1].split(".")[0]
                if network_output == "1.0":
                    possible_false_positive = True
        if possible_false_positive == True:
            FP += 1
        else:
            TN += 1

print("TP", TP, "FN", FN, "FP", FP, "TN", TN)
precision = TP/(TP + FP)
recall = TP/(TP + FN)
f1 = (2*precision*recall)/(precision+recall)
print("precision: ", precision, "recall:", recall, "F1", f1)

        
import csv
from PIL import Image
import cv2
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-csv_file_path_result", type=str, required=True) # path where is saved the csv generated from the classifier                                                       # all: if you have only source images and you need also to do segmentation and elaboration.
    parser.add_argument("-segmented_images_path", type=str, required=True) # path containing the segmented images
    parser.add_argument("-resultant_image_path", type=str, required=True)# path in which to save these visual results
    args = parser.parse_args()

    return args



def main():
    args = get_args()
    csv_file_path = args.csv_file_path_result
    processed_image_path = args.segmented_images_path
    resultant_image_path = args.resultant_image_path
    if not os.path.exists(resultant_image_path):
            os.mkdir(resultant_image_path)
    overlay_image_path = processed_image_path + "/overlay_images"
    
    with open(csv_file_path, 'r') as file:
        # Create a CSV reader
        csv_reader = csv.reader(file)

        # Iterate over each row in the CSV file
        for i,row in enumerate(csv_reader):
            image_fold = row[0]
            network_output = row[1]
            image_name = image_fold.split("/")[5] + ".png"
            patch_name = image_fold.split("/")[-1].split(".")[0]
            patch_coords = patch_name.split("-")
            if i == 0 or image_name != old_image_name:
                #overlay_image = Image.open(overlay_image_path + "/" + image_fold.split("/")[4] + "/" + image_name)
                overlay_image = cv2.imread(overlay_image_path + "/" + image_fold.split("/")[5] + "/" + image_name)
                result_image = overlay_image.copy()
            else:
                result_image = cv2.imread(resultant_image_path + "/" + image_name)
            if network_output == "1.0":
                color = (0,0,255)
            else:
                color = (0,255,0) 
            result_image = cv2.rectangle(result_image, (int(patch_coords[1]),int(patch_coords[2])), (int(patch_coords[3]),int(patch_coords[4])), color, 2)
            old_image_name = image_name
            cv2.imwrite(resultant_image_path + "/" + image_name, result_image)


if __name__ == "__main__":
    main()
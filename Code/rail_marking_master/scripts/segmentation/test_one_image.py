#!/usr/bin/env python
import os
import sys
import cv2
import datetime
from PIL import Image
import numpy as np

CURRENT_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".")
sys.path.append(os.path.join(CURRENT_DIR, "../../"))
try:
    from rail_marking.segmentation.deploy import RailtrackSegmentationHandler
    from cfg import BiSeNetV2Config
except Exception as e:
    print(e)
    sys.exit(0)


def get_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-snapshot", type=str, required=True)
    parser.add_argument("-image_path", type=str, required=True)
    parser.add_argument("-output_image_path", type=str, default="result.png")
    parser.add_argument("-num_test", type=int, default=1)

    args = parser.parse_args()

    return args


def main(snapshot, image_path, result_path, type = "generated",num_test = 1):
    #args = get_args()
    segmentation_handler = RailtrackSegmentationHandler(snapshot, BiSeNetV2Config())

    #modifica apportata, il path sarà quello delle immagini sorgenti, sul quale itereremo
    #image = cv2.imread(args.image_path)
    for image_name in os.listdir(image_path):
        image = cv2.imread(image_path + "/" + image_name)
        image = image[:, :, ::-1]
        print("processing ", image_name, "saving result in ", result_path)
        start = datetime.datetime.now()
        for i in range(num_test):
            mask, overlay, color_mask, processed_image = segmentation_handler.run(image, only_mask=False, modified_version = True)
            #il parametro modified è stato aggiunto proprio per supportare la conversione ad un problema binario a partire da questo che è un problema 
            #ternario
            #overlay è quella che visualizziamo, cioè la maschera originale con sopra sovrapposti i colori di riferimento delle parti segmentate
            processed_image = processed_image[-1]
            processed_image = processed_image.cpu().numpy()
            #print("type(processed_image)", type(processed_image), processed_image.shape)
            processed_image = (processed_image - processed_image.min()) * (255 / (processed_image.max() - processed_image.min()))
            processed_image = np.uint8(processed_image.transpose(1, 2, 0)) 
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)


            #non salvata correttamente attualmente
            if type == "generated":
                overfolder = image_name.split("_")[0] # ottengo solo il nome tipo rs0...
            else:
                overfolder=os.path.splitext(os.path.basename(image_name))[0]
            processed_folder = result_path + "/processed_images/" + overfolder
            masks_folder = result_path + "/masks/" + overfolder
            colored_folder = result_path + "/colored_masks/" + overfolder
            overlay_folder = result_path + "/overlay_images/" + overfolder

            #create_folders if not exists
            if not os.path.exists(processed_folder):
                os.makedirs(processed_folder)
            if not os.path.exists(masks_folder):
                os.makedirs(masks_folder)
            if not os.path.exists(colored_folder):
                os.makedirs(colored_folder)
            if not os.path.exists(overlay_folder):
                os.makedirs(overlay_folder)
            
            #save images
            name_to_save = os.path.splitext(os.path.basename(image_name))[0]+".png"
            cv2.imwrite(processed_folder + "/" + name_to_save, processed_image)
            cv2.imwrite(masks_folder + "/" + name_to_save, mask)
            cv2.imwrite(colored_folder + "/" + name_to_save, color_mask)
            cv2.imwrite(overlay_folder + "/" + name_to_save, overlay)
        _processing_time = datetime.datetime.now() - start
       
        
        
        # cv2.imshow("result", overlay)
        # cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #cv2.imwrite(args.output_image_path, overlay)

        print("processing segmentation time one frame {}[ms]".format(_processing_time.total_seconds() * 1000 / num_test))

    #return mask, overlay, color_mask, processed_image

if __name__ == "__main__":
    main()

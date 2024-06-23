import cv2
import numpy as np
import torch
from torchvision.utils import draw_bounding_boxes
from elaboration.connected_components import found_connected_components
import os



class Patch_extractor:
    def __init__(self, input_images_path, resultant_masks_path, resultant_patch_path, labels_path ,slice_heights):
        self.input_images_path = input_images_path #input image given to the model
        self.resultant_masks_path = resultant_masks_path #path in which al different types of masks are contained
        self.resultant_patch_path = resultant_patch_path #where to save the resultant patches
        self.slice_heights = slice_heights #static heights at which divide the image
        self.labels_path = labels_path # the path in which are saved the bounding boxes associated to the images


    def put_black_no_rail(self,to_copy_black,blck_wht_mask):
        if blck_wht_mask.shape[:2] != to_copy_black.shape[:2]:
            raise ValueError("Le dimensioni della maschera e dell'immagine devono essere uguali.")
        output = np.zeros_like(to_copy_black)

        # Copia i pixel dall'immagine originale all'immagine di output solo se la maschera è diversa 0
        output[blck_wht_mask == 0] = to_copy_black[blck_wht_mask == 0]

        return output


    def extraction(self, background_subtraction = False):
        for folder in os.listdir(self.input_images_path):
            images_path = self.input_images_path + "/" + folder
            for image_name in os.listdir(images_path):
                print("extracting patches image ", image_name)
                input_image = cv2.imread(images_path + "/" + image_name)
                height, width, _ = input_image.shape
                #print(height, width)
                #obtained_mask = cv2.imread(self.resultant_masks_path + "/masks/" + image_name, cv2.IMREAD_GRAYSCALE) #output of the segmentation model
                color_mask_grey = cv2.imread(self.resultant_masks_path + "/colored_masks/" + folder + "/" + image_name, cv2.IMREAD_GRAYSCALE)
                color_mask = cv2.imread(self.resultant_masks_path + "/colored_masks/" + folder + "/" + image_name)
                to_copy = input_image.copy()
                to_copy_black = input_image.copy()
                x1_label, x2_label, y1_label, y2_label = self.open_labels(image_name,height,width)
                if background_subtraction:
                    blck_wht_mask = cv2.imread(self.resultant_masks_path + "/masks/" + folder + "/" + image_name, cv2.IMREAD_GRAYSCALE)
                    black_to_use_src = self.put_black_no_rail(to_copy_black,blck_wht_mask)
                    black_to_use_patches = black_to_use_src.copy()
                for i,slice_height in enumerate(self.slice_heights):
                    if i == 0: 
                        start = 512-slice_height
                        end = 512
                    else:
                        end = start
                        start = start-slice_height
                    
                    #slice_to_analize = obtained_mask[start:end,:]
                    #bboxes = mask_to_bbox(slice_to_analize, color_slice)
                    color_slice = color_mask_grey[start:end,:]
                    bboxes = found_connected_components(color_slice, slice_height)
                    #print("ecco quante patch ho trovato:", len(bboxes))
                    path = self.resultant_patch_path + "/" + folder + "/" + image_name.split('.')[0]
                    if not os.path.exists(path):
                        os.makedirs(path)
                    if background_subtraction:
                        to_use_src = black_to_use_src
                        to_use_patches = black_to_use_patches
                    else:
                        to_use_src = to_copy
                        to_use_patches = input_image
                    for k, box in enumerate(bboxes):
                        #print(box[0],box[1],box[2],box[3])
                        sub_factor = sum(self.slice_heights[:i+1])
                        converted_bbox = [box[0],512 - sub_factor + box[1], box[2] , 512 - sub_factor + box[3]]
                        final_img_src = cv2.rectangle(to_use_src, (converted_bbox[0], converted_bbox[1]), (converted_bbox[2], converted_bbox[3]), (255, 0, 0), 2)
                        final_img_mask = cv2.rectangle(color_mask, (converted_bbox[0], converted_bbox[1]), (converted_bbox[2], converted_bbox[3]), (255, 0, 0), 2)
                        if x1_label!=None:
                            final_img_src_definitive = cv2.rectangle(final_img_src, (x1_label, y1_label), (x2_label,y2_label ), (0, 255, 0), 2)
                        else:
                            final_img_src_definitive=final_img_src
                        #print(converted_bbox[0], converted_bbox[1], converted_bbox[2], converted_bbox[3])
                        #stesso qui dove creiamo e salviamo il crop dobbiamo controllare ci sta intersezione tra il bounding box
                        #salvato per quell'immagine e la specifica patch che stiamo analizzando come se fosse un bounding box 
                        #e salvare nel nome della patch gia il True o False
                        if x1_label!=None:
                            result = self.check_overlap(x1_label, x2_label, y1_label, y2_label, converted_bbox[0], converted_bbox[2], converted_bbox[1],converted_bbox[3])
                        else:
                            result=False #significa che nell'immagine non ci sta nessun masso
                        crop_im = to_use_patches[converted_bbox[1]:converted_bbox[3], converted_bbox[0]:converted_bbox[2],:]
                        #print("voglio salvare il crop in: ", path + "/" + str(i) + "_"+  str(k) + "-" + str(result) + ".png")
                        coords_string = str(converted_bbox[0]) + "-" + str(converted_bbox[1]) + "-" + str(converted_bbox[2]) + "-" + str(converted_bbox[3])
                        cv2.imwrite(path + "/" + str(i) + "_" + str(k) + "-"  + coords_string + "-" +str(result) + ".png", crop_im)
                cv2.imwrite(path + "/total_src.png", final_img_src_definitive)
                cv2.imwrite(path + "/total_mask.png", final_img_mask)
                        

    def open_labels(self,image_name,height_img, width_img):
        name = os.path.splitext(os.path.basename(image_name))[0]
        if os.path.exists(self.labels_path + "/" + name + ".txt"):
            with open(self.labels_path + "/" + name + ".txt", 'r') as file:
                lines = file.readlines()
                for line in lines:
                    line = line.split(" ")
                    #print(line)
                    xc = int(float(line[1])*width_img)
                    yc = int(float(line[2])*height_img)
                    width = int(float(line[3])*width_img)
                    height = int(float(line[4])*height_img)
                    x1 = int(xc - width/2)
                    y1 = int(yc - height/2)
                    x2 = int(xc + width/2)
                    y2 = int(yc + height/2)
                    #print(x1, x2, y1, y2)
            return x1, x2, y1, y2
        else:
            return None, None, None, None

    def check_overlap(self,x1_label, x2_label, y1_label, y2_label, x1_crop, x2_crop, y1_crop, y2_crop):
         # Verifica se c'è sovrapposizione sugli assi x e y
        if (x1_label < x2_crop and x2_label > x1_crop) and (y1_label < y2_crop and y2_label > y1_crop):
            return True
        else:
            return False



# input_image = cv2.imread("/user/cspingola/Tesi/rail_marking-master/processed_image.png")
# obtained_mask = cv2.imread("/user/cspingola/Tesi/rail_marking-master/mask.png", cv2.IMREAD_GRAYSCALE)
# color_mask = cv2.imread("/user/cspingola/Tesi/rail_marking-master/color_mask.png")
# color_mask_grey = cv2.imread("/user/cspingola/Tesi/rail_marking-master/color_mask.png", cv2.IMREAD_GRAYSCALE)
# print("sono in pre elaboration")
# extractor = Patch_extractor(input_image, obtained_mask,color_mask,color_mask_grey, [160,128,96,64,32])
# extractor.extraction()
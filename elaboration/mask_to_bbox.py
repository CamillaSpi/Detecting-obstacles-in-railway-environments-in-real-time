import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from skimage.measure import label, regionprops, find_contours

""" Creating a directory """
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

""" Convert a mask to border image """
def mask_to_border(mask):
    h, w = mask.shape
    border = np.zeros((h, w))

    contours = find_contours(mask, 0)
    for contour in contours:
        for c in contour:
            x = int(c[0])
            y = int(c[1])
            border[x][y] = 255

    return border

#############################################################
##################FUNZIONI AGGIUNTE#################
def check_overlap(box1, box2):
    # Verifica se c'è sovrapposizione
    return not (box2[0] > box1[2] or box2[2] < box1[0] or box2[1] > box1[3] or box2[3] < box1[1])

def merge_boxes(box1, box2):
    # Unisce i bounding box
    x_min = min(box1[0], box2[0])
    y_min = min(box1[1], box2[1])
    x_max = max(box1[2], box2[2])
    y_max = max(box1[3], box2[3])
    return [x_min, y_min, x_max, y_max]

def calculate_overlap_area(box1, box2):
    # Calcola l'area di sovrapposizione
    x_overlap = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    overlap_area = x_overlap * y_overlap
    return overlap_area

def all_rails(selected_columns):
    for row in selected_columns:
        for element in row:
            if element != 255:
                return False
    return True

######################################################################

""" Mask to bounding boxes """
def mask_to_bbox(mask, color_mask):
    bboxes = []
    definitive_bboxes = []
    bboxes_dict = {}

    mask = mask_to_border(mask)
    lbl = label(mask)
    props = regionprops(lbl)
    i = 0
    for prop in props:
        x1 = prop.bbox[1]
        y1 = prop.bbox[0]

        x2 = prop.bbox[3]
        y2 = prop.bbox[2]
        #aggiungiamo un check sull'area minima del bounding box:
        if (x2-x1)*(y2-y1)>500:
            bboxes.append([x1, y1, x2, y2])
            bboxes_dict[i] = x1
            definitive_bboxes.append([x1, y1, x2, y2])
            i+=1
            
    # print("bboxes_dict", bboxes_dict)
    # # for ind,bbox in enumerate(bboxes):
    # #     if ind != 0  and check_overlap(bbox, bboxes[ind-1]):
    # #         merged_box = merge_boxes(bbox, bboxes[ind-1])
    # #         definitive_bboxes.append(merged_box)
    # #     else:
    # #         definitive_bboxes.append([x1, y1, x2, y2])

    # #ordiniamo il dizionario per valori
    # ordered_bboxes_dict = dict(sorted(bboxes_dict.items(), key=lambda item: item[1]))
    # print("ordered_bboxes_dict", ordered_bboxes_dict)
    # # print("len(bbox inizialmente trovati): ", len(bboxes))
    # j=0
    # for index in ordered_bboxes_dict.keys():
    #     print(index)
    #     bbox = bboxes[index]
    #     print("bbox", bbox)
    #     if j!=0:
    #         previous_coords = definitive_bboxes[j-1]
    #         print("previous_coords[2]", previous_coords[2])
    #         print("bbox[0]", bbox[0])
    #         indices = mask[-1:,previous_coords[2] : bbox[0]]
    #         colored_indices = color_mask[:,previous_coords[2] : bbox[0],:1]
    #         print("previous_coords", previous_coords)
    #         overlap_area = calculate_overlap_area(bbox,previous_coords)
    #         if overlap_area >= 800:
    #             print("c'è sovrapposizione")
    #             merged_box = merge_boxes(bbox, previous_coords)
    #             definitive_bboxes[j-1] = merged_box
    #         elif not overlap_area and all_rails(colored_indices):
    #             print("indices:", colored_indices)
    #             definitive_bboxes[j-1] = [previous_coords[0], previous_coords[1], bbox[2], previous_coords[3]]
    #         else:
    #             definitive_bboxes.append(bbox)
    #             j+=1
    #     else:
    #         definitive_bboxes.append(bbox)
    #         j+=1
    print("definitive_bboxes", definitive_bboxes)
    print("original_bboxes", bboxes)
    #         indices = mask[:,previous_coords[2] : bbox[0]]
    #         if np.all(indices == 0):
    #             definitive_bboxes[i-1] = [previous_coords[0], previous_coords[1], bbox[2], bbox[3]]
    #         else:
    #             definitive_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
    #     else:
    #         definitive_bboxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
    #         # if i != 0:
            #     previous_coords = bboxes[i-1]
            #     indices = mask[:,previous_coords[2] : x1]
            #     print("len(indices)", len(indices))
            #     if np.all(indices == 0):
            #         bboxes[i-1] = [previous_coords[0], previous_coords[1], x2, previous_coords[3]]
            #     else:        
            #         bboxes.append([x1, y1, x2, y2])
            #         i+=1
            # else:        
            #         bboxes.append([x1, y1, x2, y2])
            #         i+=1
    print("len", len(definitive_bboxes))
    print(definitive_bboxes)
    return definitive_bboxes

def parse_mask(mask):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)
    return mask

if __name__ == "__main__":
    # """ Load the dataset """
    # images = sorted(glob(os.path.join("data", "image", "*")))
    # masks = sorted(glob(os.path.join("data", "mask", "*")))

    # """ Create folder to save images """
    # create_dir("results")

    # """ Loop over the dataset """
    # for x, y in tqdm(zip(images, masks), total=len(images)):
    #     """ Extract the name """
    #     name = x.split("/")[-1].split(".")[0]

    #    """ Read image and mask """
    #input_image
    x = cv2.imread("/user/cspingola/rail_marking-master/color_mask.png", cv2.IMREAD_COLOR)
    x = x[512-160:512,:,:]
    #mask
    y = cv2.imread("/user/cspingola/rail_marking-master/mask.png", cv2.IMREAD_GRAYSCALE)
    y = y[512-160:512,:]

    #color_mask
    z = cv2.imread("/user/cspingola/rail_marking-master/color_mask.png")
    z = z[512-160:512,:]
    print(z.shape)
    """ Detecting bounding boxes """
    bboxes = mask_to_bbox(y,z) #it is a list containing the coords

    """ marking bounding box on image """
    for bbox in bboxes:
        x = cv2.rectangle(x, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)

    """ Saving the image """
    #cat_image = np.concatenate([x, parse_mask(y)], axis=1)
    cv2.imwrite("/user/cspingola/rail_marking-master/result_CIAO.JPG", x)

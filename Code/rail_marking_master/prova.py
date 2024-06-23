import cv2
import torch
from torchvision.ops import masks_to_boxes
from torchvision.utils import draw_bounding_boxes

obtained_mask = cv2.imread("/user/cspingola/rail_marking-master/mask.png", cv2.IMREAD_GRAYSCALE)
obtained_mask = torch.tensor(obtained_mask).unsqueeze(0)[:,400:512,:]
color_mask = cv2.imread("/user/cspingola/rail_marking-master/color_mask.png")
color_mask = torch.tensor(color_mask)[400:512,:,:].transpose(0, 2).transpose(1, 2)
print("obtained_mask.size", obtained_mask.size())
#print("obtained_mask", obtained_mask)
print("color_mask.size", color_mask.size())
#print("color_mask", color_mask)
# We get the unique colors, as these would be the object ids.
obj_ids = torch.unique(obtained_mask)
print(obj_ids)
# first id is the background, so remove it.
obj_ids = obj_ids[:-1]

# split the color-encoded mask into a set of boolean masks.
# Note that this snippet would work as well if the masks were float values instead of ints.
masks = obtained_mask == obj_ids[:, None, None]
print(masks.size())
print(masks)
boxes = masks_to_boxes(masks)
print(boxes.size())
print(boxes)
drawn_boxes = draw_bounding_boxes(color_mask, boxes, colors="red")
drawn_boxes = drawn_boxes.permute(1, 2, 0)
cv2.imwrite("/user/cspingola/rail_marking-master/final.png", drawn_boxes.numpy())
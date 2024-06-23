#!/usr/bin/env python
import os
import cv2
import torch
import numpy as np
from ..models import BiSeNetV2
from ..data_loader import Rs19DatasetConfig
from torchscan import summary


__all__ = ["RailtrackSegmentationHandler"]


class RailtrackSegmentationHandler:
    def __init__(self, path_to_snapshot, model_config, overlay_alpha=0.5):
        if not os.path.isfile(path_to_snapshot):
            raise Exception("{} does not exist".format(path_to_snapshot))

        self._model_config = model_config
        self._overlay_alpha = overlay_alpha

        self._data_config = Rs19DatasetConfig()
        self._model = BiSeNetV2(n_classes=self._data_config.num_classes)
        #due righe successive messe da me
        if not torch.cuda.is_available():
            self._model.load_state_dict(torch.load(path_to_snapshot, map_location = torch.device('cpu'))["state_dict"])
        else:
            self._model.load_state_dict(torch.load(path_to_snapshot)["state_dict"])
        self._model.eval()

        if torch.cuda.is_available():
            self._model = self._model.cuda()

    def run(self, image, only_mask=True, modified_version=False):
        orig_height, orig_width = image.shape[:2]
        processed_image = cv2.resize(image, (self._model_config.img_width, self._model_config.img_height))

        if not only_mask:
            overlay = np.copy(processed_image)

        processed_image = processed_image / 255.0
        processed_image = torch.tensor(processed_image.transpose(2, 0, 1)[np.newaxis, :]).float()

        if torch.cuda.is_available():
            processed_image = processed_image.cuda()
        
        #tale processed_image ha una dimensione che è pari a (1,3,512,1024)

        #print("processed_image.shape" , processed_image.shape)
        #print("rete:",print(self._model))
        output = self._model(processed_image)[0] #tensore output del modello, ha dimensioni (3,512,1024)
        #print("output.shape", output.shape)
    
        #mask dovrebbe contenere per ogni pixel un numero che è la classe di riferimento
        #e ha dimensioni (512,1024)
        mask = torch.argmax(output, axis=0)
        if modified_version:
            mask = torch.where(mask == 1, torch.tensor(mask-1), torch.where(mask == 2,torch.tensor(mask-1), mask ))

        mask = (
            mask
            .cpu()
            .numpy()
            .reshape(self._model_config.img_height, self._model_config.img_width)
        )

        #perform closing (dilation following by erosion) on the mask in order to fill incorrect empty points
        #print("mask:" , mask.shape, "type(mask)", type(mask))
        conv_mask = cv2.convertScaleAbs(mask)
        kernel = np.ones((11,11),np.uint8)
        closing = cv2.morphologyEx(conv_mask, cv2.MORPH_CLOSE, kernel)
        mask = closing
        
        
        

        if not only_mask:
            #color mask dovrebbe essere l'immagine tuta nera e con evidenziati solo i colori che sono specifici
            #degli oggetti individuati.
            #ha dimensioni (512,1024,3)
            if modified_version:
                colors = self._data_config.RS19_COLORS_MODIFIED
            else:
                colors = self._data_config.RS19_COLORS
            color_mask = np.array(colors)[mask]
            #overlay invece contiene la sovrapposizione di tali colori sull'immagine originale
            overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            overlay = (((1 - self._overlay_alpha) * overlay) + (self._overlay_alpha * color_mask)).astype("uint8")
            #overlay = cv2.resize(overlay, (orig_width, orig_height))
            return mask, overlay, color_mask, processed_image

        return mask

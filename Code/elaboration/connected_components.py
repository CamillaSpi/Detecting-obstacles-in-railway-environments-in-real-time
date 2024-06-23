import cv2
import numpy as np
from PIL import Image

def if_some_rails(patch):
    for row in patch:
        for element in row:
            if element != 0: #significa che ci sono binari
                return True
    return False

def check_if_contained(t,x1_analysed,y1_analysed,x2_analysed,y2_analysed,contained):
   #controlliamo se quello in input è contenuto all'interno di qualcun altro
    for k in range(len(contained)):
        if k!=t:
            x1_to_analyze = contained[k][0]
            y1_to_analyze = contained[k][1]
            x2_to_analyze = contained[k][2]
            y2_to_analyze = contained[k][3]
            if (x1_analysed >= x1_to_analyze and
                y1_analysed >= y1_to_analyze and
                x2_analysed <= x2_to_analyze and
                y2_analysed <= y2_to_analyze): #in questo caso è contenuto
                
                return True
    return False


def check_if_satisfied(contained,slice_height, image):
    #img = Image.fromarray(image)
    img = image
    black_mask = np.zeros((slice_height, 1024), dtype=np.uint8)
    #black_mask = Image.new('1', (1024, slice_height), 0) #creiamo una maschera grande quanto tutta la patch
    
    #print("black_mask.size", black_mask.size, "image.size", img.size)
    #analizziamo tutti i bbox che sono contenuti nella patch in esame
    to_add = []
    for t,box in enumerate(contained):
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]
        black_mask[y1:y2, x1:x2] = 1 #nella maschera settiamo a 255 i pixels corrispondenti a quei bbox
        res = check_if_contained(t,x1,y1,x2,y2,contained)
        if not res and (y2-y1)*(x2-x1)>=700: #controlliamo se quelli interni sono fra loro contenuti e che abbiano una certa area minima messa a 200
            to_add.append(box)
    # Estrai i pixel di interesse dall'immagine originale
    masked_image = cv2.bitwise_and(img, black_mask) #facciamo l'and tra le due in modo tale da ottenere bianchi solo i pixels bianchi in entrambe le maschere

    # Calcola i pixel esterni 
    external_pixels = cv2.subtract(img, masked_image) #calcoliamo i pixel che sono diversi da nero nell'immagine originale e non nella maschera

    # Cerca pixel di un certo tipo (ad esempio, colore rosso) tra i pixel esterni
    target_color = 1  
    external_pixels_red = cv2.inRange(external_pixels, target_color, target_color)

    # Controlla se ci sono pixel bianchi esterni ai bounding boxes
    if cv2.countNonZero(external_pixels_red) > 0: #controlliamo che vi siano o no pixel bianchi contenuti nell'immagine originale e non nella maschera
        return True, to_add #non tutti i pixel bianchi vengono coperti da i bbox interni, bisogna aggiungere quello esterno
    else:
        return False, to_add #tutti i pixel bianchi vengono coperti dai bbox interni, non c'è bisogno di quello esterno



def check_consistency(definitive_bboxes, slice_height, image):
    #ordiniamoli per dimensione
    max_w = 0
    max_h = 0
    for i,bbox in enumerate(definitive_bboxes):
        contained = []
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        width = x2-x1
        height = y2-y1
        if width >= max_w and height >= max_h:
            max_w = width
            max_h = height
            max_bbox = bbox
    resultant = []
    for i,bbox in enumerate(definitive_bboxes):
        contained = []
        to_check = []
        x1_analysed = bbox[0]
        y1_analysed = bbox[1]
        x2_analysed = bbox[2]
        y2_analysed = bbox[3]
        if bbox == max_bbox:
        #if y2_analysed-y1_analysed == slice_height and x2_analysed-x1_analysed == 1024: #è grande quanto tutta la patch
            for j in range(len(definitive_bboxes)):
                if j!=i:
                    x1_to_analyze = definitive_bboxes[j][0]
                    y1_to_analyze = definitive_bboxes[j][1]
                    x2_to_analyze = definitive_bboxes[j][2]
                    y2_to_analyze = definitive_bboxes[j][3]
                    #controllo se qualcuno degli altri bbox della patch è contenuto all'interno di quello analysed
                    if (x1_analysed <= x1_to_analyze and
                        y1_analysed <= y1_to_analyze and
                        x2_analysed >= x2_to_analyze and
                        y2_analysed >= y2_to_analyze):
                        #abbiamo trovato un bbox contenuto
                        contained.append(definitive_bboxes[j])
                    else:
                        to_check.append(definitive_bboxes[j])
            if len(contained) == 0: #se non abbiamo trovato bbox contenuti allora significa che c'è davvero un bbox grande quanto tutta la patch
                resultant.append(bbox)
            if len(contained) != 0: #se abbiamo trovato dei bbox contenuti dobbiamo capire se quello più esterno è davvero utile oppure no
                ret, to_add = check_if_satisfied(contained,slice_height, image) #se true significa che quello esterno serve e tutti gli altri possono essere scartati
                if ret == True:
                    resultant.append(bbox)
                else:
                    resultant = to_add #restituisco solo quelli interni che non sono fra loro contenuti
            if len(to_check)!=0:
                for c,bbox in enumerate(to_check):
                    x1_1 = bbox[0]
                    y1_1 = bbox[1]
                    x2_1 = bbox[2]
                    y2_1 = bbox[3]
                    res = check_if_contained(c, x1_1, y1_1, x2_1, y2_1,to_check)
                    if not res:
                        resultant.append(bbox)
                    

    return resultant


#gli passiamo la color mask
def found_connected_components(image,  slice_height):
    # Carica l'immagine in scala di grigi
    #image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Binarizza l'immagine (regolando la soglia secondo le tue esigenze)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Etichettatura delle componenti connesse
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)
    
    # Visualizzazione delle etichette sull'immagine originale (opzionale)
    labeled_image = np.copy(image)
    for label in range(1, num_labels):  # Ignoriamo l'etichetta 0 (background)
        labeled_image[labels == label] = label * 50  # Colore casuale per ciascuna etichetta
    
    #cv2.imwrite("/user/cspingola/Tesi/post_elaboration/connected_components_" + str(slice_height) + ".jpg", labeled_image)

    # stats contiene le informazioni sugli attributi delle componenti connesse
    # Esempio: stats[label] = [x, y, width, height, area]
    # Dove (x, y) è l'angolo in alto a sinistra del bounding box, e area è l'area della componente.
    definitive_bboxes = []
    for stat in stats:
        x = stat[0]
        y = stat[1]
        width = stat[2]
        height = stat[3]
        area = stat[4]
        # if slice_height == 22:
        #     print("stat prima dell'if",stat)
        if if_some_rails(image[y:y+height , x:x+width]):#(width!=1024 or height!=slice_height): 
            definitive_bboxes.append([x,y,x+width, y+height])

    #print("len(definitive_bboxes) prima della consistency", len(definitive_bboxes))
    # if slice_height == 22:
    #     print("definitive_bboxes prima della check consistency",definitive_bboxes)
    definitive_bboxes = check_consistency(definitive_bboxes, slice_height, image)
    # if slice_height == 22:
    #     print("definitive_bboxes dopo la check consistency",definitive_bboxes)
    #return num_labels - 1, labels, stats, centroids, labeled_image, image, definitive_bboxes
    #print("len(definitive_bboxes) dopo la consistency", len(definitive_bboxes))
    return definitive_bboxes

# Esempio di utilizzo
# image = cv2.imread("/user/cspingola/rail_marking-master/color_mask.png", cv2.IMREAD_GRAYSCALE)
# image = image[512-160:512,:]
# definitive_bboxes = found_connected_components(image,)

#print(f"Numero di componenti connesse trovate: {num_labels}")
# Fai qualcosa con le informazioni ottenute, ad esempio puoi visualizzare l'immagine con le etichette:
# cv2.imshow("Labeled Image", labeled_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#cv2.imwrite("/user/cspingola/rail_marking-master/labels.png", labeled_image)
#cv2.imwrite("/user/cspingola/rail_marking-master/hope.png", image)

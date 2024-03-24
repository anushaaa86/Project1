# -*- coding: utf-8 -*-
import numpy as np
import cv2 as cv2
print("cv2 version:",cv2.__version__)
# import os

# %matplotlib inline
# import matplotlib as mpl
import matplotlib.pyplot as plt
# mpl.rc('axes', labelsize=14)
# mpl.rc('xtick', labelsize=12)
# mpl.rc('ytick', labelsize=12)

#Routine to fix 
def fixColor(image):
    return(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

np.random.randint(0, 255, size=((10), 3))


label_file="/content/object_detection_classes_coco.txt"
LABELS = open(label_file).read().strip().split("\n")
np.random.seed(42) #Set seed so that we get the same results everytime
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

COLORS.shape==(90, 3)

threshold=0.9


LABELS[:5]


weights="/content/mask_rcnn_frozen_inference_graph.pb"
config="/content/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

print(weights=="/content/mask_rcnn_frozen_inference_graph.pb")
config=="/content/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"

!head -10 /content/mask_rcnn_frozen_inference_graph.pb

net = cv2.dnn.readNetFromTensorflow(weights, config)

str(type(net))=="<class 'cv2.dnn_Net'>"

img = cv2.imread('/content/dining_table.jpg')


blob = cv2.dnn.blobFromImage(img, swapRB=True, crop=False)


net.setInput(blob)

str(type(img))=="<class 'numpy.ndarray'>"



(boxes, masks) = net.forward(["detection_out_final",
		"detection_masks"])
# print ("Shape of boxes", boxes.shape)
# print ("Shape of masks", masks.shape)

type(masks)



for i in range(0, boxes.shape[2]): #For each detection
    classID = int(boxes[0, 0, i, 1]) #Class ID
    confidence = boxes[0, 0, i, 2] #Confidence scores
    if confidence > threshold:
        (H, W) = img.shape[:2]
        box = boxes[0, 0, i, 3:7] * np.array([W, H, W, H]) #Bounding box
        (startX, startY, endX, endY) = box.astype("int")
        boxW = endX - startX
        boxH = endY - startY

        # extract the pixel-wise segmentation for the object,       
        mask = masks[i, classID]
        plt.imshow(mask)
        plt.show()
        print ("Shape of individual mask", mask.shape)
        
        # resize the mask such that it's the same dimensions of
        # the bounding box, and interpolation gives individual pixel positions
        mask = cv2.resize(mask, (boxW, boxH), interpolation=cv2.INTER_CUBIC)
        print ("Mask after resize", mask.shape)
        # then finally threshold to create a *binary* mask
        mask = (mask > threshold)
        print ("Mask after threshold", mask.shape)
        
       
        roi = img[startY:endY, startX:endX][mask]
        print ("ROI Shape", roi.shape)
    
        color = COLORS[classID]
        blended = ((0.4 * color) + (0.6 * roi)).astype("uint8")

        # Change the colors in the original to blended color
        img[startY:endY, startX:endX][mask] = blended

        color = COLORS[classID]
        color = [int(c) for c in color]
        print (LABELS[classID], color)
        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
        text = "{}: {:.4f}".format(LABELS[classID], confidence)
        cv2.putText(img, text, (startX, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

plt.imshow(fixColor(img))

roi.shape==(13497, 3)

color==[209, 226, 77]


import cv2
import numpy as np
from skimage.filters import threshold_local
import tensorflow as tf
from skimage import measure
import imutils
import os

def sort_cont(character_contours):
    """ 
    To sort contours 
    """
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours] 
    
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, 
                                                        boundingBoxes), 
                                                    key = lambda b: b[1][i], 
                                                    reverse = False)) 
    
    return character_contours


def segment_chars(plate_img, fixed_width): 
    
    """ 
    extract Value channel from the HSV format 
    of image and apply adaptive thresholding 
    to reveal the characters on the license plate 
    """
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2] 

    thresh = cv2.adaptiveThreshold(V, 255, 
                                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY, 
                                11, 2) 
    
    thresh = cv2.bitwise_not(thresh) 

    # resize the license plate region to 
    # a canoncial size 
    plate_img = imutils.resize(plate_img, width = fixed_width) 
    thresh = imutils.resize(thresh, width = fixed_width) 
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) 

    # perform a connected components analysis 
    # and initialize the mask to store the locations 
    # of the character candidates 
    labels = measure.label(thresh, background = 0) 

    charCandidates = np.zeros(thresh.shape, dtype ='uint8') 

    # loop over the unique components 
    characters = [] 
    for label in np.unique(labels): 
        
        # if this is the background label, ignore it 
        if label == 0: 
            continue
        # otherwise, construct the label mask to display 
        # only connected components for the current label, 
        # then find contours in the label mask 
        labelMask = np.zeros(thresh.shape, dtype ='uint8') 
        labelMask[labels == label] = 255

        cnts = cv2.findContours(labelMask, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE) 

        cnts = cnts[1] if imutils.is_cv3() else cnts[0] 

        # ensure at least one contour was found in the mask 
        if len(cnts) > 0: 

            # grab the largest contour which corresponds 
            # to the component in the mask, then grab the 
            # bounding box for the contour 
            c = max(cnts, key = cv2.contourArea) 
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c) 

            # compute the aspect ratio, solodity, and 
            # height ration for the component 
            aspectRatio = boxW / float(boxH) 
            solidity = cv2.contourArea(c) / float(boxW * boxH) 
            heightRatio = boxH / float(plate_img.shape[0]) 

            # determine if the aspect ratio, solidity, 
            # and height of the contour pass the rules 
            # tests 
            keepAspectRatio = aspectRatio < 1.0
            keepSolidity = solidity > 0.15
            keepHeight = heightRatio > 0.5 and heightRatio < 0.95

            # check to see if the component passes 
            # all the tests 
            if keepAspectRatio and keepSolidity and keepHeight and boxW > 14: 
                
                # compute the convex hull of the contour 
                # and draw it on the character candidates 
                # mask 
                hull = cv2.convexHull(c) 

                cv2.drawContours(charCandidates, [hull], -1, 255, -1) 

    contours, hier = cv2.findContours(charCandidates, 
                                    cv2.RETR_EXTERNAL, 
                                    cv2.CHAIN_APPROX_SIMPLE) 
    
    if contours: 
        contours = sort_cont(contours) 
        
        # value to be added to each dimension 
        # of the character 
        addPixel = 4
        for c in contours: (x, y, w, h) = cv2.boundingRect(c) 
        if w > 14: 
                roi = thresh[y-addPixel:y+h+addPixel, 
                            x-addPixel:x+w+addPixel] 
                characters.append(roi) 
    return characters

# Load the video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    
    # Check if the frame is read correctly
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display the grayscale frame
    cv2.imshow('frame', gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
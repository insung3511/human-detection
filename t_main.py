import cv2
import time
import numpy as np
import imutils
from imutils import paths
from imutils.object_detection import non_max_suppression



def get_num_of_people(input_path, output_path):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    i = 0

    image = cv2.imread(input_path)
    image = imutils.resize(image, width=min(400, image.shape[1]))
    orig = image.copy()
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
        cv2.imwrite(output_path, image)
        i+=1

    filename = input_path[input_path.rfind("/") + 1:]
    if not i == 0:
        return True
    return False
#usage code
def main() :
    #below ones are added by yg
    import sys
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    print(get_num_of_people(input_path, output_path))
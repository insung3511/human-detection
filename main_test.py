import cv2
import time
import numpy as np
import imutils
from imutils import paths
from imutils.object_detection import non_max_suppression

start = time.time()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
timeFile = open('time.txt', 'w')
loopCounter = 0
timeAvg = 0.0

print("[INFO] ======Program Start======")

while loopCounter < 100:
    for imagePath in paths.list_images('./HO'):
        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=min(400, image.shape[1]))
        orig = image.copy()

        (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

        for (xA, yA, xB, yB) in pick:
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
            cv2.imwrite('result.jpg', image)

        filename = imagePath[imagePath.rfind("/") + 1:]
        print("[INFO] {}: {} original boxes, {} after suppression".format(filename, len(rects), len(pick)))
        log = str(time.time() - start) + '\n'
        timeFile.write(log)

        loopCounter = loopCounter + 1
        timeAvg = (timeAvg + (time.time() - start)) 

print("[INFO] Time : ", time.time() - start)
print("[DONE] Time-Avg : ", timeAvg / loopCounter)
timeFile.close()

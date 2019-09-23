import cv2
import time
import numpy as np
import imutils
from imutils import paths
from imutils.object_detection import non_max_suppression

start = time.time()
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

i = 0

for imagePath in paths.list_images('./test'):
    image = cv2.imread(imagePath)
    image = imutils.resize(image, width=min(480, image.shape[1]))
    orig = image.copy()

    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=2.06223455)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0),4)
        cv2.imwrite('result.jpg', image)
        print(imagePath)
    print("----------")

    cv2.imshow("orig", image)
    cv2.waitKey(0)
    filename = imagePath[imagePath.rfind("/") + 1:] 

print("time : ", time.time() - start)
print(i)
# import packages
import numpy as np
import cv2
import argparse

# construct the argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load SSD and ResNet network based caffe model for 300x300 dim imgs
net = cv2.dnn.readNetFromCaffe("weights-prototxt.txt", "res_ssd_300Dim.caffeModel")

# load the input image by resizing to 300x300 dims
image = cv2.imread(args["image"])
(height, width) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
	(300, 300), (104.0, 177.0, 123.0))

# pass the blob into the network
net.setInput(blob)
detections = net.forward()

	# loop over the detections to extract specific confidence
for i in range(0, detections.shape[2]):
    	# extract the confidence (i.e., probability) associated with the
	# prediction
	confidence = detections[0, 0, i, 2]

	# greater than the minimum confidence
	if confidence > 0.5:

		box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
		(x1, y1, x2, y2) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100) + " ( " + str(y2-y1) + ", " + str(x2-x1) + " )"
		y = y1 - 10 if y1 - 10 > 10 else y1 + 10
		cv2.rectangle(image, (x1, y1), (x2, y2),
			(0, 0, 255), 2)
		cv2.putText(image, text, (x1, y),
			cv2.LINE_AA, 0.45, (0, 0, 255), 2)

# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)
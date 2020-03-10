# import packages
import numpy as np
import cv2

# load SSD and ResNet network based caffe model for 300x300 dim imgs
net = cv2.dnn.readNetFromCaffe("weights-prototxt.txt", "res_ssd_300Dim.caffeModel")

# video stream initialization
vs = cv2.VideoCapture(0) 

# loop over video frames
while True:
	ret, frame = vs.read()

	# convert frame dimensions to a blob and 300x300 dim
	(height, width) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
		(300, 300), (104.0, 177.0, 123.0))
 
	# pass the blob into dnn 
	net.setInput(blob)
	detections = net.forward()

	# loop over the detections to extract specific confidence
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		# greater than the minimum confidence
		if confidence < 0.5:
			continue

		# compute the boxes (x, y)-coordinates
		box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
		(x1, y1, x2, y2) = box.astype("int")
 
		# draw the bounding box of the face along with the associated
		# probability
		text = "{:.2f}%".format(confidence * 100) + " ( " + str(y2-y1) + ", " + str(x2-x1) + " )"
		y = y1 - 10 if y1 - 10 > 10 else y1 + 10
		cv2.rectangle(frame, (x1, y1), (x2, y2),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (x1, y),
			cv2.LINE_AA, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Window", frame)
 
	# if the `w` key was pressed, break from the loop
	if cv2.waitKey(1) == ord("w"):
		break
# stop capturing
cv2.destroyAllWindows()
vs.stop()
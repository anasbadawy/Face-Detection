# Face-Detection
face detector based on OpenCV and deep learning using opencv's Caffe model
## Project Description:
Facial key-points detection project (called also facial landmarks) is a system that is based on deep learning and computer vision techniques. It is actually able to allocate eyes, nose, and mouth positions on human faces after cropping the face location. By using a small magenta dots shown on detected faces. It is 68 key-points.







<p align="center">
 <img  width="350" height="350" src="https://github.com/anasbadawy/Face-Detection/blob/master/testResult.png">
</p>





## Methodology:

First all the faces in an image are detected by using a predefined face detector which is Haar Cascade detector. Then by using CNN (Convolutional Neural Network), the detected faces are fed to model after preprocessing it by following steps:

- Normalizing: to convert a color image to grayscale values with a range of [0,1] and normalize the key-points to be in a range of about [-1, 1].
- Rescaling: to rescale an image to a desired size 224*224 pixels.
- Cropping: to crop an image such that to pass a square image to CNN.

Then CNN model passing the image through multiple type of layers (Convolution layers - Max-pooling layers - Fully-connected layers) and outputting 136 values that is used as 68 (x,y) points on detects faces.



## CNN Architecture:
- Four Convolutional layers.
- Max-pooling layers.
- Three Fully-connected layers.


## Dataset:

The dataset that is used for training CNN model on has been extracted from the YouTube Faces Dataset which includes videos of people in YouTube videos. These videos have been fed through some processing steps and turned into sets of image frames containing one face and the associated key-points. The dataset is consisting of 4232 color images separated into training and testing sets.

- 3462 of these images for training set. 
- 770 of these images for testing set.

It’s including CSV file that has the 68 points as (x,y). so it’s 136 value and image name features for each image of the dataset.




**The Result of testing an image for me with my friends on a wedding**



<p align="center">
 <img  width="350" height="350" src="https://github.com/anasbadawy/Face-Detection/blob/master/testResult.png">
</p>








## References:
https://opencv.org/

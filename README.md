# CV | Detection and identification of people by face in real time using Haar cascades

<p align="center">
  <img src="images_to_report\logo.jpeg" width="300">
</p>

In this project, I will implement real-time face recognition from my laptop camera using Haar cascades and train a model to identify specific people.

Remark: comments in the code will be written in Russian

## A brief summary of Haar Cascades' capabilities
[defolt_haar_tasks](defolt_haar_tasks.py) - code used to demonstrate some scenarios for using Haar Cascades, specifically to find eyes, smile and face in a photo, and to blur the selected area, in our case the face.

Result:

<img src="images_to_report\eyes.jpg" width="400">
<img src="images_to_report\face.jpg" width="400">
<img src="images_to_report\smile.jpg" width="400">
<img src="images_to_report\blure.jpg" width="400">


## Model training and identification
[face_train](face_train.py) - model training script

[face_detect](face_detect.py) - In this file I implemented real-time identification of me in the video stream using the LBPH (Local Binary Patterns Histograms) algorithm from the cv2 module, it is implemented by the function cv2.face.LBPHFaceRecognizer_create()

The algorithm works as follows:
* Splits the image into small regions.
* For each pixel in the region, compares its intensity with its neighbors and generates a binary pattern (local binary pattern).
* Constructs histograms of these patterns for all regions.
* Uses these histograms for comparison and face recognition.

Result:

<img src="images_to_report\detect.jpg" width="400">
<img src="images_to_report\detect2.jpg" width="400">

As we can see, the model recognizes me with acceptable confidence, while my friend's face is just made into a rectangle by the algorithm.

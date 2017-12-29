## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./writeup/sliding_windows.png
[image4]: ./writeup/heatmap.png
[image5]: ./writeup/max_heatmap.png
[image6]: ./writeup/windows_heatmap.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Code structur

I seperated the code into two main classes `model.py` and `vehicle-detection.py`. First the model gets generated via `model.py`and saved into a pickle file `Model.p`. The `vehicle-detecion-py` loads this model out of the pickle file and loads the video and detects the vehicles via the model.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines #24 through #25 of the file called `model.py`.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then split the loaded data into train and test sets in lines #30 through #38 of the file called `model.py`.

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters. First trying different colorspaces, like RGB, HLS and YUV. From previous experiences I liked the HLS the most. So I settled for it, but used only LS-channels.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using the hog features, color histogram features and binned color features.

I as well tried different non linear SVMs but the training and the performance of prediction was just really slow. On top the results weren't even better.

I achieved 96.5% on my model.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search windows only on the bottom of the screen. Having smaller windows at the top and bigger ones in the front. Like the car is bigger in the front and gets smaller in the back. With an overlap of 0.5 for the small windows and 0.8 for the big windows. I calculated the window steps in between for 5 iterations. I wanted more overlapping for the bigger cars because its harder to miss a car if the overlapping is to small, as well making the overlapping like 1.0 reduced the performance too.

![alt text][image3]

See code lines #19 through #33 in `detect-vehicles.py`.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on HLS (using only LS-channel) HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used my own algorithm using the max hits on each pixel, subtracted this max by 3 and used all hit areas. I calculated a window around them and expected them to be a car. I as well created a Car class (`car.py`) for saving the last center position of the car. As well only a car detected in 10 following frames are detected. If a car isn't detected in the last 5 frames it is deleted. Using it to find the same car in the next picture and to calculate an average window depending on the center position. Look at the code line #43 through #183 in `vehicle-detection.py`.

### Here is the heatmap:

![alt text][image4]

### Here is the heatmap with the max areas in green:
![alt text][image5]

### Here is the heatmap with the max areas in green and selected windows:
![alt text][image6]

### Video of the test video with the heatmap and the window merging:
Here's a [link to my test result with more details](./test_video_output.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The algorithm itself runs really slow because of my calculations. To run this detection in real time would not work with this. So improving this algorithm would be good.

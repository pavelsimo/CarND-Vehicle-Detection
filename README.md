## Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

My project includes the following files:

* ```README.md``` writeup summarizing the results
* ```Vehicle_Detection.ipynb``` a jupyter notebook with the vehicle detection pipeline
* ```project_video_output.mp4``` containing the vehicle detection results


[//]: # (Image References)
[img2]: ./examples/ex_hog_car.png
[img3]: ./examples/ex_hog_nocar.png
[img4]: ./examples/ex_hog_car_pixel_per_cell_16x16.png
[img5]: ./examples/ex_sliding_window_det1.png
[img6]: ./examples/ex_sliding_window_det2.png
[img7]: ./examples/ex_sliding_window_det3.png
[img8]: ./examples/ex_sliding_window_det4.png
[img9]: ./examples/ex_sliding_windows_search.png
[img10]: ./examples/ex_result.png
[video1]: ./project_video_output.mp4
 

### 1. Histogram of Oriented Gradients (HOG)

#### 1.1 Extracting HOG features from the training images.

I started by exploring different color spaces and different `skimage.hog()` parameters for the random 
samples of `vehicle` and `non-vehicle` classes. (`orientations`, `pixels_per_cell`, and `cells_per_block`).  From each 
of the two classes, I displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of  `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)` with orientations in the range 
of `[5, 10]`,:

![alt text][img2]
![alt text][img3]

The code for this step is contained in section 2 of the IPython notebook.  

#### 1.2 HOG parameters.

I tried various combinations of parameters, ended up choosing the following:

| HOG Parameter   | Value  | Reason |
|-----------------|--------|--------|
| pixels_per_cell | (8, 8) | Noticed that high caused deformation in the shape of the HOG that could influence classification |       
| cells_per_block | (2, 2) | Arbitrary choice, based on a good fit for the other two parameters |
| orientation     | 8      | The chosen value is some sort of a middle point. High values of orientation cause the vectors to differ in the direction more often, thus not resembling the real shape of the object. Low values of orientation caused the vectors to agree more often in the direction, both extremes create classification issues. 

Example of the deformation caused by ```pixels_per_cell=(16, 16)```:   

![alt text][img4]

### 2. Training the model

I trained a linear SVM using the following parameters:

```python
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [380, 656] # Min and max in y to search in slide_window()
```

The code for this step is contained in section 7 and 8 of the IPython notebook.

### 3. Sliding Window Search

I decided for the following search window configuration:

| Window          | Scale  | Overlap Percentage   |
|-----------------|--------|------------|
| (380, 600)      | 1.0    | 0%         |
| (400, 600)      | 1.5    | 90%        |
| (420, 700)      | 3.0    | 40%        |
 
As shown in the picture below:

![alt text][img9]

The criteria were simple, a car in the horizon should look smaller, so the window scale 1.0. Cars near the camera should
look bigger, thus the window scaling of 3.0. 

Ultimately I searched on three scales using YUV 3-channels HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. Here are some example images:

![alt text][img5]
![alt text][img6]
![alt text][img7]
![alt text][img8]

### 4. False-positives

To deal with false-positives two strategies were applied:
 
* Overlapping bounding boxes: As shown in the Sliding Window Search Section, when multiple overlapping bounding boxes 
agree in a particular classification, many cases of false-positive go away by simply finding were the boxes does not agree.

* Average Frame Smoothing: By keeping a record of the previous classify frames, I smoothed the heatmap by averaging it with
the current frame, thus giving a better prediction of the vehicle position.

The code for this step is contained in section 10 of the IPython notebook.

Here the resulting bounding boxes are drawn onto the last frame in the series:

![alt text][img10]

Here's a [link to my video result](./project_video_output.mp4)

### 5. Further Work

Here I'll talk about where the pipeline might fail and how I might improve it if I were going to pursue 
this project further:

* My current approach does not work well with different lighting conditions, we need to take into consideration
this kind of features.

* The current approach just consider cars from the rear, It may fail with other orientations. The model should be trained
with more data from cars in different orientations.

* Since the name of the project is Vehicle Detection, and not Car Detection, currently the classier does not take into account motorcycles, or any 2-wheels vehicle. This may not be classified for my program.

* My pipeline it's quite slow, takes one second to process one frame this is not feasible for production, further optimizations
must be made to run this real-time in a car.

* Engineering features is a daunting task, a better approach will be to applied deep learning so those features can be learned from the data. A further task may be trying vehicle detection with YOLO, SSD, etc. 
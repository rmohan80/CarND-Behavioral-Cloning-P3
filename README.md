#**Behavioral Cloning** 

---

**Behavrioal Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or readme.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network based on the [Nvidia model](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)  (model.py lines 9-62) 

The model includes RELU in each layer, and the data is normalized in the model using a Keras lambda layer (model.py line 14).

####2. Attempts to reduce overfitting in the model

The model contains a single dropout layer in the fully conncted layer in order to reduce overfitting (model.py line 53). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 163-176). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 60).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used the default Udacity training data with some augmentation techniques to make sure it stays on the road. The approaches to augment the data is discussed in the "Model Architecutre and Training Strategy" section

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with using the tried and tested Nvidia model.

My first step was to use a convolution neural network model similar to the Nvidia model. I thought this model might be appropriate because it has already been used and seemed to be simpler to implement than the comm.ai model

I couldn't collect additional data because my PC was too slow and my driving was very erratic and hence could not be used as a base. Hence I started doing research on how people have completed this project using the Udacity test data

I used the Slack forums and multiple posts to implement several different techniques in my model. 

These include the articles by:

 - [An augmentation based deep neural network approach to learn human
   driving behavior](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.ck3w11k6h)
 - [Behavioral Cloning project that teaches a car to drive autonomously using Deep Learning with Keras](https://github.com/wonjunee/behavioral-cloning)
 - [Self-driving car in a simulator with a tiny neural network](https://medium.com/@xslittlegrass/self-driving-car-in-a-simulator-with-a-tiny-neural-network-13d33b871234#.o0ybbzlcg)
 - [My first self-driving car](https://machinelearnings.co/my-first-self-driving-car-e9cd5c04f0f2#.2loood768)
 - [Cloning a car to mimic human driving](https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff#.2qptrlkhu)
 - And countless others on the slack and forums....!!!
 
Based on all these, I created a simple model that was able to drive till the dirt road after the bridge. The car continued straight on the dirt road and was driving well there, but off course that's a fail :)

After doing some more research, I found out that the issue is with the straight bias that many recommend dropping off the data set. I achieved this by using a combination of 2 things 

 1. Flip the image (discussed in #3 below on why this is done in the first place) only when the steering angles are high enough, i.e. >0.25 in either direction (model.py lines 104 - 107) and
 2. Drop / Replace minor steering angles with 0.0 as steering angle. This reduces the straight bias (model.py lines 110 - 111)

At the end of this process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 9-62) consisted of a convolution neural network with the following layers and layer sizes:

![Final Model](/images/model.png?raw=true)

####3. Creation of the Training Set & Training Process
Since I was unable to create additional training data due to my PC being too slow and keyboard not supporting proper driving I did the following:

Used image flipping technique at a 50% probability. That way, the left bias is reduced and gives additional right bias since the training set is heavily skewed on the left

The below shows the Udacity training set distribution
![Data set distribution](/images/steering_angles.png?raw=true)

The below image shows the left and corresponding flipped right image

![Left Image](/images/left_image.png?raw=true)
![Flipped Image](/images/flipped_image.png?raw=true)

I then also incorporated a technique to pick up a random camera - either left, right or center for each row in the data. This helped reduce the bias in a specific camera angle. There was a +/- 0.25 added to the left and right camera to make sure that the steering angle was compensated (as it is always recorded from a center camera perspective).

The images of left, center and right at a random position is given below


![Left Camera](/images/camera_left.png?raw=true)
![Center Camera](/images/camera_center.png?raw=true)
![Right Camera](/images/camera_right.png?raw=true)

I also added slight noise to the training data by adding minor random shift to both the image and steering angle. The shift is very subtle that it doesn't ruin the steering angle, but good enough to augment the data points in terms of noise

An example of this shift is shown below. The original and shifted image is shown below:
![Original Image](/images/orig_image.png?raw=true)
![Transformed Image](/images/trans_image.png?raw=true)

Using the above techniques in conjunction with normal programming technique, I obtained 20,000 samples with just 8,000 images. Theoretically, I could obtain any number of images with these techniques. The upper limit is limited by the memory that my CPU has

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was between 3 to 5 as evidenced by the fact that the training loss started increasing after this and validation set wasn't performing any better.

I used an adam optimizer so that manually training the learning rate wasn't necessary.


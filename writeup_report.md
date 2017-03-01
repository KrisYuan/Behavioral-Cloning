#**Behavioral Cloning** 


###The goal of this project is to teach a car driving automatically through deep learning. The steering angle can be adjusted by a car by using convolutional neural network.

---

**Behavioral Cloning Project**

The steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/center_2017_03_01_10_13_56_000.jpg "Center Image"
[image3]: ./examples/left_2017_03_01_10_13_56_000.jpg "Left Image"
[image4]: ./examples/right_2017_03_01_10_13_56_000.jpg "Right Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points 
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 64 (model.py lines 69-95) 

The model includes RELU layers to introduce nonlinearity (code line 70), and the data is normalized in the model using a Keras lambda layer (code line 66). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 87). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 97). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 96).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving,  left lane driving and right lane driving.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a 3 layers convolutional neural network. 

My first step was to use a convolution neural network model similar to the VGG-16. I thought this model might be appropriate because it is suitable to build an efficient image classifier based on small dataset.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. I try to reduce overfitting by using dropout and reduce the epoch from 10 to 5.

The final step was to run the simulator to see how well the car was driving around track one. The vehicle doesn't keep on the center of the road. To improve the driving behavior, I added left image and right image to the dataset. I also adjusted both left and right steering angle according to the following formula.

* LeftAngle = CenterAngle + correction 

* RightAngle =  CenterAngle - correction 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 64-95) consisted of a convolution neural network with the following layers and layer sizes.

____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
____________________________________________________________________________________________________
cropping2d_1 (Cropping2D)        (None, 80, 320, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 80, 320, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 78, 318, 32)   896         lambda_1[0][0]                   
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 78, 318, 32)   0           convolution2d_1[0][0]            
____________________________________________________________________________________________________
maxpooling2d_1 (MaxPooling2D)    (None, 39, 159, 32)   0           activation_1[0][0]               
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 37, 157, 32)   9248        maxpooling2d_1[0][0]             
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 37, 157, 32)   0           convolution2d_2[0][0]            
____________________________________________________________________________________________________
maxpooling2d_2 (MaxPooling2D)    (None, 18, 78, 32)    0           activation_2[0][0]               
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 16, 76, 64)    18496       maxpooling2d_2[0][0]             
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 16, 76, 64)    0           convolution2d_3[0][0]            
____________________________________________________________________________________________________
maxpooling2d_3 (MaxPooling2D)    (None, 8, 38, 64)     0           activation_3[0][0]               
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 19456)         0           maxpooling2d_3[0][0]             
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 64)            1245248     flatten_1[0][0]                  
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 64)            0           dense_1[0][0]                    
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 64)            0           activation_4[0][0]               
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 16)            1040        dropout_1[0][0]                  
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 16)            0           dense_2[0][0]                    
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 16)            0           activation_5[0][0]               
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 1)             17          dropout_2[0][0]                  
____________________________________________________________________________________________________
Total params: 1,274,945
Trainable params: 1,274,945
Non-trainable params: 0

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I also recorded the vehicle driving situation from the left side and right sides of the road and steering angle for both sides. Thus the vehicle would learn to handle other situation as well. 

![alt text][image3]
![alt text][image4]

After the collection process, I had 5580 number of data points. I split this data to training set and validation set and got 4464 samples on training, 1116 samples on validate. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. Finally the validation loss of the model is 0.0400. I used an adam optimizer so that manually training the learning rate wasn't necessary.

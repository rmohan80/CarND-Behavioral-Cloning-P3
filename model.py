import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D, Activation
import cv2
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.visualize_util import plot

# Using Nvidia model with additional dropuot layer
def nvidia_model():
  row, col, depth = 64, 64, 3
  model = Sequential()

  # normalize image values between -.5 : .5
  model.add(Lambda(lambda x: x/255 - .5, input_shape=(row, col, depth), output_shape=(row, col, depth)))
  
  #Layer 1 = Input 64x64x3 - Output 30x30x24
  model.add(Convolution2D(24, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode='valid'))
  model.add(Activation('relu'))

  #Layer 2 = Input 30x30x24 - Output 13x13x26
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(Activation('relu'))

  #Layer 3 = Input 13x13x26 - Output 5x5x48
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(Activation('relu'))
  
  #Layer 4 = Input 5x5x48 - Output 3x3x64
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
  model.add(Activation('relu'))
  
  #Layer 5 = Input 3x3x64 - Output 1x1x64
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
  model.add(Activation('relu'))

  #Fully Connected Layer1 = Input 1x1x64 - Output 64
  model.add(Flatten())
  model.add(Activation('relu'))

  #Fully Connected Layer 2 = Input 64 - Output 100
  model.add(Dense(100))
  model.add(Activation('relu'))

  #Fully Connected Layer 3 = Input 100 - Output 50
  model.add(Dense(50))
  model.add(Activation('relu'))

  #Fully Connected Layer 2 = Input 50 - Output 10
  model.add(Dense(10))
  model.add(Activation('relu'))

  #Add a dropout layer not available in Nvidia model to reduce overfitting
  model.add(Dropout(0.5)) 

  #Fully Connected Layer 2 = Input 10 - Output 1
  model.add(Dense(1))


  #compile with normal adam optimizer and return
  model.compile(optimizer="adam", loss='mse')  
  model.summary()
 
  plot(model, to_file='images/model.png', show_shapes=True)
  return model

# Helper function to add random horizontal and vertical shift to images
# Default input of transformation randomness set at 0.004 if nothing supplied
def trans_image(image,steer, trans_range=0.004):
    
	# Apply some randomness to the steering angle (currently using default 40 as value to modify)
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2

    # Apply some randomness to the image itself (using 40 as above
    tr_y = 40*np.random.uniform()-40/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1],image.shape[0]))
    
	# Return the modified image and steering angle
    return image_tr,steer_ang

# Helper function to return 1 image along with steering angle
# Input - image paths for every row in the csv
# Output - processed image + steering angle for that particular image
def get_row(row):
    
	# Get the steering angle for the specified row
	steering = row['steering']

	# randomly choose the camera to take the image from
	camera = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left and right cameras by an offset (Currently hardcoded at +/-0.25)
	if camera == 'left':
		steering += 0.25
	elif camera == 'right':
		steering -= 0.25
	
	#Load the specified image
	image = load_img("./data/" + row[camera].strip())
	image = img_to_array(image)

	# Randomly flip 50% of images to augment data
	flip_prob = np.random.random()
	
	if flip_prob > 0.5 and (steering < -0.25 or steering > 0.25):
		# flip the image and reverse the steering angle only for sharp turns 50% of times
		steering = -1*steering
		image = cv2.flip(image, 1)

    #Drop almost straight steering angles with a bias towards the left
	if steering > -0.25 and steering < 0.10:
		return cv2.resize(image, (64,64)), 0.0

	# Add random horizontal and vertical shifts to images
	tr_image, steering = trans_image(image,steering)

	# Resize modified image to 64x64 to get out final image
	final_image = cv2.resize(tr_image, (64,64))
	
	# Return the final image along with the steering angle
	return final_image, steering


# Data generator to get our training and validation data
# Default batch size of 32 is being used
# Input is a chunk from csv that includes different camera images + steering image for that particular image
def data_generator(data_frame, batch_size=32):

	#Number of batches is identified using a function of batch size and 
	N = data_frame.shape[0]
	batches_per_epoch = N // batch_size
	
	i = 0
	while(True):
		start = i*batch_size
		end = start+batch_size - 1
		
		# Make sure our array is pre-filled with the correct shape, i.e. 64x64x3
		X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
		y_batch = np.zeros((batch_size,), dtype=np.float32)
		
		j = 0

        # create a chunk from the dataframe equivalent to batch_size
		for index, row in data_frame.loc[start:end].iterrows():
			X_batch[j], y_batch[j] = get_row(row)
			j += 1
			
		i += 1
		if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
			i = 0
		yield X_batch, y_batch


if __name__ == "__main__":

	# Use pandas helper function to read the CSV file
    data_frame = pd.read_csv('./data/driving_log.csv', usecols=[0, 1, 2, 3])

    # shuffle the data
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    # Take 80% of the data as training data, rest (20%) is validation
    training_split = 0.8

	# Figure out the number of rows for the training data
    num_rows_training = int(data_frame.shape[0]*training_split)

    training_data = data_frame.loc[0:num_rows_training-1]
    validation_data = data_frame.loc[num_rows_training:]

    data_frame = None  # release the main data_frame from memory

    # Generate the training and validation data
    training_generator = data_generator(training_data, 32)
    validation_data_generator = data_generator(validation_data, 32)

    model = nvidia_model()

    samples_per_epoch = (20000//32)*32

    # Use fit_generator to simplify data generation on the fly
	# Many values are hard-coded here which should probably be defined as STATIC VARIABLES
    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=5, nb_val_samples=3000)

    print("Saving model weights and configuration file.")
    model.save('model.h5')  # always save your weights after training 
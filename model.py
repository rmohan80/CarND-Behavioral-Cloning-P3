import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, Lambda, MaxPooling2D, Activation
import cv2
from keras.preprocessing.image import img_to_array, load_img

def nvidia_model():
  row, col, depth = 64, 64, 3
  model = Sequential()

  # normalize image values between -.5 : .5
  model.add(Lambda(lambda x: x/255 - .5, input_shape=(row, col, depth), output_shape=(row, col, depth)))
  
  #Use Nvidia as base
  model.add(Convolution2D(24, 5, 5, input_shape=(64, 64, 3), subsample=(2, 2), border_mode='valid'))
  model.add(Activation('relu'))
  #model.add(Dropout(0.1)) 

  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(Activation('relu'))
  #model.add(ELU())

  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid'))
  model.add(Activation('relu'))
  #model.add(ELU())
  #model.add(Dropout(0.2)) 
  
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
  model.add(Activation('relu'))
  #model.add(ELU())
  
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='valid'))
  #model.add(ELU())
  model.add(Activation('relu'))


  model.add(Flatten())
  model.add(Activation('relu'))
  #model.add(ELU())
  #model.add(Dropout(0.3)) 


  model.add(Dense(100))
  model.add(Activation('relu'))
  #model.add(ELU())

  model.add(Dense(50))
  model.add(Activation('relu'))
  #model.add(ELU())
  #model.add(Dropout(0.4)) 

  model.add(Dense(10))
  model.add(Activation('relu'))
  #model.add(ELU())

  #Add a dropout layer not available in Nvidia model
  model.add(Dropout(0.5)) 

  model.add(Dense(1))


  #compile with normal adam optimizer and return
  model.compile(optimizer="adam", loss='mse')  
  model.summary()
  return model

def rotate_image(img, x_translation):
	# Randomly compute a Y translation
	y_translation = (40 * np.random.uniform()) - (40 / 2)

	# Form the translation matrix
	translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])

	# Translate the image
	image_tr = cv2.warpAffine(img, translation_matrix, (img.shape[1], img.shape[0]))
	return image_tr

def trans_image(image,steer, trans_range=0.004):
    # Translation
    tr_x = trans_range*np.random.uniform()-trans_range/2
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 40*np.random.uniform()-40/2
    #tr_y = 0
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(image.shape[1],image.shape[0]))
    
    return image_tr,steer_ang

def get_row(row):
    
	steering = row['steering']

	# randomly choose the camera to take the image from
	camera = np.random.choice(['center', 'left', 'right'])

    # adjust the steering angle for left anf right cameras
	if camera == 'left':
		steering += 0.25
	elif camera == 'right':
		steering -= 0.25
	
	image = load_img("./data/" + row[camera].strip())

	image = img_to_array(image)

	flip_prob = np.random.random()
	if flip_prob > 0.5:
		# flip the image and reverse the steering angle
		steering = -1*steering
		image = cv2.flip(image, 1)

	# Add random horizontal and vertical shifts to 50% of images
	#trans_prob = np.random.random()
	#if trans_prob > 0.5:
	tr_image, steering = trans_image(image,steering)

	#Randomly rotate 50% of images
	#rotate_prob = np.random.random()
	#if rotate_prob > 0.5:
		#image = trans_image(image, (100 * np.random.uniform()) - (100 / 2))

	#final_image = pre_process(image)
	final_image = cv2.resize(tr_image, (64,64))

	return final_image, steering

def pre_process(image):

	grayscale_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
	resized_image = cv2.resize(grayscale_image,(64,64))

	return resized_image

def data_generator(data_frame, batch_size=32):
    N = data_frame.shape[0]
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i*batch_size
        end = start+batch_size - 1

        X_batch = np.zeros((batch_size, 64, 64, 3), dtype=np.float32)
        y_batch = np.zeros((batch_size,), dtype=np.float32)

        j = 0

        # create a `batch_size` sized chunk from the dataframe
        for index, row in data_frame.loc[start:end].iterrows():
            X_batch[j], y_batch[j] = get_row(row)
            j += 1

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0
        yield X_batch, y_batch


if __name__ == "__main__":

    data_frame = pd.read_csv('./data/driving_log.csv', usecols=[0, 1, 2, 3])

    # shuffle the data
    data_frame = data_frame.sample(frac=1).reset_index(drop=True)

    # 80-20 training validation split
    training_split = 0.8

    num_rows_training = int(data_frame.shape[0]*training_split)

    training_data = data_frame.loc[0:num_rows_training-1]
    validation_data = data_frame.loc[num_rows_training:]

    # release the main data_frame from memory
    data_frame = None

    training_generator = data_generator(training_data, 32)
    validation_data_generator = data_generator(validation_data, 32)

    model = nvidia_model()

    samples_per_epoch = (20000//32)*32

    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=5, nb_val_samples=3000)

    print("Saving model weights and configuration file.")

    model.save('model.h5')  # always save your weights after training or during training
    #with open('model.json', 'w') as outfile:
        #outfile.write(model.to_json())
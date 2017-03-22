# Convolutional Neural Network (CNN)
# Images are from kaggle

### Building the CNN ###
# Import libraries
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense

# Initializing CNN
classifier = Sequential()

# CNN: Convolution > Max Pooling > Flattening > Full Connection

# 1) Build a Convolution Layer
classifier.add(Convolution2D(filters = 32, kernel_size = (3, 3) , padding = 'same', 
                             input_shape = (64,64, 3), activation = 'relu'))
# The tensors in Tensorflow and theano for tensor is different (X,Y, Channel -- or Z)
# input_shape(64,64,3); need to 
# include input_shape because we dont have anything infront of it i.e. input layer

# 2) Build a Pooling Layer
# Max Pooling
# Reduce the size of feature map. Pooling from feature map to Pooled Feature map
# Stride of 2
# Reduce # of nodes > less computer intensive
classifier.add(MaxPooling2D(pool_size = (2,2))) 
# Still hold consistent on obtaining the max number


'''
Add 2nd pair of convolution layer and pooling 
To prove accuracy
'''

classifier.add(Convolution2D(filters = 32, kernel_size = (3, 3) , padding = 'same', activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
 
# 3) Build a flattening layer
# Flattening
# Turn all entries in the pooling map to a huge single vector
# This huge vector will be an input vector for fully connected ANN
classifier.add(Flatten())

# 4) Build a classic fully connected ANN (Hidden Layer) to classify Dog/Cats
# Build a fully connected ANN with Dense(*)
classifier.add(Dense(units = 128, activation = 'relu'))
# Output Layer; sigmoid for binary outcome
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compile the CNN with Stochastic Gradient Descent (Adam Algorithm)l 
# loss(logirthmic loss) - Binary_crossentropy for 2 classes
# For more than 2 classes, categorical_crossentropy
classifier.compile(optimizer ='adam', loss = 'binary_crossentropy',
                   metrics = ['accuracy'])


### Fitting the CNN to the image using keras documentation###
"""Image augmentation - image Preprocessing with Keras Documenation
if we dont do image preprocessing well, overfitting might occur
Use data augmentation trick to make more diverse images(transformed images)
to reduce overfitting
Resource: keras documentation
"""
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True) 
# Image Augmentation
# Rescale the pixel value for feature scaling
# Sheer_range, zoom_range, horizontal_flip are some transformation to have more diverse images

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',target_size=(64, 64),batch_size=32, class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set', target_size=(64, 64), batch_size=32, class_mode='binary')
# Tensor size, or image size, (64,64,3)
# choose higher target size will definitely get more accuracy as we have more data

classifier.fit_generator(training_set, steps_per_epoch=8000,
                         epochs=25, validation_data=test_set,
                         nb_val_samples=2000)
# samples_per_epoch number of training set, num_val_samples number of test set







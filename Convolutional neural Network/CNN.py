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

# 2) Build a Pooling Layer
# Max Pooling
# Reduce the size of feature map. Pooling from feature map to Pooled Feature map
# Stride of 2
# Reduce # of nodes > less computer intensive
classifier.add(MaxPooling2D(pool_size = (2,2))) 
# Still hold consistent on obtaining the max number

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
"""Image augmentation - image Preprocessing
if we dont do image preprocessing well, overfitting might occur
Use data augmentation trick to make more diverse images(transformed images)
to reduce overfitting
Resource: keras documentation
"""

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        'data/validation',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

model.fit_generator(
        train_generator,
        samples_per_epoch=2000,
        epochs=50,
        validation_data=validation_generator,
        num_val_samples=800)







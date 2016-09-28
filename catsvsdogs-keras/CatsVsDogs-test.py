##
# TODO: Add Data folder
##

from keras.models import Model
from keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
from imagenet_utils import preprocess_input,decode_predictions

img_width, img_height = 150, 150


model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, img_width, img_height)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))







model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model.load_weights('first_try.h5')

#img_path = 'cat.122.jpg'
img_path1 = 'data/train/dogs/dog.122.jpg'
img_path2 = 'data/train/cats/cat.322.jpg'
img_path3 = 'data/train/dogs/dog.152.jpg'
img_path4 = 'data/train/cats/cat.222.jpg'
img_path5 = 'data/train/cats/cat.782.jpg'
img_path6 = 'data/train/cats/cat.902.jpg'
img_path7='cat-2.jpg'
img_path8='dog-2.jpg'


img_path11 = 'data/train/dogs/dog.244.jpg'
img_path12 = 'data/train/cats/cat.678.jpg'
img_path13 = 'data/train/dogs/dog.455.jpg'
img_path14 = 'data/train/cats/cat.900.jpg'
img_path15 = 'data/train/cats/cat.999.jpg'
img_path16 = 'data/train/cats/cat.345.jpg'



#new image from web
img_path21 = 'test/dog/dog123.jpg'
img_path22 = 'test/dog/dog124.jpg'
img_path23 = 'test/cat/cat123.jpg'
img_path24 = 'test/cat/cat124.jpg'

#11-16
#actual - 101000
#Predicted -101100



#2-7
#actual-1010000
#1,1,1,0,0,0,1

def image_convert(path):
        img = image.load_img(path, target_size=(img_width, img_height))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        return x




#cat-0 , dog-1

# dimensions of our images.
#img_width, img_height = 150, 150
#img = image.load_img(img_path4, target_size=(img_width, img_height))
#x = image.img_to_array(img)
#x = np.expand_dims(x, axis=0)
#x = preprocess_input(x)

x=image_convert(img_path21)
print model.predict(x)
#print('Predicted:', decode_predictions(model.predict(x)))

x=image_convert(img_path22)
print model.predict(x)
#print('Predicted:', decode_predictions(model.predict(x)))

x=image_convert(img_path23)
print model.predict(x)
#print('Predicted:', decode_predictions(model.predict(x)))

x=image_convert(img_path24)
print model.predict(x)








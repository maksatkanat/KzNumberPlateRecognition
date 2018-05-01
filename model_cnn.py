from keras.models import Sequential
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D

def build_model():

    # Model parameters    
    nb_classes = 36
    image_h, image_w = 28, 28
    

    #Define sequential models 
    model = Sequential()
    #Add two Convolution and Pooling layers
    model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(image_h,image_w,1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #Flatten the image data into vector so that they can be processed by Dense layers
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(nb_classes, activation='softmax'))

    print(model.summary())

    return model



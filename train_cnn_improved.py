from keras.optimizers import SGD,Adam
from loadDataset import load_data_2d as load_data
from model_cnn_reg import build_model
from keras.models import load_model

from keras.preprocessing.image import ImageDataGenerator

#Train tensorflow model for all characters and gives "h5" NN model.

batch_size = 128
nb_epoch = 50
Data_augmentation=True #True
# Load data
(X_train, y_train, X_test, y_test) = load_data()


    # Load and compile model
model = build_model()

adam=Adam(lr=0.001)
model.compile(loss='categorical_crossentropy', optimizer=adam,
              metrics=['accuracy'])

if not Data_augmentation:
    print('Not using Data Augmentation')
    model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch,
          verbose=1, validation_data=(X_test, y_test))
else:
    print('Using Data Augmentation')
    datagen = ImageDataGenerator(
   #         featurewise_center=True,
   #         featurewise_std_normalization=True,
            rotation_range=10,
            width_shift_range=0.2,
            height_shift_range=0.1,
            #shear_range=0.3,
            #zoom_range=0.3,
            horizontal_flip=False,
            vertical_flip=False)

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    print(X_train.shape)
    datagen.fit(X_train)
    print(X_train.shape)
    # fits the model on batches with real-time data augmentation:
    model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=len(X_train) / batch_size, epochs=nb_epoch, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=1)

model.save('my_model_syntezed_3004_with_inv.h5')
#print(dir(score))
print("Accuracy:", score[1])

from keras.models import Sequential
from keras import  layers
from Paramaters import *
import pandas as pd
from  GridDataset import *
import keras

'''main functions'''
def Resnet34(shape=(128,128,1)):
    # Step 1 (Setup Input Layer)
    x_input = layers.Input(shape)
    x = layers.ZeroPadding2D((3, 3))(x_input)
    # Step 2 (Initial Conv layer along with maxPool)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='same')(x)
    # Define size of sub-blocks and initial filter size
    block_layers = [3, 4, 6, 3]
    filter_size = 64
    # Step 3 Add the Resnet Blocks
    for i in range(4):
        if i == 0:
            # For sub-block 1 Residual/Convolutional block not needed
            for j in range(block_layers[i]):
                x = identity_block(x, filter_size)
        else:
            # One Residual/Convolutional Block followed by Identity blocks
            # The filter size will go on increasing by a factor of 2
            filter_size = filter_size * 2
            x = conv_block(x, filter_size)
            for j in range(block_layers[i] - 1):
                x = identity_block(x, filter_size)
    # Step 4 End Dense Network
    x = layers.AveragePooling2D((2, 2), padding='same')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dense(1, activation='relu')(x)
    model = keras.Model(inputs=x_input, outputs=x, name="ResNet34")
    return model

def VGG16():
    sequential=Sequential(name="VGG16")
    activation="relu"
    sequential.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation=activation,input_shape=(image_size,image_size,1),padding='same'))
    sequential.add(layers.Conv2D(filters=64,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.MaxPooling2D((2,2),strides=(2,2)))


    sequential.add(layers.Conv2D(filters=128,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.Conv2D(filters=128,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.MaxPooling2D((2,2),strides=(2,2)))

    sequential.add(layers.Conv2D(filters=256,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.Conv2D(filters=256,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.Conv2D(filters=256,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.MaxPooling2D((2,2),strides=(2,2)))

    sequential.add(layers.Conv2D(filters=512,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.Conv2D(filters=512,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.Conv2D(filters=512,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.MaxPooling2D((2,2),strides=(2,2)))

    sequential.add(layers.Conv2D(filters=512,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.Conv2D(filters=512,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.Conv2D(filters=512,kernel_size=(3,3),activation=activation,padding='same'))
    sequential.add(layers.MaxPooling2D((2,2),strides=(2,2)))

    sequential.add(layers.Flatten())
    sequential.add(layers.Dense(units=4096,activation=activation))
    sequential.add(layers.Dense(units=4096,activation=activation))
    sequential.add(layers.Dense(units=1,activation=activation))

    return sequential

def LeNet5():
    sequential=Sequential(name="LeNet5")
    activation="relu"
    sequential.add(layer=layers.Conv2D(filters=6,kernel_size=(5,5),activation=activation,input_shape=(image_size,image_size,1)))
    sequential.add(layers.AveragePooling2D((2,2),strides=(2,2)))

    sequential.add(layer=layers.Conv2D(filters=16,kernel_size=(5,5),activation=activation))
    sequential.add(layers.AveragePooling2D((2,2),strides=(2,2)))

    sequential.add(layers.Flatten())
    sequential.add(layers.Dense(units=120,activation=activation))
    sequential.add(layers.Dense(units=84,activation=activation))
    sequential.add(layers.Dense(units=1,activation="relu"))

    return sequential

def TrainModel(model,datasetName):
    dataset = Dataset()
    X_train, X_test, y_train, y_test = dataset.get_train_test_data(datasetName)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=30),
        keras.callbacks.ModelCheckpoint(filepath='./models/' + model.name + '.h5', save_weights_only=False,
                                        save_best_only=True, mode='min', monitor='val_loss', period=5),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, verbose=0, min_lr=0.0000001),
        # keras.callbacks.CSVLogger('./logs/' + parameter_path + '.csv', append=True, separator=';'),
    ]
    model.fit(X_train, y_train, validation_split=0.2, verbose=1, batch_size=8, epochs=100, callbacks=callbacks)

def TestModel(datasetName,modelName):
    dataset = Dataset()
    X_train, X_test, y_train, y_test = dataset.get_train_test_data(datasetName)
    model = keras.models.load_model('./models/' + modelName + '.h5')
    preds = model.predict(X_test)
    dataset.evaluateResults(modelName, y_test, preds)



'''Axularity functions'''

def identity_block(x,filter):
    #copy tensor to variable
    x_skip=x
    #layer1
    x=layers.Conv2D(filter,(3,3),padding='same')(x)
    x=layers.BatchNormalization(axis=3)(x)
    x=layers.Activation(activation="relu")(x)

    #layer2
    x=layers.Conv2D(filter,(3,3),padding='same')(x)
    x=layers.BatchNormalization(axis=3)(x)

    #add residue
    x=layers.Add()([x,x_skip])
    x=layers.Activation(activation="relu")(x)

    return x

def conv_block(x,filter):
    x_skip=x

    #layer 1
    x=layers.Conv2D(filter,(3,3),padding='same',strides=(2,2))(x)
    x=layers.BatchNormalization(axis=3)(x)
    x=layers.Activation(activation="relu")(x)
    #layer 2
    x=layers.Conv2D(filter,(3,3),padding='same' )(x)
    x=layers.BatchNormalization(axis=3)(x)

    x_skip=layers.Conv2D(filter,(1,1),strides=(2,2))(x_skip)
    x=layers.Add()([x,x_skip])
    x=layers.Activation('relu')(x)
    return x


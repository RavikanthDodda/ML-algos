import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import matplotlib.pyplot as plt

num_classes = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Function defined for training and evaluation part to reuse the code while changing parameters
def cnn(feature_maps_1, feature_maps_2 , k_size_1, k_size_2, lr, batch_size, epochs):
    model = Sequential()
    model.add(Conv2D(feature_maps_1, kernel_size=k_size_1,
                    activation='relu',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(feature_maps_2, k_size_2, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))

    model.add(Dense(num_classes, activation='softmax'))

    # https://keras.io/optimizers/ 
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(lr=lr, rho=0.95, decay=0.0),
                metrics=['accuracy'])

    # capturing the history object which contains training error and accuracy returned by fit() for plotting
    result = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    result.history['loss'] = [i*100 for i in result.history['loss']]
    result.history['accuracy'] = [i*100 for i in result.history['accuracy']]

    # Printing training error and accuracy for each epoch in neat format
    print("Epoch   Training error(%)      Training accuracy(%)")
    for i in range(len(result.history['loss'])):
        print(i+1,"      ",result.history['loss'][i],"    ",result.history['accuracy'][i])

    # Printing test error/loss and accuracy 
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print()

    # Returning the history object
    return result
    


# Baseline code parameters
print("Baseline parameters:")
res = cnn(6,16,(3,3),(3,3),0.1,128,12) 
# Plotting training loss and accuracy on Y-Axis with distinguishable symbols
plt.plot(res.history['loss'],label = 'training error',marker='^',color='r')
plt.plot(res.history['accuracy'],label = 'training accuracy',marker='s',color='b')
# setting scale for X - axis as 1 to 12
plt.xticks(range(12),[i+1 for i in range(12)])
# Title of the graph
plt.title("Baseline")
# Naming X and Y axes of plot/graph
plt.xlabel('Number of Epochs')
plt.ylabel('Loss/Accuracy(%)')
# Shows legend of the plot/graph
plt.legend()
# Displays the plot itself
plt.show()


# Changing kernel size of 2 convolutinal layers to 5 x 5
print("5 X 5 kernel:")
res = cnn(6,16,(5,5),(5,5),0.1,128,12)
# Plotting training loss and accuracy on Y-Axis with distinguishable symbols
plt.plot(res.history['loss'],label = 'training error',marker='^',color='r')
plt.plot(res.history['accuracy'],label = 'training accuracy',marker='s',color='b')
# setting scale for X - axis as 1 to 12
plt.xticks(range(12),[i+1 for i in range(12)])
# Title of the graph
plt.title("5 X 5 kernel")
# Naming X and Y axes of plot/graph
plt.xlabel('Number of Epochs')
plt.ylabel('Loss/Accuracy(%)')
# Shows legend of the plot/graph
plt.legend()
# Displays the plot itself
plt.show()


# Case where number of feature maps are changed 
print("Different number of feature maps:")
res = cnn(10,25,(3,3),(3,3),0.1,128,12)
# Plotting training loss and accuracy on Y-Axis with distinguishable symbols
plt.plot(res.history['loss'],label = 'training error',marker='^',color='r')
plt.plot(res.history['accuracy'],label = 'training accuracy',marker='s',color='b')
# setting scale for X - axis as 1 to 12
plt.xticks(range(12),[i+1 for i in range(12)])
# Title of the graph
plt.title("Different number of feature maps")
# Naming X and Y axes of plot/graph
plt.xlabel('Number of Epochs')
plt.ylabel('Loss/Accuracy(%)')
# Shows legend of the plot/graph
plt.legend()
# Displays the plot itself
plt.show()

# Case with different learning rate, batch size and feature maps
print("Learning rate - 0.22, batch size - 98, feature maps-10,25")
res = cnn(10,25,(3,3),(3,3),0.22,98,12)
# Plotting training loss and accuracy on Y-Axis with distinguishable symbols
plt.plot(res.history['loss'],label = 'training error',marker='^',color='r')
plt.plot(res.history['accuracy'],label = 'training accuracy',marker='s',color='b')
# setting scale for X - axis as 1 to 12
plt.xticks(range(12),[i+1 for i in range(12)])
# Title of the graph
plt.title("Learning rate - 0.22, batch size - 98, feature maps-10,25")
# Naming X and Y axes of plot/graph
plt.xlabel('Number of Epochs')
plt.ylabel('Loss/Accuracy(%)')
# Shows legend of the plot/graph
plt.legend()
# Displays the plot itself
plt.show()
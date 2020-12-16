import tensorflow as tf
from tensorflow import keras
import  numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist  =  keras.datasets.mnist

(x_train, y_train),(x_test ,y_test) = mnist.load_data() #hand writen digits dataset 
print(x_train.shape , y_train.shape)

#normalize the data 0,255 -> 0,1
x_train , x_test = x_train/255 , x_test/255

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)), #reduce dimension from original i.e (28,28) of the images
    keras.layers.Dense(128,activation='relu'), #1st hidden layer 
    keras.layers.Dense(10) #final layer 10 because we have 10 classifications   
    ])

#loss and optimizers

loss  =  keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim =  keras.optimizers.Adam(lr=0.001)
metrics = ['accuracy']
model.compile(loss=loss , optimizer=optim , metrics=metrics)

#hyperparameters 
batch_size=64
epochs = 5

model.fit(x_train , y_train , batch_size=batch_size,epochs=epochs,shuffle=True,verbose=2)

#evaluate model
model.evaluate(x_test , y_test , batch_size=batch_size , verbose=2)

#predictions with the model
probability_model = keras.models.Sequential([
    model , 
    keras.layers.Softmax()])

# to show the predictions 
predictions = probability_model(x_test)
pred0 = predictions[0]
print(pred0) # prints an array of probabilities of all the digits 
label0 = np.argmax(pred0) # select the highest probability from the array, returns index
print(label0) 
plt.imshow(x_test[label0],cmap='gray') # check which number is it
print(y_test[label0]) # verify if it is correctly identified.

plt.show()

'''
#model + softmax different method to do the same as above 

predictions = model(x_test)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0) 

'''

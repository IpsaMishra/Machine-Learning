import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from mxnet import gluon

def transform(data, label):
    return data.astype(np.float32)/255, label.astype(np.float32)

def multiclassMetrics(y, yhat):
    # y is the true label data, Example y = 1
    # yhat is the predicted class label. Example yhat = 4
    # Assumes that the classes are indexed from 0 to nClass - 1
    
    y = np.array(y)
    yhat = np.array(yhat)
    
    nClass = len(np.unique(y))  # Total number of classes
    conf_matr = np.zeros((nClass, nClass))
    PrecisionVal = np.zeros(nClass)
    RecallVal = np.zeros(nClass)
    
    if len(y) != len(yhat):
        print('True label set and Predicted class label set size must match')
        return None
    
    Match = 0
    m = len(y)
    for i in range(m):
        conf_matr[y[i], yhat[i]] += 1
        if y[i] == yhat[i]:
            Match += 1
    
    for i in range(nClass):
        PrecisionVal[i] = conf_matr[i,i] / np.sum(conf_matr[:,i])
        RecallVal[i] = conf_matr[i,i] / np.sum(conf_matr[i,:])
    
    accr = Match / np.sum(conf_matr)
    return PrecisionVal, RecallVal, accr

# fix a random seed for reproducibility
np.random.seed(9)

# user inputs
num_classes = 5               #Total number of class labels or classes involved in the classification problem
batch_size = 128               #Number of images given to the model at a particular instance
nepochs = 100

# Process dataset
mnist_train = gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test = gluon.data.vision.MNIST(train=False, transform=transform)

Xtrain = []
ytrain = []
for data in mnist_train:
    x, y = data
    if y <= 4:      # Filter out data with class label 5 or more
        x = x.reshape(1,784).asnumpy()
        x = x[0]
        Xtrain.append(x)
        ytrain.append(y)
Xtrain = np.array(Xtrain)
Xtrain = np.concatenate((Xtrain,np.ones((np.shape(Xtrain)[0],1))),axis=1)
ytrain = [np.int(val) for val in ytrain]

Xtest = []
ytest = []
for data in mnist_test:
    x, y = data
    if y <= 4:      # Filter out data with class label 5 or more
        x = x.reshape(1,784).asnumpy()
        x = x[0]
        Xtest.append(x)
        ytest.append(y)    
Xtest = np.array(Xtest)
Xtest = np.concatenate((Xtest,np.ones((np.shape(Xtest)[0],1))),axis=1)
ytest = [np.int(val) for val in ytest]                                   #normalize the floating point variables in the range (0,1)

# convert class vectors into binary form representation
mTrainLabels = np_utils.to_categorical(ytrain, num_classes)
mTestLabels = np_utils.to_categorical(ytest, num_classes)

# create the model using MLP
model = Sequential()
model.add(Dense(256 ,input_shape=(785,)))
model.add(Activation("relu"))
model.add(Dense(num_classes))
model.add(Activation("softmax"))

# summarize the model
model.summary()

# compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# fit the model
history = model.fit(Xtrain, mTrainLabels, validation_data=(Xtest, mTestLabels), batch_size = batch_size, epochs = nepochs, verbose=2)

# print the history keys
print(history.history.keys())

# Evaluate the model
yhat_classes = model.predict_classes(Xtest, verbose=0)
p,r,a = multiclassMetrics(ytest, yhat_classes)

# print the results
print('Accuracy = ', a)
for i in range(5):
    print('Precision for class ', i, 'is ', p[i])
for i in range(5):
    print('Recall for class ', i, 'is ', r[i])
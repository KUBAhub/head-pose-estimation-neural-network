import numpy as np
import matplotlib.pyplot as plt
import theano
from theano import tensor as T
import itertools

import os
import sys
from pandas.io.parsers import read_csv
from lasagne import layers
from lasagne.updates import momentum
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet


try:
    import cPickle as pickle
except ImportError:  # Python 3
    import pickle

plt.ion()

cmatrix = []

CLASSES = ['Front', 'Right', 'Left', 'Top', 'Bottom', 'Top Right', 'Top Left', 'Bottom Right', 'Bottom Left']


DATA = 'dataset/classification_dataset.csv'


cmatrix = np.zeros((9, 9), dtype='int32')


x=T.vector('x')
classes = T.scalar('n_classes')
onehot = T.eq(x.dimshuffle(0,'x'),T.arange(classes).dimshuffle('x',0))
oneHot = theano.function([x,classes],onehot)
y = T.matrix('y')
y_pred = T.matrix('y_pred')
confMat = T.dot(y.T,y_pred)
confusionMatrix = theano.function(inputs=[y,y_pred],outputs=confMat)



def confusion_matrix(x,y,n_class):
    return confusionMatrix(oneHot(x,n_class),oneHot(y,n_class))



def load():
    """
    	load and split data
    """
    print 'loading test data'
    # load dataframe 
    df = read_csv(os.path.expanduser(DATA))

    # conver pixel strings to numpy arrays
    df['image'] = df['image'].apply(lambda im: np.fromstring(im, sep=' '))

    # split dataset into train and test
    train=df.sample(frac=0.8,random_state=200)
    test=df.drop(train.index)


    train_X = np.vstack(train['image'].values) / 255.      # scale pixel values between 0 and 1
    train_X = train_X.astype(np.float32)

    train_Y = train['direction'].values
    train_Y = train_Y.astype(np.int32)          # convert int64 to int32
    
    #train_Y = train_Y.reshape(len(train_Y),1)

    
    test_X = np.vstack(test['image'].values) / 255.      # scale pixel values between 0 and 1
    test_X = test_X.astype(np.float32)

    test_Y = test['direction'].values
    test_Y = test_Y.astype(np.int32)          # convert int64 to int32
    
    #test_Y = test_Y.reshape(len(test_Y),1)
    
    return (train_X,train_Y) , (test_X, test_Y)	

def showConfusionMatrix():
    #new figure
    plt.figure(figsize=(15,15))
    plt.clf()

    #show matrix
    plt.imshow(cmatrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    #tick marks
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES)
    plt.yticks(tick_marks, CLASSES)

    #labels
    thresh = cmatrix.max() / 2.
    for i, j in itertools.product(range(cmatrix.shape[0]), range(cmatrix.shape[1])):
        plt.text(j, i, cmatrix[i, j], 
                 horizontalalignment="center",
                 color="white" if cmatrix[i, j] > thresh else "black")

    #axes labels
    plt.ylabel('Target label')
    plt.xlabel('Predicted label')
	
    #save
    plt.savefig('cmatrix.png')
    plt.pause(0.5)

# Calculate accuracy percentage
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


model_file = 'trained_network/net1_softmax.pickle'

with open(model_file) as f:
    print('Load pretrained weights from %s...' % model_file)
    net = pickle.load(f)

train, test = load()


# perform test
test_x = test[0]
actual = test[1]



predicted = []

for row in test_x:
    # predict label
    label = net.predict([row])
    print label
    predicted.append(label[0])


# calculate accuracy
#accuracy = calculate_accuracy(actual, predicted)
#print "Accuracy: ",accuracy,"%"



cmatrix = confusion_matrix(actual,predicted,len(CLASSES))
showConfusionMatrix()
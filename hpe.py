try:
    import cPickle as pickle
except ImportError:  # Python 3
    import pickle

import os
import sys
import numpy as np
from pandas.io.parsers import read_csv
from lasagne import layers
from lasagne.updates import momentum
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet


DATA = 'dataset/classification_dataset.csv'


def load():
    """
    	load and split data
    """

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




def build_network():
    
    net = NeuralNet(
        # three layers: one hidden layer
        layers=[('input', layers.InputLayer),   # This layer holds a symbolic variable that represents a network input.
                ('hidden', layers.DenseLayer),  # Fully connected hidden layer.
                ('output', layers.DenseLayer),  # Fully connected output layer.
        ],

        # layer parameters:
        input_shape=(None, 90*82), # 90x82 input pixels
        hidden_num_units=100,  # number of neurons in hidden layer
        output_nonlinearity=nonlinearities.softmax,     #activation function
        output_num_units=9,  # neurons in output layer

        # optimization method:
        update=momentum,    # Stochastic Gradient Descent (SGD) updates with momentum
        update_learning_rate=0.01,
        update_momentum=0.3,

        max_epochs=400, # we want to train this many epochs
        verbose=1, # print out debugging messages to the command line
    )

    return net

# Calculate accuracy percentage
def calculate_accuracy(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0


def main():

    # load training and test data
    train, test = load()

    # build 3 layer neural network
    net = build_network()

    # Train the network
    x = train[0]    # list of images in 1d pixels
    y = train[1]    # list of labels


    net.fit(x,y)

    # training for 400 epochs will take time. 
    # pickle the trained model so that we can load it back later 
    with open('net1_softmax.pickle', 'wb') as f:
        pickle.dump(net, f, -1)

    # perform test
    test_x = test[0]
    actual = test[1]

    predicted = []

    for row in test_x:
        # predict label
        label = net.predict([row])
        predicted.append(label[0])

    # calculate accuracy
    accuracy = calculate_accuracy(actual, predicted)
    print "Accuracy: ",accuracy,"%"



if __name__ == '__main__':
    main()



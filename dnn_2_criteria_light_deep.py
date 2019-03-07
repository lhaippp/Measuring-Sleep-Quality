import pandas as pd 
import numpy as np
import math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tf_utils import random_mini_batches,convert_to_one_hot


# buffer_df = pd.read_csv('SleepQuality_DNN_Classification_3Critere.csv')
buffer_df = pd.read_csv('SleepQuality_DNN_Classification_binary.csv')

# select features and calculate their coefficients
corr = buffer_df[['avg23bpd_s2','avg23bps_s2','ai_all','rdi0p', 'nsupinep', 'pctlt75', 'pctlt80', 'pctlt85', 'pctlt90',
                     'slp_eff', 'slp_lat', 'slpprdp', 'supinep', 'times34p', 'timest1p', 'timest2p', 'waso']+['ms204c']].corr()['ms204c'].abs().sort_values(ascending = False)

cols = corr.index.values.tolist()[1:18]
trainSet_X = buffer_df.loc[:,cols]

# Sleep Quality from light to dark: 1 -> 5
trainSet_y = buffer_df.loc[:,['ms204a']]
# trainSet_y = buffer_df.loc[:,['ms204b']]
# trainSet_y = buffer_df.loc[:,['ms204c']]

# split training set and test set
X_train, X_test, y_train, y_test = train_test_split(trainSet_X, trainSet_y, test_size=0.2, shuffle=True)

# Dataframe to numpy array
buffer_y_train = y_train.values
buffer_y_test = y_test.values

# float to int
buffer_y_train = buffer_y_train.astype(int)
buffer_y_test = buffer_y_test.astype(int)
buffer_y_train = buffer_y_train.T
buffer_y_test = buffer_y_test.T
y_train = convert_to_one_hot(buffer_y_train,2)
y_test = convert_to_one_hot(buffer_y_test,2)
# from dataframe to numpy array
X_train = X_train.values
X_test = X_test.values
y_train = y_train.astype(int)
y_test = y_test.astype(int)
X_train = X_train.T
X_test = X_test.T


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- 17
    n_y -- 5
    
    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"
    
    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    ### START CODE HERE ### (approx. 2 lines)
    X = tf.placeholder(tf.float32, [n_x, None], name="X")
    Y = tf.placeholder(tf.float32, [n_y, None], name="Y")
    ### END CODE HERE ###
    
    return X, Y
 
def initialize_parameters():
    """
    Initializes parameters to build a neural network with tensorflow. The shapes are:
                        
                        W1 : [30, 9] 第一层隐藏层神经元25个，输入特征：17个
                        b1 : [30, 1] 
                        
                        W2 : [12, 30] 第二层隐藏层神经元12个，第一层隐藏层神经元25个
                        b2 : [12, 1]
                        
                        W3 : [20, 12] 第三层隐藏层神经元20个，第二层隐藏层神经元12个
                        b3 : [20, 1]
                        
                        W4 : [20, 20] 第4层隐藏层神经元20个，第三层隐藏层神经元20个
                        b4 : [20, 1]
                        
                        W5 : [5, 20] 输出神经元5个，第4层隐藏层神经元20个
                        b5 : [5, 1]
    
    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3， W4, b4
    """
        
    ### START CODE HERE ### (approx. 6 lines of code)
    W1 = tf.get_variable("W1", [100, 17], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [100, 1], initializer = tf.zeros_initializer())
    
    W2 = tf.get_variable("W2", [100, 100], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [100, 1], initializer = tf.zeros_initializer())
    
    W3 = tf.get_variable("W3", [100, 100], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [100, 1], initializer = tf.zeros_initializer())
    
    W4 = tf.get_variable("W4", [100, 100], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [100, 1], initializer = tf.zeros_initializer())
    
    W5 = tf.get_variable("W5", [2, 100], initializer = tf.contrib.layers.xavier_initializer())
    b5 = tf.get_variable("b5", [2, 1], initializer = tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3,
                  "W4": W4,
                  "b4": b4,
                  "W5": W5,
                  "b5": b5}
    
    return parameters
  
def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> TANH
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    W4 = parameters['W4']
    b4 = parameters['b4']
    W5 = parameters['W5']
    b5 = parameters['b5']
    
    ### START CODE HERE ### (approx. 5 lines)              # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,a2) + b3
    A3 = tf.nn.relu(Z3)                                    # A3 = relu(Z3)
    
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    A4 = tf.nn.relu(Z4)
    
    Z5 = tf.add(tf.matmul(W5, A4), b5)
    ### END CODE HERE ###
    
    return Z5
  
def compute_cost(Z5, Y):
    """
    Computes the cost
    
    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (3, number of examples)
    Y -- "sleep quality" labels vector placeholder, same shape as Z3
    
    Returns:
    cost - Tensor of the cost function
    """
    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z5)
    labels = tf.transpose(Y)
    
    ### START CODE HERE ### (1 line of code)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    ### END CODE HERE ###
    
    return cost
  
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.00006,
          num_epochs = 20000, minibatch_size = 2041, print_cost = True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.
    
    Arguments:
    X_train -- training set, of shape (input size = 17, number of training examples = 2113)
    Y_train -- test set, of shape (output size = 5, number of training examples = 2113)
    X_test -- training set, of shape (input size = 17, number of training examples = 529)
    Y_test -- test set, of shape (output size = 5, number of test examples = 529)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    # tf.set_random_seed(1)                             # to keep consistent results
    # seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost
    learning_rate_origin = learning_rate
    
    # Create Placeholders of shape (n_x, n_y)
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters()
    
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z5 = forward_propagation(X, parameters)
    
    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z5, Y)
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        
        # Run the initialization
        sess.run(init)
        
        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:

                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                
                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer" and the "cost", the feedict should contain a minibatch for (X,Y).
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                
                epoch_cost += minibatch_cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        print(learning_rate_origin)        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z5), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        print ("Test Presicon:", )
        
        #sess.close()
        
        return parameters

parameters = model(X_train, y_train, X_test, y_test)
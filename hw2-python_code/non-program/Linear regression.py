# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 14:23:17 2018

@author: Kaai
"""

import numpy as np

# %%
# Here we make the functions that can calculate the gradiant based on the
# solution dJ/dw_j = -sum_{i=1}^m (y_i-w^T x_i)x_{ij} (pg 31 lecture 8)

def batch_gradiant(weights, x, y):
    '''
    Used in the batch gradient decent function. 
    
    Parameters
    ------------
    weights: numpy.matrix,
        Matrix of size (1 x N) where N is the number of features + 1, calulated 
        from len([b]+[w]) where b is the bias andd w is the weights.
        
    x: numpy.matrix,
        Matrix of size (M x N) where M is the number of instances, and N is the
        number of features + 1.
        
    y: numpy.matrix,
        Matrix of size (M x 1) where M is the number of instances. This is the 
        label of the data used for training.

    Returns
    ------------
    gradiant: list,
        This list contains the gradiant [dJ/dw_j, dJ/dw_j+1, ....., dJ/dw_N].
        (Optimized weights should produce a vector of zeros [0, 0, ...., 0])
    '''
    gradiant = []
    for j in range(0, weights[0,:].shape[1]):
        grad_j = 0
        for i in range(0, y[0,:].shape[1]):
            grad_j += (y[0,i] - weights*x[i].transpose())*x[i,j]
        gradiant.append(-grad_j[0,0])
    return gradiant

def predict(weights, x_i):
    '''
    Function should be used to predict label with trained weights
    
    Parameters
    ------------
    weights: numpy.matrix,
        Matrix of size (1 x N) where N is the number of features + 1 calulated 
        from len([b]+[w]) where b is the bias andd w is the weights. Trained
        weights should produce results similar to the label.
        
    x_i: list,
        List of size (1 x N-1) N is the number of features + 1. This should be
        a single feature vector assocaited with one instance of data.

    Returns
    ------------
    prediction: float,
        Returnss the expected label calculated from the model weights and 
        features.
    '''
    x_i = np.matrix(x_i)
    x_i = np.hstack((np.ones((1,1)), x_i))
    prediction = weights*x_i.transpose()
    return prediction[0,0]

def lms_error(weights, x, y):
    '''
    Used in the batch and stochastic gradient decent functions.
    
    Parameters
    ------------
    weights: numpy.matrix,
        Matrix of size (1 x N) where N is the number of features + 1, calulated
        from len([b]+[w]) where b is the bias andd w is the weight.
        
    x: numpy.matrix,
        Matrix of size (M x N) where M is the number of instances, and N is the
        number of features + 1.
        
    y: numpy.matrix,
        Matrix of size (M x 1) where M is the number of instances. This is the
        label of the data used for training.

    Returns
    ------------
    error: float,
        This return the lms error for the entire dataset. This is done by
        summing the squared difference between the prediction and the label
        values for each instance.
    '''
    error = 0
    for i in range(0, y[0,:].shape[1]):
        error += 0.5 * (y[0,i]-weights*x[i].transpose())**2
    return error[0, 0]

def batch_gradient_descent(weights, x, y, error_threshold=0.5,
                           learning_rate=0.1, max_iterations=500):
    '''
    Function to perform batch gradiant descent on linear data. 

    Parameters
    ------------
    weights: list,
        List of size (1 x N) where N is the number of features + 1, calulated
        from len([b]+[w]) where b is the bias andd w is the weights.

    x: list,
        List of size (M x N) where M is the number of instances, and N is the
        number of features + 1.

    y: list,
        List of size (M x 1) where M is the number of instances. This is the
        label of the data used for training.
    
    error_threshold: float,
        Set the stopping error for the model.

    learning_rate: float,
        This is the rate at which the gradiant is subtracted from the weights 
        during optimization. Larger numbers can allow for faster convergence, 
        but may also result in ocsillations. Smaller values allow for closer 
        convergence to actual minimum.

    max_iterations: int:
        Set the upper limit for how many iterations the program will run. This
        is useful when the minimum-error can't be reached, or the learning rate
        is unstable or too slow.

    Returns
    ------------
    weights: float,
        This returns the optimized weights using the batch gradiaent decent 
        technique.
    '''
    weights = np.matrix(weights)
    x = np.matrix(x)
    x = np.hstack((np.ones((len(y), 1)), x))
    y = np.matrix(y)
    error = lms_error(weights, x, y)
    count = 0
    while error >= error_threshold:
        if count >= max_iterations:
            break
        weights = weights - learning_rate * np.array(batch_gradiant(weights,
                                                                    x, y))
        error = lms_error(weights, x, y)
        count += 1
    return weights


def stochastic_gradiant(weights, x, y, i):
    '''
    Used in the stochastic gradient decent function. 
    
    Parameters
    ------------
    weights: numpy.matrix,
        Matrix of size (1 x N) where N is the number of features + 1, calulated 
        from len([b]+[w]) where b is the bias andd w is the weights.
        
    x: numpy.matrix,
        Matrix of size (M x N) where M is the number of instances, and N is the
        number of features + 1.
        
    y: numpy.matrix,
        Matrix of size (M x 1) where M is the number of instances. This is the 
        label of the data used for training.
    i: integer,
        used to keep track of the data instance for which we are performing our 
        calculation.
        
    Returns
    ------------
    gradiant: list,
        This list contains the gradiant [dJ/dw_j, dJ/dw_j+1, ....., dJ/dw_N].
        (Optimized weights should produce a vector of zeros [0, 0, ...., 0])
    '''
    gradiant = []
    weights = np.matrix(weights)
    for j in range(0, weights[0,:].shape[1]):
        grad_j = -(y[0, i] - weights*x[i].transpose())*x[i, j]
        gradiant.append(grad_j[0,0])
    return gradiant

def stochastic_gradiant_descent(weights, x, y, error_threshold=0.5,
                                learning_rate=0.01, max_iterations=500, print_text=False):
    '''
    Function to perform batch gradiant descent on linear data. 

    Parameters
    ------------
    weights: list,
        List of size (1 x N) where N is the number of features + 1, calulated
        from len([b]+[w]) where b is the bias andd w is the weights.

    x: list,
        List of size (M x N) where M is the number of instances, and N is the
        number of features + 1.

    y: list,
        List of size (M x 1) where M is the number of instances. This is the
        label of the data used for training.
    
    error_threshold: float,
        Set the stopping error for the model.

    learning_rate: float,
        This is the rate at which the gradiant is subtracted from the weights 
        during optimization. Larger numbers can allow for faster convergence, 
        but may also result in ocsillations. Smaller values allow for closer 
        convergence to actual minimum.

    max_iterations: int:
        Set the upper limit for how many iterations the program will run. This
        is useful when the minimum-error can't be reached, or the learning rate
        is unstable or too slow.

    Returns
    ------------
    weights: float,
        This returns the optimized weights using the batch gradiaent decent 
        technique.
    '''
    weights = np.array(weights)
    x = np.matrix(x)
    x = np.hstack((np.ones((len(y),1)), x))
    y = np.matrix(y)
    error = lms_error(weights, x, y)
    count = 0
    while error >= error_threshold:
        for i in range(0, y[0,:].shape[1]):
            if count >= max_iterations:
                return weights
            if print_text == True:
                print('Feature:', x[i])
                print('w:', weights, 'gradiant:', stochastic_gradiant(weights, x, y, i))
            weights = weights - learning_rate * np.array(stochastic_gradiant(weights, x, y, i))
            if print_text == True:
                print('w_t:', weights)
            error = lms_error(weights, x, y)
            count += 1
    return weights


# %%
#This is the trainind data given in the problem 2-1

x = [ 
     [1, -1, 2],
     [1, 1, 3],
     [-1, 1, 0],
     [1, 2, -4],
     [3, -1, -1]
     ]

x_bw = [ 
     [1, 1, -1, 2],
     [1, 1, 1, 3],
     [1, -1, 1, 0],
     [1, 1, 2, -4],
     [1, 3, -1, -1]
     ]

y = [1, 4, -1, -2, 0]


# %%
#    (a)
# Here we test three different combincations of the wieght and bias as insturcted
# in the homework
w = [0, 0, 0]
b = [0]
weights = b + w

optimized_weights  = batch_gradient_descent(weights, x, y, error_threshold=0.001, learning_rate=0.01, max_iterations=1000)
print('gradient:', batch_gradiant(np.matrix(weights), np.matrix(x_bw), np.matrix(y)))
print('LMS error:', lms_error(optimized_weights, np.matrix(x_bw), np.matrix(y)), '\nb,w:', optimized_weights)
# %%
#    (b)
w = [-1, 1, -1]
b = [-1]
weights = b + w

optimized_weights  = batch_gradient_descent(weights, x, y, error_threshold=0.001, learning_rate=0.01, max_iterations=1000)

print('gradient:', batch_gradiant(np.matrix(weights), np.matrix(x_bw), np.matrix(y)))
print('LMS error:', lms_error(optimized_weights, np.matrix(x_bw), np.matrix(y)), '\nb,w:', optimized_weights)
# %%
#    (c)
w = [1/2, -1/2, 1/2]
b = [1]
weights = b + w

optimized_weights  = batch_gradient_descent(weights, x, y, error_threshold=0.001, learning_rate=0.01, max_iterations=1000)
optimized_weights  = [-1,1,1,1]
print('gradient:', batch_gradiant(np.matrix(weights), np.matrix(x_bw), np.matrix(y)))
print('LMS error:', lms_error(optimized_weights, np.matrix(x_bw), np.matrix(y)), '\nb,w:', optimized_weights)
# %%
weights = [0,0,0,0]
stochastic_weights = stochastic_gradiant_descent(weights, x, y, error_threshold=0.001, learning_rate=0.1, max_iterations=5, print_text=True)

# %% 
# Check the predictions here. Should get everythin in the training set correct

weights = [0,0,0,0]
batch_weights = batch_gradient_descent(weights, x, y, error_threshold=0.001, learning_rate=0.01, max_iterations=1000)

stochastic_weights = stochastic_gradiant_descent(weights, x, y, error_threshold=0.001, learning_rate=0.01, max_iterations=1000)


x = [ 
     [1, -1, 2],
     [1, 1, 3],
     [-1, 1, 0],
     [1, 2, -4],
     [3, -1, -1]
     ]

y = [1, 4, -1, -2, 0]

print('Batch error:', lms_error(batch_weights, np.matrix(x_bw), np.matrix(y)))
for x_vector, y_vector in zip(x, y):
    x_ = predict(batch_weights, x_vector)
    print('input', x_vector, 'actual', y_vector, 'predicted', x_)


print('Stochastic error:', lms_error(stochastic_weights, np.matrix(x_bw), np.matrix(y)))
for x_vector, y_vector in zip(x, y):
    x_ = predict(stochastic_weights, x_vector)
    print('input', x_vector, 'actual', y_vector, 'predicted', x_)

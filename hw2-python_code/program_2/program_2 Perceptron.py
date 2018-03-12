# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 16:04:19 2018

@author: Kaai
"""

import numpy as np
import pandas as pd
import random as rnd
import matplotlib.pyplot as plt
# %%

# =============================================================================
# STANDARD PERCEPTRON
# =============================================================================


def train_standard_perceptron(x, y, epochs=10, r=0.1):
    '''
    Used in to train a standard linear perceptrion
    
    Parameters
    ------------
    x: numpy.matrix,
        Matrix of size (M x N) where M is the number of instances, and N is the
        number of features (no bias term).
        
    y: numpy.matrix,
        Matrix of size (M x 1) where M is the number of instances. This is the 
        label of the data used for training.
    
    epochs: int,
        This determines the number of unique runs performed by algorithm

    Returns
    ------------
    weights: float,
        This returns the optimized weights using the perceptron.
    '''
    # add the bias term to the features
    xx = pd.DataFrame(np.matrix(np.hstack((np.ones((len(y), 1)), x))))
    # concatenate x and y data to allow for random sorting during each epoch
    xy = pd.concat([xx, y], axis=1)
    # get weights of the same size as features + bias
    w = np.zeros(len(xx.transpose()))

    for T in range(epochs):
        df_shuffle = xy.sample(frac=1)
        df_shuffle.columns = list(np.arange(0, df_shuffle.shape[1]))
        columns = df_shuffle.columns.values
        for instance in df_shuffle.index.values:
            x_i = np.matrix(df_shuffle.iloc[instance][columns[:-1]])
            y_i = df_shuffle.iloc[instance][columns[-1]]
            if y_i == 0:
                y_i = -1
            check = y_i * w * x_i.transpose()
            if check <= 0:
                w = w + r * y_i*x_i
    weights = w
    return weights


def predict_standard_perceptron(x, weights):
    '''
    Used to predict with standard linear perceptrion given weights
    
    Parameters
    ------------
    x: pandas.DataFrame
        DataFrame of size (M x N) where M is the number of instances, and N is the
        number of features (no bias term).
        
    weights: numpy.matrix,
        input trained weights to get accurate predictions
        
    Returns
    ------------
    y: numpy.array,
        perceptron output
    '''
    y = []
    for i in x.index.values:
        x_i = np.matrix([1] + list(x.iloc[i]))
        y_i = weights*x_i.transpose()
        y.append(np.sign(y_i.mean()))
    return y

# =============================================================================
# VOTED PERCEPTRON
# =============================================================================


def train_voted_perceptron(x, y, epochs=10, r=0.1):
    '''
    Used in to train a voted linear perceptrion

    Parameters
    ------------
    x: numpy.matrix,
        Matrix of size (M x N) where M is the number of instances, and N is the
        number of features (no bias term).

    y: numpy.matrix,
        Matrix of size (M x 1) where M is the number of instances. This is the 
        label of the data used for training.

    epochs: int,
        This determines the number of unique runs performed by algorithm
        
    Returns
    ------------
    weights: float,
        This returns the optimized weights using the perceptron.
    '''
    # add the bias term to the features
    xx = pd.DataFrame(np.matrix(np.hstack((np.ones((len(y), 1)), x))))
    # concatenate x and y data to allow for random sorting during each epoch
    xy = pd.concat([xx, y], axis=1)
    # get weights of the same size as features + bias
    w = np.zeros(len(xx.transpose()))

    weight_vote = []
    for T in range(epochs):
        xy = xy.sample(frac=1)
        xy.columns = list(np.arange(0, xy.shape[1]))
        columns = xy.columns.values
        c_m = 1
        for instance in xy.index.values:
            x_i = np.matrix(xy.iloc[instance][columns[:-1]])
            y_i = xy.iloc[instance][columns[-1]]
            if y_i == 0:
                y_i = -1
            check = y_i * w * x_i.transpose()
            if check <= 0:
                weight_vote.append([w, c_m])
                w = w + r * y_i*x_i
                c_m = 1
            else:
                c_m += 1
                
            
    return weight_vote


def predict_voted_perceptron(x, weight_vote):
    '''
    Used to predict with standard linear perceptrion given weights
    
    Parameters
    ------------
    x: pandas.DataFrame
        DataFrame of size (M x N) where M is the number of instances, and N is the
        number of features (no bias term).
        
    weight_vote: numpy.matrix,
        input trained weight_vote to get accurate prediction
        
    Returns
    ------------
    y: numpy.array,
        perceptron output
    '''
    y = []
    for i in x.index.values:
        x_i = np.matrix([1] + list(x.iloc[i]))
        summation = 0
        for weight in weight_vote:
            summation += weight[1]*np.sign(weight[0]*x_i.transpose())
        y_i = np.sign(summation)
        y.append(np.sign(y_i.mean()))
    return y

# =============================================================================
# AVERAGE PERCEPTRON
# =============================================================================


def train_averaged_perceptron(x, y, epochs=10, r=0.1):
    '''
    Used in to train an averaged linear perceptrion
    
    Parameters
    ------------
    x: numpy.matrix,
        Matrix of size (M x N) where M is the number of instances, and N is the
        number of features (no bias term).
        
    y: numpy.matrix,
        Matrix of size (M x 1) where M is the number of instances. This is the 
        label of the data used for training.
    
    epochs: int,
        This determines the number of unique runs performed by algorithm

    Returns
    ------------
    weights: float,
        This returns the optimized weights using the perceptron.
    '''
    # add the bias term to the features
    xx = pd.DataFrame(np.matrix(np.hstack((np.ones((len(y), 1)), x))))
    # concatenate x and y data to allow for random sorting during each epoch
    xy = pd.concat([xx, y], axis=1)
    # get weights of the same size as features + bias
    w = np.zeros(len(xx.transpose()))
    a = np.matrix(w)
    for T in range(epochs):
        df_shuffle = xy.sample(frac=1)
        df_shuffle.columns = list(np.arange(0, df_shuffle.shape[1]))
        columns = df_shuffle.columns.values
        for instance in df_shuffle.index.values:
            x_i = np.matrix(df_shuffle.iloc[instance][columns[:-1]])
            y_i = df_shuffle.iloc[instance][columns[-1]]
            if y_i == 0:
                y_i = -1
            check = y_i * w * x_i.transpose()
            if check <= 0:
                w = w + r * y_i*x_i
            a += w
    return a


def predict_averaged_perceptron(x, weights):
    '''
    Used to predict with averaged linear perceptrion given weights
    
    Parameters
    ------------
    x: pandas.DataFrame
        DataFrame of size (M x N) where M is the number of instances, and N is the
        number of features (no bias term).
        
    weights: numpy.matrix,
        input trained weights to get accurate predictions
        
    Returns
    ------------
    y: numpy.array,
        perceptron output
    '''
    y = []
    for i in x.index.values:
        x_i = np.matrix([1] + list(x.iloc[i]))
        y_i = weights*x_i.transpose()
        y.append(np.sign(y_i.mean()))
    return y

# =============================================================================
# RESULT ANALYSIS
# =============================================================================

def convert_label(y):
    '''
    used in avg_prediction_error function**
    '''
    for i in range(len(y)):
        if y[i] == -1:
            y[i] = 0
        else:
            y[i] = 1
    return y

def avg_prediction_error(y, y_pred):
    '''
    Parameters
    ------------
    y_predicted: 
    '''
    y_pred = convert_label(y_pred)
    incorrect = sum(abs(np.array(y).ravel()-np.array(y_pred)))
    total = len(np.array(y).ravel())
    avg_error = incorrect/total
    return avg_error

# %%
df_train = pd.read_csv('train.csv', header=None)
df_test = pd.read_csv('test.csv', header=None)

columns_train = df_train.columns.values
X_train = df_train[columns_train[:-1]]
y_train = df_train[columns_train[-1]]

columns_test = df_test.columns.values
X_test = df_test[columns_test[:-1]]
y_test = df_test[columns_test[-1]]

print('____voted____')
weight_vote = train_voted_perceptron(X_train, y_train, r=0.1, epochs=10)
y_predicted = list(predict_voted_perceptron(X_test, weight_vote))
avg_error = avg_prediction_error(y_test, y_predicted)
for vector in weight_vote:
    print('\n- weight vector (voted):',  vector[0].tolist()[0], ', count:', vector[1] )
print('\n- voted perceptron error:', avg_error)

print('\n\n____averaged____')
a = train_averaged_perceptron(X_train, y_train, r=0.1, epochs=10)
y_predicted = list(predict_standard_perceptron(X_test, a))
avg_error = avg_prediction_error(y_test, y_predicted)
print('- learned weight vector (averaged):', a.tolist()[0])
print('\n- averaged perceptron error:', avg_error)

print('\n\n____standard____')
weights = train_standard_perceptron(X_train, y_train, r=0.1, epochs=10)
y_predicted = list(predict_standard_perceptron(X_test, weights))
avg_error = avg_prediction_error(y_test, y_predicted)
print('- learned weight vector (standard):', weights.tolist()[0])
print('\n- standard perceptron error:', avg_error)

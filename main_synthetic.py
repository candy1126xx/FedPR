import json
import math
import numpy as np
import os
import sys
import random


NUM_USER = 4
ALPHA = BETA = 0.0

def softmax(x):
    ex = np.exp(x)
    sum_ex = np.sum( np.exp(x))
    return ex/sum_ex


def generate_synthetic(alpha, beta):

    dimension = 64
    NUM_CLASS = 4
    
    samples_per_user = np.random.lognormal(4, 2, (NUM_USER)).astype(int) + 500
    num_samples = np.sum(samples_per_user)

    X_split = [[] for _ in range(NUM_USER)]
    y_split = [[] for _ in range(NUM_USER)]


    #### define some eprior ####
    mean_W = np.random.normal(0, alpha, NUM_USER)
    mean_b = mean_W
    B = np.random.normal(0, beta, NUM_USER)
    mean_x = np.zeros((NUM_USER, dimension))

    diagonal = np.zeros(dimension)
    for j in range(dimension):
        diagonal[j] = np.power((j+1), -1.2)
    cov_x = np.diag(diagonal)

    for i in range(NUM_USER):
        mean_x[i] = np.random.normal(B[i], 1, dimension)

    for i in range(NUM_USER):
        W = np.random.normal(mean_W[i], 1, (dimension, NUM_CLASS))
        b = np.random.normal(mean_b[i], 1,  NUM_CLASS)
        xx = np.random.multivariate_normal(mean_x[i], cov_x, samples_per_user[i])
        yy = np.zeros(samples_per_user[i])
        for j in range(samples_per_user[i]):
            tmp = np.dot(xx[j], W) + b
            yy[j] = np.argmax(softmax(tmp))
        X_split[i] = xx.tolist()
        y_split[i] = yy.tolist()

    return X_split, y_split


X, y = generate_synthetic(alpha=ALPHA, beta=BETA)     # synthetiv (0,0)

# Create data structure
train_data = {'users': [], 'user_data':{}, 'num_samples':[]}
test_data = {'users': [], 'user_data':{}, 'num_samples':[]}
    
for i in range(NUM_USER):

    combined = list(zip(X[i], y[i]))
    random.shuffle(combined)
    X[i][:], y[i][:] = zip(*combined)
    num_samples = len(X[i])
    train_len = int(0.8 * num_samples)
    test_len = num_samples - train_len
        
    train_data['users'].append(i) 
    train_data['user_data'][i] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
    train_data['num_samples'].append(train_len)
    test_data['users'].append(i)
    test_data['user_data'][i] = {'x': X[i][train_len:], 'y': y[i][train_len:]}
    test_data['num_samples'].append(test_len)
    

with open("data/synthetic_"+str(NUM_USER)+"_"+str(BETA)+"/train.json",'w') as outfile:
    json.dump(train_data, outfile)
with open("data/synthetic_"+str(NUM_USER)+"_"+str(BETA)+"/test.json", 'w') as outfile:
    json.dump(test_data, outfile)

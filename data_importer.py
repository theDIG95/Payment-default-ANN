#! /usr/bin/python3

import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Path for the datafile
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FILE_PATH = BASE_PATH + '/Data/data.xls'

def synthetic_data():
    """Synthetic data for testing the model, Four 2D Gaussian clouds
    centered at corners of square b/w (5,5) and (-5,-5). Each cloud has sigma^2 of 6.
    
    Returns:
        (numpy.ndarray, numpy.ndarray) -- (The inputs, Target labels)
    """

    # Four gaussian clouds
    w = np.random.rand(100, 2) + np.array([-5, 5]) * 6
    x = np.random.rand(100, 2) + np.array([5, 5]) * 6
    y = np.random.rand(100, 2) + np.array([-5, -5]) * 6
    z = np.random.rand(100, 2) + np.array([5, -5]) * 6
    # Combine into single array
    inputs = np.vstack([w, x, y, z])
    # Create output indicator matrix
    targets = np.zeros([400, 1])
    targets[100:200] = 1
    targets[200:300] = 2
    targets[300:] = 3
    # return values
    return (inputs, targets)

def _one_hot_encode(vals):
    """One-hot encoding for categorical data.
    
    Arguments:
        vals {numpy.ndarray} -- Categorical column of the data
    
    Returns:
        numpy.ndarray -- One-hot encoded columns
    """

    enc = np.zeros([vals.shape[0], np.amax(vals) + 1])
    for idx, val in enumerate(vals):
        enc[idx, val] = 1
    return enc

def get_data():
    """Retrieve the data for payment default dataset. 
    Available at https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    
    Returns:
        (numpy.ndarray, numpy.ndarray) -- (Input data, Target labels)
    """

    # Data from file
    pd_df = pd.read_excel(DATA_FILE_PATH)

    # Given credit                          index 0
    lim_bal = np.expand_dims(pd_df['LIMIT_BAL'].values, axis=1)

    # Gender                                index 1
    gen = np.expand_dims(pd_df['SEX'].values - 1, axis=1)

    # Education 1-4                         index 2-5
    edu = pd_df['EDUCATION'].values
    edu[np.argwhere(edu > 4)] = 4   # 5, 6 also considered unknown
    edu[np.argwhere(edu < 1)] = 4   # 0 also considered unknown
    edu = _one_hot_encode(edu - 1)

    # Married 1-3                           index 6-8
    marr = pd_df['MARRIAGE'].values
    marr = _one_hot_encode(marr - 1)

    # Age                                   index 9
    age = np.expand_dims(pd_df['AGE'].values, axis=1)

    # Payment of month delayed by months    index 10-15
    pay_delay = pd_df[['PAY_1', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']]

    # Amount of payment due per month       index 16-21
    prev_bill = pd_df[['BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6']]

    # Amount of payment made per month      index 22-27
    prev_pay = pd_df[['PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']]

    # Defaulted 0-1
    default = np.expand_dims(pd_df['default payment next month'].values, axis=1)

    # Stack values
    inputs = np.hstack([
        lim_bal, gen, edu, marr, age,
        pay_delay, prev_bill, prev_pay
    ])

    # Normalize values
    inputs = preprocessing.minmax_scale(inputs)
    default = preprocessing.minmax_scale(default)

    return inputs.astype(np.float32), default.astype(np.float32)

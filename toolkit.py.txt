# Copyright (c) 2024 Harshita Sharma
# Licensed under the MIT License (see LICENSE file for details)

"""
Data Transformation and Analysis Toolkit
----------------------------------------

This toolkit provides a collection of functions built with numpy for data manipulation, 
cleaning, statistical analysis, and efficient computations.

Modules:
    - Array Manipulation: Functions for array creation, reshaping, and concatenation.
    - Data Cleaning: Functions for handling outliers, filling missing values, and normalization.
    - Statistical Analysis: Functions for calculating statistics, correlations, and aggregations.
    - Efficient Computations: Functions for broadcasting, matrix operations, and vectorized calculations.

Usage:
    This toolkit is designed for educational and practical purposes, allowing users to understand
    and apply core numpy functionalities to real-world data processing and analysis.

Author:
    Harshita Sharma

License:
    This code is licensed under the MIT License. See LICENSE file in the repository for details.
"""


import numpy as np

### 1. Array Creation and Manipulation

def create_custom_array(shape, fill_value=0):
    """
    Creates a numpy array with a specified shape and fill value.
    
    Parameters:
    shape (tuple): Shape of the array (e.g., (2, 3) for a 2x3 array).
    fill_value (int/float, optional): Value to fill the array with. Default is 0.
    
    Returns:
    numpy.ndarray: Array filled with the specified value.
    """
    return np.full(shape, fill_value)

def reshape_array(array, new_shape):
    """
    Reshapes a numpy array to a specified new shape.
    
    Parameters:
    array (numpy.ndarray): Input array to reshape.
    new_shape (tuple): Target shape for the array.
    
    Returns:
    numpy.ndarray: Reshaped array with the specified shape.
    """
    return np.reshape(array, new_shape)

def concatenate_arrays(array1, array2, axis=0):
    """
    Concatenates two arrays along a specified axis.
    
    Parameters:
    array1 (numpy.ndarray): First array to concatenate.
    array2 (numpy.ndarray): Second array to concatenate.
    axis (int, optional): Axis along which to concatenate. Default is 0 (rows).
    
    Returns:
    numpy.ndarray: Concatenated array along the specified axis.
    """
    return np.concatenate((array1, array2), axis=axis)


### 2. Data Cleaning

def remove_outliers(data, threshold=3):
    """
    Removes outliers from a 1D numpy array based on a standard deviation threshold.
    
    Parameters:
    data (numpy.ndarray): Input 1D data array.
    threshold (float, optional): Number of standard deviations to use as threshold. Default is 3.
    
    Returns:
    numpy.ndarray: Array with outliers removed.
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    lower_limit = mean - threshold * std_dev
    upper_limit = mean + threshold * std_dev
    return data[(data >= lower_limit) & (data <= upper_limit)]

def fill_missing_values(data, method='mean'):
    """
    Fills missing values (NaNs) in a numpy array using a specified method.
    
    Parameters:
    data (numpy.ndarray): Input data array containing NaNs.
    method (str, optional): Method to fill missing values ('mean', 'median', or 'zero'). Default is 'mean'.
    
    Returns:
    numpy.ndarray: Array with missing values filled based on the chosen method.
    """
    if method == 'mean':
        fill_value = np.nanmean(data)
    elif method == 'median':
        fill_value = np.nanmedian(data)
    elif method == 'zero':
        fill_value = 0
    else:
        raise ValueError("Method not recognized. Use 'mean', 'median', or 'zero'.")
    return np.where(np.isnan(data), fill_value, data)

def normalize_data(data):
    """
    Normalizes a 1D numpy array to a range between 0 and 1.
    
    Parameters:
    data (numpy.ndarray): Input data array to normalize.
    
    Returns:
    numpy.ndarray: Normalized array with values between 0 and 1.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    return (data - min_val) / (max_val - min_val)


### 3. Statistical Analysis

def calculate_statistics(data):
    """
    Calculates basic statistics (mean, median, variance, standard deviation) for a numpy array.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    
    Returns:
    dict: Dictionary containing 'mean', 'median', 'variance', and 'std_dev' of the data.
    """
    return {
        'mean': np.mean(data),
        'median': np.median(data),
        'variance': np.var(data),
        'std_dev': np.std(data)
    }

def calculate_correlation(data1, data2):
    """
    Calculates the Pearson correlation coefficient between two data arrays.
    
    Parameters:
    data1 (numpy.ndarray): First data array.
    data2 (numpy.ndarray): Second data array.
    
    Returns:
    float: Pearson correlation coefficient between the two arrays.
    """
    return np.corrcoef(data1, data2)[0, 1]

def aggregate_data(data, operation='sum'):
    """
    Aggregates data based on a specified operation.
    
    Parameters:
    data (numpy.ndarray): Input data array.
    operation (str, optional): Aggregation operation ('sum', 'count', 'min', or 'max'). Default is 'sum'.
    
    Returns:
    float/int: Result of the aggregation operation.
    """
    if operation == 'sum':
        return np.sum(data)
    elif operation == 'count':
        return data.size
    elif operation == 'min':
        return np.min(data)
    elif operation == 'max':
        return np.max(data)
    else:
        raise ValueError("Operation not recognized. Use 'sum', 'count', 'min', or 'max'.")


### 4. Efficient Computations

def broadcast_multiply(array_2d, array_1d, axis=0):
    """
    Multiplies a 2D array with a 1D array using broadcasting along a specified axis.
    
    Parameters:
    array_2d (numpy.ndarray): 2D array to multiply.
    array_1d (numpy.ndarray): 1D array for broadcasting.
    axis (int, optional): Axis for broadcasting (0 for column-wise, 1 for row-wise). Default is 0.
    
    Returns:
    numpy.ndarray: Result of broadcasting multiplication.
    """
    if axis == 0:
        return array_2d * array_1d[:, np.newaxis]
    elif axis == 1:
        return array_2d * array_1d
    else:
        raise ValueError("Axis must be 0 or 1.")

def multiply_matrices(matrix_a, matrix_b):
    """
    Multiplies two matrices using numpy's dot product.
    
    Parameters:
    matrix_a (numpy.ndarray): First matrix.
    matrix_b (numpy.ndarray): Second matrix.
    
    Returns:
    numpy.ndarray: Matrix product of the two input matrices.
    """
    return np.dot(matrix_a, matrix_b)

def euclidean_norm(vector):
    """
    Calculates the Euclidean norm (magnitude) of a vector.
    
    Parameters:
    vector (numpy.ndarray): Input vector.
    
    Returns:
    float: Euclidean norm of the vector.
    """
    return np.sqrt(np.sum(vector ** 2))

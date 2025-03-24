
# Fundamntals of tensors using tensorflow

# Imports
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


#%%
print (tf.__version__)

#Create a tensor
scalar = tf.constant(7)
print(scalar)
print(scalar.ndim)
#%%
#Create a vector
vector = tf.constant([10,10])
print(vector)
print(vector.ndim)
#%%
#Create a matrix
matrix = tf.constant([[10,7],[7,10]])
print(matrix)
print(matrix.ndim)
#%%
# Creating tensors with tf.Variable
changeable_tensor=tf.Variable([10,7])
changeable_tensor[0].assign(7)
#%%
#Creating random tensors
random_1 = tf.random.Generator.from_seed(42)
random_1 = random_1.normal(shape=(3,2)) # Creating from  normal distribution
print(random_1)

random_2 = tf.random.Generator.from_seed(42)
random_2 = random_2.uniform(shape=(3,2)) # Creating from uniform distribution
print(random_2)
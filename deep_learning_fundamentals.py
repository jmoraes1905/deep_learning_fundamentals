
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

#Shuffling a tensor on its first shape dimension
tf.random.set_seed(42) # Keeps the same shuffling by setting the global level seed
tf.random.shuffle(random_2, seed=42) #Operating level seed
#%%
# Creating tensors from numpy arrays
tf.zeros(shape=(3,4),dtype=np.int32)

array = np.arange(1,25,dtype=np.int32)

# Changing it to a multidimensional tensor
tensor = tf.constant(array,shape=(2,3,4))
print(tensor)

# Re-shaping the tensor
tensor = tf.constant(array, shape=(3,8))
print(tensor)

#Getting info from tensors

# Create a rank 4 tensor

rank_4 = tf.zeros(shape=(2,3,4,5))

print(rank_4.shape)
print(rank_4.ndim)
print(tf.size(rank_4)) # 2*3*4*5=120
print("Datatype of the elements:",rank_4.dtype)
#%%

#Matrix multiplication

tensor = tf.ones(shape=(2,2))

multi = tf.linalg.matmul(tensor,tensor)
print(multi)

print(tensor*tensor) # Element wise

print(tensor @ tensor) # Actual matrixes multiplication

#Reshaping the tensors
tensor = tf.reshape(tensor, shape=(1,4))
tensor_transposed = tf.reshape(tensor,shape=(4,1)) # Carefull, essentially reshaping is not transposing!

print(tensor)
print(tensor_transposed)
print(tf.matmul(tensor,tensor_transposed))

#You can also transpose using tf.transpose
tensor_transpose = tf.transpose(tensor)
print(tf.matmul(tensor,tensor_transposed))

#%%

#Changing tensors' data type
print(tensor.dtype) #float32

# Altering to dtype float 16 consumes less memory and make operations run faster
tensor = tf.cast(tensor,dtype=tf.float16)
print(tensor.dtype)

# We can also cast to int types
tensor=tf.cast(tensor,dtype=tf.int16)
print(tensor.dtype)
#%%

random_3 = tf.constant(np.random.randint(0,100,size=50))
print(random_3)
random_3=tf.cast(random_3,dtype=tf.int16)
print(random_3.dtype)

#Find the minimum
print(tf.reduce_min(random_3))

#Find the maximum
print(tf.reduce_max(random_3))

#Find the mean
print(tf.reduce_mean(random_3))

#Find the sum
print(tf.reduce_sum(random_3))

#Find the variance
print(tf.math.reduce_variance(tf.cast(random_3,dtype=tf.float16))) # Requires input to be real number

#Find the standard deviation
print(tf.math.reduce_std(tf.cast(random_3,dtype=tf.float16)))
#%%

# Finding the positional maximum and minimum of a tensor

print(tf.argmax(random_3))

print(tf.reduce_max(random_3)==random_3[tf.argmax(random_3)])

if(tf.reduce_max(random_3)==random_3[tf.argmax(random_3)]):
    print(random_3[tf.argmax(random_3)])

print(tf.argmin(random_3))

if(tf.reduce_min(random_3)==random_3[tf.argmin(random_3)]):
    print(random_3[tf.argmin(random_3)])
#%%
#One-hot-encoding

aux_list = [1,2,3] # Onliy numerical values are allowed
one_hot_tensor =tf.one_hot(aux_list,depth=3)
print(one_hot_tensor)
#%%
#You can convert tensorflow types to numpy arrays and back
random_3 = random_3.numpy()
print(type(random_3))

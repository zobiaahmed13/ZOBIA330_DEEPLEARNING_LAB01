# ZOBIA(330)_DEEPLEARNING_LAB01

#CODES:
#TASK 01:
PART(A):
#ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 01 / PART A:
import tensorflow as tf
print("ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 01 / PART A:\n")
# Create a 3D tensor with random float values
random_tensor = tf.random.uniform(shape=(3, 2, 3), minval=0, maxval=1, dtype=tf.float32)
# Calculate the sum, mean, and maximum of the tensor
tensor_sum = tf.reduce_sum(random_tensor)
tensor_mean = tf.reduce_mean(random_tensor)
tensor_max = tf.reduce_max(random_tensor)
# Print the results
print("Random Tensor:\n", random_tensor.numpy())
print("\nSum:", tensor_sum.numpy())
print("Mean:", tensor_mean.numpy())
print("Max:", tensor_max.numpy())

PART(B):
#ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 01 / PART B:
print("ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 01 / PART B:\n")
# Create another 3D tensor for slicing
tensor_3d = tf.constant([
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
    [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
])
# Access a specific element (e.g., element at position [1, 2, 1])
specific_element = tensor_3d[1, 2, 1]
# Access a 2D slice (e.g., slice of the first layer)
two_d_slice = tensor_3d[0, :, :]
# Access a 1D slice (e.g., first row of the second layer)
one_d_slice = tensor_3d[2, 1, :]
# Print the accessed elements and slices
print("Specific Element [1, 2, 1]:", specific_element.numpy())
print("2D Slice (First Layer):\n", two_d_slice.numpy())
print("1D Slice (First Row of Second Layer):", one_d_slice.numpy())

#TASK: 02:
#ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 02:
print("ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 02:\n")
import tensorflow as tf
# Define the tensor operations function
@tf.function
def tensor_operations(tensor1, tensor2):
    addition = tf.add(tensor1, tensor2)
    subtraction = tf.subtract(tensor1, tensor2)
    multiplication = tf.multiply(tensor1, tensor2)
    division = tf.divide(tensor1, tensor2)   
    return {
        "addition": addition,
        "subtraction": subtraction,
        "multiplication": multiplication,
        "division": division
    }
# Test the function with the provided tensors
tensor1 = tf.constant([10, 20, 30], dtype=tf.float32)
tensor2 = tf.constant([5, 10, 15], dtype=tf.float32)
# Call the tensor_operations function
results = tensor_operations(tensor1, tensor2)
# Print the results
print("Results of Tensor Operations:")
for operation, result in results.items():
    print(f"{operation.capitalize()}: {result.numpy()}")

#TASK 03:
PART(A):
#ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 03 / PART A:
print("ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 03 / PART A:\n")
import numpy as np
# Create a NumPy array with random float values
random_array = np.random.rand(10)  # Array of 10 random floats
# Sort the array in ascending order
sorted_ascending = np.sort(random_array)
# Sort the array in descending order
sorted_descending = sorted_ascending[::-1]  # Reverse the sorted array
# Print the results
print("Random Array:", random_array)
print("\nSorted Ascending:", sorted_ascending)
print("\nSorted Descending:", sorted_descending)

PART(B):
#ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 03 / PART B:
print("ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 03 / PART B:\n")
# Create a 2x2 NumPy matrix
matrix_2x2 = np.array([[1, 2], [3, 4]])
# Calculate its transpose
matrix_transpose = np.transpose(matrix_2x2)
# Calculate its inverse
matrix_inverse = np.linalg.inv(matrix_2x2)
# Print the results
print("2x2 Matrix:\n", matrix_2x2)
print("Transpose of Matrix:\n", matrix_transpose)
print("Inverse of Matrix:\n", matrix_inverse)

PART(C):
#ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 03 / PART C:
print("ZOBIA AHMED / 2022F-BSE-330 / LAB 01 / TASK 03 / PART C:\n")
# Create a 2D NumPy array
array_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
# Calculate the sum of each row
row_sums = np.sum(array_2d, axis=1)
# Calculate the mean of each column
column_means = np.mean(array_2d, axis=0)
# Calculate the standard deviation of the entire array
std_deviation = np.std(array_2d)
# Print the results
print("2D Array:\n", array_2d)
print("\nSum of Each Row:", row_sums)
print("Mean of Each Column:", column_means)
print("Standard Deviation of the Entire Array:", std_deviation)

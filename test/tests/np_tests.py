import numpy as np
import time

# Example array
arr = np.random.randint(1, 6, size=1000000)

# Mapping dictionary and array
mapping = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50}
mapping_array = np.array([0, 10, 20, 30, 40, 50])

# Method 1: Using np.vectorize
vectorized_mapping = np.vectorize(lambda x: mapping[x])
start = time.time()
new_arr_vectorize = vectorized_mapping(arr)
end = time.time()
print(f"np.vectorize time: {end - start:.6f} seconds")

# Method 2: Using np.take
start = time.time()
new_arr_take = mapping_array.take(arr)
end = time.time()
print(f"np.take time: {end - start:.6f} seconds")

# Method 3: Using Boolean Indexing
start = time.time()
new_arr_indexing = np.copy(arr)
new_arr_indexing[arr == 1] = 10
new_arr_indexing[arr == 2] = 20
new_arr_indexing[arr == 3] = 30
new_arr_indexing[arr == 4] = 40
new_arr_indexing[arr == 5] = 50
end = time.time()
print(f"Boolean Indexing time: {end - start:.6f} seconds")

# Method 4: Using Dictionary and List Comprehension
start = time.time()
new_arr_list_comp = np.array([mapping[x] for x in arr])
end = time.time()
print(f"List Comprehension time: {end - start:.6f} seconds")
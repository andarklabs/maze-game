import math

""" Turns a 2d array into a 1d flattened array """
def flatten(array) -> list: 
    new_array = []
    for row in array:
        for elm in row:
            new_array.append(elm)

    return new_array 

""" turns a 1d flattened array back into a 2d array """
def unflatten(array) -> list:
    size = len(array)
    dim = math.sqrt(size)
    outer_array = []

    for i in range(dim):
        inner_array = []
        for j in range(dim):
            inner_array.append(array[i*dim + j])
        outer_array.append(inner_array)

    return outer_array

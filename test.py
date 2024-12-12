from concurrent.futures import ProcessPoolExecutor
from functools import partial

def some_function(y, param1, param2):
    x = y[1]
    print(f"Processing {y[0]} with params: {param1}, {param2}")
    return x * param1 + param2

if __name__ == "__main__":
    inputs = {1:2, 2:3, 3:4, 4:5, 5:6}
    param1 = 10
    param2 = 20

    # Create a partial function with param1 and param2 fixed
    partial_function = partial(some_function, param1=param1, param2=param2)

    with ProcessPoolExecutor() as executor:
        results = executor.map(partial_function, inputs.items())

    print(list(results))

import inspect

def my_func():
    print("My function name is:", inspect.getframeinfo(inspect.currentframe()).function)
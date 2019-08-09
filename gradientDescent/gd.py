'''
inputs:
x is cost???  input to f(x) = x**2 + 5
gradx is gradient
lr
'''

def gradient_descent_update(x, gradx, learning_rate):
    """
    Performs a gradient descent update.
    """
    # Implement gradient descent.
    x = x - learning_rate * gradx
    # Return the new value for x
    return x

import numpy as np

def linear(x1,x2):
    return np.dot(x1,x2)

def gaussian(x1,x2, sigma=0.5):
    return np.exp(-np.linalg.norm(x1-x2)**2/(2*(sigma**2)))
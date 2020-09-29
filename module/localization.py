import numpy as np
from numpy import sqrt
import matplotlib.pyplot as plt

# localization用の関数
def calc_dist(i, j, J=40):
    return sqrt(min([(i-j)**2, (i+J-j)**2]))

def polynomial(x, coefs):
    return np.array([coefs[i]*x**(i) for i in range(len(coefs))]).sum()

def gaspari_cohn(d, c):
    x = d/c
    if d < c:
        return polynomial(x, [1, 0, -5/3, 5/8, 1/2, -1/4])
    elif c < d and d < 2*c:
        return polynomial(x, [-2/3, 4, -5, 5/3, 5/8, -1/2, 1/12])/x
    else:
        return 0

def plot_gaspari_cohn(c):
    fig, ax = plt.subplots()
    x_arr = np.linspace(0, 20, 100)
    y_func = [gaspari_cohn(x, c=c) for x in x_arr]
    plt.plot(x_arr, y_func, label='gaspari cohn')
    plt.title('Localization function')
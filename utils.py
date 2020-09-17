# よく使う関数をまとめたもの
import numpy as np

# Runge-Kuta
def rk4(t, y, dt, f):
    k1 = f(t, y)
    k2 = f(t+dt/2, y + k1*dt/2)
    k3 = f(t+dt/2, y + k2*dt/2)
    k4 = f(t+dt, y + k3*dt)
    dy = y + (k1 + 2*k2 + 2*k3 + k4)*dt/6
    return dy

# Euler
def euler(t, x, dt, f):
    return x + f(t, x)*dt

# Lorenz96
# F, J を指定した関数lorenz(t,x)を返す
def make_lorenz96(F, J=40):
    return lambda t,x: lorenz96(t, x, F, J)

def lorenz96(t, x, F, J):
    dx = np.zeros(J)
    for j in range(J):
        dx[j] = (x[(j+1)%40] - x[(j-2)%40])*x[(j-1)%40] - x[j]
    dx += F
    return dx

# 各時刻でrmseを計算
def error_series_kf(true, estimate, time_index, J=40):
    return np.array([np.linalg.norm(estimate[t] - true[t])/np.sqrt(J) for t in time_index])

def search_optimal_param(params, errors):
    errors = np.array(errors)
    idx = errors.argmin()
    return params[idx], idx
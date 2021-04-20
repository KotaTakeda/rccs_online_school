# よく使う関数をまとめたもの
# TODO: ファイル分割
import numpy as np
import matplotlib.pyplot as plt

# 4次Runge-Kuta
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

"""
true: 真の値
obs: 観測値
kf_results: list (ndarray); kfに推定値(複数可)
legendes: list (string); kf_resultsのlegend
trPs: list (ndarray); 
legends_trP: 
例: plot_error_KF (true, obs, [result1, result2], legends=['result1', 'result2'], trP=[trP1], legends_trP=['])
"""
def plot_error_KF(true, obs, kf_results, legends=['filtering error'], trPs=[], legends_trP=['trP']):
    results = np.array(kf_results)
    T = results.shape[1]

    # plot
    fig, ax = plt.subplots(figsize=(20,3))
    # 観測誤差
    obs_error = error_series_kf(true, obs, range(T))
    ax.plot(obs_error, label='obs error', lineWidth=0.5)
    # 推定誤差
    for k in range(len(results)):
        filtering_error = error_series_kf(true, results[k], range(T))
        ax.plot(filtering_error, label=legends[k].format(k), lineWidth=0.5)
    # 推定誤差共分散のtrace
    for i in range(len(trPs)):
        ax.plot(trPs[i], label=legends_trP[i].format(k), lineWidth=0.5)

    ax.set_xlabel('step')
    ax.set_ylabel('$ error $')
    plt.title('obs and filtering error')
    plt.legend()
    
    plt.show()

"""
推定のパラメータ依存性を調べる
引数: resultsとparamsの長さは等しい．
仕様: filteringが馴染んだ後の200step以降のrmseの時間平均をとる．
"""
def estimate_error_with_params(true, results, params, param_name, plot=True, log=False,):
    results = np.array(results)
    T = results.shape[1] - 200
    filtering_errors = []

    for k in range(len(params)):
        filtering_error = error_series_kf(true, results[k], np.arange(T)+200).mean()
        filtering_errors.append(filtering_error)
    optimal_param, optimal_idx = search_optimal_param(params, filtering_errors)
    optimal_error = filtering_errors[optimal_idx]

    if plot:
        fig, ax = plt.subplots()
        ax.plot(params, filtering_errors)
        ax.scatter(optimal_param, optimal_error)
        ax.set_xlabel(param_name)
        ax.set_ylabel('rmse time mean')
        if log:
            plt.xscale('log')
        plt.title('rmse time mean by {}'.format(param_name))
        fig.text(0, 0, 'optimal param: {}, optimal rmse: {}'.format(optimal_param, optimal_error))
    return optimal_param, optimal_idx, optimal_error

import numpy as np
# from numpy import multivariate_normal
import pdb
            
class ExtendedKalmanFilter:
    def __init__(self, M, H, Q, R, y, x_0, P_0, dim_x=2, dt=0.05, delta=1e-3, var3d=False, inflation_factor=0):
        self.M = M
        self.H = H
        self.Q = Q
        self.R = R
        self.A = inflation_factor*np.identity(dim_x)
        self.y = y
        self.dt = dt
        self.dim_x = dim_x # todo : x_0から計算
        self.P = P_0
        self.P_hist = []
        self.x_a = x_0
        self.x = []
        self.delta = delta
        self.var3d = var3d
        
  # 逐次推定を行う
    def forward_estimation(self, verbose=False):
        count = 0
        for y_obs in self.y:
            self._forecast()
            self._update(y_obs)
            
            if verbose:
                if count%10 == 0:
                    print('step: {}, x: {}, P11: {}'.format(count, self.x_a, self.P[0,0]))
                count += 1

    # 更新/解析
    def _update(self, y_obs):
        self.P = self.P + self.A # 誤差共分散膨張, P = self.P + self.Aとするとうまくいかない．
        P = self.P
        H = self.H
                
        # Kalman gain 
        K = P@H.T@np.linalg.inv(H@P@H.T + self.R)
        
        # 誤差共分散更新
        if not self.var3d:
            self.P -= K@H@P

        # x 更新
        self.x_a = self.x_f + K@(y_obs - H@self.x_f)

        # 更新した値を保存
        self.x.append(self.x_a)
        self.P_hist.append(self.P)

    # 予報/時間発展
    def _forecast(self, log=False):
        x_a = self.x_a; dt = self.dt; M = self.M
        N = self.dim_x
        
        # 予報
        self.x_f = self.M(x_a, dt)
        
        if not self.var3d:
            # 線形化， サイクルを変化させるとうまくいかなくなる
            JM = np.zeros((N, N))
            for j in range(N):
                dx = self.delta*np.identity(N)[:, j]
                JM[:, j] = (M(x_a + dx, dt) - M(x_a, dt))/self.delta

            self.P = JM@self.P@JM.T + self.Q

        if log:
            self.x.append(self.x_f)
    
    # 追加の推定(観測値なし)
    def additional_forecast(self, step):
        for _ in range(step):
            self._forecast(log=True)

# ========================
# 未完成
# ========================
# Ensemble Kalman Filter PO法
"""
Parameters

M: callable(x, dt)
  状態遷移関数

H: ndarray(dim_y, dim_x)
  観測関数
  
Q: ndarray(dim_x, dim_x)
  モデルの誤差共分散行列
  
R: ndarray(dim_y, dim_y)
  観測の誤差共分散行列

N: アンサンブルメンバーの数

x: ndarray(dim_x)

"""
class EnsembleKalmanFilter:
    def __init__(self, M, H, Q, R, y, x_0, P_0, dim_x=2, dim_y=1, N=10, dt=0.05):
        self.M = M
        self.H = H
        self.Q = Q
        self.R = R
        self.y = y
        self.N = N
        self.dt = dt
        self.dim_x = dim_x # todo : x_0から計算
        self.dim_y = dim_y # todo : yから計算
        self.mean = np.zeros(self.dim_x)
        self.mean_y = np.zeros(self.dim_y)
        self.x_mean = np.zeros(self.dim_x)
        self.x = []

        self._initialize(x_0, P_0, N)

  #　初期状態
    def _initialize(self, x_0, P_0, N):
        self.ensemble = np.zeros((self.N, self.dim_x))
        e = np.random.multivariate_normal(self.mean, P_0, N)
        for i in range(N):
            self.ensemble[i] = x_0 + e[i]
    
        self.x = np.mean(self.ensemble, axis=0)
    
  # 逐次推定を行う
    def forward_estimation(self):
        for y_obs in self.y:
            self._forecast()
            self._update(y_obs)

    # 更新/解析
    def _update(self, y_obs):
        x_f = self.x_mean
        X_f = self.ensemble
        H = self.H
        N = self.N

        #dx
        dX = X_f - x_f

        # P_f: 予報誤差共分散を計算
        P_f = (dX.T@dX) / (N-1)
        
        # Kalman gain 
        K = P_f@H.T@np.linalg.inv(H@P_f@H.T + self.R)

        # アンサンブルで x(k) 更新, ノイズを加える．
        e = np.random.multivariate_normal(self.mean_y, self.R, size=N)
        self.ensemble = X_f + K@(y_obs + e - H@X_f)

        # 更新した値のアンサンブル平均　x を保存
        self.x.append(self.x_mean)

    # 予報/時間発展
    def _forecast(self, log=False):
    # アンサンブルで x(k) 予測
        for i, s in enumerate(self.ensemble):
            self.ensemble[i] = self.M(s, self.dt)

        self.x_mean = self.ensemble.mean(axis=0)

        if log:
            self.x.append(self.x_mean)
    
    # 追加の推定(観測値なし)
    def additional_forecast(self, step):
        for _ in range(step):
            self._forecast(log=True)

# ========================
# 未完成
# ========================
# Ensemble Kalman Filter SRF法
"""
Parameters

M: callable(x, dt)
  状態遷移関数

H: callable(x)
  観測関数
  
Q: ndarray(dim_x, dim_x)
  モデルの誤差共分散行列
  
R: ndarray(dim_y, dim_y)
  観測の誤差共分散行列

N: アンサンブルメンバーの数

x: ndarray(dim_x)

"""
class EnSquareRootFilter:
    def __init__(self, M, H, Q, R, y, x_0, P_0, dim_x=2, dim_y=1, N=10, dt=0.05):
        self.M = M
        self.H = H
        self.Q = Q
        self.R = R
        self.y = y
        self.N = N
        self.dt = dt
        self.dim_x = dim_x # todo : x_0から計算
        self.dim_y = dim_y # todo : yから計算
        self.mean = np.zeros(self.dim_x)
        self.x = np.zeros(self.dim_x)
        self.x_log = []

        self._initialize(x_0, P_0, N)

  #　初期状態
    def _initialize(self, x_0, P_0, N):
        self.ensemble = np.zeros((self.N, self.dim_x))
        e = np.random.multivariate_normal(self.mean, P_0, N)
        for i in range(N):
            self.ensemble[i] = x_0 + e[i]
    
        self.x = np.mean(self.ensemble, axis=0)
    
  # 逐次推定を行う
    def forward_estimation(self):
        for y_obs in self.y:
            self._forecast()
            self._update(y_obs)
            
    # 更新/解析
    def _update(self, y_obs):
        N = self.N

        #dx dy
        dX_f = self.ensemble - self.x
        P_f = (dX_f.T@dX_f)/(N-1)
        
        # Kalman gain 
        K = P_f@H.T@np.linalg.inv(H@P_f@H.T + R)

        # K' TODO:
        # 2通りの方法で実装が考えられる．
        K_dash = P_f@H()

        # アンサンブルで x(k) 更新
        x_a = self.x + K@(y_obs - self.H(self.x))
        dX_a = dX_f - K_dash@H.T@dX_f
        self.ensemble = dX_a + x_a

        # 更新した値のアンサンブル平均　x を保存
        self.x_log.append(self.x)

    # 予報/時間発展
    def _forecast(self, log=False):
    # アンサンブルで x(k) 予測
        for i, s in enumerate(self.ensemble):
            self.ensemble[i] = self.M(s, self.dt)

        self.x = np.mean(self.ensemble, axis=0)

        if log:
            self.x_log.append(self.x)
    
    # 追加の推定(観測値なし)
    def additional_forecast(self, step):
        for _ in range(step):
            self._forecast(log=True)
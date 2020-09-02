import numpy as np

# Linear Kalman Filter 
"""
Parameters

M: (dim_x, dim_x)
  状態遷移行列

H: (dim_y, dim_x)
  観測行列
  
Q: ndarray(dim_x, dim_x)
  モデルの誤差共分散行列
  
R: ndarray(dim_y, dim_y)
  観測の誤差共分散行列

x: ndarray(dim_x)

"""
class LinearKalmanFilter:
    def __init__(self, M, H, Q, R, y, x_0, P_0, dim_x=2, dt=0.1):
        self.M = M
        self.H = H
        self.Q = Q
        self.R = R
        # self.A = A # 加法的誤差共分散膨張行列
        self.y = y
        self.dt = dt
        self.dim_x = dim_x # todo : x_0から計算
        self.P = P_0
        self.x_a = x_0
        self.x = []
        
  # 逐次推定を行う
    def forwardEstimation(self, verbose=False):
        count = 0
        for y_obs in self.y:
            self._forecast()
            self._update(y_obs)

            if verbose:
                if count%10 == 0:
                    print('step: {}, x: {}'.format(count, self.x_a))
                count += 1

    # 更新/解析
    def _update(self, y_obs):
        P = self.P # ここで予報誤差に定数を加えることができる．（誤差共分散膨張)
        H = self.H
                
        # Kalman gain 
        K = P@H.T@np.linalg.inv(H@P@H.T + self.R)
        
        # 誤差共分散更新
        self.P -= K@H@P

        # アンサンブルで x(k) 更新
        self.x_a = self.x_f + K@(y_obs - H@self.x_f)

        # 更新した値を保存
        self.x.append(self.x_a)

    # 予報/時間発展
    def _forecast(self, log=False):
        self.x_f = self.M@self.x_a
        self.P = self.M@self.P@self.M.T + self.Q

        if log:
            self.x.append(self.x_f)
    
    # 追加の推定(観測値なし)
    def additional_forecast(self, step):
        for _ in range(step):
            self._forecast(log=True)
            
class ExtendedKalmanFilter:
    def __init__(self, M, H, Q, R, y, x_0, P_0, dim_x=2, dt=0.1, delta=1e-3):
        self.M = M
        self.H = H
        self.Q = Q
        self.R = R
        # self.A = A # 加法的誤差共分散膨張行列
        self.y = y
        self.dt = dt
        self.dim_x = dim_x # todo : x_0から計算
        self.P = P_0
        self.x_a = x_0
        self.x = []
        self.delta = delta
        
  # 逐次推定を行う
    def forwardEstimation(self, verbose=False):
        count = 0
        for y_obs in self.y:
            self._forecast()
            self._update(y_obs)

            if verbose:
                if count%10 == 0:
                    print('step: {}, x: {}'.format(count, self.x_a))
                count += 1

    # 更新/解析
    def _update(self, y_obs):
        P = self.P # ここで予報誤差に定数を加えることができる．（誤差共分散膨張)
        H = self.H
                
        # Kalman gain 
        K = P@H.T@np.linalg.inv(H@P@H.T + self.R)
        
        # 誤差共分散更新
        self.P -= K@H@P

        # アンサンブルで x(k) 更新
        self.x_a = self.x_f + K@(y_obs - H@self.x_f)

        # 更新した値を保存
        self.x.append(self.x_a)

    # 予報/時間発展
    def _forecast(self, log=False):
        x_a = self.x_a; dt = self.dt; M = self.M
        N = self.dim_x
        
        # 予報
        self.x_f = self.M(x_a, dt)
        
        # 線形化
        JM = np.zeros((N,N))
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

import numpy as np
from numpy import sqrt, trace, zeros, identity, exp, random
from numpy.random import multivariate_normal, choice
from numpy.linalg import inv
import scipy
from scipy.linalg import sqrtm
import pdb
            
class ExtendedKalmanFilter:
    def __init__(self, M, H, Q, R, y, x_0, P_0, dt=0.05, delta=1e-3, var3d=False, alpha=1, cut_obs_size=0):
        self.M = M
        self.H = H
        self.Q = Q
        self.R = R
        self.y = y
        self.dt = dt
        self.dim_x = Q.shape[0]
        self.P = P_0
        self.trP = []
        self.x_a = x_0
        self.x = []
        self.delta = delta
        self.var3d = var3d
        #self.A = alpha*identity(dim_x) #　加法的誤差共分散膨張
        self.alpha = alpha # 1以上
        self.cut_obs_size = cut_obs_size
        
  # 逐次推定を行う
    def forward_estimation(self):
        count = 0
        for y_obs in self.y:
            self._forecast()
            self._update(y_obs)

    # 更新/解析
    def _update(self, y_obs):
        # self.P = self.P + self.A # 加法的誤差共分散膨張, P = self.P + self.Aとするとうまくいかない． 乗法的方法に変更
        self.P = self.alpha*self.P # 乗法的
        P = self.P

        H = self.H.copy()
        # 観測値をcut_obs_sizeだけランダムに間引く
        obs_choice = choice(range(self.dim_x), self.cut_obs_size, replace=False)
        for n in obs_choice:
            H[n, n] = 0
                
        # Kalman gain 
        K = P@H.T@inv(H@P@H.T + self.R)
        
        # 誤差共分散更新
        if not self.var3d:
            self.P -= K@H@P

        # x 更新
        self.x_a = self.x_f + K@(y_obs - H@self.x_f)

        # 更新した値を保存
        self.x.append(self.x_a)
        self.trP.append(sqrt(trace(self.P)/40)) # traceを正規化して保存

    # 予報/時間発展
    def _forecast(self, log=False):
        x_a = self.x_a; dt = self.dt; M = self.M; N = self.dim_x
        
        # 予報
        self.x_f = self.M(x_a, dt) #保存しておく
        
        if not self.var3d:
            # 線形化， dtを大きくするとうまくいかなくなる
            JM = zeros((N, N))
            for j in range(N):
                dx = self.delta*identity(N)[:, j]
                JM[:, j] = (M(x_a + dx, dt) - self.x_f)/self.delta # ここでJM[:, j] = (M(x_a + dx, dt) - self.M(x_a, dt))/self.deltaとするとすごく遅くなる

            self.P = JM@self.P@JM.T + self.Q

        if log:
            self.x.append(self.x_f)
    
    # 追加の推定(観測値なし)
    def additional_forecast(self, step):
        for _ in range(step):
            self._forecast(log=True)

# TODO: kalman_filters.pyに決定版を保存
# ========================
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
    def __init__(self, M, H, Q, R, y, x_0, P_0, N=10, dt=0.05, alpha=1, localization=True):
        self.M = M
        self.H = H
        self.Q = Q
        self.R = R
        self.y = y
        self.N = N # アンサンブルメンバー数
        self.dt = dt
        
        # 実装で技術的に必要
        self.dim_x = Q.shape[0]
        self.dim_y = R.shape[0]
        self.mean_y = zeros(self.dim_y)
        
        self.alpha = alpha # inflation用の定数
        self.localization = localization
        if localization:
            self.loc_mat = self.make_loc_mat() # localization用の行列

        # filtering実行用
        self.x = [] # 記録用
        self.trP = []

        self._initialize(x_0, P_0, N)

    #　初期状態
    def _initialize(self, x_0, P_0, N):
        random.seed(0)
        self.X = x_0 + multivariate_normal(zeros(self.dim_x), P_0, N)
        self.x_mean = self.X.mean(axis=0)
    
    # 逐次推定を行う
    def forward_estimation(self):
        for y_obs in self.y:
            self._forecast()
            self._update(y_obs)

    # 更新/解析
    def _update(self, y_obs):
        X_f = self.X; x_f = self.x_mean; H = self.H; N = self.N; R = self.R

        # P_f: 予報誤差共分散を計算
        dX = X_f - x_f # (N, dim_x)
        P_f = (dX.T@dX) / (N-1) # (dim_x, dim_x) dim_xが大きい場合はP_fをメモリ上に持たない方が良い
        
        # localizationとinflation
        if self.localization:
            P_f = self.loc_mat*P_f
        P_f = self.alpha*P_f
        
        # Kalman gain 
        K = P_f@H.T@inv(H@P_f@H.T + R) # (dim_x, dim_y)

        # アンサンブルで x(k) 更新, ノイズを加える．
        e = multivariate_normal(self.mean_y, R, N) # (N, dim_x)
        for i in range(N):
            self.X[i] = X_f[i] + K@(y_obs + e[i] - H@X_f[i]) # dim_x

        # 記録: 更新した値のアンサンブル平均xを保存, 推定誤差共分散P_fのtraceを保存
        self.x.append(self.X.mean(axis=0))
        self.trP.append(sqrt(trace(P_f)/40))

    # 予報/時間発展
    def _forecast(self):
        # アンサンブルで x(k) 予測
        for i, s in enumerate(self.X):
            self.X[i] = self.M(s, self.dt)

        self.x_mean = self.X.mean(axis=0)

    def make_loc_mat(self):
        J = self.dim_x
        mat = zeros((J,J))
        for i in range(J):
            for j in range (J):
                mat[i, j] = exp(-0.5*(i-j)**2)
        return mat


# ========================
# EnsembleSquareRootFilter
# ========================
"""
Parameters
M: callable(x, dt)
  状態遷移関数
H: ndarray(dim_y, dim_x)
  観測行列  
Q: ndarray(dim_x, dim_x)
  モデルの誤差共分散行列 
R: ndarray(dim_y, dim_y)
  観測の誤差共分散行列
x_0: 状態変数の初期値
P_0: 誤差共分散の初期値
N: アンサンブルメンバーの数
dt: 同化時間step幅
alpha: inflation factor
localization: localizationの設定
"""
class EnsembleSquareRootFilter:
    def __init__(self, M, H, Q, R, y, x_0, P_0, N=40, dt=0.05, alpha=1):
        self.M = M
        self.H = H
        self.Q = Q
        self.R = R
        self.y = y
        self.N = N # アンサンブルメンバー数
        self.dt = dt
        
        # 実装で技術的に必要
        self.dim_x = Q.shape[0]
        self.I = identity(N)
        
        self.alpha = alpha # inflation用の定数

        # filtering実行用
        self.x = [] # 記録用
        self.trP = []

        self._initialize(x_0, P_0, N)

  #　初期状態
    def _initialize(self, x_0, P_0, N):
        random.seed(0)
        self.X = x_0 + multivariate_normal(np.zeros(self.dim_x), P_0, N) # (N, J)
        self.x_mean = self.X.mean(axis=0)
    
  # 逐次推定を行う
    def forward_estimation(self):
        for y_obs in self.y:
            self._forecast()
            self._update(y_obs)

    # 更新/解析
    def _update(self, y_obs):
        X_f = self.X; x_f = self.x_mean; alpha = self.alpha; H = self.H; R = self.R; N = self.N; I = self.I

        # dX, dYを計算
        dX_f = X_f - x_f # (N, dim_x)
        dX_f = sqrt(alpha)*dX_f # inflation
        dY = (H@dX_f.T).T # (N, dim_y)
        
        # Kalman gain 
        K = dX_f.T@dY@inv(dY.T@dY + (N-1)*R) # (dim_x, dim_y)
        # 平均を更新
        x_a = x_f + K@(y_obs - H@x_f) # dim_x

        # dXを変換, I - dY^t(dYdY^t + (N-1)R)dYの平方根をとる
        # sqrtmの内部ではユニタリー変換により上三角化を行い平方根を計算する．(scipy.linalg.sqrtm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.sqrtm.html)
        T = sqrtm(I - dY@inv(dY.T@dY + (N-1)*R)@dY.T) # (N, N)
        dX_a = (dX_f.T@T).T # (N, dim_x)
        self.X = x_a + dX_a # (N, dim_x)

        # 記録: 更新した値のアンサンブル平均xを保存, 推定誤差共分散P_fのtraceを保存
        self.x.append(self.X.mean(axis=0))
        self.trP.append(sqrt(trace(dX_f.T@dX_f)/40))

    # 予報/時間発展
    def _forecast(self):
        # アンサンブルで x(k) 予測
        for i, s in enumerate(self.X):
            self.X[i] = self.M(s, self.dt)

        self.x_mean = self.X.mean(axis=0)
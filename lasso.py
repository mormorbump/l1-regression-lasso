import numpy as np



# ソフト閾値関数S(x, y)
# S(x, y) = sgn(x)max(0,||x||-y)
def soft_thresholding(x, y):
    return np.sign(x) * max(abs(x) - y, 0)


class Lasso:

    """
    ラッソ回帰を行うクラス。
    ソフト閾値関数S
    S(p, q) = sgn(p)max(0,||p||-q)
    (sgnはpの大小によって符号が変わる符号関数)
    より、ωの更新しきは
    ω_ = S( Σ(yi - w0 - Σxij*ωj)xik, λ ) / Σxik^2
    """
    def __init__(self, lambda_, tol=0.0001, max_iter=1000):
        """
        wが更新したい重み。
        :param lambda_: ハイパーパラメタ
        :param tol: 重みの一個ずれの差を見たとき、どの程度差が減ったら処理を終わらせるかの値。収束判定のためのトレランス（許容度）という。
        :param max_iter: 重みの一個ずれの差がなかなか減らなかった時に強制的に学習を終わらせる回数
        """
        self.lambda_ = lambda_
        self.tol = tol  # 学習の損失
        self.max_iter = max_iter
        self.w_ = None

    def fit(self, X, t):
        """
        学習を行うメソッド。
        :param X: 実際のデータ
        :param t: 教師データ
        :return:
        """
        n, d = X.shape  # nはデータ数、dは次元数。
        self.w_ = np.zeros(d + 1)
        avgl1 = 0.
        for _ in range(self.max_iter):
            avgl1_prev = avgl1
            self._update(n, d, X, t)
            avgl1 = np.abs(self.w_).sum() / self.w_.shape[0]  # |Σw / d| より、ωの平均をavglに代入。
            if abs(avgl1 - avgl1_prev) <= self.tol:
                break

    def _update(self, n, d, X, t):
        """
        全ての座標についてのwの更新を行う。
        更新しきはこれ
        Σ(yi - w0 - Σxij*ωj) * xik - λ / Σxik^2 (λの符号は変わるよ)
        :param n: データ数
        :param d:　次元数
        :param X: データそのもの
        :param t: 教師データ
        :return:
        """
        self.w_[0] = (t - np.dot(X, self.w_[1:])).sum() / n  # ω0の時
        w0vec = np.ones(n) * self.w_[0]  # Xはオフセットなので、w0とかけ合わさるのは1 (X = {1, x1, x2, ..., xd}の１項目)

        # 二個目のdにかんするΣを回す。一つ目のnに関するΣは、X[:, k]とすることでまとめて計算しているので省略可能。
        # むしろ、二つ目のΣはd * n通りの値があるので、省略することができない。
        for k in range(d):
            ww = self.w_[1:]  # w0以外を取得
            ww[k] = 0  # 初期値は0
            q = np.dot(t - w0vec - np.dot(X, ww), X[:, k])  # カッコ内とxikを掛け算
            r = np.dot(X[:, k], X[:, k])  # 更新式の分母
            self.w_[k + 1] = soft_thresholding(q / r, self.lambda_)  # 一つ先のオメガを更新

    def predict(self, X):
        if X.ndim == 1:
            X = X.reshape(X.shape[0], 1)
        Xtil = np.c_[np.ones(X.shape[0]), X]  # オフセットの(X = {1, x1, x2, ..., xd})Xを作成。
        return np.dot(Xtil, self.w_)  # いつも通り線形回帰の定義式Xtill*w。この計算結果が未知のデータの教師データと近ければ良い。


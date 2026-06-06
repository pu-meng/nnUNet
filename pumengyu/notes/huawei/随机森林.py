import numpy as np

# ========== 1. 决策树桩（最简单的决策树，只分一次）==========
class DecisionStump:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        best_gini = float('inf')

        # 随机选部分特征（随机森林的"随机"）
        feature_indices = np.random.choice(n_features,
                                            size=max(1, n_features // 2),
                                            replace=False)

        for feat in feature_indices:
            thresholds = np.unique(X[:, feat])
            for thresh in thresholds:
                left = y[X[:, feat] <= thresh]
                right = y[X[:, feat] > thresh]
                gini = self._gini(left, right, len(y))
                if gini < best_gini:
                    best_gini = gini
                    self.feat = feat
                    self.thresh = thresh

    def _gini(self, left, right, total):
        def g(arr):
            if len(arr) == 0:
                return 0
            p = np.bincount(arr) / len(arr)
            return 1 - np.sum(p ** 2)
        return (len(left) * g(left) + len(right) * g(right)) / total

    def predict(self, X):
        return np.where(X[:, self.feat] <= self.thresh, 0, 1)


# ========== 2. 随机森林 ==========
class SimpleRandomForest:
    def __init__(self, n_trees=10):
        self.n_trees = n_trees
        self.trees = []

    def fit(self, X, y):
        for _ in range(self.n_trees):
            # Bootstrap 有放回采样
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_sub, y_sub = X[idx], y[idx]

            tree = DecisionStump()
            tree.fit(X_sub, y_sub)
            self.trees.append(tree)

    def predict(self, X):
        # 每棵树投票
        votes = np.array([tree.predict(X) for tree in self.trees])
        # 多数票决定结果
        return np.round(votes.mean(axis=0)).astype(int)


# ========== 3. 测试 ==========
np.random.seed(42)

# 造一个简单数据集：x1+x2>1 则为类别1
X = np.random.rand(100, 2)
y = ((X[:, 0] + X[:, 1]) > 1).astype(int)

# 训练
rf = SimpleRandomForest(n_trees=20)
rf.fit(X, y)

# 预测
y_pred = rf.predict(X)
accuracy = np.mean(y_pred == y)
print(f"准确率: {accuracy:.2f}")

# 测试新样本
test = np.array([[0.8, 0.8],   # 0.8+0.8=1.6 > 1 → 应预测1
                 [0.1, 0.2]])  # 0.1+0.2=0.3 < 1 → 应预测0
print(f"预测结果: {rf.predict(test)}")  # 期望: [1, 0]

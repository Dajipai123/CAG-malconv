
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from skopt import BayesSearchCV

# 定义超参数搜索空间
param_space = {
    'n_estimators': (10, 100),
    'max_depth': (1, 10),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10),
}

# 加载数据
X = np.load('data.npy')
y = np.load('labels.npy')

# 定义模型
model = RandomForestClassifier()

# 使用贝叶斯优化进行超参数搜索
opt = BayesSearchCV(model, param_space, n_iter=50, cv=5)
opt.fit(X, y)

# 输出最佳超参数和对应的交叉验证得分
print("Best parameters found: ", opt.best_params_)
print("Best cross-validation score: ", opt.best_score_)


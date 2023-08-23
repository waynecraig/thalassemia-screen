import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy import interp

# 生成随机的真实标签和预测概率
np.random.seed(0)
y_true = np.random.randint(2, size=100)
y_scores = np.random.rand(100)

# 计算FPR、TPR和阈值
fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# 计算AUC
roc_auc = auc(fpr, tpr)

# 计算95%置信区间
tprs = []
aucs = []
n_bootstraps = 1000
rng = np.random.RandomState(0)
for i in range(n_bootstraps):
    indices = rng.randint(0, len(y_scores), len(y_scores))
    fpr_bootstrap, tpr_bootstrap, _ = roc_curve(y_true[indices], y_scores[indices])
    tprs.append(interp(fpr, fpr_bootstrap, tpr_bootstrap))
    tprs[-1][0] = 0.0
    roc_auc_bootstrap = auc(fpr_bootstrap, tpr_bootstrap)
    aucs.append(roc_auc_bootstrap)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

# 绘制ROC曲线
plt.figure()
plt.plot(fpr, tpr, color='b', label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.fill_between(fpr, mean_tprs - 1.96 * std, mean_tprs + 1.96 * std, color='grey', alpha=0.3, label='95% CI')
plt.plot([0, 1], [0, 1], color='r', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")

plt.show()

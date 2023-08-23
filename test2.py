import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc

# 生成一组随机数据作为示例
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# 将数据集划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化三个分类器
rf_classifier = RandomForestClassifier(random_state=42)
lr_classifier = LogisticRegression(random_state=42)
svm_classifier = SVC(probability=True, random_state=42)

# 训练三个分类器
rf_classifier.fit(X_train, y_train)
lr_classifier.fit(X_train, y_train)
svm_classifier.fit(X_train, y_train)

# 分别预测测试集的概率
rf_probs = rf_classifier.predict_proba(X_test)[:, 1]
lr_probs = lr_classifier.predict_proba(X_test)[:, 1]
svm_probs = svm_classifier.predict_proba(X_test)[:, 1]

# 计算各自的FPR和TPR
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
svm_fpr, svm_tpr, _ = roc_curve(y_test, svm_probs)

# 计算AUC值
rf_auc = auc(rf_fpr, rf_tpr)
lr_auc = auc(lr_fpr, lr_tpr)
svm_auc = auc(svm_fpr, svm_tpr)

# 初始化Voting集成模型
voting_classifier = VotingClassifier(estimators=[('rf', rf_classifier), ('lr', lr_classifier), ('svm', svm_classifier)], voting='soft')

# 训练Voting集成模型
voting_classifier.fit(X_train, y_train)

# 预测测试集的概率
voting_probs = voting_classifier.predict_proba(X_test)[:, 1]

# 计算Voting集成模型的FPR和TPR
voting_fpr, voting_tpr, _ = roc_curve(y_test, voting_probs)
voting_auc = auc(voting_fpr, voting_tpr)

# 绘制ROC曲线
plt.plot(rf_fpr, rf_tpr, label='Random Forest (AUC = %0.2f)' % rf_auc)
plt.plot(lr_fpr, lr_tpr, label='Logistic Regression (AUC = %0.2f)' % lr_auc)
plt.plot(svm_fpr, svm_tpr, label='SVM (AUC = %0.2f)' % svm_auc)
plt.plot(voting_fpr, voting_tpr, label='Voting (AUC = %0.2f)' % voting_auc)

# 绘制对角线
plt.plot([0, 1], [0, 1], 'k--')

# 设置图形属性
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")

# 显示图形
plt.show()

# [“魔镜杯”风控算法大赛](https://www.kesci.com/apps/home/#!/competition/56cd5f02b89b5bd026cb39c9/content/0)

> 拍拍贷“魔镜风控系统”从平均400个数据维度评估用户当前的信用状态，给每个借款人打出当前状态的信用分，在此基础上，再结合新发标的信息，打出对于每个标的6个月内逾期率的预测，为投资人提供了关键的决策依据，促进健康高效的互联网金融。拍拍贷首次开放丰富而真实的历史数据，邀你PK“魔镜风控系统”，通过机器学习技术，你能设计出更具预测准确率和计算性能的违约预测算法吗？

**我的成绩**：在第一阶段数据集（没有使用第二阶段数据集）得到auc(官方确定衡量标准)：**0.794587**，接近[比赛冠军分数](https://www.kesci.com/apps/home/#!/competition/56cd5f02b89b5bd026cb39c9/leaderboard/1)，因为比赛已经结束无法提交，所以这个结果不具有严格可对比性，不过很大程度上也已经很接近了。

# 一、思路

## 1.1 数据清洗

- 删除数据缺失比例很大的列，比如超过20%为nan
- 删除数据缺失比例大的行，并保持删除的行数不超过总体的1%
- 填补剩余缺失值，通过value_count观察是连续/离散变量，然后用最高频/平均数填补nan。这里通过观察，而不是判断类型是否object，更贴近实际情况

## 1.2 feature分类

- 所有的分类中，如果其中最大频率的值出现超过一定阈值（50%）,则把这列转化成为2值。比如[0,1,2,0,0,0,4,0,3]转化为[0,1,1,0,0,0,1,0,1]
- 剩余的feature中，根据dtype，把所有features分为numerical和categorical 2类
- numerical中，如果unique num不超过10个，也归属为categorical分类

## 1.3 outlier删除

- 所有的numerical feature，画出在不同target下的分布图，stripplot(with jitter)，类似于boxplot，不过更方便于大值outlier寻找。

```
melt = pd.melt(train_master, id_vars=['target'], value_vars = [f for f in numerical_features])
g = sns.FacetGrid(data=melt, col="variable", col_wrap=4, sharex=False, sharey=False)
g.map(sns.stripplot, 'target', 'value', jitter=True, palette="muted")
```

- 绘制所有numerical features的密度图，并且可以观察出，它们都可以通过求对数转化为更接近正态分布

```
for f in numerical_features_log:
    train_master[f + '_log'] = np.log1p(train_master[f])
```

- 转化为log分布后，可以再删除一些极小的outlier。

## 1.4 Feature Engineering

### other 2 datasets

train_loginfo：对Idx做group，提取记录数，LogInfo1独立数，活跃日期数，日期跨度

train_userinfo：对于Idx做group，提取记录数，UserupdateInfo1独立数、UserupdateInfo1/UserupdateInfo2独立数，日期跨度。以及每种UserupdateInfo1/UserupdateInfo2的数量。

### 解析日期

用`arrow` lib，把日期解析成年、月、日、周、星期几、月初/月中/月末。带入模型前进行one-hot encoding

### 新feature

- at_home，猜测UserInfo_2和UserInfo_8可能表示用户的当前居住地和户籍地，从而判断用户是否在老家。

## 1.5 训练前准备

### 指定one-hot encoding features

这里不要自动推算get_dummies所使用的列，pandas会自动选择object类型，而有些非object feature，实际含义也是categorical的，也需要被one-hot encoding

```
train_master_ = pd.get_dummies(train_master_, columns=finally_dummy_columns)
```

### normalized

```
X_train = StandardScaler().fit_transform(X_train)
```

## 1.6 训练评估

### Cross Validation

使用StratifiedKFold保证预测target的分布合理，并且shuffle随机。

```
cv = StratifiedKFold(n_splits=3, shuffle=True)
```

### AUC评估

```
auc = cross_val_score(estimator, X_train, y_train, scoring='roc_auc', cv=cv).mean()
```

### 模型算法

- XGBClassifier
- RidgeClassifier
- LogisticRegression
- AdaBoostClassifier
- VotingClassifier组合上面4种，做Ensembling

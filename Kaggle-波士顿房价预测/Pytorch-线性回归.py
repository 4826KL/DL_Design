#!/usr/bin/env python
# coding: utf-8

# ## 导入必要的运行库

# In[ ]:


import numpy as np
import pandas as pd
import torch
from torch import nn
from d2l import torch as d2l
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import norm, skew


# In[12]:


train_data = pd.read_csv("./data/train.csv")  # [1460 rows x 81 columns]
test_data = pd.read_csv("./data/test.csv")  # [1459 rows x 80 columns]


# In[13]:


train_data.head()   #查看数据集信息


# In[15]:


train_data['SalePrice'].describe()


# In[16]:


sns.distplot(train_data['SalePrice'])


# In[17]:


print("Skewness: %f" % train_data['SalePrice'].skew())
print("Kurtosis: %f" % train_data['SalePrice'].kurt())


# In[18]:


train_data['SalePrice'] = np.log1p(train_data['SalePrice'])
sns.distplot(train_data['SalePrice'], fit=norm)


# In[21]:


sns.boxplot(data=train_data, x='OverallQual', y='SalePrice')


# In[22]:


plt.figure(figsize=(12,4),dpi=130)
sns.boxplot(data=train_data, x='YearBuilt', y='SalePrice')
plt.xticks(rotation=90)
plt.show()


# In[24]:


sns.scatterplot(data=train_data,x='TotalBsmtSF',y='SalePrice')


# In[25]:


sns.scatterplot(data=train_data, x='GrLivArea', y='SalePrice')


# In[26]:


print(train_data.corr()["SalePrice"].sort_values(ascending=False)[1:])


# In[27]:


corrmat = train_data.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# In[28]:


corr = train_data.corr()
highest_corr_features = corr.index[abs(corr["SalePrice"])>0.5]
plt.figure(figsize=(10, 10))
g = sns.heatmap(train_data[highest_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()


# ‘OverQual’ in the top of highest correlation it’s 0.79!
# 
# ‘GarageCars’ & ‘GarageArea’ like each other (correlation between them is 0.88)
# 
# ‘TotalBsmtSF’ & ‘1stFlrSF’ also like each other (correlation betwwen them is 0.82), so we can keep either one of them or add the1stFlrSF to the Toltal.
# 
# ‘TotRmsAbvGrd’ & ‘GrLivArea’ also has a strong correlation (0.83), I decided to keep ‘GrLivArea’ because it’s correlation with ‘SalePrice’ is higher.
# 

# In[29]:


cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
sns.pairplot(train_data[cols])
plt.show()


# In[30]:


y_train = train_data['SalePrice']
test_id = test_data['Id']
all_data = pd.concat([train_data, test_data], axis=0, sort=False)
all_data = all_data.drop(['Id', 'SalePrice'], axis=1)


# In[31]:


Total = all_data.isnull().sum().sort_values(ascending=False)
percent = (all_data.isnull().sum() / all_data.isnull().count() * 100).sort_values(ascending=False)
missing_data = pd.concat([Total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_data.head(25))


# In[32]:


all_data.drop((missing_data[missing_data['Total'] > 5]).index, axis=1, inplace=True)
print(all_data.isnull().sum().max())


# In[33]:


total = all_data.isnull().sum().sort_values(ascending=False)
total.head(19)


# In[38]:


numeric_missed = ['BsmtFinSF1',
                  'BsmtFinSF2',
                  'BsmtUnfSF',
                  'TotalBsmtSF',
                  'BsmtFullBath',
                  'BsmtHalfBath',
                  'GarageArea',
                  'GarageCars']

for feature in numeric_missed:
    all_data[feature] = all_data[feature].fillna(0)


# In[39]:


categorical_missed = ['Exterior1st',
                      'Exterior2nd',
                      'SaleType',
                      'MSZoning',
                      'Electrical',
                      'KitchenQual']

for feature in categorical_missed:
    all_data[feature] = all_data[feature].fillna(all_data[feature].mode()[0])


# In[40]:


# Fill in the remaining missing values with the values that are most common for this feature.
all_data['Functional'] = all_data['Functional'].fillna('Typ')


# In[41]:


print(all_data.isnull().sum().max())


# In[42]:


numeric_feats = all_data.dtypes[all_data.dtypes != 'object'].index
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
high_skew = skewed_feats[abs(skewed_feats) > 0.5]
print(high_skew)


# In[43]:


for feature in high_skew.index:
    all_data[feature] = np.log1p(all_data[feature])


# In[44]:


all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']


# In[45]:


all_data = pd.get_dummies(all_data)
all_data.head()


# In[46]:


x_train = all_data[:len(y_train)]  # (1460, 219)
x_test = all_data[len(y_train):]  # (1459, 219)


# In[47]:


train_features = torch.tensor(x_train.values, dtype=torch.float32)  # torch.Size([1460, 219])
test_features = torch.tensor(x_test.values, dtype=torch.float32)  # torch.Size([1459, 219])

train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)  # torch.Size([1460, 1])


# In[48]:


loss = nn.MSELoss()
in_features = train_features.shape[1]  

def get_net():
    net = nn.Sequential(nn.Linear(in_features, 1))
    # 模型参数初始化
    for param in net.parameters():
        nn.init.normal_(param, mean=0, std=0.01)
    return net


# In[49]:


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = d2l.load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_ls.append(loss(net(train_features), train_labels).item())
        if test_labels is not None:
            test_ls.append(loss(net(test_features), test_labels).item())
    return train_ls, test_ls


# In[50]:


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


# In[51]:


def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
           batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
                     xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
                     legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
              f'验证log rmse{float(valid_ls[-1]):f}')
    return train_l_sum / k, valid_l_sum / k


# In[52]:


k, num_epochs, lr, weight_decay, batch_size = 5, 200, 0.0002, 0, 64
train_l, valid_l = k_fold(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')
d2l.plt.show()


# In[55]:


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.plot(np.arange(1, num_epochs + 1), [train_ls], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs], yscale='log')
    print(f'训练log rmse：{float(train_ls[-1]):f}')
    # 将网络应用于测试集。
    preds = net(test_features).detach().numpy()
    preds = np.exp(preds) - 1
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


# In[56]:


train_and_pred(train_features, test_features, train_labels, test_data,
               num_epochs, lr, weight_decay, batch_size)
d2l.plt.show()


# https://github.com/Ming7959/Kaggle

# In[ ]:





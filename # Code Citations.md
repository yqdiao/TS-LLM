# Code Citations

## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: unknown
https://github.com/RRisto/learning/blob/a3ea5b2772c63742956e9f35440a5f8ea8e98fd6/nn_learn/transformers/transformers_from_scratch.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: unknown
https://github.com/RRisto/learning/blob/a3ea5b2772c63742956e9f35440a5f8ea8e98fd6/nn_learn/transformers/transformers_from_scratch.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: unknown
https://github.com/RRisto/learning/blob/a3ea5b2772c63742956e9f35440a5f8ea8e98fd6/nn_learn/transformers/transformers_from_scratch.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: unknown
https://github.com/RRisto/learning/blob/a3ea5b2772c63742956e9f35440a5f8ea8e98fd6/nn_learn/transformers/transformers_from_scratch.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin
```


## License: unknown
https://github.com/RRisto/learning/blob/a3ea5b2772c63742956e9f35440a5f8ea8e98fd6/nn_learn/transformers/transformers_from_scratch.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
```


## License: unknown
https://github.com/RRisto/learning/blob/a3ea5b2772c63742956e9f35440a5f8ea8e98fd6/nn_learn/transformers/transformers_from_scratch.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self
```


## License: unknown
https://github.com/RRisto/learning/blob/a3ea5b2772c63742956e9f35440a5f8ea8e98fd6/nn_learn/transformers/transformers_from_scratch.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self
```


## License: unknown
https://github.com/RRisto/learning/blob/a3ea5b2772c63742956e9f35440a5f8ea8e98fd6/nn_learn/transformers/transformers_from_scratch.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.
```


## License: unknown
https://github.com/RRisto/learning/blob/a3ea5b2772c63742956e9f35440a5f8ea8e98fd6/nn_learn/transformers/transformers_from_scratch.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.
```


## License: MIT
https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/4c48a3304932d58c5349434e7b0085f48dcb4be4/captioning/models/cachedTransformer.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        encoder_layer = nn.Transformer
```


## License: unknown
https://github.com/RRisto/learning/blob/a3ea5b2772c63742956e9f35440a5f8ea8e98fd6/nn_learn/transformers/transformers_from_scratch.py

```
# 分钟级股票交易数据：小模型 vs 时间序列大模型 对比方案

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示"大模型过度依赖"的资源浪费问题。
2. **预测任务**：基于历史特征预测未来30分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

---

## 二、数据描述与预处理

### 2.1 原始特征（5个）

| 特征 | 含义 | 说明 |
|------|------|------|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征工程

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/feature_engineering.py
import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    基于原始5个特征构建衍生特征
    df 需包含: trade_volume, best_ask, ask_size, best_bid, bid_size
    """
    # ---- 1. 价差与中间价 ----
    df['mid_price'] = (df['best_ask'] + df['best_bid']) / 2
    df['spread'] = df['best_ask'] - df['best_bid']
    df['spread_ratio'] = df['spread'] / df['mid_price']
    
    # ---- 2. 订单不平衡 (Order Imbalance) ----
    df['order_imbalance'] = (df['bid_size'] - df['ask_size']) / (df['bid_size'] + df['ask_size'] + 1e-8)
    
    # ---- 3. 成交量加权价格压力 ----
    df['volume_price_pressure'] = df['trade_volume'] * df['order_imbalance']
    
    # ---- 4. 滚动统计特征 (多时间窗口) ----
    for window in [5, 10, 30, 60]:
        df[f'mid_price_return_{window}'] = df['mid_price'].pct_change(window)
        df[f'volume_ma_{window}'] = df['trade_volume'].rolling(window).mean()
        df[f'volume_std_{window}'] = df['trade_volume'].rolling(window).std()
        df[f'spread_ma_{window}'] = df['spread'].rolling(window).mean()
        df[f'order_imbalance_ma_{window}'] = df['order_imbalance'].rolling(window).mean()
        df[f'mid_price_volatility_{window}'] = df['mid_price'].rolling(window).std()
    
    # ---- 5. 动量特征 ----
    df['momentum_5'] = df['mid_price'] / df['mid_price'].shift(5) - 1
    df['momentum_30'] = df['mid_price'] / df['mid_price'].shift(30) - 1
    
    # ---- 6. VWAP 近似 ----
    df['vwap_proxy'] = (df['trade_volume'] * df['mid_price']).rolling(30).sum() / \
                        (df['trade_volume'].rolling(30).sum() + 1e-8)
    df['price_vwap_deviation'] = (df['mid_price'] - df['vwap_proxy']) / df['vwap_proxy']
    
    return df
````

### 2.3 标签定义（预测目标）

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/label_generation.py
import numpy as np
import pandas as pd

def generate_labels(df: pd.DataFrame, horizon: int = 30, threshold: float = 0.001) -> pd.DataFrame:
    """
    生成未来30分钟的交易信号标签
    
    方式一: 三分类信号 (UP / DOWN / NEUTRAL)
    方式二: 五分位区间 (Strong Down / Down / Neutral / Up / Strong Up)
    """
    # 未来30分钟的收益率
    df['future_return_30'] = df['mid_price'].shift(-horizon) / df['mid_price'] - 1
    
    # ---- 方式一: 三分类 ----
    df['signal_3class'] = np.where(
        df['future_return_30'] > threshold, 2,     # UP
        np.where(df['future_return_30'] < -threshold, 0, 1)  # DOWN / NEUTRAL
    )
    
    # ---- 方式二: 五分位区间分类 ----
    df['signal_5class'] = pd.qcut(
        df['future_return_30'].rank(method='first'), 
        q=5, labels=[0, 1, 2, 3, 4]
    ).astype(float)
    
    # ---- 方式三: 回归目标 (直接预测收益率) ----
    # 直接使用 future_return_30
    
    return df
````

---

## 三、小模型方案（Baseline & Lightweight Models）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|------|--------|------|----------|----------|
| **LightGBM** | ~50K | 树模型 | 秒级 | 表格特征强 |
| **XGBoost** | ~50K | 树模型 | 秒级 | 同上 |
| **1D-CNN** | ~10K-50K | 深度学习 | 分钟级 | 局部模式 |
| **LSTM (2层)** | ~50K-200K | 深度学习 | 分钟级 | 序列依赖 |
| **TCN (轻量)** | ~30K-100K | 深度学习 | 分钟级 | 长距离依赖 |
| **Transformer-Tiny** | ~100K-500K | 深度学习 | 分钟级 | 注意力机制 |
| **Linear / Ridge** | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 核心小模型实现

#### 3.2.1 LightGBM (最强表格基线)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lightgbm_model.py
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def train_lightgbm(X_train, y_train, X_val, y_val, num_class=3):
    """
    LightGBM 分类模型
    参数量约 50K 左右
    """
    params = {
        'objective': 'multiclass',
        'num_class': num_class,
        'metric': 'multi_logloss',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 8,
        'min_child_samples': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42
    }
    
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100)
        ]
    )
    
    y_pred = model.predict(X_val).argmax(axis=1)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"LightGBM - Accuracy: {acc:.4f}, F1: {f1:.4f}")
    print(f"模型参数量估算: ~{model.num_trees() * params['num_leaves'] * 2} 参数")
    
    return model
````

#### 3.2.2 轻量 LSTM

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/lstm_model.py
import torch
import torch.nn as nn

class LightLSTM(nn.Module):
    """
    轻量LSTM: 约 50K-150K 参数
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, 
                 num_classes=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
        # 打印参数量
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightLSTM 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步
        out = lstm_out[:, -1, :]
        out = self.fc(out)
        return out
````

#### 3.2.3 轻量 Temporal Convolutional Network (TCN)

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tcn_model.py
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=self.padding)
    
    def forward(self, x):
        out = self.conv(x)
        return out[:, :, :x.size(2)]  # 截断未来信息

class LightTCN(nn.Module):
    """
    轻量TCN: 约 30K-100K 参数
    """
    def __init__(self, input_dim, num_channels=[32, 32, 32], 
                 kernel_size=3, num_classes=3, dropout=0.2):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            in_ch = input_dim if i == 0 else num_channels[i-1]
            layers.append(CausalConv1d(in_ch, out_ch, kernel_size, dilation=2**i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], num_classes)
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"LightTCN 总参数量: {total_params:,}")
    
    def forward(self, x):
        # x: (batch, seq_len, input_dim) -> (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)
        out = out[:, :, -1]  # 取最后时间步
        return self.fc(out)
````

#### 3.2.4 Tiny Transformer

````python
# filepath: /Users/yuiqin/Desktop/股票dataset/models/tiny_transformer.py
import torch
import torch.nn as nn
import math

class TinyTransformer(nn.Module):
    """
    微型Transformer: 约 100K-500K 参数
    对照大模型的 attention 机制
    """
    def __init__(self, input_dim, d_model=64, nhead=4, 
                 num_layers=2, num_classes=3, dropout=0.1, max_len=120):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # 位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        encoder_layer = nn.Transformer
```


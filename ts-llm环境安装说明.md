# ts-llm 环境安装说明

本文档汇总「时序大模型 + DeepLOB + 轻量模型」在 **ts-llm** 虚拟环境中的安装步骤与常见报错修正。环境路径为项目目录下的 `ts-llm`（Python 建议 3.11，以便安装 TimesFM）。

---

## 一、环境准备

### 1.1 目录与路径

- 项目目录：`/Users/yuiqin/Desktop/股票dataset`（注意是 **dataset**，末尾有字母 **t**）。
- 虚拟环境：`ts-llm` 位于上述目录内，即 `股票dataset/ts-llm/`。

### 1.2 创建并激活环境（若尚未创建）

```bash
cd /Users/yuiqin/Desktop/股票dataset

# 使用 Python 3.11 创建环境（TimesFM 需要 3.10–3.11）
python3.11 -m venv ts-llm

# 激活环境（zsh）
source ts-llm/bin/activate
```

若当前在 **Desktop** 或其他目录，需先 `cd` 到 `股票dataset`，或使用绝对路径激活：

```bash
source /Users/yuiqin/Desktop/股票dataset/ts-llm/bin/activate
```

### 1.3 升级 pip

```bash
pip install --upgrade pip
```

---

## 二、按顺序安装依赖

以下命令均在 **已激活 ts-llm** 且当前目录为 `股票dataset` 的前提下执行。**不要在同一行 pip install 后写注释**（如 `# 说明`），否则会报 `Invalid requirement: '#'`。

### 2.1 Chronos

```bash
pip install chronos-forecasting
```

验证：`python -c "from chronos import Chronos2Pipeline; print('Chronos OK')"`

### 2.2 TimesFM（可选，需 Python 3.10/3.11）

```bash
pip install timesfm
```

若报错 `No matching distribution found for jaxlib==0.4.26`，说明当前 Python 或平台不受支持（如 3.13 或部分 arm64），可跳过 TimesFM，改用 Chronos/Lag-Llama 做零样本对比。

### 2.3 PatchTST

```bash
cd /Users/yuiqin/Desktop/股票dataset
git clone https://github.com/yuqinie98/PatchTST.git
cd PatchTST/PatchTST_supervised
pip install -r requirements.txt
cd ../..
```

说明：`requirements.txt` 在子目录 **PatchTST_supervised** 内，不在仓库根目录。若报错 `torch==1.11.0` 找不到，已改为 `torch>=2.0.0`，直接重新执行 `pip install -r requirements.txt` 即可。

### 2.4 Informer2020

```bash
cd /Users/yuiqin/Desktop/股票dataset
git clone https://github.com/zhouhaoyi/Informer2020.git
cd Informer2020
pip install -r requirements.txt
cd ..
```

说明：原版 requirements 为 2019 年版本，在 Python 3.11 下会构建失败；已改为兼容 3.11 的宽松版本（如 `numpy>=1.21.0`、`torch>=2.0.0` 等），直接安装即可。

### 2.5 Lag-Llama

```bash
cd /Users/yuiqin/Desktop/股票dataset
git clone https://github.com/time-series-foundation-models/lag-llama.git
cd lag-llama
pip install -r requirements.txt
cd ..
```

### 2.6 DeepLOB（二选一）

**方式一：论文作者官方仓库（FI-2010 + Jupyter）**

```bash
cd /Users/yuiqin/Desktop/股票dataset
git clone https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books.git
```

**方式二：S-razmi 实现（BTC 永续 LOB）**

```bash
cd /Users/yuiqin/Desktop/股票dataset
git clone https://github.com/S-razmi/DeepLOB.git
```

注意：仓库名是 **Deep-Convolutional-Neural-Networks-for-Limit-Order-Books**，不是 `Deep-Learning-for-Limit-Order-Books`（后者不存在，会报 Repository not found）。

### 2.7 PyTorch（DeepLOB 等若需单独装 torch）

只执行下面一行，**不要在同一行后面加任何注释**：

```bash
pip install torch torchvision torchaudio
```

### 2.8 轻量模型依赖

```bash
pip install lightgbm xgboost
```

---

## 三、常见报错与修正

| 报错或现象 | 原因 | 修正 |
|------------|------|------|
| `cd 股票datase` → no such file or directory | 目录名少写字母 **t** | 使用 `股票dataset` |
| `source ts-llm/bin/activate` → no such file | 当前目录不是 股票dataset，ts-llm 在其子目录下 | 先 `cd /Users/yuiqin/Desktop/股票dataset`，再 `source ts-llm/bin/activate`；或使用绝对路径 `source .../股票dataset/ts-llm/bin/activate` |
| `Invalid requirement: '#'` | 把 `pip install ... # 注释` 整行复制，注释被当成包名 | 只执行 `pip install 包名`，注释另起一行或删除 |
| TimesFM: No matching distribution for jaxlib==0.4.26 | TimesFM 仅支持 Python 3.10–3.11，且依赖旧版 JAX | 使用 Python 3.11 创建 ts-llm，或跳过 TimesFM |
| PatchTST: No such file or directory: requirements.txt | requirements 在子目录 | 执行 `cd PatchTST/PatchTST_supervised` 后再 `pip install -r requirements.txt` |
| PatchTST: torch==1.11.0 not found | 旧版 torch 无 Python 3.11 / arm64 wheel | 已改为 `torch>=2.0.0`，重新安装即可 |
| Informer: Failed to build pandas / pkg_resources | 原 requirements 过旧，从源码构建失败 | 已放宽版本，重新 `pip install -r requirements.txt` |
| DeepLOB: Repository not found (zcakhaa/...Deep-Learning-for...) | 仓库名写错 | 使用 **Deep-Convolutional-Neural-Networks-for-Limit-Order-Books** 或 S-razmi/DeepLOB |

---

## 四、验证环境

激活 ts-llm 后，可逐条验证（按你已安装的项）：

```bash
python -c "from chronos import Chronos2Pipeline; print('Chronos OK')"
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import lightgbm; print('LightGBM OK')"
python -c "import xgboost; print('XGBoost OK')"
```

---

## 五、文档与代码引用

- 七个时序模型角色与下一步：见 `时序大模型七个开源库说明与下一步.md`。
- 方案提纲与引用：见 `Code_Citations_CN.md`。

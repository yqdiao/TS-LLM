# TS-LLM

时序大模型与分钟级股票预测：环境说明、七个开源库概览与「小模型 vs 大模型」对比方案。

---

## 一、项目概述

本仓库围绕**分钟级股票预测**，对比**轻量模型**与**时序大模型/专用模型**，论证在該场景下是否值得依赖大模型。

### 模型角色划分

| 类型 | 模型 | 说明 |
|------|------|------|
| **小模型（基线）** | LightGBM / XGBoost、小 LSTM/TCN、AR | 证明「小就够用」 |
| **预训练零样本** | TimesFM、Chronos、Lag-Llama | 现成 TS 基础模型，零样本/少样本即可跑 |
| **需自训练的 Transformer** | PatchTST、Informer | 在本数据上训练，中等规模时序模型 |
| **订单簿专用（可选）** | DeepLOB | 订单簿 CNN/TCN，领域专用基线 |

### 七个库简要

- **TimesFM**（Google）：decoder-only 时序基础模型，零样本多变量/长 context，适合分钟级 5 元特征 → 未来 30 分钟预测。  
- **Chronos**（Amazon）：预训练模型族，零样本；与 TimesFM 并列作为「零样本时序大模型」对比。  
- **Lag-Llama**：概率时序基础模型，零样本或微调，可调 context length 适配分钟级。  
- **PatchTST**（ICLR 2023）：patch 化 Transformer，多变量、长序列；需自备数据并训练。  
- **Informer**（AAAI 2021）：长序列 ProbSparse 注意力，支持分钟频；需在本数据上训练。  
- **DeepLOB**：订单簿专用 CNN/TCN（FI-2010 或 BTC LOB），可选用于 LOB 场景对比。  
- **Amazon Forecast Samples**：云服务示例，非本地大模型代码，方案中可暂不采用。

### 建议的下一步（五步）

1. **数据与标签**：确认 5 元特征列，PySpark 分钟聚合 + 未来 30 分钟收益/方向标签，输出分钟级表并严格按时间划分（如训练=1 月，测试=2 月第一周）。  
2. **小模型基线**：LightGBM/XGBoost、小 LSTM/AR 等，记录精度与训练/推理时间、内存。  
3. **TimesFM 零样本**：用同一分钟级数据与标签，对齐预测与指标，记录推理资源。  
4. **PatchTST 或 Informer**：在 1 月数据上训练，2 月测试，同一指标与资源对比。  
5. **对比与结论**：做对比表（精度、训练时间、推理延迟、显存），判断小模型是否足以媲美大模型。

详细步骤与产出见：`时序大模型七个开源库说明与下一步.md`。

---

## 二、环境安装（摘自《ts-llm 环境安装说明》）

在项目目录下使用 **ts-llm** 虚拟环境（建议 **Python 3.11**，以便安装 TimesFM）。

### 创建并激活环境

```bash
cd /path/to/股票dataset
python3.11 -m venv ts-llm
source ts-llm/bin/activate   # zsh
pip install --upgrade pip
```

### 按顺序安装

| 顺序 | 组件 | 命令 / 说明 |
|------|------|--------------|
| 1 | Chronos | `pip install chronos-forecasting` |
| 2 | TimesFM（可选） | `pip install timesfm`（仅 3.10/3.11；若 jaxlib 报错可跳过） |
| 3 | PatchTST | 进入 `PatchTST/PatchTST_supervised` 后 `pip install -r requirements.txt`（torch 已放宽为 ≥2.0） |
| 4 | Informer2020 | 进入 `Informer2020` 后 `pip install -r requirements.txt` |
| 5 | Lag-Llama | 进入 `lag-llama` 后 `pip install -r requirements.txt` |
| 6 | DeepLOB | 二选一：`zcakhaa/Deep-Convolutional-Neural-Networks-for-Limit-Order-Books` 或 `S-razmi/DeepLOB` |
| 7 | PyTorch | `pip install torch torchvision torchaudio`（不要在同一行加注释） |
| 8 | 轻量模型 | `pip install lightgbm xgboost` |

### 常见问题

- 目录名是 **股票dataset**（末尾有 **t**）。  
- `requirements.txt` 在 **PatchTST_supervised** 子目录内。  
- DeepLOB 官方仓库名是 **Deep-Convolutional-Neural-Networks-for-Limit-Order-Books**。  
- 执行 `pip install` 时不要在同一行写 `# 注释`，否则会报 `Invalid requirement: '#'`。

完整安装步骤与报错修正见：`ts-llm环境安装说明.md`。

---

## 三、文档与仓库结构

| 文档 | 内容 |
|------|------|
| **时序大模型七个开源库说明与下一步.md** | 七个库定位、与分钟级股票的契合度、五步行动与对照表 |
| **ts-llm环境安装说明.md** | 环境路径、按顺序安装、常见报错与修正、验证命令 |
| **Code_Citations_CN.md** | 方案提纲与代码引用 |

仓库中还包含：**PatchTST**、**Informer2020**、**lag-llama**、**DeepLOB** 等子目录，**LASSO操作流程**、**PySpark 高频数据与滞后收益预测** 的 notebook 与样本数据（大体积数据目录 `merged_300_common_stocks/` 未纳入 Git，见 `.gitignore`）。

---

## 四、快速验证

激活 ts-llm 后可按需验证：

```bash
python -c "from chronos import Chronos2Pipeline; print('Chronos OK')"
python -c "import torch; print('PyTorch', torch.__version__)"
python -c "import lightgbm; print('LightGBM OK')"
python -c "import xgboost; print('XGBoost OK')"
```

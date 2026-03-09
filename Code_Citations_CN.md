# Code Citations
## 代码引用与方案提纲

**作者**：分钟级股票：小模型 vs 时间序列大模型

---

## 摘要

本文档为「分钟级股票交易数据：小模型 vs 时间序列大模型」对比方案的**引用与提纲**，不含完整实现代码；具体实现见项目内脚本与 Notebook。

---

## 一、研究目标

1. **证明命题**：在分钟级股票交易信号预测任务中，精心设计的小模型可以达到甚至超越时间序列大模型的表现，揭示「大模型过度依赖」的资源浪费问题。
2. **预测任务**：基于历史特征预测未来 30 分钟的**交易信号区间**（涨/跌/震荡 或 具体价格变动区间）。

## 二、数据描述与预处理

### 2.1 原始特征（5 个）

| 特征 | 含义 | 说明 |
|---|---|---|
| `trade_volume` | 成交量 | 当前分钟的交易量 |
| `best_ask` | 最优卖价 | 卖一价 |
| `ask_size` | 最优卖价挂单量 | 卖一量 |
| `best_bid` | 最优买价 | 买一价 |
| `bid_size` | 最优买价挂单量 | 买一量 |

### 2.2 衍生特征（概要）

- 价差与中间价：`mid_price`, `spread`, `spread_ratio`
- 订单不平衡：`order_imbalance`；成交量加权压力：`volume_price_pressure`
- 滚动统计：多窗口（5/10/30/60）的 return、volume_ma/std、spread_ma、波动率等
- 动量、VWAP 近似与价格偏离

*实现参考：项目内特征工程脚本（如 `feature_engineering.py` 或 PySpark 笔记本）。*

### 2.3 标签定义（预测目标）

- 未来 30 分钟收益率：`future_return_30 = mid_price.shift(-30) / mid_price - 1`
- 三分类信号：UP / DOWN / NEUTRAL（按阈值）
- 可选：五分位区间分类或回归目标

*实现参考：项目内标签生成脚本（如 `label_generation.py`）或 PySpark 笔记本。*

## 三、小模型方案（概要）

### 3.1 模型选型矩阵

| 模型 | 参数量 | 类型 | 训练时间 | 适用场景 |
|---|---:|---|---|---|
| LightGBM | ~50K | 树模型 | 秒级 | 表格特征强 |
| XGBoost | ~50K | 树模型 | 秒级 | 同上 |
| 1D-CNN | ~10K–50K | 深度学习 | 分钟级 | 局部模式 |
| LSTM (2 层) | ~50K–200K | 深度学习 | 分钟级 | 序列依赖 |
| TCN (轻量) | ~30K–100K | 深度学习 | 分钟级 | 长距离依赖 |
| Linear / Ridge | <1K | 线性模型 | 即时 | 最简基线 |

### 3.2 实现说明

- **LightGBM**：多分类/回归，early stopping，记录准确率/F1 或 MSE/MAE 及参数量估算。
- **轻量 LSTM/TCN**：滑动窗口输入 `(seq_len, input_dim)`，输出分类或回归；控制 hidden/channels 以保持参数量在 10K–200K 量级。

*具体实现见项目内 `models/` 或 Notebook，此处不贴代码。*

## 四、时序大模型与相关开源引用（七选）

以下为对比实验中「时序大模型」的**官方开源网址与简要说明**，用于引用与复现；前四个为原方案主选，后三个为扩展可选（Chronos、Lag-Llama、DeepLOB）。

**重要说明**：上述大模型均为**已预训练 / 已训练好**的模型（pre-trained），在本方案中默认先以**未微调**形态参与对比——例如 TimesFM 直接零样本推理，PatchTST/Informer 在 1 月数据上做的是「从零训练」或「加载预训练权重后再训练」，而非在通用预训练基础上针对本数据集做**微调（fine-tuning）**。若后续加入微调实验，则形成「预训练零样本 vs 微调后 vs 小模型」的三方对比，便于进一步论证资源与收益是否匹配。

### 4.1 TimesFM（Google Research）

- **网址**：<https://github.com/google-research/timesfm>
- **说明**：预训练时间序列基础模型（decoder-only），约 200M 参数，context 最长 16k，horizon 最长 256。输入为 1D 序列列表，输出点预测与可选分位数。
- **与分钟级股票**：高。零样本/少样本即可；将过去 N 分钟的 5 元特征或 mid_price 做成序列，`model.forecast(horizon=30, inputs=[...])` 预测未来 30 分钟。
- **在方案中的角色**：代表「现成 TS-Foundation Model」，无需自训练。
- **参考**：ICML 2024；Hugging Face 集合：<https://huggingface.co/collections/google/timesfm-release-66e4be5fdb56e960c1e482a6>

### 4.2 Amazon Forecast Samples（AWS）

- **网址**：<https://github.com/aws-samples/amazon-forecast-samples/tree/master>
- **说明**：云服务示例，演示通过 Amazon Forecast API 做预测（非本地大模型代码）。仓库已归档。
- **与分钟级股票**：低。需 AWS、S3、Forecast；不利于固定评估协议与公平对比。
- **在方案中的角色**：可选；做「小模型 vs 时序大模型」公平对比时建议**暂不采用**。

### 4.3 PatchTST（ICLR 2023）

- **网址**：<https://github.com/yuqinie98/PatchTST>
- **说明**：时间序列切 patch 作 token，channel-independent Transformer，支持多变量与可调 look-back / prediction length。脚本在 `scripts/PatchTST/`。
- **与分钟级股票**：高。过去 30–60 分钟的 5 元特征作多 channel，预测未来 30 步；需在本地数据上训练。
- **在方案中的角色**：代表「可训练的时序 Transformer」，与 LightGBM、TimesFM 一起对比。

### 4.4 Informer2020（AAAI 2021）

- **网址**：<https://github.com/zhouhaoyi/Informer2020>
- **说明**：长序列时序 Transformer（ProbSparse 注意力），支持多变量与分钟频（`--freq t`）。`main_informer.py` 配置 `--seq_len`、`--pred_len` 等。
- **与分钟级股票**：高。`seq_len=60`、`pred_len=30` 做未来 30 分钟预测；需在本地数据上训练。
- **在方案中的角色**：同上，作为「可训练时序大模型」参与对比。

### 4.5 Chronos（Amazon Science）

- **网址**：<https://github.com/amazon-science/chronos-forecasting>
- **说明**：预训练时序预测模型族，含 Chronos-2（零样本支持单变量/多变量/带协变量）、Chronos-Bolt（patch 编码、更快更省显存）。提供 8M–710M 多种规模，Hugging Face 可下载；`pip install chronos-forecasting`，支持 `Chronos2Pipeline.from_pretrained` 与 `predict_df`。
- **与分钟级股票**：高。零样本即可；可将分钟级 5 元特征或 mid_price 作为 target/context，指定 `prediction_length=30` 得到未来 30 步预测；可选协变量（如 volume）。
- **在方案中的角色**：与 TimesFM 并列的「预训练零样本时序模型」，可一并对比推理效果与资源。
- **参考**：Chronos (TMLR 2024)，Chronos-2 (arXiv 2025)；<https://huggingface.co/amazon/chronos-2>

### 4.6 Lag-Llama（Time-Series Foundation Models）

- **网址**：<https://github.com/time-series-foundation-models/lag-llama>
- **说明**：面向概率时序预测的开源基础模型，基于 Llama 架构；支持**零样本预测**与**微调**，任意频率、任意预测长度。建议调节 context length（如 32/64/128/256/512/1024）与 RoPE scaling 以适配分钟级长度。
- **与分钟级股票**：高。零样本或在本项目 1 月数据上微调后，用历史分钟序列预测未来 30 步分布；官方提供 Colab 零样本与微调示例。
- **在方案中的角色**：预训练 + 可微调的时序基础模型，可与 TimesFM/Chronos 零样本及 PatchTST/Informer 一起参与「零样本 vs 微调 vs 小模型」对比。
- **参考**：Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting (arXiv 2023)。

### 4.7 DeepLOB（订单簿专用，可选）

- **网址**：<https://github.com/S-razmi/DeepLOB>
- **说明**：基于论文「DeepLOB: Deep Convolutional Neural Networks for Limit Order Books」的实现，面向**订单簿（LOB）**数据（bid/ask 价格与档位）；该仓库为 BTC 永续合约 LOB 的 1 秒频数据上的 CNN/TCN 实现（TensorFlow）。非大语言模型，而是**领域专用的深度学习基线**。
- **与分钟级股票**：高。你的 best_ask、best_bid、size 即为 LOB 核心字段；可将分钟级 LOB 特征构造为模型输入，预测未来价格方向或收益区间，作为「面向 LOB 的专用模型」与通用时序大模型、小模型对比。
- **在方案中的角色**：可选；代表「订单簿/高频场景下的专用深度学习模型」，用于补充对比或论证在 LOB 场景下轻量模型是否足以媲美专用结构。

### 4.8 时序模型在股票价格预测上的适用性

上述模型（含 TimesFM、PatchTST、Informer、Chronos、Lag-Llama、DeepLOB 等）中，通用时序模型均可用于股票价格时间序列预测，但**均非专为股票价格设计**，故属「可用但有前提」而非「完全、无条件适用」。TimesFM 为通用时序基础模型，在股票上属零样本迁移，效果取决于与预训练分布的接近程度。Amazon Forecast 为通用预测云服务，可做股价/收益预测但非股票专用且依赖 AWS。PatchTST、Informer 为通用长序列模型，论文实验多为电力、天气等；用于股价预测需在**本数据集上自行训练或微调**，并注意金融特性（非平稳、波动聚集、交易时段等）。预训练数据多为日/小时频，分钟级高频价量的分布与噪声与常见预训练数据不同，零样本或直接迁移未必最优。若在本实验中**小模型不逊于或优于**这些大模型，可说明在分钟级股票场景下通用时序大模型并非「完全适用」，盲目上大模型可能是资源浪费；若要提高在股票预测上的适用性，建议在自有股票数据上训练或微调（如 PatchTST/Informer），或选用面向金融/高频的时序模型。

## 五、大模型微调（扩展研究方向）

在「已预训练大模型」基础上，可增加**微调（fine-tuning）**部分，用于研究：微调能否显著提升分钟级股票预测效果、以及微调带来的额外计算与数据成本是否合理。

### 5.1 为何要做微调对比

- **预训练 vs 微调**：TimesFM、PatchTST 等在大规模通用时序数据上已训练好；在**本项目的分钟级股票数据**上微调，可检验领域适配是否带来明显增益。
- **与命题的关系**：若微调后大模型仍无法显著优于小模型，或提升幅度远小于其额外成本（数据量、GPU、时间），可进一步支撑「过度依赖大模型是资源浪费」的论点。

### 5.2 哪些模型适合做微调

| 模型 | 预训练形态 | 微调支持与方式 |
|---|---|---|
| **TimesFM** | 已预训练，零样本/少样本使用 | 官方主要提供推理 API；若需微调需查阅是否开放 adapter / 低秩微调或后续更新。可先记录「零样本 vs 少样本 prompt」对比。 |
| **PatchTST** | 提供自监督预训练脚本（如 `patchtst_pretrain.py`）+ 下游微调（`patchtst_finetune.py`） | **支持**：先在其他时序数据或 1 月子集上预训练，再在 1 月全量上 fine-tune；或直接加载官方/已有权重后在 1 月数据上微调。 |
| **Informer** | 通常从零在目标数据上训练，亦有预训练扩展工作 | 可在 ETT 等公开数据上预训练，再在本项目分钟级数据上微调；或仅记录「从零训练」作为基线，与「预训练+微调」对比。 |
| 通用 LLM（时间序列版） | 如 Time-LLM、LLM4TS 等将时序转文本/token 后用 LLM | 微调方式多为 LoRA/Adapter；需将分钟级序列转为模型输入格式，再在 1 月数据上微调。可作为**扩展对比**（非上述四库）。 |
| **Chronos** | 已预训练，零样本/多变量/协变量 | 官方以推理为主；若开放微调接口可做「零样本 vs 微调」对比。 |
| **Lag-Llama** | 已预训练，零样本 + 官方提供微调脚本与 Colab | **支持**：零样本或按官方 best practices 在 1 月数据上微调，context length 与学习率需调参。 |
| **DeepLOB** | 订单簿专用 CNN/TCN，需在 LOB 数据上训练 | 非预训练大模型；在本项目 LOB 风格数据上从零训练，作为专用基线对比。 |

### 5.3 微调实验建议记录项

- **数据**：微调所用样本量（如 1 月分钟数、是否做增强/采样）。
- **设置**：学习率、epoch 数、batch size、是否冻结部分层、LoRA/Adapter 秩（若用）。
- **资源**：微调耗时、显存峰值、与「零样本/从零训练」的差异。
- **指标**：微调前后在**同一测试集**（2 月第一周）上的准确率/MSE/MAE；若有可能，与「未微调预训练模型」对比。

### 5.4 对比维度扩展（加入微调后）

- **零样本预训练模型**（如 TimesFM 直接 forecast）
- **微调后大模型**（如 PatchTST 预训练 + 在本数据上 fine-tune）
- **小模型**（LightGBM、小 LSTM 等）

从「精度—成本」曲线可判断：微调带来的提升是否值得额外数据和算力；若小模型在微调后大模型附近甚至更优，则强化「不必过度依赖大模型」的结论。

## 六、下一步该进行做啥（按顺序）

| 步骤 | 内容 | 产出 |
|---|---|---|
| **第一步** | 在原始 CSV 中确认 trade volume、size_best_ask、size_best_bid 的列索引；在 PySpark 中按 (date, stock_code, minute) 聚合 5 元特征，沿用「完整分钟网格 + 前向填充」，构建衍生特征与「未来 30 分钟收益/方向」标签；训练=1 月，测试=2 月第一周。 | 分钟级表 + 训练/测试划分 |
| **第二步** | 用分钟级表训练 LightGBM/XGBoost、小 LSTM/TCN 或线性/AR；记录准确率/MSE/MAE/R² 及训练时间、推理延迟、内存。 | 小模型精度与资源指标 |
| **第三步** | 接入 TimesFM（或 Chronos、Lag-Llama）：安装并从 Hugging Face 加载模型；从分钟级表构造过去一段序列作为 inputs，`forecast(horizon=30)` 或 Chronos `predict_df` / Lag-Llama 零样本；同一测试集与标签算同一批指标，记录推理耗时与 GPU/内存。 | TimesFM/Chronos/Lag-Llama 精度与推理资源 |
| **第四步** | 接入 PatchTST 或 Informer（及可选 DeepLOB）：将分钟级表整理为模型所需 CSV；设置 look-back（如 60）、pred_len=30（或 seq_len/pred_len）；1 月训练、2 月第一周测试；记录精度与训练/推理资源。 | PatchTST/Informer/DeepLOB 精度与资源 |
| **第五步** | 制对比表（LightGBM、小 LSTM、PatchTST/Informer、TimesFM、Chronos、Lag-Llama（及可选 DeepLOB）× 准确率/MSE、训练时间、推理延迟、显存/内存）；写结论：小模型是否在更少资源下达到或超过大模型。 | 对比表与报告 |

建议先完成第一步，再按 2→3→4→5 推进；TimesFM 与 PatchTST/Informer 可并行；微调（第五节）可在基线跑通后再加入。

## 七、可行性检查

- **数据**：现有 `merged_15_common_stocks` 与 PySpark 笔记本已具备 time、best_ask、best_bid；需在原始 CSV 中确认 volume 与 ask/bid size 的列索引并扩展读取。可行性高。
- **标签**：未来 30 分钟收益/方向仅依赖已有分钟级价格与时间对齐，与现有 lag_return 逻辑一致。可行性高。
- **小模型**：LightGBM、小 LSTM/TCN、线性模型均为常见库与结构，本地可训练。可行性高。
- **TimesFM**：开源、提供 pip/Hugging Face，输入为序列列表，与分钟级序列兼容；需注意序列长度与降采样。可行性高。
- **PatchTST / Informer**：开源、需自备 CSV 与训练；数据格式与脚本可适配分钟级多变量。可行性高。
- **Amazon Forecast**：依赖 AWS 与 API，可选；公平对比建议暂不纳入。

**结论**：在确认原始数据中 volume 与 size 列的前提下，整体方案可行；建议优先完成数据与标签构建，再依次跑小模型与 TimesFM/PatchTST/Informer；微调部分在基线稳定后接入。

## 八、可补充的研究方向与信息（值得进一步研究）

以下为可写入文档或作为后续研究的问题与维度，不改变当前方案主干，但能增强论证深度与引用价值。

- **领域与频率适配**：预训练多基于日/小时频；分钟级、高频价量的分布差异对零样本/微调效果的影响（可做简单统计或可视化）。
- **数据量—微调效果**：用 1 月的 10%/30%/100% 做微调，看准确率或 MSE 随样本量的变化，讨论「小数据下大模型是否仍优于小模型」。
- **推理成本与延迟**：对 TimesFM、PatchTST、Informer 记录单条/批量推理的时延与显存，与小模型对比，量化「资源浪费」程度。
- **稳健性与鲁棒性**：在测试集上做子时段（如按日/按股票）拆解，看小模型与大模型在不同时段的表现方差与最差情况。
- **可解释性**：LightGBM 特征重要性、PatchTST/Informer 注意力或通道贡献（若可实现），用于解释「为何小模型足够」或「大模型在哪些情形失效」。
- **标签与阈值敏感性**：三分类阈值、未来 30 分钟 vs 5/15 分钟窗口对结论的影响；可做少量敏感性实验并写进报告。
- **其他时序/金融 LLM**：Time-LLM、Chronos、StockTime、LLM4TS 等若有机会接入，可一并纳入「预训练 vs 微调 vs 小模型」对比表与参考文献。

## 九、参考文献（时序模型与 LOB 相关）

- **TimesFM**：Google Research, *A decoder-only foundation model for time-series forecasting*, ICML 2024. <https://github.com/google-research/timesfm>
- **PatchTST**：Nie et al., *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, ICLR 2023. <https://github.com/yuqinie98/PatchTST>
- **Informer**：Zhou et al., *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021. <https://github.com/zhouhaoyi/Informer2020>
- **Amazon Forecast Samples**：AWS 云服务示例。<https://github.com/aws-samples/amazon-forecast-samples>
- **Chronos**：Amazon Science, *Chronos: Learning the Language of Time Series* (TMLR 2024), *Chronos-2: From Univariate to Universal Forecasting* (arXiv 2025). <https://github.com/amazon-science/chronos-forecasting>
- **Lag-Llama**：Rasul et al., *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting*, arXiv 2023. <https://github.com/time-series-foundation-models/lag-llama>
- **DeepLOB**：Zhang et al. 论文实现，S-razmi 在 BTC 永续 LOB 上的实现。<https://github.com/S-razmi/DeepLOB>

（若加入微调或通用时序 LLM，可在本节补充对应论文与仓库链接。）

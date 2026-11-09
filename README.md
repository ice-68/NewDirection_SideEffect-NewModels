# ASD 前沿时序模型与站点效应

本项目使用 ABIDE 的 CC200 脑区时间序列，在严格的
Leave-One-Site-Out（LOSO）协议下比较时序模型、图模型及其组合，并区分
ASD 相关信号与站点捷径。

## 当前状态

- 工作集：877 名受试者、20 个站点、CC200、T=78。
- 定义闸：已冻结，见 `METRICS.md`。
- 已完成：B0、B1（strict inductive）、B2、B3 的 5-seed LOSO。
- 下一步：B4（Mamba + 图模型）。

| 模型 | site-macro AUC | Balanced ACC | 正式结果 |
|---|---:|---:|---|
| B0 FC + logistic regression | 0.7044 ± 0.0008 | 0.6469 | `outputs/stage2_b0_v2/` |
| B1 strict inductive GCA | 0.6362 ± 0.0142 | 0.5978 | `outputs/stage2_b1_inductive/` |
| B2 bidirectional GRU | 0.5775 ± 0.0221 | 0.5460 | `outputs/stage2_b2_v2/` |
| B3 unidirectional Mamba | 0.6148 ± 0.0282 | 0.5695 | `outputs/stage2_b3_v1/` |

上述数值仅为描述性结果，模型优劣和科学解释需人工复核。

## 目录

```text
src/            可复现分析代码
tests/          回归测试
cache/          可重建缓存与 LOSO 划分
data/           原始数据，只读
outputs/        当前四套正式实验结果
logs/           变更记录
Papers/         文献材料
METRICS.md      Stage 2 评估契约
```

历史 smoke、seed0 和旧版实验目录已清理；原始数据未改动。

## 环境

正式解释器：

```text
C:\Users\28439\miniconda3\python.exe
```

版本与运行说明见 `ENVIRONMENT.md`。

## 常用命令

```powershell
& 'C:\Users\28439\miniconda3\python.exe' -m pytest -q
& 'C:\Users\28439\miniconda3\python.exe' -m src.b0
& 'C:\Users\28439\miniconda3\python.exe' -m src.b1
& 'C:\Users\28439\miniconda3\python.exe' -m src.b2
& 'C:\Users\28439\miniconda3\python.exe' -m src.b3
& 'C:\Users\28439\miniconda3\python.exe' -m src.verify_stage2
```

任何标准化、调参、特征选择或 harmonization 都只能在训练折内拟合。

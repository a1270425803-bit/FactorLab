# FactorLab 系统架构

## 概述

FactorLab 是一个 AI 驱动的 A 股量化因子挖掘系统，通过大语言模型自动生成、评分、回测和入库量化因子。

## 核心模块

| 模块 | 职责 |
|------|------|
| `engine.py` | AI 因子生成引擎 |
| `pipeline.py` | 完整执行管道（单轮） |
| `batch_pipeline.py` | 批量运行引擎 |
| `score.py` | 8 维评分系统 |
| `backtest.py` | 冲击成本回测 |
| `robustness_checker.py` | 4 维稳健性检验 |
| `diversity_gate.py` | 多样性门控（防止因子同质化） |
| `checker.py` | AST 安全沙箱 |
| `sandbox.py` | 代码执行沙箱 |
| `database.py` | SQLite 数据库 |

## 数据流

```
用户输入 → engine.py 生成因子 → checker.py 合规检查 → sandbox.py 沙箱执行
  → score.py 8维评分 → backtest.py 回测 → diversity_gate.py 多样性门控
  → database.py 入库 → html_reporter.py 生成报告
```

## 技术栈

- Python 3.9+
- pandas / numpy（数据处理）
- AKShare（A 股数据）
- DeepSeek API（因子生成）
- SQLite（因子数据库）

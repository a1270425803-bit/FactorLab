# FactorLab

> **AI-driven quantitative factor mining for China's A-share market.**
>
> Let AI generate, validate, score, backtest and store quantitative factors — you only set the research direction.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![AKShare](https://img.shields.io/badge/data-AKShare-orange)](https://www.akshare.xyz/)

[English](#english) | [中文](#中文)

---

<a name="中文"></a>

## 中文介绍

FactorLab 是一套**全自动 A 股量化因子挖掘系统**。核心理念：

```
人类设定研究方向 → AI 生成因子代码 → AST 安全沙箱执行 → 8 维量化评分 → 冲击成本回测 → 4 维稳健性检验 → 自动入库
```

你只需告诉它"我要找反转类因子"，系统便自动生成 Python 代码，在 **1,482 只 A 股、15 年历史** 上跑回测，全部通过则自动存入数据库。跑完 50 轮后自动生成一份带图表的 HTML 研究报告。

### 核心亮点

| 特性 | 说明 |
|------|------|
| **AI 自动生成因子** | 接入 DeepSeek API，根据研究方向自动生成 Python 因子代码 |
| **AST 安全沙箱** | 纯语法树级别代码审查，4 类危险模式拦截，零误报 |
| **8 维评分关口** | IC、IR、覆盖率、相关性、换手率、方向性、稳定性、衰减比 — 全部达标才入库 |
| **真实冲击成本** | 逐日逐股计算冲击成本，非固定扣费，大盘股 < 0.05%，小盘股 > 0.5% |
| **4 维稳健性检验** | 单调性、样本外稳定性、IC 衰减、分年验证 |
| **多样性门控** | 新因子与库存因子 Spearman 相关 > 0.70 自动拒绝，防止同质化 |
| **全自动批量** | 一键 50 轮，双层熔断 + 断点续跑，单轮成本 ≈ ¥0.10 |
| **HTML 报告** | 单文件 base64 内嵌图表，双击打开，无需网络 |
| **自然语言检索** | "有哪些量价背离的因子？"→ 自动分词匹配 |

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/YippeeXu/FactorLab.git
cd FactorLab

# 2. 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 API Key
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY

# 5. 下载数据（可选，首次约 10 分钟）
python scripts/download_data.py --fetch

# 6. 运行演示（不消耗 API 额度）
python src/main.py --batch 5 --dry-run

# 7. 实战挖因子
python src/main.py --batch 50
```

### 项目结构

```
FactorLab/
├── src/               # 核心源码（29 个 Python 模块）
│   ├── main.py        # CLI 入口
│   ├── engine.py      # AI 因子生成引擎
│   ├── checker.py     # AST 安全审查
│   ├── sandbox.py     # 代码执行沙箱
│   ├── score.py       # 8 维评分系统
│   ├── backtest.py    # 冲击成本回测
│   ├── database.py    # SQLite 数据库
│   └── ...
├── tests/             # 测试套件
├── docs/              # 文档
│   ├── QUICKSTART.md  # 快速上手
│   ├── ARCHITECTURE.md # 系统架构
│   ├── API.md         # 接口文档
│   └── FAQ.md         # 常见问题
├── examples/          # 示例报告
├── scripts/           # 工具脚本
│   ├── download_data.py  # 数据下载
│   └── schema.sql     # 数据库结构
├── data/              # A 股数据（.gitignore 忽略）
└── requirements.txt
```

### 数据说明

A 股历史行情数据来源于 [AKShare](https://www.akshare.xyz/)（基于东方财富/同花顺公开数据）。首次运行需下载约 **169MB**（1,482 只 × 15 年日线 OHLCV）。数据仅供研究学习，不构成投资建议。

### 技术栈

- Python 3.9+
- pandas / numpy（数据处理）
- [AKShare](https://www.akshare.xyz/)（A 股数据）
- DeepSeek API（因子生成）
- SQLite + WAL（因子数据库）
- matplotlib（图表生成）

### 安全设计

| 层级 | 机制 |
|------|------|
| **代码安全** | AST 语法树审查，4 类危险模式拦截，白名单验证 |
| **执行隔离** | 沙箱超时 30s，异常不崩溃主程序 |
| **权力分割** | 人类锁死核心规程，AI 只能写建议 |
| **熔断机制** | 单轮 3 次失败 + 连续 10 轮无入库自动暂停 |
| **防注入** | 100% 参数化 SQL |

### 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库并创建分支：`git checkout -b feature/my-feature`
2. 确保代码通过测试：`pytest tests/ -v`
3. 遵循现有代码风格（PEP 8）
4. 提交 PR，描述清楚改动内容和动机

详细规范请参见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

### 许可证

[MIT License](LICENSE) — 2026 YippeeXu

---

<a name="english"></a>

## English Introduction

FactorLab is a **fully automated quantitative factor mining system** for China's A-share market. The core idea:

```
Human sets direction → AI generates factor code → AST sandbox execution → 8-dimension scoring → Impact cost backtest → 4-dimension robustness check → Auto storage
```

Just tell it "I want reversal factors", and the system generates Python code, runs backtests on **1,482 stocks over 15 years**, and stores passing factors automatically. After 50 rounds, it generates an HTML research report with embedded charts.

### Key Features

| Feature | Description |
|---------|-------------|
| **AI Factor Generation** | DeepSeek API integration, generates Python factor code from research directions |
| **AST Security Sandbox** | Pure syntax-tree level code review, 4-category danger interception, zero false positives |
| **8-Dimension Scoring** | IC, IR, coverage, correlation, turnover, directionality, stability, decay ratio — all must pass |
| **Real Impact Cost** | Per-day per-stock impact cost calculation, not fixed fee deduction |
| **4-Dimension Robustness** | Monotonicity, out-of-sample stability, IC decay, yearly validation |
| **Diversity Gate** | Auto-reject new factors with Spearman correlation > 0.70 to existing ones |
| **Full Auto Batch** | One-click 50 rounds, dual fuse + resume support, ~¥0.10 per round |
| **HTML Report** | Single-file base64 embedded charts, open offline |
| **NL Query** | "Show me divergence factors" → auto token matching |

### Quick Start

```bash
# 1. Clone
git clone https://github.com/YippeeXu/FactorLab.git
cd FactorLab

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Configure API Key
cp .env.example .env
# Edit .env with your DEEPSEEK_API_KEY

# 5. Download data (first run, ~10 min)
python scripts/download_data.py --fetch

# 6. Demo (no API cost)
python src/main.py --batch 5 --dry-run

# 7. Mine factors
python src/main.py --batch 50
```

### Project Structure

```
FactorLab/
├── src/               # Core source (29 Python modules)
│   ├── main.py        # CLI entry
│   ├── engine.py      # AI factor generation engine
│   ├── checker.py     # AST security review
│   ├── sandbox.py     # Code execution sandbox
│   ├── score.py       # 8-dimension scoring
│   ├── backtest.py    # Impact cost backtest
│   ├── database.py    # SQLite database
│   └── ...
├── tests/             # Test suite
├── docs/              # Documentation
│   ├── QUICKSTART.md  # Quick start guide
│   ├── ARCHITECTURE.md # System architecture
│   ├── API.md         # API docs
│   └── FAQ.md         # FAQ
├── examples/          # Sample reports
├── scripts/           # Utility scripts
│   ├── download_data.py  # Data download
│   └── schema.sql     # Database schema
├── data/              # A-share data (.gitignore ignored)
└── requirements.txt
```

### Data Source

Historical A-share data from [AKShare](https://www.akshare.xyz/) (based on East Money/Flush public data). First download ~**169MB** (1,482 stocks × 15 years daily OHLCV). For research only, not investment advice.

### Tech Stack

- Python 3.9+
- pandas / numpy (data processing)
- [AKShare](https://www.akshare.xyz/) (A-share data)
- DeepSeek API (factor generation)
- SQLite + WAL (factor database)
- matplotlib (chart generation)

### Security Design

| Layer | Mechanism |
|-------|-----------|
| **Code Safety** | AST syntax tree review, 4-category danger interception, whitelist validation |
| **Execution Isolation** | Sandbox timeout 30s, exceptions don't crash main program |
| **Power Separation** | Humans lock core protocols, AI can only write suggestions |
| **Fuse Mechanism** | Single round 3 failures + 10 consecutive rounds without storage auto-pause |
| **Anti-Injection** | 100% parameterized SQL |

### Contributing

Issues and Pull Requests are welcome!

1. Fork this repo and create a branch: `git checkout -b feature/my-feature`
2. Ensure tests pass: `pytest tests/ -v`
3. Follow existing code style (PEP 8)
4. Submit PR with clear description of changes and motivation

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed guidelines.

### License

[MIT License](LICENSE) — 2026 YippeeXu

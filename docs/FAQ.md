# FactorLab 常见问题

## 环境配置

**Q: 如何配置 DeepSeek API Key？**
A: 复制 `.env.example` 为 `.env`，填入你的 API Key：
```bash
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

**Q: 数据从哪里来？**
A: A 股历史数据通过 AKShare 获取（基于东方财富/同花顺公开数据）。运行 `scripts/download_data.py` 自动下载。

## 评分系统

**Q: 8 维评分是什么？**
A: 覆盖：信息系数(IC)、稳定性(IR)、夏普比率、最大回撤、换手率、收益分布、参数敏感性、交易成本。

**Q: 什么样的因子能通过门控？**
A: 综合评分 ≥ 0.6，且与已入库因子的相似度 < 0.8。

## 数据

**Q: 完整数据集有多大？**
A: 1,482 只 A 股历史数据约 169MB。首次下载需要一定时间。

**Q: 数据库可以重建吗？**
A: 可以。运行 `sqlite3 db/factorlab.db < scripts/schema.sql` 即可初始化数据库结构。

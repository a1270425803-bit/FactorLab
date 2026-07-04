# FactorLab 快速上手指南 (v3.1.0)

## 1. 配置环境

```bash
source venv_mac/bin/activate       # Mac
pip install -r requirements.txt
```

## 2. 配置 API Key

编辑项目根目录的 `.env` 文件：

```bash
DEEPSEEK_API_KEY=sk-your-real-key-here
```

没有 Key？先用 `--dry-run` 模式跑通全流程，零成本验证。

## 3. 数据准备

```bash
python data_fetcher_v2.py --check   # 数据完整性校验（1482只+）
python data_fetcher_v2.py --fetch   # 下载数据（断点续传）
```

## 4. 验证管道（不消耗 API 额度）

```bash
python main.py --batch 3 --dry-run
```

## 5. 全自动批量挖掘

```bash
python main.py --batch 50           # 50 轮全自动
python main.py --batch 50 --resume  # 中断后续跑
```

## 6. 模块自检

```bash
python score.py                     # 评分系统自检（v2.1 反馈增强版）
python check_code_uniqueness.py     # 代码相似度门禁（6项测试）
python backtest.py --demo           # 冲击成本回测
python robustness_checker.py --demo # 稳健性检验
```

## 7. 查看成果

```bash
python database.py                  # 查看数据库表结构
python nl_query.py --demo           # 自然语言检索因子库
ls reports/                         # HTML 报告
```

## 8. 调整方向

用编辑器打开 `program.md`，修改前三章（研究方向/因子定义域/评分标准），保存后重启系统。

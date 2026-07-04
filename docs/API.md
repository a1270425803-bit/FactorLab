# FactorLab API 文档

## 核心接口

### engine.py

```python
from engine import create_engine

engine = create_engine(model="deepseek-chat")
factor_code = engine.generate_factor(prompt="生成一个基于成交量的因子")
```

### score.py

```python
from score import score_factor

result = score_factor(factor_code, df_1800)
# result: ScoreResult 包含 8 个维度评分
```

### backtest.py

```python
from backtest import simple_backtest

result = simple_backtest(factor_series, close_df, holding_period=5)
```

## 数据库接口

### database.py

```python
from database import get_conn, init_db, insert_factor

conn = get_conn()
init_db(conn)
insert_factor(conn, factor_id, factor_code, summary, scores)
```

更多接口详见源码注释。

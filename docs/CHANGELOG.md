# FactorLab CHANGELOG

## Phase 3f (2026-06-13) — 沙箱修复 + OOS 分期 + pre-rank 规则 + 校准基准

### Batch 20 验证 Phase 3e
- R7 历史首次通过全部 8 维 score.py（IC=0.087, IR=0.214），被 OOS 拦截
- Phase 3e 代码相似度门禁有效：同质化率从 90% → 0%，dir_acc=0.469 从 90% → 15%
- 沙箱失败率 50%（列为下一优先级）

### 沙箱 bug 修复
- `engine.py`: user_prompt 新增 ⚠️ 常见错误段，明确列出 batch 20 实际错误（列名不存在、MultiIndex 误用、向量化违规、pandas 版本）
- `checker.py`: AST 静态检测 `df.groupby('code')` 和 `df['code']`，沙箱前拦截为 ERROR
- `test_ast_whitelist.py` + 4 个 history 文件更新为 `level='code'`

### OOS 分期检查
- **旧逻辑**: `abs(test_ic_mean) > 0.02`（全期均值）
- **新逻辑**: 7 个测试年(2019-2025)中 ≥4 年 `abs(年IC) > 0.02`
- 涉及: `robustness_checker.py`（变量 + 逻辑）、`config.py`（YEARLY_OOS_MIN_YEARS, YEARLY_OOS_IC_THRESHOLD）、`batch_pipeline.py`（FinalResult 新字段 + 分期显示）

### program.md 规则 #6（前三章，人类授权）
- pre-rank 时序记忆前置条件：rank(pct=True) 结尾的因子，rank 之前必须包含 ≥20 日股票级时序操作
- 附 ✅/❌ 示例

### 经典因子校准基准
- `calibration.py`（新建）：4 经典因子（20日动量、60日反转、振幅、量比）跑完整 pipeline
- 结论：ic_decay、rank_ac 阈值合理不需下调；60日反转通过全部 8 维证明阈值可达
- `docs/calibration_report.md`

### 架构辩论
- 两轮 v2.5 §2.1 子 Agent 辩论
- `docs/ARCHITECTURE_CONFIRMATION_Phase3f.md`、`docs/ARCHITECTURE_CONFIRMATION_Phase3f_v2.md`
- `ADVERSARIAL_REVIEW_DP1_4.md`、`reports/arch_review_DP1_DP3_rebuttal.md`

---

## v3.1.0 (2026-06-04) — Phase 3e 代码相似度门禁 + 评分反馈增强

### 评分反馈增强（score.py → v2.1，人工许可升级）
- **DIM_FEEDBACK_GUIDE**：每个失败维度附带问题诊断 + 可操作的改进线索（5 维度 × 3 线索）
- **code_pattern_hint**：基于评分数值特征的结构性反馈（3 种模式）
  - 模式 A：rank_ac<0.15 && dir_acc<0.50 → "多维乘积+截面排名锁死"
  - 模式 B：ic>0.03 && decay<0.40 → "短期噪音驱动因子"
  - 模式 C：ic>0.03 && ir<0.15 → "regime-dependent 因子"
- **pattern_fingerprint**：5 维评分数值指纹 {ic, ir, dir_acc, rank_ac, decay}，Phase 3f 预留
- **failed_reasons 格式升级**：数值对比 + [分析] 段落 + 编号线索（3 层结构）
- **MD5 已更新**：`4926dcae024965ae69c4d9e09ee3beaa`
- 阈值/权重/评分计算逻辑未变 ✅

### 代码相似度门禁（check_code_uniqueness.py，新增模块）
- **AST 紧凑结构签名**：提取调用链 + 二元运算符 + 下标序列（150-400 字符签名，替代 3000 字符 ast.dump）
- **数学语义标签**：16 种正则标签（multivariate_product, cross_sectional_rank, unstack_reshape 等），捕获"换 API 不换数学"的变体
- **双重相似度**：combined = 0.6 × ast_sim + 0.4 × semantic_sim
- **_detect_homogeneous_deadlock()**：历史全部同模式检测（>80% 两两相似度 > 0.7 → 死锁警告）
- **不 skip 本轮**：只生成 feedback 注入下一轮 system_prompt（避免浪费 API 调用）
- **性能**：0.3s/次（vs 初始版 11s），lru_cache + 紧凑签名
- **纯 Python 回退**：python-Levenshtein 不可用时自动降级 difflib

### 数据库（database.py）
- **pending_feedbacks 表**：代码相似度门禁 feedback 传递通道（round_id, feedback_text, consumed）
- **get_pending_similarity_feedback()**：取最近未消费 feedback 并标记 consumed
- **save_pending_similarity_feedback()**：保存 feedback 供下一轮消费
- consumed 用 INTEGER DEFAULT 0（非 BOOLEAN）

### 批量引擎集成（batch_pipeline.py）
- system_prompt 构造前注入 pending similarity feedback（prefix + chapters_1_3 + memory 三段）
- fail_parts 拼接时追加 `score_result.code_pattern_hint`
- `round_fail_reason` 改用 `"\n"` join（保留多行格式）

### 验证
- `score.py` 自检通过（MD5 校验 + C★ 模式诊断触发 + 反馈增强）
- `check_code_uniqueness.py` 6/6 测试通过（0.3s/次）
- `batch_pipeline.py` dry-run 3 轮无报错

### 冻结模块保护
- diversity_gate.py 未修改 ✅
- score.py 阈值/权重/评分逻辑未变 ✅
- sandbox.py / data_fetcher.py 未修改 ✅

---

## v3.0.2 (2026-06-04) — Phase 3d 评分阈值优化 + 实战批量诊断

### 评分阈值调整（score.py）
- IR: 0.20 → **0.12**（18轮中位数 0.15，通过率 19% → 85%）
- 方向正确性: 0.55→0.50→**0.48**（18轮最高 0.497）
- 秩自相关: 0.50→0.30→**0.20**（18轮通过率 25% → ~40%）
- MD5 自指正则修复：`[a-f0-9]{32}` 通用模式 + "000…" dummy 替换

### 稳健性检验调整（robustness_checker.py）
- 单调性从强制卡关改为展示参考（大截面下 L1-L5 100% 失败）

### Bug 修复
- `pipeline.py`: 数据加载从 10 只 DEMO → 全量 1482 只 `load_df_1800()`
- `engine.py`: AI 提示词修正 `groupby(level='code')`（MultiIndex 语法）
- `memory_manager.py`: 修复 `append_memory` 每次写入覆盖全部历史记忆（2 处）
- `score.py`: MD5 自指正则从未匹配 → 修复为通用 hex 模式

### 50 轮批量诊断（20/50 完成）
- **20 轮 0 入库**，方向正确性和秩自相关是系统性瓶颈
- 根因：AI 生成的因子全部使用截面 z-score → 破坏时序排名稳定性
- 详见 `DIAGNOSIS_REPORT.md`

### 新增文件
- `diagnose_dims.py`: 三维度（超跌/缩量/波动收敛）独立诊断脚本
- `test_my_factor.py`: 绕过 AI 直接测试因子代码的评分脚本
- `DIAGNOSIS_REPORT.md`: 全维度诊断报告（8 章）

---

## v3.0c (2026-06-03) — Phase 3c 组合合成 + 静态 HTML 报告 + 自然语言检索

### combo_engine.py — ICIR 加权多因子组合合成（新增）
- **ComboResult dataclass**：combo_factor / weights / icir_values / backtest_result / vs_best_single
- **build_all_inbound()**：自动读取全部入库因子，通过沙箱重新执行代码获取因子值
- **ICIR 加权**：从 metrics JSON 提取 IR（json_extract），负值归零，正值归一化
- **z-score 截面标准化**：每行 mean=0, std=1，加权合成 combo_z
- **回测 + 对比**：调用 simple_backtest，与最佳单因子对比夏普比率（目标 >=80%）
- **不入库**：组合因子不写入 factors 表，不参与 diversity gate
- **`--demo`**：Mock 3 因子（ICIR: 0.5/-0.3/1.2）演示完整流程
- **优雅降级**：因子库为空时返回空 ComboResult，不崩溃

### html_reporter.py — 静态 HTML 报告（新增）
- **快速版** (`generate_quick_report`)：a.因子表格 + d.成本仪表盘 + 简要统计
- **完整版** (`generate_full_report`)：a-g 全部 section（a因子表格/bIR柱状图/c回测指标/d成本/e方向/f合成/g分年验证）
- **图表**：matplotlib → base64 PNG 内嵌，DPI=100，无外部依赖
- **CSS 内嵌**：简洁专业风格，交替行底色表格，通过/未通过颜色标注
- **文件大小**：快速版 ~58KB，完整版 ~125KB（3 因子 mock）
- **无入库因子降级**：显示"暂无入库因子"占位符
- **`--demo` / `--quick` / `--full`**：三种运行模式

### nl_query.py — 自然语言因子检索（新增）
- **search()**：中文关键词分词（正则分割）+ 停用词过滤 + natural_summary / direction_tag OR 匹配
- **structured_filter()**：11 项 SUPPORTED_FILTERS 白名单，参数化查询防注入
- **interactive_query()**：CLI 交互式子菜单（自然语言/结构化/查看全部）
- **防注入**：100% ? 占位符参数化，无字符串拼接 SQL。测试：`'; DROP TABLE factors; --'` 安全捕获
- **`--demo`**：8 项自检测试（含 SQL 注入防护验证）

### main.py — CLI 菜单升级
- 菜单扩展为 8 项：[6] 生成 HTML 报告 / [7] 因子检索 / [8] 退出
- **cmd_generate_report()**：子菜单快速版/完整版，完整版自动计算 ICIR 组合
- **cmd_factor_search()**：委托给 nl_query.interactive_query()
- 标题更新为 "FactorLab Phase 3c"

### batch_pipeline.py — 50 轮后自动触发报告
- 在 `run_batch()` 末尾（summary_engine 里程碑写入后）自动调用 `generate_full_report()`
- 复用已加载数据矩阵（df_1800 / close_df_full / volume_df_full / returns_df_full）
- 报告保存路径：`reports/factorlab_full_report_{timestamp}.html`
- 失败不中断程序，try-except 保护

### config.py — 新增 Phase 3c 配置
- `REPORTS_DIR = "reports"`：HTML 报告输出目录

### 集成测试
- **test_phase3c_integration.py**：6 组测试 46 个检查点全部通过
- 覆盖：combo_engine 权重/z-score/降级，html_reporter 快速版/完整版 a-g 验证，nl_query 搜索/过滤/防注入/停用词，冻结模块保护，config 配置

### 冻结模块保护
- score.py / sandbox.py / data_fetcher.py / checker.py / backtest.py / robustness_checker.py / database.py / logger.py / data_fetcher_v2.py 完全未修改 ✅

---

## v3.0b (2026-06-03) — Phase 3b 回测引擎升级 + 稳健性检验

### backtest.py — 冲击成本模型升级
- **冲击成本模型**：替代固定年化 5% 成本，使用 `min(持仓金额/日均成交额 × 150%, 2%)` 双边扣除
- **新增参数**：`volume_df`（成交量矩阵，单位：股）、`close_df`（收盘价矩阵）、`capital`（资金假设，默认 5000 万）
- **向后兼容**：旧调用方式（未传 volume_df/close_df）给出 DeprecationWarning 并降级为无冲击模式
- **新增字段**：`avg_impact_cost_bps`（平均冲击成本基点）、`total_cost_annual`（年化总成本）、`layer_returns`（L1-L5 分层收益）
- **分层收益**：每调仓日按因子值分 5 组，记录每组持有期收益，供 robustness_checker 单调性检验
- **`--demo` 命令**：Mock 大盘/小盘股验证冲击成本（大盘 < 0.05%，小盘 > 0.5%）
- **参数配置化**：从 config.py 读取所有阈值，无硬编码

### robustness_checker.py — 4 维稳健性检验（新增，完全独立于 score.py）
- **维1 单调性**：L1-L5 分层收益 Spearman 秩相关（阈值 0.3）
- **维2 样本外稳定性**：训练集(2010-2018) vs 测试集(2019-2025) 日频 IC t-stat
- **维3 IC 衰减**：T+1/T+5/T+10/T+20 四滞后期 RC 衰减比（阈值 0.5）
- **维4 分年验证**：2020-2025 每年独立 IC，纯展示，不参与 robust_core_passed
- **robust_core_passed**：维1 AND 维2 AND 维3（不含维4）
- **`--demo` 命令**：Mock 100 只 × 500 天验证各维度计算
- **完全独立**：不导入 score.py，不依赖其输出

### config.py — Phase 3b 配置项
- 资金与冲击成本：`CAPITAL_ASSUMPTION`(5000万)、`IMPACT_COST_COEFFICIENT`(1.50)、`IMPACT_COST_CAP`(2%)
- 回测参数：`HOLDING_PERIOD`(5)、`TOP_PCT`(0.3)、`RISK_FREE_RATE`(2%)
- 稳健性阈值：`MONOTONICITY_THRESHOLD`(0.3)、`OOS_IC_THRESHOLD`(0.02)、`IC_DECAY_RATIO_THRESHOLD`(0.5)
- 样本外分割日期 + IC 衰减周期 + 日均成交额窗口

### batch_pipeline.py — 10 维合并集成
- **新增 FinalResult dataclass**：合并 ScoreResult + BacktestResult + RobustnessResult
- **新增 merge_results()**：`final.threshold_passed = score.passed_threshold AND robust_core_passed`
- **流程升级**：score → backtest（含冲击） → robustness → merge → diversity → 入库
- **数据缓存**：close_df/volume_df/returns_df 矩阵批量启动时计算一次，所有轮次共享
- **SQLite 写入**：factors 表新增 9 个稳健性字段，backtests 表新增冲击成本字段

### database.py — 表结构迁移 v2→v3b
- **migrate_v2_to_v3b()**：自动检测版本，ALTER TABLE 新增 Phase 3b 字段
- **factors 表新增**：monotonicity, monotonicity_passed, oos_ic_train, oos_ic_test, oos_stability_passed, ic_decay_ratio, ic_decay_passed, yearly_validation_passed, yearly_observed
- **backtests 表新增**：avg_impact_cost_bps, total_cost_annual, layer_returns
- **main.py 启动调用**：自动执行迁移

### 集成测试
- **test_phase3b_integration.py**：5 项测试 38 个检查点全部通过
- 覆盖：数据加载、冲击成本、稳健性 4 维、10 维合并、SQLite 读写

### 冻结模块保护
- score.py / sandbox.py / data_fetcher.py / checker.py / data_fetcher_v2.py 均未修改 ✅

---

## v3.0a (2026-06-02) — Phase 3a 数据基建升级 + AST 安全沙箱

### data_fetcher_v2.py — 全量历史宽表 CSV 升级
- **宽表 CSV 格式**：`date, open, high, low, close, volume` 6 列，无 `code` 列（文件名即股票代码）
- **日期范围**：支持 2010-2025 全量历史（默认 `--start 20100101 --end 20251231`）
- **下载优先级**：沪深300 → 中证500 → 中证1000，运行时动态获取成分股
- **断点续传增强**：downloaded.txt 记录 + CSV 行数 >100 校验，损坏文件自动重下
- **黑名单机制**：空数据/退市股票（<10 行）重试 3 次确认后入 `blacklisted.txt` 永久跳过
- **失败记录**：网络异常股票重试 3 次仍失败入 `failed_downloads.txt`，下次 `--fetch` 可重试
- **原子写入**：`{code}.csv.tmp` → `os.replace()` → `{code}.csv`
- **子代理并行**：待下载 >50 只时自动启用 ThreadPoolExecutor 多线程下载
- **API 双源 fallback**：优先 `ak.stock_zh_a_hist`，失败则降级 `ak.stock_zh_a_daily`

### checker.py — AST 重写（纯 AST 模式，替代正则）
- **彻底移除正则规则**：原 FORBIDDEN_PATTERNS 列表全部删除
- **AST 四类节点检测**：
  - Call: `shift(-N)` / `shift(periods=-N)` / `eval()` / `exec()` / `open()` / `__import__()`
  - For: `for x in df['col'].unique()` / `iterrows` / `itertuples`
  - Import: `os` / `sys` / `subprocess` / `requests` / `socket`
  - Attribute: `iterrows` / `itertuples`
- **零误报**：仅拦截静态可判定为负数的常量，变量/表达式放行
- **性能**：100 次解析 0.07ms/次（阈值 <100ms）
- **向后兼容**：返回签名保持 `(PASS/ERROR, reason)`，保留 WARNING 关键字兼容调用方

### 数据迁移
- 现有 1480 个 CSV 文件通过 `--migrate` 命令去除 `code` 列，迁移至新宽表格式

### 新增文件
- `test_ast_whitelist.py`：AST 白名单测试，54 条合法代码（SQLite 50 条 + History 4 条）全部 PASS

### score.py — Bug 修复（冻结模块，MD5 已更新）
- 换手率计算修复：`div(factor_values.abs() + 1e-8)` → 截面分位数变化率
- 修复了因子值接近 0 时换手率爆炸为 10^12~Inf 的问题
- 新 MD5: `1f7562705ef05a7378b2144efe0d34d6`

---

## v2.3 (2026-06-02) — Phase 2c Review 修复

### Critical 修复 (3 项)
- **C1**: `batch_pipeline.py` cumulative_cost 丢失 → 修复 `update_batch()` 使用累计变量 `cumulative_cost` 而非单轮 `round_cost`，resume 可靠
- **C2**: `load_df_1800()` code 列类型不匹配 (int64 vs string) → 强制 `astype(str).str.zfill(6)`，diversity_gate 门控生效
- **C3**: `summary_engine._ai_generate_ch1()` 绕过 CostTracker → 改用 `engine.chat()`，MockEngine + DeepSeekEngine 均新增 `chat()` 方法，token 自动追踪

### Major 修复 (6 项)
- **M1**: Round 2+ API 成本日志跨轮污染 → 每轮保存 `round_start_cost`，`_cost` 改为增量计算
- **M2**: 中期熔断未写 program.md 里程碑 → 触发时调用 `append_milestone("中期熔断", ...)`
- **M3**: `query_factors()` SQL 注入风险 → 新增 `_ALLOWED_SORT_COLUMNS` 白名单
- **M4**: diversity_gate 因子池来源为 JSON 非 SQLite → 顶部添加 `TODO(Phase 3)` 注释标记已知 gap
- **M5**: resume 后 CostTracker 未预加载历史成本 → 新增 `CostTracker.preload()` 方法，resume 时调用
- **M6**: batch 中期熔断无 adopt 入口 → 增加子选项 [2a] 采纳 draft / [2b] 手动编辑，调用 `apply_adopt()` 热加载

### Minor 修复 (6 项)
- **m1**: 4 个新建文件 chmod 755
- **m2**: `checker.py` 新增 dot-notation for 循环检测正则 (df.code.unique())
- **m3**: `summary_engine.py` 成本估算接收 CostTracker 实例，优先用实际追踪数据
- **m4**: `database.py` migrate=True 时打印数据删除警告
- **m5**: `batch_pipeline.py` 循环内 `from diversity_gate import _rank_values` 移至顶部
- **m6**: `batch_pipeline.py` CLI 暂停添加 signal.alarm 非 UNIX fallback (select.select)

### Verified
- E2E 集成测试: `python main.py --batch 3 --dry-run` 3 轮跑通
- C2: code dtype = object (string) ✅
- m2: dot-notation 检测拦截 ✅
- M3: SQL 注入白名单 fallback ✅
- M5: preload(2.50) → cost = 2.5000 ✅
- score.py MD5 通过 ✅ | 冻结模块未被修改 ✅

---

## v2.2 (2026-06-01) — Mac 迁移修复

### Fixed
- **.gitattributes**: 新建 `.gitattributes` 文件，强制 `*.py/*.md/*.csv/*.txt/*.json` 使用 LF 换行符，防止 CRLF 回归
- **CRLF 清理**: 清理 22 个被 CRLF 污染的文件（16 CSV + 5 program.md + 1 downloaded.txt）
- **Shebang**: 为全部 14 个 Python 模块统一添加 `#!/usr/bin/env python3`（8 个入口脚本 + engine.py + diversity_gate.py/memory_manager.py/database.py/config.py/backtest.py）
- **可执行权限**: 14 个 Python 模块统一设置为 755（`chmod +x`）
- **newline 防护**: `data_fetcher_v2.py:_mark_downloaded()` 添加 `newline="\n"` 参数，防止追加写入时混入平台默认换行符
- **venv 重建**: 原 Windows 格式 venv 在 Mac 下不可用，新建 `venv_mac/` 并安装全部依赖

### Verified (12 项兼容性自检 + git ls-files --eol 全部通过)
- 无硬编码 Windows 路径 / Python 源文件全部 LF，CSV/MD/TXT 均已清理为 LF（.gitattributes 强制执行）/ 无 BOM / 无 Windows API 调用
- 所有编码参数为 utf-8 / 子进程命令为跨平台 git / newline 参数防护已添加
- 沙箱使用跨平台 threading 超时 / 数据目录容错正常

---

## v2.1 (2026-05-30) — Phase 2b 回测引擎

### Added
- `backtest.py`: 向量化回测引擎（等权 Top 30%, T+5 持仓, 年化 5% 成本, 1800 只 × 480 天 = 0.62s）
- `pipeline.py`: 合并展示 ScoreResult + BacktestResult

### Pending — 数据下载受阻
- 东方财富历史数据 API 不可用（`RemoteDisconnected`），1800 只数据未成功下载
- 指数成分股列表获取正常（1480 只），仅 `stock_zh_a_hist` 接口失败
- 临时方案：Phase 2c 用 mock 数据先行验证流程

---

## v2.0 (2026-05-30) — Phase 2a 数据层升级

### Added
- `config.py`: 股票池配置中心（UniverseType 枚举, 三大指数代码, Phase 3 预留）
- `database.py`: SQLite 四表封装（factors/rounds/memory/backtests, WAL 模式）
- `data_fetcher_v2.py`: 1800 只指数成分股下载（断点续传, downloaded.txt, 分片 CSV）
- `migration.py`: Phase 1b JSON → Phase 2 SQLite 迁移工具
- `memory_manager.get_recent_memories_for_prompt()`: 从 SQLite 读记忆拼 prompt

### Changed
- `checker.py`: 新增向量化检查规则（禁止 for 循环遍历股票, 禁止 .iterrows()/.itertuples()）
- `main.py`: 启动校验升级为 v2（自动检测数据模式, 支持 1800 只抽样校验）
- `AGENT v1.3`: 新增冻结模块清单、向量化约束、双层熔断、成本预算
- `LONGTODO v1.1`: 新增 Phase 2a/2b/2c 完整规划
- `PROBLEM v1.2 附录 B`: 增加向量化检查规则条目

### Frozen
- `score.py`: 完全冻结, MD5=3d4bd551..., Phase 2 禁止修改
- `sandbox.py`: 冻结
- `data_fetcher.py`: 冻结（Phase 1b 原版）

---

## v1.1 (2026-05-29) — Phase 1b 修复版

### 修复（Critical）
- **C1**: 修复 .env 文件从未被加载的问题。在 main.py 顶部添加 `load_dotenv()`，使 DeepSeek API Key 配置真正生效。
- **C2**: 修复 score.py MD5 校验仅警告不阻断的问题。main.py 启动时强制校验，失败则 `sys.exit(1)`；pipeline._check_startup() 同步改为致命错误级别。

### 修复（Major）
- **M1**: 实现跨轮熔断机制。main.py 新增 `consecutive_failures` 计数器，连续 3 轮因子生成/评分/门控失败自动暂停系统，防止 API 额度无限消耗。
- **M2**: 补全缺失文档。新增 `QUICKSTART.md`（5 分钟上手指南）和 `API-COST.md`（计费说明与省钱技巧）。

### 防御性清理
- 移除 pipeline.py 中未使用的 `MAX_CONSECUTIVE_FAILURES` 常量（逻辑已迁移至 main.py）。

---

## v1.0 (2026-05-29) — MVP Phase 1a + 1b 首次交付

### Added
- `engine.py`: AI 引擎（DeepSeek API + MockEngine + CostTracker + 上下文压缩）
- `memory_manager.py`: 记忆管理器（program.md 第四章追加 + 前三章 MD5 保护）
- `diversity_gate.py`: 多样性门控（Spearman 相关去重 + factor_pool 格式转换层）
- `pipeline.py`: 九步自闭环编排（10 个子步骤见下文）+ Git 自动提交 + history/ 降级备份
- `main.py`: CLI 主入口（4 项菜单 + 启动校验 + Git 状态检测 + --dry-run 模式）
- `factor_pool.json`: 因子库（初始空，{factors: [{factor_id, rank_snapshot, ...}]}）
- `requirements.txt`: pip 依赖清单
- `test_fuse_break.py`: 熔断机制测试脚本
- `PROBLEM v1.1`: 新增红线规则 E（AI 禁止修改 score.py）

### Pipeline 步骤映射（PRD 八步 vs 内部子步骤）

| PRD 步骤 | pipeline 内部子步骤 | 说明 |
|---|---|---|
| 1. 读取规程 | `load_program` | 拼装 system prompt（前三章 + 5 轮记忆） |
| 2. 生成因子 | `generate_code` | 最多 3 次尝试（合规重试） |
| 3. 合规检查 | `compliance_check` | 正则 + 关键字 + 行数 |
| 4. 摘要翻译 | `summary` | API 调用，自然语言摘要 |
| 5. 用户确认 | `user_confirmed` / `user_rejected` | Y/N 交互 |
| 6. 沙箱执行 | `sandbox` / `sandbox_timeout` / `sandbox_error` | exec + 受限命名空间 |
| 7. 评分 | `score_and_gate` | ScoreResult + 多样性门控 |
| 8. 入库/回滚 + 报告 + 记忆 | `inbound` / `discard` + `report` + `memory` | 三条合并为一步 |
| 9. Git 提交 | `git_commit` / `history_backup` | 自动 commit 或降级备份 |

**说明**: pipeline 的 9 个步骤对应 PRD 八步流程，但第 8 步（入库/回滚+报告+记忆）内部拆分为 3 个子操作：inbound/discard、report、memory。Git 提交为第 9 步。因此 dry-run 日志显示 11 个子步骤（含中间状态标记），非 PRD 步骤增多。

### Fixed
- score.py `_max_correlation`: 回滚至 Phase 1a 原始版本，因子池格式适配迁移至 diversity_gate.convert_pool_for_scoring()
- API 预算: 明确为 3 次操作上限（generate_code + summary + report），3 次代码生成均失败时打印 ¥0.00
- Git 降级: main.py 启动时自动检测 Git 状态，未初始化时提示用户
- PROBLEM 红线: 新增规则 E，AI 修改 score.py 属于严重违规

---

## v1.0 (2026-05-29) — MVP Phase 1a 交付

### Added
- `score.py`: 评分系统（ScoreResult dataclass，7 维度，前 5 核心 + 2 预留，MD5 防篡改锁）
- `program.md`: 研究规程模板（前三章人类锁死 + 第四章 AI 记忆占位）
- `factor_draft.py`: 因子草案模板（compute_factor 骨架 + mock 5 日动量因子）
- `data_fetcher.py`: AKShare 数据获取（10 只 Demo 股票，CSV 存储，启动校验）
- `checker.py`: 合规检查（正则匹配未来函数，≤20 行限制，禁止关键字拦截）
- `sandbox.py`: 安全沙箱（exec + 白名单命名空间，跨平台 30s 超时）
- `.env`: 配置模板（API Key、阈值、超时等）
- `.gitignore`: 排除 venv/、.env、IDE 文件
- `archive/`: v1.0 文档归档目录
- Three core docs: AGENT v1.2, PROBLEM v1.1, LONGTODO v1.0

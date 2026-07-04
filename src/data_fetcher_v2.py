#!/usr/bin/env python3
"""Phase 3a 数据获取 — 全量历史宽表 CSV，断点续传，黑名单，子代理并行。

宽表 CSV 格式（每个股票一个文件）：
  列: date, open, high, low, close, volume（不含 code 列）
  排序: 按 date 升序
  写入: index=False, encoding='utf-8', lineterminator='\n'
  读取: pd.read_csv(path, index_col='date', parse_dates=True)

用法:
  python data_fetcher_v2.py --fetch [--start 20100101] [--end 20251231] [--no-subagents]
  python data_fetcher_v2.py --check
"""

import os
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import akshare as ak
import pandas as pd

from config import get_index_universe, INDEX_CODES

# ── 路径常量 ─────────────────────────────────────────────────
from config import PROJECT_ROOT
DATA_DIR = PROJECT_ROOT / "data"
DOWNLOADED_FILE = DATA_DIR / "downloaded.txt"
BLACKLISTED_FILE = DATA_DIR / "blacklisted.txt"
FAILED_FILE = DATA_DIR / "failed_downloads.txt"
REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]
CSV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]
DEFAULT_START = "20100101"
DEFAULT_END = "20251231"
MIN_VALID_ROWS = 100  # 断点续传：行数 > 此值视为有效
MIN_STOCK_ROWS = 10   # 退市判定：行数 < 此值视为空数据/退市


# ── 工具函数 ─────────────────────────────────────────────────
def _to_sina_code(code: str) -> str:
    """纯数字代码 → AKShare 新浪格式: sh600519 / sz000001。"""
    code = str(code).replace("sh", "").replace("sz", "").zfill(6)
    if code.startswith(("6", "9")):
        return f"sh{code}"
    else:
        return f"sz{code}"


def _to_pure_code(code: str) -> str:
    """去除可能的前缀，统一为 6 位纯数字字符串。"""
    return str(code).replace("sh", "").replace("sz", "").zfill(6)


def _file_append(path: Path, line: str):
    """线程安全地追加一行到文件（追加模式天然原子）。"""
    with open(path, "a", newline="\n") as f:
        f.write(line.strip() + "\n")


def _load_set(file_path: Path) -> Set[str]:
    """读取文件每行去重。"""
    if not file_path.exists():
        return set()
    with open(file_path, "r") as f:
        return set(line.strip() for line in f if line.strip())


# ── 断点续传状态管理 ────────────────────────────────────────
class ResumeState:
    """管理 downloaded / blacklisted / failed 三个状态文件。"""

    def __init__(self):
        self.downloaded: Set[str] = _load_set(DOWNLOADED_FILE)
        self.blacklisted: Set[str] = _load_set(BLACKLISTED_FILE)
        self.failed: Set[str] = _load_set(FAILED_FILE)

    def is_valid_csv(self, code: str, start_date: str = "") -> bool:
        """检查已下载的 CSV 是否有效（存在、行数 > MIN_VALID_ROWS、日期覆盖 start_date）。"""
        csv_path = DATA_DIR / f"{code}.csv"
        if not csv_path.exists():
            return False
        try:
            df = pd.read_csv(csv_path)
            if len(df) <= MIN_VALID_ROWS:
                return False
            # 检查日期范围是否覆盖请求的起始日期
            if start_date:
                df["date"] = pd.to_datetime(df["date"])
                csv_start = df["date"].min()
                req_start = pd.to_datetime(start_date)
                # CSV 起始日期必须在请求起始日期的 30 天内
                if csv_start > req_start + pd.Timedelta(days=30):
                    return False
            return True
        except Exception:
            return False

    def mark_downloaded(self, code: str):
        self.downloaded.add(code)
        _file_append(DOWNLOADED_FILE, code)

    def mark_blacklisted(self, code: str):
        self.blacklisted.add(code)
        _file_append(BLACKLISTED_FILE, code)

    def mark_failed(self, code: str):
        self.failed.add(code)
        _file_append(FAILED_FILE, code)

    def clear_all(self):
        """resume=False 时清空所有记录。"""
        for f in [DOWNLOADED_FILE, BLACKLISTED_FILE, FAILED_FILE]:
            if f.exists():
                f.unlink()
        self.downloaded.clear()
        self.blacklisted.clear()
        self.failed.clear()


# ── 单只股票下载 ─────────────────────────────────────────────
def download_stock(
    code: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    retries: int = 3,
) -> Optional[str]:
    """下载单只股票全量日线数据（前复权）。

    优先使用 ak.stock_zh_a_hist，失败则 fallback 到 ak.stock_zh_a_daily。

    Args:
        code: 6 位纯数字股票代码（如 "000001"）
        start: 起始日期 YYYYMMDD
        end: 结束日期 YYYYMMDD
        retries: 网络异常重试次数

    Returns:
        None — 下载成功
        "empty" — 退市/空数据（行数 < MIN_STOCK_ROWS）
        "network" — 网络/API 异常（重试耗尽）
    """
    pure = _to_pure_code(code)
    sina = _to_sina_code(code)

    for attempt in range(retries):
        df = None

        # 主接口：stock_zh_a_hist（纯数字代码）
        try:
            df = _call_api_hist(pure, start, end)
        except Exception:
            pass

        # Fallback：stock_zh_a_daily（新浪格式）
        if df is None or df.empty:
            try:
                df = _call_api_daily(sina, start, end)
            except Exception:
                pass

        if df is not None and not df.empty:
            break

        # 重试间隔
        if attempt < retries - 1:
            time.sleep(2 + random.random() * 2)

    else:
        # 3 次都失败 — 区分网络异常 vs 空数据
        # 用最后一次尝试判断
        try:
            df = _call_api_hist(pure, start, end)
            if df is None or df.empty:
                df = _call_api_daily(sina, start, end)
        except Exception:
            pass
        return "network"

    # ── 数据清洗 ───────────────────────────────────────────
    # 列名标准化
    col_map = {
        "日期": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume",
    }
    df = df.rename(columns=col_map)

    # 确保所需列存在
    for col in CSV_COLUMNS:
        if col not in df.columns:
            return "network"  # 返回数据格式异常

    # 日期解析 & 排序
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # 截取日期范围
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)]

    # 空数据/退市检查
    if len(df) < MIN_STOCK_ROWS:
        return "empty"

    # ── 原子写入 ───────────────────────────────────────────
    csv_path = DATA_DIR / f"{pure}.csv"
    tmp_path = DATA_DIR / f"{pure}.csv.tmp"
    df[CSV_COLUMNS].to_csv(
        tmp_path, index=False, encoding="utf-8", lineterminator="\n"
    )
    os.replace(tmp_path, csv_path)

    return None  # 成功


def _call_api_hist(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """调用 ak.stock_zh_a_hist（东方财富源，纯数字代码）。"""
    result = [None]
    error = [None]

    def _target():
        try:
            result[0] = ak.stock_zh_a_hist(
                symbol=code, period="daily",
                start_date=start, end_date=end, adjust="qfq",
            )
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        return None
    if error[0]:
        return None
    return result[0]


def _call_api_daily(code_sina: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """调用 ak.stock_zh_a_daily（新浪源，sh/sz 前缀代码）。"""
    result = [None]
    error = [None]

    def _target():
        try:
            result[0] = ak.stock_zh_a_daily(
                symbol=code_sina, adjust="qfq",
            )
        except Exception as e:
            error[0] = e

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=30)
    if t.is_alive():
        return None
    if error[0]:
        return None
    return result[0]


# ── 获取排序后的股票列表 ─────────────────────────────────────
def _get_ordered_stocks(stock_list: List[str]) -> List[str]:
    """按 沪深300 → 中证500 → 中证1000 优先级排序。

    运行时动态获取指数成分股，与传入 stock_list 取交集。
    """
    stock_set = set(_to_pure_code(c) for c in stock_list)
    ordered = []
    seen = set()

    index_funcs = [
        ("000300", "沪深300"),
        ("000905", "中证500"),
        ("000852", "中证1000"),
    ]

    for idx_code, idx_name in index_funcs:
        try:
            stocks = _get_index_constituents(idx_code)
            print(f"  {idx_name} ({idx_code}): {len(stocks)} 只成分股")
            for c in stocks:
                c = _to_pure_code(c)
                if c in stock_set and c not in seen:
                    seen.add(c)
                    ordered.append(c)
        except Exception as e:
            print(f"  [警告] 获取 {idx_name} ({idx_code}) 成分股失败: {e}")

    # 补充不在三大指数中的股票
    for c in sorted(stock_set):
        if c not in seen:
            ordered.append(c)

    print(f"  排序后待处理: {len(ordered)} 只")
    return ordered


def _get_index_constituents(index_code: str) -> List[str]:
    """获取指数成分股列表。

    优先使用 index_stock_cons_csindex，失败则降级到 index_stock_cons。
    """
    # 尝试 index_stock_cons_csindex
    try:
        df = ak.index_stock_cons_csindex(index_code)
        if "成分券代码" in df.columns:
            return df["成分券代码"].astype(str).str.zfill(6).tolist()
        elif "品种代码" in df.columns:
            return df["品种代码"].astype(str).str.zfill(6).tolist()
        else:
            return df.iloc[:, 0].astype(str).str.zfill(6).tolist()
    except Exception:
        pass

    # 降级到 index_stock_cons
    try:
        df = ak.index_stock_cons(symbol=index_code)
        if "品种代码" in df.columns:
            return df["品种代码"].astype(str).str.zfill(6).tolist()
        else:
            return df.iloc[:, 0].astype(str).str.zfill(6).tolist()
    except Exception as e:
        raise RuntimeError(f"无法获取指数 {index_code} 成分股: {e}")


# ── 批量下载 ─────────────────────────────────────────────────
def fetch_all(
    stock_list: List[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    resume: bool = True,
    use_subagents: bool = True,
) -> Dict[str, str]:
    """批量下载全量历史数据。

    Args:
        stock_list: 股票代码列表
        start: 起始日期 YYYYMMDD
        end: 结束日期 YYYYMMDD
        resume: True=断点续传，False=强制重新下载
        use_subagents: True=启用子代理并行（待下载>50只时触发）

    Returns:
        {code: "success"|"empty"|"network"|"blacklisted"}
    """
    DATA_DIR.mkdir(exist_ok=True)
    state = ResumeState()

    if not resume:
        state.clear_all()
        print("resume=False: 已清空下载记录，将重新下载全部股票")

    # 排序
    print("\n获取指数成分股并排序...")
    ordered = _get_ordered_stocks(stock_list)

    # 筛选待下载列表
    pending = []
    skipped_blacklisted = 0
    skipped_valid = 0

    for code in ordered:
        if code in state.blacklisted:
            skipped_blacklisted += 1
            continue
        if resume and code in state.downloaded and state.is_valid_csv(code, start):
            skipped_valid += 1
            continue
        # resume 但 CSV 损坏 → 从 downloaded 中移除，重新下载
        if resume and code in state.downloaded:
            state.downloaded.discard(code)
        pending.append(code)

    total = len(ordered)
    print(f"总计: {total} 只 | 黑名单跳过: {skipped_blacklisted} | "
          f"已下载有效: {skipped_valid} | 待下载: {len(pending)}")

    if not pending:
        print("无需下载，全部就绪。")
        return {}

    # 子代理模式判定
    SUBAGENT_THRESHOLD = 50
    if use_subagents and len(pending) > SUBAGENT_THRESHOLD:
        print(f"\n待下载 > {SUBAGENT_THRESHOLD} 只，启用子代理并行模式")
        return _fetch_with_subagents(pending, state, start, end)
    else:
        print(f"\n待下载 ≤ {SUBAGENT_THRESHOLD} 只，使用串行模式")
        return _fetch_serial(pending, state, start, end)


def _fetch_serial(
    pending: List[str],
    state: ResumeState,
    start: str,
    end: str,
) -> Dict[str, str]:
    """串行下载所有待处理股票。"""
    results: Dict[str, str] = {}
    t_start = time.time()

    for i, code in enumerate(pending):
        t_before = time.time()
        result = download_stock(code, start, end)
        elapsed = time.time() - t_before

        if result is None:
            state.mark_downloaded(code)
            results[code] = "success"
        elif result == "empty":
            # 重试 3 次确认
            confirmed_empty = True
            for _ in range(2):
                r = download_stock(code, start, end)
                if r is None:
                    confirmed_empty = False
                    state.mark_downloaded(code)
                    results[code] = "success"
                    break
                if r == "network":
                    confirmed_empty = False
                    state.mark_failed(code)
                    results[code] = "network"
                    break
                time.sleep(1)
            if confirmed_empty:
                state.mark_blacklisted(code)
                results[code] = "blacklisted"
                print(f"跳过退市/空数据股票: {code}（已加入黑名单）")
        else:  # "network"
            state.mark_failed(code)
            results[code] = "network"

        # 进度显示（每 100 只）
        done = i + 1
        total = len(pending)
        if done % 100 == 0 or done == total:
            total_elapsed = time.time() - t_start
            avg_time = total_elapsed / done
            print(f"进度: {done}/{total} | "
                  f"已用 {total_elapsed/60:.0f}min | "
                  f"平均 {avg_time:.1f}s/只 | "
                  f"当前: {code}")

        # 请求间隔
        time.sleep(1.5 + random.random() * 2)

    return results


def _fetch_with_subagents(
    pending: List[str],
    state: ResumeState,
    start: str,
    end: str,
) -> Dict[str, str]:
    """子代理并行下载。

    按指数分组，每组分配给一个子代理并行下载。
    由于 workflow 子代理无法直接操作本地文件系统，
    此处使用多线程模拟并行：将 pending 分为 N 组，每组独立线程串行下载。
    """
    import concurrent.futures

    # 分组：按指数优先级自然分组
    n_groups = min(4, len(pending))
    group_size = (len(pending) + n_groups - 1) // n_groups
    groups = [pending[i:i + group_size] for i in range(0, len(pending), group_size)]

    print(f"分组: {n_groups} 组 | 每组约 {group_size} 只\n")

    results: Dict[str, str] = {}
    t_start = time.time()
    done_count = [0]  # 用列表以在闭包中修改

    def download_group(group: List[str], group_id: int) -> Dict[str, str]:
        """子代理任务：下载一组股票。"""
        local_results = {}
        local_state = ResumeState()  # 重新加载状态以同步
        for code in group:
            result = download_stock(code, start, end)

            if result is None:
                local_state.mark_downloaded(code)
                local_results[code] = "success"
            elif result == "empty":
                confirmed = True
                for _ in range(2):
                    r = download_stock(code, start, end)
                    if r is None:
                        confirmed = False
                        local_state.mark_downloaded(code)
                        local_results[code] = "success"
                        break
                    time.sleep(1)
                if confirmed:
                    local_state.mark_blacklisted(code)
                    local_results[code] = "blacklisted"
            else:
                local_state.mark_failed(code)
                local_results[code] = "network"

            done_count[0] += 1
            if done_count[0] % 100 == 0:
                elapsed = time.time() - t_start
                avg = elapsed / max(done_count[0], 1)
                print(f"进度: {done_count[0]}/{len(pending)} | "
                      f"已用 {elapsed/60:.0f}min | "
                      f"平均 {avg:.1f}s/只 | "
                      f"子代理: {group_id}")

            time.sleep(1.5 + random.random() * 2)

        return local_results

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_groups) as executor:
        futures = {
            executor.submit(download_group, group, i): i
            for i, group in enumerate(groups)
        }
        for future in concurrent.futures.as_completed(futures):
            group_id = futures[future]
            try:
                group_results = future.result()
                results.update(group_results)
            except Exception as e:
                print(f"[错误] 子代理 {group_id} 失败: {e}")

    return results


# ── 数据完整性检查 ───────────────────────────────────────────
def check_data() -> Tuple[bool, List[str]]:
    """检查 data/ 目录下 CSV 文件的完整性。

    Returns:
        (是否通过, 问题列表)
    """
    issues = []
    DATA_DIR.mkdir(exist_ok=True)

    # 统计
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    n_csv = len(csv_files)
    total_size = sum(f.stat().st_size for f in csv_files)

    state = ResumeState()

    print(f"=== 数据完整性检查 ===\n")
    print(f"CSV 文件数:   {n_csv}")
    print(f"downloaded:   {len(state.downloaded)}")
    print(f"blacklisted:  {len(state.blacklisted)}")
    print(f"failed:       {len(state.failed)}")
    print(f"总大小:       {total_size / 1024 / 1024:.1f} MB")

    # 抽样检查列完整性和日期范围
    if csv_files:
        sample = random.sample(csv_files, min(20, len(csv_files)))
        date_min, date_max = None, None
        bad_cols = 0
        bad_rows = 0

        for f in sample:
            try:
                df = pd.read_csv(f)
                for col in CSV_COLUMNS:
                    if col not in df.columns:
                        bad_cols += 1
                        issues.append(f"{f.name}: 缺少列 {col}")
                if len(df) < MIN_VALID_ROWS:
                    bad_rows += 1
                    issues.append(f"{f.name}: 行数不足 ({len(df)})")
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    if date_min is None or df["date"].min() < date_min:
                        date_min = df["date"].min()
                    if date_max is None or df["date"].max() > date_max:
                        date_max = df["date"].max()
            except Exception as e:
                issues.append(f"{f.name}: 读取失败 ({e})")

        print(f"\n--- 抽样检查 ({len(sample)} 个文件) ---")
        print(f"日期范围:    {date_min.date() if date_min else 'N/A'} ~ "
              f"{date_max.date() if date_max else 'N/A'}")
        print(f"列异常:      {bad_cols}")
        print(f"行数不足:    {bad_rows}")

    # 对比 downloaded vs 实际 CSV
    expected_from_downloaded = state.downloaded - state.blacklisted
    actual_csv_codes = {f.stem for f in csv_files}
    missing = expected_from_downloaded - actual_csv_codes
    extra = actual_csv_codes - expected_from_downloaded

    if missing:
        issues.append(f"downloaded 记录但 CSV 缺失: {len(missing)} 只")
    if extra:
        issues.append(f"CSV 存在但不在 downloaded: {len(extra)} 只")

    print(f"\n--- 一致性检查 ---")
    print(f"downloaded 记录但 CSV 缺失: {len(missing)}")
    print(f"CSV 存在但不在 downloaded:  {len(extra)}")

    ok = len(issues) == 0
    print(f"\n结论: {'PASS' if ok else 'WARN'} "
          f"({'无问题' if ok else f'{len(issues)} 项问题'})")
    if issues:
        for i in issues[:10]:
            print(f"  - {i}")

    return ok, issues


# ── 数据迁移：去除旧 CSV 中的 code 列 ──────────────────────────
def migrate_csv_format():
    """将现有 data/*.csv 文件从旧格式（含 code 列）迁移到新格式（纯宽表）。

    旧格式: date, code, open, high, low, close, volume
    新格式: date, open, high, low, close, volume
    """
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    migrated = 0
    skipped = 0
    errors = 0

    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if "code" not in df.columns:
                skipped += 1
                continue

            # 移除 code 列
            df = df[[c for c in CSV_COLUMNS if c in df.columns]]
            tmp_path = DATA_DIR / f"{f.name}.tmp"
            df.to_csv(tmp_path, index=False, encoding="utf-8",
                      lineterminator="\n")
            os.replace(tmp_path, f)
            migrated += 1
        except Exception:
            errors += 1

    print(f"迁移完成: {migrated} 个文件已更新 | "
          f"{skipped} 个无需迁移 | {errors} 个失败")


# ── Demo / CLI ───────────────────────────────────────────────
if __name__ == "__main__":
    if "--check" in sys.argv:
        check_data()

    elif "--migrate" in sys.argv:
        migrate_csv_format()

    elif "--fetch" in sys.argv:
        # 解析参数
        start = DEFAULT_START
        end = DEFAULT_END
        use_subagents = True

        for i, arg in enumerate(sys.argv):
            if arg == "--start" and i + 1 < len(sys.argv):
                start = sys.argv[i + 1]
            if arg == "--end" and i + 1 < len(sys.argv):
                end = sys.argv[i + 1]
            if arg == "--no-subagents":
                use_subagents = False

        print(f"=== Phase 3a 全量历史下载 ===\n")
        print(f"日期范围: {start} ~ {end}")

        print(f"\n获取指数成分股...")
        stocks = get_index_universe()
        print(f"股票总数: {len(stocks)}")

        results = fetch_all(
            stock_list=stocks,
            start=start,
            end=end,
            resume=True,
            use_subagents=use_subagents,
        )

        # 汇总
        success = sum(1 for v in results.values() if v == "success")
        empty = sum(1 for v in results.values() if v == "blacklisted")
        failed = sum(1 for v in results.values() if v == "network")
        print(f"\n=== 下载完成 ===")
        print(f"成功: {success} | 退市/空数据: {empty} | 网络失败: {failed}")

    else:
        # 默认：数据检查
        check_data()

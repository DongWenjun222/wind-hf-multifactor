from __future__ import annotations

"""行情与外部数据读取模块。

本模块集中管理：
- 主品种分钟 K 线本地缓存读取与 Wind 拉取。
- 相关品种分钟 K 线读取。
- 宏观/资金利率/指数等 Wind 日频数据读取与缓存。

因子公式和因子矩阵拼装仍放在 factors.py 与 factor_builders/ 中。
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import BacktestConfig

try:
    from WindPy import w
except ImportError:
    w = None


def ensure_wind_started() -> None:
    """启动 WindPy 连接。"""
    if w is None:
        raise ImportError("未检测到 WindPy，请先安装 Wind 客户端和 WindPy。")

    result = w.start()
    if result.ErrorCode != 0:
        raise RuntimeError(f"Wind 启动失败，错误码: {result.ErrorCode}")


def stop_wind() -> None:
    """关闭 WindPy 连接。"""
    if w is not None:
        w.stop()


def safe_symbol_name(symbol: str) -> str:
    """把 Wind 标的代码转换成适合用于文件名或因子名的安全片段。"""
    return symbol.replace(".", "_").replace("/", "_").replace("-", "_").lower()


def get_data_cache_path(config: BacktestConfig) -> Path:
    """根据标的代码和 K 线周期生成本地行情缓存路径。"""
    safe_symbol = config.symbol.replace(".", "_").replace("/", "_")
    return Path(config.data_cache_dir) / f"{safe_symbol}_{config.bar_size}min_data.csv"


def get_local_data_candidates(config: BacktestConfig) -> list[Path]:
    """返回可尝试读取的本地行情文件列表。"""
    return [get_data_cache_path(config)]


def normalize_intraday_data(data: pd.DataFrame) -> pd.DataFrame:
    """标准化分钟行情数据。"""
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data.columns = [str(col).strip().lower() for col in data.columns]
    data = data[~data.index.duplicated(keep="last")]

    required_cols = {"open", "high", "low", "close"}
    missing = required_cols.difference(data.columns)
    if missing:
        raise ValueError(f"数据缺少必要字段: {sorted(missing)}")

    if "amt" not in data.columns and "amount" in data.columns:
        data["amt"] = data["amount"]

    for optional_col in ["volume", "amt", "amount"]:
        if optional_col not in data.columns:
            data[optional_col] = np.nan

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["open", "high", "low", "close"])
    market_cols = ["open", "high", "low", "close", "volume", "amt", "amount"]
    return data[[col for col in market_cols if col in data.columns]]


def validate_local_data_bar_size(
    data: pd.DataFrame,
    config: BacktestConfig,
    data_path: Path,
) -> None:
    """校验本地数据的实际 K 线周期是否与配置一致。"""
    if len(data.index) < 3:
        return

    interval_minutes = (
        pd.Series(data.index, index=data.index)
        .sort_index()
        .diff()
        .dropna()
        .dt.total_seconds()
        .div(60.0)
    )
    interval_minutes = interval_minutes[interval_minutes > 0].round().astype(int)
    if interval_minutes.empty:
        return

    inferred_bar_size = int(interval_minutes.mode().iloc[0])
    if inferred_bar_size != config.bar_size:
        raise ValueError(
            "本地数据周期与当前配置不一致: "
            f"{data_path} 推断周期约为 {inferred_bar_size} 分钟, "
            f"当前配置为 {config.bar_size} 分钟"
        )


def load_local_intraday_data(config: BacktestConfig) -> pd.DataFrame:
    """读取并裁剪本地缓存行情数据。"""
    last_error = None
    start_time = pd.to_datetime(config.start_time)
    end_time = pd.to_datetime(config.end_time)

    for data_path in get_local_data_candidates(config):
        if not data_path.exists():
            continue

        try:
            data = pd.read_csv(data_path, index_col=0, parse_dates=True)
            data = normalize_intraday_data(data)
            validate_local_data_bar_size(data, config, data_path)
            data = data.loc[(data.index >= start_time) & (data.index <= end_time)]
            if data.empty:
                raise ValueError(
                    f"本地数据没有覆盖配置时间段: {config.start_time} 到 {config.end_time}"
                )
            print(f"优先使用本地行情数据: {data_path}")
            return data
        except Exception as exc:
            last_error = exc

    if last_error is not None:
        raise ValueError(f"本地行情数据读取失败: {last_error}") from last_error
    raise FileNotFoundError(f"没有找到可用的本地行情数据: {get_local_data_candidates(config)}")


def save_local_intraday_data(data: pd.DataFrame, config: BacktestConfig) -> Path:
    """把 Wind 拉取到的行情保存为本地 CSV 缓存。"""
    cache_path = get_data_cache_path(config)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path, encoding="utf-8-sig")
    return cache_path


def fetch_intraday_data_from_wind(config: BacktestConfig) -> pd.DataFrame:
    """从 Wind 拉取分钟行情并做标准化。"""
    options = f"BarSize={config.bar_size}"
    error_code, raw = w.wsi(
        config.symbol,
        config.price_fields,
        config.start_time,
        config.end_time,
        options,
        usedf=True,
    )
    if error_code != 0:
        raise RuntimeError(f"Wind 分钟数据获取失败，错误码: {error_code}")
    return normalize_intraday_data(raw)


def fetch_intraday_data(config: BacktestConfig) -> pd.DataFrame:
    """获取回测行情数据，优先本地缓存，失败后尝试 Wind。"""
    if config.prefer_local_data:
        try:
            return load_local_intraday_data(config)
        except (FileNotFoundError, ValueError) as exc:
            print(f"未能使用本地行情数据，将从 Wind 获取。原因: {exc}")

    print("从 Wind 获取行情数据...")
    ensure_wind_started()
    data = fetch_intraday_data_from_wind(config)
    saved_path = save_local_intraday_data(data, config)
    print(f"Wind 行情已保存到本地: {saved_path}")
    return data


def get_macro_data_cache_path(config: BacktestConfig, symbol: str) -> Path:
    """根据 Wind 宏观代码生成本地日频缓存路径。"""
    safe_symbol = safe_symbol_name(symbol)
    safe_field = safe_symbol_name(getattr(config, "macro_state_field", "close"))
    return Path(config.data_cache_dir) / f"macro_{safe_symbol}_{safe_field}_daily.csv"


def normalize_macro_daily_data(data: pd.DataFrame, field_name: str) -> pd.DataFrame:
    """标准化 Wind 日频宏观数据，统一输出 close 列。"""
    data = data.copy()
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data.columns = [str(column).strip().lower() for column in data.columns]
    lower_field = str(field_name).strip().lower()
    if lower_field in data.columns:
        value = data[lower_field]
    elif "close" in data.columns:
        value = data["close"]
    elif len(data.columns):
        value = data.iloc[:, 0]
    else:
        raise ValueError("宏观日频数据没有可用字段")
    result = pd.DataFrame({"close": pd.to_numeric(value, errors="coerce")}, index=data.index)
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])


def load_local_macro_daily_data(config: BacktestConfig, symbol: str) -> pd.DataFrame:
    """读取本地宏观日频缓存。"""
    cache_path = get_macro_data_cache_path(config, symbol)
    if not cache_path.exists():
        raise FileNotFoundError(f"没有找到宏观缓存: {cache_path}")
    data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
    return normalize_macro_daily_data(data, getattr(config, "macro_state_field", "close"))


def fetch_macro_daily_data_from_wind(config: BacktestConfig, symbol: str) -> pd.DataFrame:
    """通过 Wind wsd 拉取日频宏观、指数或利率代理数据。"""
    ensure_wind_started()
    field_name = getattr(config, "macro_state_field", "close")
    error_code, raw = w.wsd(
        symbol,
        field_name,
        config.start_time,
        config.end_time,
        "",
        usedf=True,
    )
    if error_code != 0:
        raise RuntimeError(f"Wind 宏观日频数据获取失败: {symbol}, 错误码: {error_code}")
    return normalize_macro_daily_data(raw, field_name)


def save_local_macro_daily_data(data: pd.DataFrame, config: BacktestConfig, symbol: str) -> Path:
    """保存宏观日频数据缓存。"""
    cache_path = get_macro_data_cache_path(config, symbol)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path, encoding="utf-8-sig")
    return cache_path


def fetch_macro_state_data(config: BacktestConfig) -> dict[str, pd.DataFrame]:
    """读取或拉取配置中的全部宏观状态数据。"""
    macro_data: dict[str, pd.DataFrame] = {}
    symbols = list(getattr(config, "macro_state_symbols", []) or [])
    if not getattr(config, "enable_macro_state_factors", False) or not symbols:
        return macro_data

    for symbol in symbols:
        symbol = str(symbol).strip()
        if not symbol:
            continue
        try:
            if getattr(config, "prefer_local_data", True):
                try:
                    macro_data[symbol] = load_local_macro_daily_data(config, symbol)
                    print(f"优先使用本地宏观数据: {get_macro_data_cache_path(config, symbol)}")
                    continue
                except (FileNotFoundError, ValueError):
                    pass
            data = fetch_macro_daily_data_from_wind(config, symbol)
            saved_path = save_local_macro_daily_data(data, config, symbol)
            print(f"Wind 宏观数据已保存到本地: {saved_path}")
            macro_data[symbol] = data
        except Exception as exc:
            message = f"宏观状态数据已跳过: {symbol}, 原因: {exc}"
            if getattr(config, "macro_state_strict", False):
                raise RuntimeError(message) from exc
            print(message)
    return macro_data


def fetch_related_intraday_data(config: BacktestConfig) -> dict[str, pd.DataFrame]:
    """获取或读取配置中的全部相关期货行情数据。"""
    related_data: dict[str, pd.DataFrame] = {}
    related_symbols = list(getattr(config, "related_symbols", []) or [])
    if not getattr(config, "enable_cross_asset_factors", False) or not related_symbols:
        return related_data

    main_symbol = str(config.symbol).upper()
    for symbol in related_symbols:
        symbol = str(symbol).strip()
        if not symbol or symbol.upper() == main_symbol:
            continue

        symbol_config = replace(config, symbol=symbol)
        try:
            related_data[symbol] = fetch_intraday_data(symbol_config)
        except Exception as exc:
            message = f"相关品种数据已跳过: {symbol}, 原因: {exc}"
            if getattr(config, "cross_asset_strict", False):
                raise RuntimeError(message) from exc
            print(message)

    return related_data

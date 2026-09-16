from __future__ import annotations

"""行情与外部数据读取模块。

本模块集中管理：
- 主品种分钟 K 线本地缓存读取与 Wind 拉取。
- 相关品种分钟 K 线读取。
- 宏观/资金利率/指数等 Wind 日频数据读取与缓存。

因子公式和因子矩阵拼装放在 framework/factors.py 与 framework/factor_builders/ 中。
"""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import BacktestConfig
from framework.output_layout import get_frequency_key, is_daily_frequency

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
    frequency = get_frequency_key(config)
    return (
        Path(config.data_cache_dir)
        / "market"
        / frequency
        / f"{safe_symbol}_{frequency}_data.csv"
    )


def get_local_data_candidates(config: BacktestConfig) -> list[Path]:
    """返回新版和旧版可尝试读取的本地行情文件列表。"""
    safe_symbol = config.symbol.replace(".", "_").replace("/", "_")
    frequency = get_frequency_key(config)
    filename = f"{safe_symbol}_{frequency}_data.csv"
    legacy_minute_filename = f"{safe_symbol}_{config.bar_size}min_data.csv"
    candidates = [
        get_data_cache_path(config),
        Path(config.data_cache_dir) / filename,
        Path(config.output_dir) / filename,
    ]
    if not is_daily_frequency(config):
        candidates.extend(
            [
                Path(config.data_cache_dir) / "market" / legacy_minute_filename,
                Path(config.data_cache_dir) / legacy_minute_filename,
                Path(config.output_dir) / legacy_minute_filename,
            ]
        )
    return list(dict.fromkeys(candidates))


def normalize_intraday_data(data: pd.DataFrame) -> pd.DataFrame:
    """标准化分钟行情数据。"""
    if data is None or data.empty:
        raise ValueError(
            "行情数据为空，请检查 Wind 品种代码、交易所后缀和请求时间段。"
            "股指期货应使用 IF.CFE、IH.CFE、IC.CFE 或 IM.CFE。"
        )
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

    for optional_col in ["volume", "amt", "amount", "oi", "open_interest", "settle"]:
        if optional_col not in data.columns:
            data[optional_col] = np.nan

    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=["open", "high", "low", "close"])
    if data.empty:
        raise ValueError("行情 OHLC 字段没有任何完整记录，不能继续生成因子或回测。")
    market_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amt",
        "amount",
        "oi",
        "open_interest",
        "settle",
    ]
    return data[[col for col in market_cols if col in data.columns]]


def filter_completed_daily_bars(
    data: pd.DataFrame,
    config: BacktestConfig,
    now: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """仅保留已经完成的日线，防止实盘信号读取当天未收盘数据。"""
    if not is_daily_frequency(config) or data.empty:
        return data
    current = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    ready_text = str(getattr(config, "daily_bar_ready_time", "15:30") or "15:30")
    ready_offset = pd.Timestamp(f"2000-01-01 {ready_text}") - pd.Timestamp("2000-01-01")
    latest_complete_date = current.normalize()
    if current < current.normalize() + ready_offset:
        latest_complete_date -= pd.Timedelta(days=1)
    completed = data.index.normalize() <= latest_complete_date
    return data.loc[completed].copy()


def validate_local_data_bar_size(
    data: pd.DataFrame,
    config: BacktestConfig,
    data_path: Path,
) -> None:
    """校验本地数据的实际 K 线周期是否与配置一致。"""
    if is_daily_frequency(config):
        return
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
            data = filter_completed_daily_bars(data, config)
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
    """从 Wind 拉取当前频率行情并做标准化。"""
    if is_daily_frequency(config):
        fields = str(getattr(config, "daily_price_fields", config.price_fields))
        error_code, raw = w.wsd(
            config.symbol,
            fields,
            config.start_time,
            config.end_time,
            "",
            usedf=True,
        )
        if error_code != 0:
            raise RuntimeError(f"Wind 日频数据获取失败，错误码: {error_code}")
        daily_data = filter_completed_daily_bars(normalize_intraday_data(raw), config)
        if daily_data.empty:
            raise ValueError(
                f"Wind 未返回 {config.symbol} 的已完成日线，"
                "请检查代码、时间范围或 daily_bar_ready_time。"
            )
        return daily_data

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
    """获取当前研究频率行情，优先本地缓存，失败后尝试 Wind。"""
    if config.prefer_local_data:
        try:
            return load_local_intraday_data(config)
        except (FileNotFoundError, ValueError) as exc:
            print(f"未能使用本地行情数据，将从 Wind 获取。原因: {exc}")

    print(f"从 Wind 获取 {get_frequency_key(config)} 行情数据...")
    ensure_wind_started()
    data = fetch_intraday_data_from_wind(config)
    if data.empty:
        raise ValueError(
            f"Wind 未返回 {config.symbol} 的有效行情，已停止写入空缓存。"
        )
    saved_path = save_local_intraday_data(data, config)
    print(f"Wind 行情已保存到本地: {saved_path}")
    return data


def get_macro_data_cache_path(config: BacktestConfig, symbol: str) -> Path:
    """根据 Wind 宏观代码生成本地日频缓存路径。"""
    safe_symbol = safe_symbol_name(symbol)
    safe_field = safe_symbol_name(getattr(config, "macro_state_field", "close"))
    return Path(config.data_cache_dir) / "macro" / f"macro_{safe_symbol}_{safe_field}_daily.csv"


def get_macro_data_cache_candidates(config: BacktestConfig, symbol: str) -> list[Path]:
    """返回新版和旧版宏观日频缓存候选。"""
    filename = get_macro_data_cache_path(config, symbol).name
    candidates = [
        get_macro_data_cache_path(config, symbol),
        Path(config.data_cache_dir) / filename,
        Path(config.output_dir) / filename,
    ]
    return list(dict.fromkeys(candidates))


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
    for cache_path in get_macro_data_cache_candidates(config, symbol):
        if cache_path.exists():
            data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return normalize_macro_daily_data(data, getattr(config, "macro_state_field", "close"))
    raise FileNotFoundError(f"没有找到宏观缓存: {get_macro_data_cache_candidates(config, symbol)}")


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


def normalize_external_source(source: dict | str, default_lag: int) -> dict[str, str | int]:
    """规范化外部日频数据源配置。"""
    if isinstance(source, str):
        source = {"name": source, "symbol": source, "field": "close"}
    if not isinstance(source, dict):
        raise ValueError(f"外部日频数据源配置必须是 dict 或字符串: {source}")

    symbol = str(source.get("symbol", "")).strip()
    if not symbol:
        raise ValueError(f"外部日频数据源缺少 symbol: {source}")
    name = str(source.get("name") or symbol).strip()
    field = str(source.get("field") or "close").strip()
    lag = int(source.get("lag", default_lag) or 0)
    return {"name": name, "symbol": symbol, "field": field, "lag": max(0, lag)}


def get_external_daily_cache_path(config: BacktestConfig, source: dict[str, str | int]) -> Path:
    """根据外部数据源配置生成本地缓存路径。"""
    safe_name = safe_symbol_name(str(source["name"]))
    safe_symbol = safe_symbol_name(str(source["symbol"]))
    safe_field = safe_symbol_name(str(source.get("field", "close")))
    return (
        Path(config.data_cache_dir)
        / "external"
        / f"external_{safe_name}_{safe_symbol}_{safe_field}_daily.csv"
    )


def get_external_daily_cache_candidates(
    config: BacktestConfig,
    source: dict[str, str | int],
) -> list[Path]:
    """返回新版和旧版外部日频缓存候选。"""
    filename = get_external_daily_cache_path(config, source).name
    candidates = [
        get_external_daily_cache_path(config, source),
        Path(config.data_cache_dir) / filename,
        Path(config.output_dir) / filename,
    ]
    return list(dict.fromkeys(candidates))


def normalize_external_daily_data(data: pd.DataFrame, field_name: str) -> pd.DataFrame:
    """标准化 Wind 外部日频数据，统一输出 value 列。"""
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
        raise ValueError("外部日频数据没有可用字段")
    result = pd.DataFrame({"value": pd.to_numeric(value, errors="coerce")}, index=data.index)
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=["value"])


def load_local_external_daily_data(
    config: BacktestConfig,
    source: dict[str, str | int],
) -> pd.DataFrame:
    """读取本地外部日频数据缓存。"""
    for cache_path in get_external_daily_cache_candidates(config, source):
        if cache_path.exists():
            data = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return normalize_external_daily_data(data, str(source.get("field", "close")))
    raise FileNotFoundError(
        f"没有找到外部日频缓存: {get_external_daily_cache_candidates(config, source)}"
    )


def fetch_external_daily_data_from_wind(
    config: BacktestConfig,
    source: dict[str, str | int],
) -> pd.DataFrame:
    """通过 Wind wsd 拉取外部日频数据。"""
    ensure_wind_started()
    symbol = str(source["symbol"])
    field_name = str(source.get("field", "close") or "close")
    error_code, raw = w.wsd(
        symbol,
        field_name,
        config.start_time,
        config.end_time,
        "",
        usedf=True,
    )
    if error_code != 0:
        raise RuntimeError(f"Wind 外部日频数据获取失败: {symbol}, 字段: {field_name}, 错误码: {error_code}")
    return normalize_external_daily_data(raw, field_name)


def save_local_external_daily_data(
    data: pd.DataFrame,
    config: BacktestConfig,
    source: dict[str, str | int],
) -> Path:
    """保存外部日频数据缓存。"""
    cache_path = get_external_daily_cache_path(config, source)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(cache_path, encoding="utf-8-sig")
    return cache_path


def fetch_external_daily_data(config: BacktestConfig) -> dict[str, pd.DataFrame]:
    """读取或拉取配置中的全部外部日频数据。"""
    external_data: dict[str, pd.DataFrame] = {}
    raw_sources = list(getattr(config, "external_daily_sources", []) or [])
    if not getattr(config, "enable_external_daily_factors", False) or not raw_sources:
        return external_data

    default_lag = max(0, int(getattr(config, "external_daily_lag_daily_bars", 1) or 0))
    for raw_source in raw_sources:
        try:
            source = normalize_external_source(raw_source, default_lag)
            source_name = str(source["name"])
            if getattr(config, "prefer_local_data", True):
                try:
                    data = load_local_external_daily_data(config, source)
                    data.attrs["lag_daily_bars"] = int(source.get("lag", default_lag) or 0)
                    external_data[source_name] = data
                    print(f"优先使用本地外部日频数据: {get_external_daily_cache_path(config, source)}")
                    continue
                except (FileNotFoundError, ValueError):
                    pass
            data = fetch_external_daily_data_from_wind(config, source)
            data.attrs["lag_daily_bars"] = int(source.get("lag", default_lag) or 0)
            saved_path = save_local_external_daily_data(data, config, source)
            print(f"Wind 外部日频数据已保存到本地: {saved_path}")
            external_data[source_name] = data
        except Exception as exc:
            message = f"外部日频数据已跳过: {raw_source}, 原因: {exc}"
            if getattr(config, "external_daily_strict", False):
                raise RuntimeError(message) from exc
            print(message)
    return external_data


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


def resolve_related_symbols(config: BacktestConfig) -> list[str]:
    """按主品种解析跨品种数据源，并保持配置顺序去重。"""
    main_symbol = str(config.symbol).strip().upper().replace("_", ".")
    overrides = getattr(config, "related_symbols_by_symbol", {}) or {}
    normalized_overrides = {
        str(symbol).strip().upper().replace("_", "."): values
        for symbol, values in overrides.items()
    }
    configured = normalized_overrides.get(
        main_symbol,
        getattr(config, "related_symbols", []) or [],
    )

    resolved: list[str] = []
    seen: set[str] = set()
    for symbol in configured:
        normalized = str(symbol).strip().upper().replace("_", ".")
        if not normalized or normalized == main_symbol or normalized in seen:
            continue
        resolved.append(normalized)
        seen.add(normalized)
    return resolved


def fetch_related_intraday_data(config: BacktestConfig) -> dict[str, pd.DataFrame]:
    """获取或读取配置中的全部相关期货行情数据。"""
    related_data: dict[str, pd.DataFrame] = {}
    related_symbols = resolve_related_symbols(config)
    if not getattr(config, "enable_cross_asset_factors", False) or not related_symbols:
        return related_data

    for symbol in related_symbols:
        symbol_config = replace(config, symbol=symbol)
        try:
            related_data[symbol] = fetch_intraday_data(symbol_config)
        except Exception as exc:
            message = f"相关品种数据已跳过: {symbol}, 原因: {exc}"
            if getattr(config, "cross_asset_strict", False):
                raise RuntimeError(message) from exc
            print(message)

    return related_data

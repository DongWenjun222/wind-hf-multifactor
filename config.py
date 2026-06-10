from __future__ import annotations

from dataclasses import dataclass, field
import datetime as dt
from typing import Optional


# 中国商品期货中相对常用、流动性较好的主力连续合约代码池。
# 这里使用 Wind 常见主力连续代码写法，例如 C.DCE、CU.SHF、TA.CZC。
# 如果某些新品种在当前 Wind 权限或本地缓存中不可用，数据读取层会按配置跳过或报错。
LIQUID_COMMODITY_MAIN_SYMBOLS_BY_EXCHANGE: dict[str, list[str]] = {
    # 上海期货交易所：有色、贵金属、黑色建材、能源化工。
    "SHF": [
        "CU.SHF",
        "AL.SHF",
        "ZN.SHF",
        "PB.SHF",
        "NI.SHF",
        "SN.SHF",
        "AO.SHF",
        "AU.SHF",
        "AG.SHF",
        "RB.SHF",
        "HC.SHF",
        "SS.SHF",
        "RU.SHF",
        "BR.SHF",
        "BU.SHF",
        "FU.SHF",
        "SP.SHF",
    ],
    # 大连商品交易所：农产品、油脂油料、黑色、化工、畜牧。
    "DCE": [
        "A.DCE",
        "B.DCE",
        "C.DCE",
        "CS.DCE",
        "M.DCE",
        "Y.DCE",
        "P.DCE",
        "JD.DCE",
        "LH.DCE",
        "L.DCE",
        "V.DCE",
        "PP.DCE",
        "EG.DCE",
        "EB.DCE",
        "PG.DCE",
        "I.DCE",
        "J.DCE",
        "JM.DCE",
    ],
    # 郑州商品交易所：农产品、软商品、煤化工、建材化工。
    "CZC": [
        "CF.CZC",
        "SR.CZC",
        "OI.CZC",
        "RM.CZC",
        "AP.CZC",
        "CJ.CZC",
        "TA.CZC",
        "MA.CZC",
        "FG.CZC",
        "UR.CZC",
        "SA.CZC",
        "PF.CZC",
        "PK.CZC",
        "SF.CZC",
        "SM.CZC",
        "SH.CZC",
        "PX.CZC",
    ],
    # 上海国际能源交易中心：原油、低硫燃料油、20号胶、国际铜、集运指数。
    "INE": [
        "SC.INE",
        "LU.INE",
        "NR.INE",
        "BC.INE",
        "EC.INE",
    ],
    # 广州期货交易所：新能源和新材料相关品种。
    "GFE": [
        "SI.GFE",
        "LC.GFE",
        "PS.GFE",
    ],
}


LIQUID_COMMODITY_MAIN_SYMBOLS: list[str] = [
    symbol
    for symbols in LIQUID_COMMODITY_MAIN_SYMBOLS_BY_EXCHANGE.values()
    for symbol in symbols
]


SYMBOL_UNIVERSES: dict[str, list[str]] = {
    "liquid_commodity": LIQUID_COMMODITY_MAIN_SYMBOLS,
    "commodity_liquid": LIQUID_COMMODITY_MAIN_SYMBOLS,
    "all_liquid_commodity": LIQUID_COMMODITY_MAIN_SYMBOLS,
    "all_commodity": LIQUID_COMMODITY_MAIN_SYMBOLS,
}


def resolve_symbol_universe(symbols: str | list[str]) -> list[str]:
    """解析品种池别名或手工品种列表，返回去重后的 Wind 代码列表。"""
    if isinstance(symbols, str):
        raw_items = [item.strip() for item in symbols.split(",") if item.strip()]
    else:
        raw_items = [str(item).strip() for item in symbols if str(item).strip()]

    resolved: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        key = item.lower()
        candidates = SYMBOL_UNIVERSES.get(key, [item])
        for symbol in candidates:
            normalized = str(symbol).strip().upper()
            if normalized and normalized not in seen:
                resolved.append(normalized)
                seen.add(normalized)
    return resolved


@dataclass
class BacktestConfig:
    # ----------------------------
    # 数据读取与基础回测设置
    # ----------------------------

    # 回测标的代码。当前为万得或本地数据中使用的合约代码，例如 "C.DCE"。
    symbol: str = "C.DCE"

    # 多品种批量回测标的列表。仅 multi_symbol_backtest.py 使用；
    # 单独运行 single_factor_backtest.py 或 composite_factor_backtest.py 时仍只使用 symbol。
    symbols: list[str] = field(
        default_factory=lambda: LIQUID_COMMODITY_MAIN_SYMBOLS.copy()
    )

    # 多品种批量回测时是否先为每个品种运行单因子流程并更新该品种自己的 active 因子库。
    multi_symbol_run_single_factor: bool = True

    # 多品种批量回测时是否为每个品种运行 XGBoost 综合因子流程。
    multi_symbol_run_composite: bool = True

    # 多品种批量回测时，每个品种是否使用独立输出目录。
    # 开启后输出目录形如 output_dir/by_symbol/C_DCE/，避免不同品种的因子库和结果互相覆盖。
    multi_symbol_separate_output_dirs: bool = True

    # 多品种批量回测的品种子目录名。仅在 multi_symbol_separate_output_dirs=True 时生效。
    multi_symbol_output_subdir: str = "by_symbol"

    # 多品种批量回测是否跳过已经完成的品种流程。
    # 开启后，如果品种目录中已经存在对应结果文件，就直接复用，适合长任务中断后的断点续跑。
    multi_symbol_skip_existing: bool = True

    # 多品种组合权重是否使用滚动历史窗口估计。
    # 开启后 inverse_vol / positive_sharpe 不再使用整段测试集计算权重，避免组合层未来函数。
    multi_symbol_use_rolling_portfolio_weights: bool = True

    # 多品种滚动组合权重的历史窗口长度，单位为 K 线根数。
    multi_symbol_portfolio_weight_window: int = 480

    # 估计滚动组合权重所需的最少历史样本数。
    # 样本不足时会自动退化为当期可用品种等权。
    multi_symbol_portfolio_min_weight_samples: int = 120

    # 单一品种在组合中的最大权重上限。
    # 设为 1 表示不限制；例如 0.5 表示任一品种权重最多 50%。
    multi_symbol_portfolio_max_symbol_weight: float = 0.5

    # 多品种组合是否启用横截面机会选择。
    # 开启后，每根 K 线只保留机会评分最高的一部分品种参与组合，弱机会品种权重会被压到 0。
    multi_symbol_use_opportunity_selection: bool = True

    # 每根 K 线最多交易多少个机会最强的品种。
    # 设为 0 或负数表示不限制数量，只使用最低机会评分过滤。
    multi_symbol_opportunity_top_n: int = 8

    # 横截面机会评分的最低阈值。
    # 0 表示只要有有效评分即可；如果希望更保守，可以设为 0.02、0.05 等。
    multi_symbol_opportunity_min_score: float = 0.0

    # 机会评分模式。
    # edge：使用上一根综合模型校准概率差绝对值；
    # edge_probability：使用上一根概率差绝对值 * 方向概率；
    # edge_rank：使用上一根概率差绝对值 * 近期置信度分位；
    # position：使用当前实际仓位绝对值作为兜底机会强度。
    multi_symbol_opportunity_score_mode: str = "edge_probability"

    # 机会评分对原组合权重的放大幂次。
    # 1 表示线性加权；大于 1 会让权重更集中到最高机会品种。
    multi_symbol_opportunity_weight_power: float = 1.0

    # 多品种组合是否启用波动率目标控制。
    # 开启后会根据组合过去一段时间的已实现波动率动态缩放总风险暴露。
    multi_symbol_use_vol_target: bool = True

    # 多品种组合目标年化波动率。0.12 表示目标年化波动约 12%。
    multi_symbol_target_annual_vol: float = 0.12

    # 估计组合已实现波动率的滚动窗口，单位为 K 线根数。
    multi_symbol_vol_target_window: int = 240

    # 组合风控允许的最大风险杠杆倍数。
    # 设为 1 表示只降风险不加杠杆；大于 1 允许低波动期适度放大。
    multi_symbol_max_portfolio_leverage: float = 1.0

    # 多品种组合是否启用回撤降仓。
    multi_symbol_use_drawdown_control: bool = True

    # 组合回撤达到该水平后开始线性降仓。例如 -0.05 表示回撤超过 5% 开始降风险。
    multi_symbol_drawdown_reduce_start: float = -0.05

    # 组合回撤达到该水平后降到 0 仓位。例如 -0.12 表示回撤超过 12% 暂停组合风险暴露。
    multi_symbol_drawdown_stop: float = -0.12

    # 回测开始时间。格式建议使用 "YYYY-MM-DD HH:MM:SS"。
    start_time: str = "2025-01-02 09:00:00"

    # 回测结束时间。默认使用运行脚本时的当前时间，适合持续增量更新数据。
    end_time: str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # K线周期，单位为分钟。当前为 30 分钟线。
    bar_size: int = 30

    # 从万得或本地数据中需要使用的行情字段。
    # 开盘价/最高价/最低价/收盘价用于收益和形态类因子，成交量/成交额用于量价类因子。
    price_fields: str = "open,high,low,close,volume,amt"

    # 单因子信号阈值。因子标准化分数高于该值做多，低于负该值做空，中间为空仓。
    # 调大：信号更少、更保守；调小：信号更多、交易更频繁。
    signal_threshold: float = 0.7

    # 基础滚动标准化窗口。部分因子会使用该窗口做滚动均值/标准差标准化。
    # 调大：因子更平滑、更慢；调小：因子更灵敏、噪声也可能更多。
    zscore_window: int = 120

    # 单边手续费，单位 bps。1 bps = 0.01%。当前设为 0，表示不考虑手续费。
    commission_bps: float = 0#0.5#0.5

    # 单边滑点，单位 bps。当前设为 0，表示不考虑滑点。
    slippage_bps: float = 0#0.5

    # 综合回测成本压力测试档位，单位 bps。
    # 每个值表示“手续费+滑点”的单边总成本，用于在不重新训练模型的情况下重算策略表现。
    cost_stress_bps_list: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 3.0]
    )

    # 年化交易日数量。用于把单根K线收益年化成夏普、年化收益、年化波动。
    annual_trading_days: int = 252

    # 回测输出目录。汇总表、明细表、图表、因子库等都会写入该目录。
    output_dir: str = "wind_hf_multifactor_output"

    # 是否为每次运行额外生成独立实验目录。
    # 开启时会在输出目录的 runs 子目录下保存运行配置、结果快照和因子库快照。
    enable_experiment_run_dirs: bool = True

    # 手工指定实验编号。留空表示运行时自动用当前时间生成。
    run_id: Optional[str] = None

    # 行情数据缓存目录。优先复用本地缓存，避免每次从万得重新拉取。
    data_cache_dir: str = "wind_hf_multifactor_output"

    # 是否优先读取本地缓存数据。开启表示先找本地 CSV，找不到再尝试万得。
    prefer_local_data: bool = True

    # ----------------------------
    # 跨品种数据与跨品种因子设置
    # ----------------------------

    # 是否启用跨品种因子。开启时会尝试读取相关品种列表的行情并生成联动因子。
    enable_cross_asset_factors: bool = True

    # 相关品种列表。建议优先选择产业链、替代品、同板块或可能领先主品种的期货。
    # 如果本地没有缓存，会按“优先本地数据”的配置逻辑尝试从万得获取并缓存。
    related_symbols: list[str] = field(
        default_factory=lambda: ["CS.DCE", "M.DCE", "Y.DCE", "P.DCE"]
    )

    # 跨品种因子滚动窗口。窗口越短越敏感，窗口越长越稳定。
    cross_asset_factor_windows: list[int] = field(
        default_factory=lambda: [2, 3, 5, 8, 13, 21, 34, 55]
    )

    # 相关品种数据对齐到主品种时间轴时，最多向前填充多少根K线。
    # 只允许向前填充，不允许向后填充，避免把未来相关品种数据填到过去。
    cross_asset_max_ffill_bars: int = 2

    # 跨品种因子最大数量。留空表示不截断；设置数字可控制因子数量膨胀。
    cross_asset_max_factors: Optional[int] = None

    # 相关品种读取失败时是否直接中断。
    # 关闭：跳过失败品种并继续跑主流程；开启：任一相关品种失败就报错。
    cross_asset_strict: bool = False

    # ----------------------------
    # 资金利率与宏观状态因子设置
    # ----------------------------

    # 是否启用资金利率、指数和宏观状态类因子。
    # 这些数据通过 Wind 日频接口读取，默认整体滞后一日后再对齐到 30 分钟K线，
    # 避免盘中使用当天收盘后才知道的日频数据。
    enable_macro_state_factors: bool = True

    # Wind 宏观/市场状态代理代码列表。可以按自己的 Wind 权限和研究方向调整。
    # 默认给出常用市场状态代理：A股宽基、创业板、人民币汇率和中国债券指数。
    # 如果某个代码在你的 Wind 权限中不可用，默认会跳过，不中断主流程。
    macro_state_symbols: list[str] = field(
        default_factory=lambda: ["000300.SH", "000001.SH", "399006.SZ", "USDCNY.IB", "CBA00101.CS"]
    )

    # Wind 日频字段。多数指数/汇率可以使用 close；利率类数据可按 Wind 字段实际情况调整。
    macro_state_field: str = "close"

    # 宏观因子的滚动窗口，单位为日频数据点。
    macro_state_windows: list[int] = field(
        default_factory=lambda: [3, 5, 10, 20, 40, 60]
    )

    # 宏观日频数据整体滞后几天再对齐到分钟线。
    # 建议至少为 1，避免使用当天尚未收盘的宏观/指数日频数据。
    macro_state_lag_daily_bars: int = 1

    # 宏观数据读取失败时是否中断。
    # False 表示跳过失败代码继续运行；True 表示任一宏观代码失败就报错。
    macro_state_strict: bool = False

    # ----------------------------
    # 单因子测试与因子库设置
    # ----------------------------

    # 单因子测试范围。
    # "all"：测试全部因子；"new"：只测试编号 >= single_factor_new_factor_start_index 的新因子；
    # "selected"：只测试 single_factor_selected_factors 中指定的因子。
    single_factor_scope: str = "new"  # 可选："all"、"new"、"selected"

    # 当 single_factor_scope="selected" 时使用的单因子名单。
    # 留空表示不额外指定；如果启用 selected，建议填入因子列名列表。
    single_factor_selected_factors: Optional[list[str]] =  None

    # 因子库最多保留的活跃因子数量。当前为只保留综合排名前 200 个。
    single_factor_keep_top_n: int = 200

    # 是否启用 active 因子库的家族配额控制。
    # 开启后，即使某一类因子整体排名靠前，也不会无限占满 active 因子库。
    factor_library_enable_family_quota: bool = True

    # 各因子家族在 active 因子库中的数量上限。
    # 这些上限只限制 active 入库数量，不影响单因子回测和 rejected/all 记录。
    # None 或缺失家族表示不单独限制；所有家族合计仍受 single_factor_keep_top_n 约束。
    factor_library_family_max_counts: dict[str, Optional[int]] = field(
        default_factory=lambda: {
            "basic": 10,
            "parametric": 60,
            "non_cross_complex": 45,
            "cross_asset": 45,
            "calendar": 25,
            "macro_state": 15,
        }
    )

    # 是否为所有单因子生成报告图。
    # 关闭：先完整回测，再只给排名前 N 的入库因子出图，速度更快、文件更少；
    # 开启：每个被测试的因子都生成图，适合少量因子的精细检查。
    single_factor_plot_all: bool = False

    # 当不全量出图时，只给入库排名前 N 的因子生成图表。
    # 设为 0 可完全关闭单因子图表生成。
    single_factor_plot_top_n: int = 50

    # 因子库子目录名。如果是相对路径，会放在 output_dir 下面。
    factor_library_dir: str = "factor_library"

    # 综合因子是否使用冻结版 active 因子库。
    # 开启后 composite_factor_backtest.py 不再读取当前 factor_library/active_factors.csv，
    # 而是读取 frozen_active_library_path 指向的历史快照，便于做严格样本外检验。
    use_frozen_active_library: bool = False

    # 冻结版 active 因子库路径。可以是绝对路径，也可以是相对 output_dir 的路径。
    # 例如 "runs/20260510_120000_single/active_factors_snapshot.csv"。
    frozen_active_library_path: Optional[str] = None

    # 因子入库的最低测试夏普要求。低于该值的因子会被拒绝入库。
    factor_library_min_sharpe: float = 1.0

    # 因子入库的最低测试累计收益要求。低于该值的因子会被拒绝入库。
    factor_library_min_total_return: float = 0.0

    # 因子入库的最低训练夏普要求。
    # 和测试门槛同时生效，避免只在测试段偶然表现好、训练段完全无效的因子入库。
    factor_library_min_train_sharpe: float = 1.0

    # 因子入库的最低训练累计收益要求。
    factor_library_min_train_total_return: float = 0.0

    # 因子入库的最低训练胜率要求。
    # 使用严格大于判断，例如 0.5 表示训练胜率必须大于 50%，等于 50% 不入库。
    factor_library_min_train_win_rate: float = 0.5

    # 因子入库的最低测试侧胜率要求。
    # 若存在验证集表现，则优先约束验证胜率；否则回退约束测试胜率。
    factor_library_min_test_win_rate: float = 0.5

    # 因子入库的最低测试交易次数要求。
    # 用于过滤“只交易极少数几次但偶然收益很好”的不稳定因子；0 表示不限制。
    factor_library_min_test_trades: int = 0

    # 因子入库的最低训练交易次数要求；0 表示不限制。
    factor_library_min_train_trades: int = 0

    # 因子入库的最低测试信号覆盖率要求。
    # 例如 0.05 表示测试集中至少 5% K线有非零信号；0 表示不限制。
    factor_library_min_test_signal_coverage: float = 0.0

    # 因子入库的最低训练信号覆盖率要求；0 表示不限制。
    factor_library_min_train_signal_coverage: float = 0.0

    # 因子入库的最低初筛 RankIC 要求。
    # None 表示不强制过滤；设置为 0 或正数后，会要求训练/验证综合后的 RankIC 达标。
    factor_library_min_selection_rank_ic: Optional[float] = None

    # 因子入库的最低初筛分组单调性要求。
    # None 表示不强制过滤；越接近 1，代表 Q1 到 Q5 的未来收益越接近严格单调递增。
    factor_library_min_selection_monotonicity: Optional[float] = None

    # 因子入库允许的最大测试回撤。
    # None 表示不限制；例如 -0.10 表示测试最大回撤低于 -10% 的因子会被拒绝。
    factor_library_max_test_drawdown: Optional[float] = None

    # 因子入库允许的最大训练回撤。None 表示不限制。
    factor_library_max_train_drawdown: Optional[float] = None

    # 因子去重相关性阈值。候选因子与已入库因子的相关性高于该值时会被拒绝。
    # 调低：因子库更多样，但可能错过同类强因子；调高：保留更多相似因子。
    factor_library_max_corr: float = 0.80

    # 是否使用因子原始连续值计算相关性，用于识别数值形态相似的因子。
    factor_library_use_value_corr: bool = True

    # 是否使用因子多空信号计算相关性，用于识别实际交易行为相似的因子。
    factor_library_use_signal_corr: bool = True
    
    # 当 single_factor_scope="new" 时，从这个因子编号开始测试。
    # 适合自动生成新因子后，只增量回测新加入的一批因子。
    single_factor_new_factor_start_index: int = 126978

    # 不同品种可以单独覆盖 single_factor_new_factor_start_index。
    # 适合多品种持续扩因子时使用：例如 C.DCE 已经测到 126978，
    # 但 P.DCE 可能只测到 106885，可以在这里单独指定。
    # key 支持 "C.DCE"、"C_DCE"、"c_dce" 这几种写法，程序会自动归一化匹配。
    single_factor_new_factor_start_index_by_symbol: dict[str, int] = field(
        default_factory=lambda: {
            "C.DCE": 126978,
            "M.DCE": 126978,
            "Y.DCE": 126978,
            "P.DCE": 126978,
            "CS.DCE": 126978,
        }
    )

    # 是否启用因子自动淘汰清单。
    # 多品种回测结束后，如果某个因子在足够多品种中都表现很差，程序会把它写入淘汰清单；
    # 后续 factors.py 构建因子矩阵时会自动跳过这些因子，避免候选库越来越臃肿。
    enable_factor_pruning: bool = True

    # 因子淘汰清单文件。相对路径会放在 output_dir 下；多品种独立目录运行时仍使用总 output_dir 下的全局清单。
    factor_prune_list_path: str = "factor_prune_list.csv"

    # 至少需要在多少个品种上完成单因子测试，才允许判断该因子是否应该淘汰。
    # 设置得越大越保守，避免因为少数品种短样本表现差就过早删除。
    factor_pruning_min_tested_symbols: int = 2

    # 如果某个因子在任一品种的 active_factors.csv 中出现，则默认保护，不加入淘汰清单。
    # 这符合“不是必须所有品种都 active，但至少某处有价值就保留”的原则。
    factor_pruning_keep_if_active_any_symbol: bool = True

    # 保留门槛：只要某个因子在任一品种上的初筛夏普达到该值，就不会被淘汰。
    # 0 表示只要某个品种上有非负预测/交易贡献，就先保留观察。
    factor_pruning_keep_min_sharpe: float = 0.0

    # 保留门槛：只要某个因子在任一品种上的初筛累计收益达到该值，就不会被淘汰。
    factor_pruning_keep_min_total_return: float = 0.0

    # 是否在 build_factors 阶段应用淘汰清单。
    # 关闭后仍会生成淘汰报告，但不会实际从候选因子矩阵中过滤。
    factor_pruning_apply_to_build: bool = True

    # 综合因子或 XGBoost 默认使用的人工精选因子列表。
    # 注释中的数字是因子编号，方便和单因子汇总表、因子库文件对应。
    selected_factors: Optional[list[str]] = None

    # ----------------------------
    # 单因子训练/测试切分与筛选
    # ----------------------------

    # 是否在训练集上自动判断因子方向。
    # 开启：训练集表现正向则顺用，反向更好则乘以 -1；关闭：全部按原始方向使用。
    auto_detect_factor_direction: bool = True

    # 早期单因子筛选用的最低夏普要求。当前主要因子库逻辑更依赖 factor_library_min_sharpe。
    min_select_sharpe: float = 1

    # 早期单因子筛选用的最低累计收益要求。当前主要因子库逻辑更依赖 factor_library_min_total_return。
    min_select_total_return: float = 0.0

    # 单因子训练/测试切分比例。0.7 表示前 70% 样本用于判断方向和训练内表现，后 30% 用于样本外测试。
    auto_select_train_ratio: float = 0.7

    # 单因子验证集比例。验证集位于训练集之后、最终测试集之前。
    # 因子入库优先使用验证集表现，最终测试集只作为留存评估，降低测试集污染。
    auto_select_validation_ratio: float = 0.15

    # ----------------------------
    # XGBoost 综合因子模型设置。
    # ----------------------------

    # 综合因子滚动训练时要对比的模型列表。
    # xgboost 是原主模型；logistic_regression 是线性基准；
    # random_forest 和 extra_trees 是两类不依赖额外安装包的树集成模型。
    composite_model_names: list[str] = field(
        default_factory=lambda: [
            "xgboost",
            "logistic_regression",
            "random_forest",
            "extra_trees",
        ]
    )

    # sklearn 树模型的树数量，供 random_forest 和 extra_trees 使用。
    composite_sklearn_n_estimators: int = 120

    # sklearn 模型使用的并行线程数。-1 表示尽量使用全部 CPU 线程。
    composite_sklearn_n_jobs: int = -1

    # 逻辑回归的最大迭代次数。样本或特征较多时可适当调大。
    composite_logistic_max_iter: int = 1000

    # 逻辑回归的 L2 正则强度倒数。越小正则越强，越不容易过拟合。
    composite_logistic_c: float = 1.0

    # XGBoost 滚动训练窗口长度，单位为 K线根数。
    # 当前 240*5 表示约 1200 根 30分钟K线。调大通常更稳定，但对市场变化反应更慢。
    xgboost_train_window: int = 240*5

    # 每次训练 XGBoost 所需的最低样本数。样本不足时跳过预测。
    xgboost_min_train_samples: int = 120*5

    # XGBoost 重新训练间隔，单位为 K线根数。
    # 当前 5*5 表示每 25 根K线重新训练一次，中间复用上一次模型滚动预测。
    xgboost_retrain_every: int = 5*5

    # XGBoost 树的数量。调大可能提升拟合能力，但更慢且更容易过拟合。
    xgboost_n_estimators: int = 80

    # XGBoost 树构建算法。"hist" 通常比默认精确算法更快，适合当前滚动训练场景。
    xgboost_tree_method: str = "hist"

    # XGBoost 使用的线程数。-1 表示尽量使用全部 CPU 线程。
    xgboost_nthread: int = -1

    # 单棵树最大深度。调大能捕捉更复杂非线性关系，但高频数据里过拟合风险会明显上升。
    xgboost_max_depth: int = 3

    # 学习率。调小更稳但通常需要更多树；调大收敛更快但可能不稳定。
    xgboost_learning_rate: float = 0.05

    # 每棵树训练时抽样的样本比例。小于 1 可增加随机性，降低过拟合。
    xgboost_subsample: float = 0.8

    # 每棵树训练时抽样的特征比例。小于 1 可降低对少数特征的依赖。
    xgboost_colsample_bytree: float = 0.8

    # 树分裂的最小子节点权重。数值越大，模型越不容易被少量噪声样本带偏。
    xgboost_min_child_weight: float = 5.0

    # 节点继续分裂所需的最小损失下降。数值越大，树结构越保守。
    xgboost_gamma: float = 0.1

    # 叶子权重的 L2 正则强度。
    xgboost_reg_lambda: float = 2.0

    # 叶子权重的 L1 正则强度。
    xgboost_reg_alpha: float = 0.0

    # 是否在每个滚动训练窗口末尾留出一段时间顺序验证集。
    xgboost_use_validation_early_stopping: bool = True

    # 启用早停时，滚动训练窗口中用作验证集的比例。
    xgboost_validation_ratio: float = 0.20

    # 验证集 mlogloss 连续多少轮不改善时提前停止。
    xgboost_early_stopping_rounds: int = 15

    # 随机种子。固定后便于复现实验结果。
    xgboost_random_state: int = 42

    # XGBoost 自变量因子来源。
    # "selected"：使用 selected_factors；"all"：使用全部因子；"best"：使用因子库或单因子表现最好的因子。
    xgboost_feature_scope: str = "best"  # 可选："selected"、"all"、"best"

    # 综合因子回测是否只按 active 因子库构建需要的因子列。
    # 开启后可避免先生成全部海量因子，通常能显著提升 composite_factor_backtest.py 运行速度。
    composite_build_active_only: bool = True

    # 综合因子回测是否缓存 active 因子矩阵。
    # 开启后，相同品种、周期、时间索引和 active 因子集合再次运行时会直接读取缓存。
    composite_use_factor_cache: bool = True

    # XGBoost 特征形式。
    # "signal"：只使用每个因子的 -1/0/1 多空信号；
    # "continuous"：只使用因子连续值；
    # "both"：同时使用信号和连续值。
    xgboost_feature_mode: str = "both" #"both"  # 可选："signal"、"continuous"、"both"

    # 是否为 XGBoost 增加因子状态特征。
    # 状态特征包括因子变化量、上一期值、上一期信号、信号变化和连续同向信号长度。
    xgboost_include_factor_state_features: bool = False#True

    # 当 xgboost_feature_scope="best" 时，候选因子数量上限。
    xgboost_best_top_n: int = 50

    # 是否在每个滚动训练窗口内重新做特征选择。
    # 开启更符合样本外逻辑，也能适应阶段变化；关闭更快，但特征集合更静态。
    xgboost_walk_forward_feature_selection: bool = False#True

    # 滚动特征选择时，先取 best_top_n 的多少倍作为候选池，再做相关性去重。
    # 例如 best_top_n=50 且候选倍数=5，则先看前 250 个候选。
    xgboost_candidate_multiplier: int = 5

    # XGBoost 滚动特征选择中的最大允许特征相关性。
    # 高于该值的相似因子会被过滤，减少模型输入的冗余。
    xgboost_max_feature_corr: float = 0.85

    # XGBoost 交易信号的概率优势阈值。
    # 只有多空概率差超过该值才交易；调大更保守，调小更积极。
    xgboost_trade_min_edge: float = 0.01

    # XGBoost 交易信号的最低类别概率阈值。
    # 设为 0 表示不额外限制；调高后只有模型足够自信才开仓。
    xgboost_trade_min_probability: float = 0 #0.55

    # 是否让每个滚动 XGBoost 模型在训练窗口内自动校准交易阈值。
    xgboost_auto_calibrate_trade_thresholds: bool = False#True

    # 滚动阈值校准使用的候选概率差阈值。
    xgboost_trade_edge_grid: list[float] = field(
        default_factory=lambda: [0.06, 0.08, 0.10, 0.12, 0.15, 0.20]
    )

    # 滚动阈值校准使用的候选方向概率阈值。
    xgboost_trade_probability_grid: list[float] = field(
        default_factory=lambda: [0.50, 0.55, 0.60, 0.65]
    )

    # 每组候选阈值至少需要产生的窗口内交易次数。
    xgboost_threshold_min_trades: int = 0#5#10

    # 是否在开仓前应用简单的市场状态过滤。
    xgboost_trade_use_market_filters: bool = False#True

    # 用于估计波动率和流动性状态分位的滚动窗口。
    xgboost_trade_filter_window: int = 120

    # 允许交易所需的日内绝对收益滚动分位下限。
    xgboost_trade_min_volatility_rank: float = 0#0.05#0.10

    # 允许交易所需的成交额或成交量滚动分位下限。
    xgboost_trade_min_liquidity_rank: float = 0#0.05#0.10

    # 是否按模型置信度动态调整仓位，而不是信号通过后总是满仓。
    xgboost_use_dynamic_position_sizing: bool = False

    # 信号通过全部交易过滤后的最小仓位。
    # 该参数避免有效信号被压缩成接近零的敞口。
    xgboost_position_size_min: float = 0.5

    # 置信度映射为仓位时使用的幂次。大于 1 会让仓位更集中在强信号上。
    xgboost_position_size_power: float = 1.0

    # 综合策略允许的最大绝对仓位。
    xgboost_position_size_max: float = 1.0

    # 是否启用综合策略持仓规则优化。
    # 开启后会在模型目标仓位生成后，应用最小持仓期、反转冷却、小变化忽略和可选仓位平滑。
    xgboost_use_position_rules: bool = True

    # 最小持仓 K 线数。
    # 新开仓后，在达到该持仓期之前，默认不允许因为短期信号噪声直接反向或清仓。
    xgboost_min_holding_bars: int = 2

    # 反转或清仓后的冷却 K 线数。
    # 冷却期内不允许马上重新开仓，用于降低 1 -> -1 -> 1 这类高频抖动。
    xgboost_reentry_cooldown_bars: int = 1

    # 目标仓位变化小于该阈值时忽略变化。
    # 仅对同方向仓位调整生效；例如 0.10 表示仓位变化小于 10% 时沿用上一期目标仓位。
    xgboost_min_position_change: float = 0.10

    # 仓位平滑系数。
    # 1 表示不平滑；0.5 表示新目标仓位和上一期目标仓位各占一半。
    xgboost_position_smoothing_alpha: float = 1.0

    # 是否要求当前模型置信度相对近期历史处于较高分位才交易。
    xgboost_trade_use_confidence_rank_filter: bool = False

    # 用于计算近期置信度强度分位的滚动窗口。
    xgboost_trade_confidence_rank_window: int = 240

    # 允许开仓所需的绝对置信度滚动分位下限。
    xgboost_trade_min_confidence_rank: float = 0.3#0.50

    # 是否对模型训练样本应用同样的市场状态过滤。
    # 开启后模型会更关注更容易预测、也更适合交易的 K 线。
    xgboost_train_use_market_filters: bool = False

    # 每个滚动训练窗口内至少需要的非中性标签数量。
    # 该参数避免在几乎全是中性样本的窗口上训练方向模型。
    xgboost_train_min_directional_samples: int = 80

    # XGBoost 训练时施加给非中性标签（-1 和 1）的类别权重。
    # 大于 1 有助于避免模型过度偏向中性类别。
    xgboost_train_nonzero_class_weight: float = 1.5

    # XGBoost 训练时施加给中性标签（0）的类别权重。
    # 小于 1 可以降低中性类别在噪声数据中的主导性。
    xgboost_train_neutral_class_weight: float = 0.8

    # 是否在训练窗口内自动校准 XGBoost 信号方向。
    # 开启后会根据训练窗口表现决定是否反向；如果你认为方向校准不符合研究原则，可关闭。
    xgboost_auto_calibrate_signal_direction: bool = True

    # 等权投票基准策略的最低绝对投票分数。
    # 只有因子投票强度超过该值才交易；0 表示只要有方向就允许交易。
    benchmark_vote_min_abs_score: float = 0.0

    # 是否在训练窗口内自动校准等权投票基准的信号方向。
    benchmark_vote_auto_calibrate_direction: bool = True

    # XGBoost 预测目标的未来收益跨度，单位为 K线根数。
    # 1 表示预测下一根 K 线从开盘到收盘的收益方向；3/5 通常比单根方向有更高信噪比。
    xgboost_target_horizon: int = 1

    # XGBoost 分类目标中的中性区间，单位 bps。
    # 留空表示使用手续费和滑点之和作为涨跌中性阈值；
    # 设为 0 则只按未来 horizon 根累计收益正负号分类为 -1/0/1。
    xgboost_target_neutral_bps: Optional[float] = 3.0  # 留空表示使用手续费和滑点之和

    # ----------------------------
    # qcut 分组检验设置
    # ----------------------------

    # qcut 分组数量。5 表示把因子值分成 Q1-Q5 五组，观察单调性和分组收益。
    qcut_groups: int = 5

    # qcut 滚动分位数窗口，单位为 K线根数。
    # 用滚动窗口计算分位，避免使用未来全样本分布造成信息泄露。
    qcut_window: int = 240*2

    # qcut 分组所需的最小历史样本数。样本不足时不生成有效分组。
    qcut_min_periods: int = 120*2

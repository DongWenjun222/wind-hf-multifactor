# 高频多因子研究框架项目文档

本文档用于说明当前项目的整体设计、主要文件职责、核心方法、运行流程、输出文件、风控/研究注意事项，以及后续接入 AI 自动生成因子的推荐工作流。

当前项目是一个面向期货 30 分钟 K 线数据的高频/日内多因子研究框架。它的核心闭环是：

```text
行情数据读取
-> 因子批量生成
-> 单因子训练/验证/测试回测
-> 基于训练集和验证集的因子库入库与相关性去重
-> XGBoost 多因子滚动训练预测
-> 综合因子回测、诊断和模型对比
```

重要提醒：这个框架适合做因子研究和模型验证，不建议在未充分加入成本、滑点、稳定性检验和样本外验证前直接用于实盘。

## 0. 快速入口：文件职责与推荐运行顺序

如果只是想快速使用项目，优先看本节即可。当前项目推荐通过 `cli.py` 统一运行，直接运行 `single_factor_backtest.py`、`composite_factor_backtest.py`、`multi_symbol_backtest.py` 也可以，但 CLI 更适合频繁切换品种、时间段、模型参数和实验编号。

### 0.1 根目录 Python 文件作用

| 文件 | 是否建议直接运行 | 作用 |
| --- | --- | --- |
| `cli.py` | 是，推荐入口 | 统一命令行入口。支持运行单因子、综合因子、多品种流程，并可通过命令行参数或 JSON 覆盖 `config.py`。 |
| `config.py` | 否 | 全局配置中心。控制数据源、品种、时间区间、因子库门槛、XGBoost/模型参数、图表、实验目录、多品种组合风控等。 |
| `data_loader.py` | 否 | 数据读取层。负责本地 CSV 缓存、Wind 分钟 K 线、相关品种行情、Wind 日频宏观/利率/指数数据读取与缓存。 |
| `data_quality_report.py` | 是，回测前推荐 | 数据质量报告工具。检查主品种、相关品种、宏观数据的缺失、重复时间戳、K线间隔异常、OHLC异常、零成交量和极端收益。 |
| `factors.py` | 否 | 因子总装配入口。负责因子编号、按需构建、软淘汰过滤、因子矩阵拼装；同时兼容转导部分旧的数据读取函数。 |
| `factor_library.py` | 否 | 因子库管理。根据单因子训练/验证表现、收益/夏普/胜率门槛和相关性去重，维护 `active/rejected/all` 三类因子库。 |
| `single_factor_backtest.py` | 可以 | 单因子批量回测入口。读取数据、构建因子、训练/验证/测试分段回测、生成单因子报告并更新因子库。 |
| `composite_factor_backtest.py` | 可以 | 综合因子回测入口。只在 active 因子池内选因，使用 XGBoost/逻辑回归/随机森林等模型滚动训练预测，并输出综合策略效果。 |
| `multi_symbol_backtest.py` | 可以 | 多品种批量入口。对多个期货品种独立运行单因子和综合因子流程，并生成多品种组合层汇总、图表和风控组合结果。 |
| `runtime_utils.py` | 否 | 运行追踪工具。把控制台输出同步写入日志，并生成 `execution_manifest.json`，记录运行状态、耗时、配置哈希、错误堆栈和输出文件。 |
| `project_fingerprint.py` | 否 | 源码指纹工具。计算影响因子构建的源码哈希，用于让因子矩阵缓存随公式变更自动失效。 |
| `experiment_utils.py` | 否 | 实验快照工具。负责创建 `runs/` 实验目录、保存配置、复制关键输出、快照 active 因子库和因子数量。 |
| `factor_metadata.py` | 可以 | 因子元数据导出工具。生成 `factor_metadata.csv` 和因子家族汇总，用于解释、聚类、治理和 AI 因子管理。 |
| `leakage_audit.py` | 是，改代码后推荐 | 未来函数/数据泄露静态审计工具。扫描 `shift(-n)`、`bfill`、全样本统计等高风险写法并输出审计报告。 |
| `hard_prune_factors.py` | 谨慎运行 | 因子硬删除工具。读取淘汰池，扫描 `factor_builders/*.py` 中可安全定位的公式行，预演或执行源码级删除。默认先预演，不加 `--apply` 不会改代码。 |
| `smoke_test.py` | 是，改代码后推荐 | 轻量冒烟测试。用于快速检查数据读取、因子构建、active 因子池和小规模综合模型是否能跑通。 |

### 0.2 因子构造文件作用

| 文件 | 是否建议直接运行 | 作用 |
| --- | --- | --- |
| `factor_builders/__init__.py` | 否 | 因子构造模块导出入口，把各类 `add_*_factors` 函数统一暴露给 `factors.py`。 |
| `factor_builders/basic.py` | 否 | 基础量价因子，例如 `momentum`、`reversal`、`breakout`、`volume_confirm`。 |
| `factor_builders/parametric.py` | 否 | 主品种自身参数化量价因子，覆盖收益、波动、量价、K 线结构、流动性 proxy、资金流、gap 等。 |
| `factor_builders/non_cross.py` | 否 | 非跨品种复杂因子，包括 `ultra_`、`hyper_`、`omega_` 等高阶量价结构。 |
| `factor_builders/cross_asset.py` | 否 | 跨品种联动因子，包括相关品种收益、价差、beta、相关性、滞后联动、成交活跃度差异等。 |
| `factor_builders/calendar.py` | 否 | 交易日历/季节性因子，包括日内时段、周/月/季度/年度位置，以及时间状态与量价状态交互。 |
| `factor_builders/macro_state.py` | 否 | 资金利率、指数、汇率、债券等 Wind 日频宏观状态代理因子。默认日频数据滞后一日再对齐到分钟线。 |
| `factor_builders/common.py` | 否 | 因子构造共享工具，例如滚动 z-score、跨品种对齐、宏观日频对齐、代码名清洗等。 |

### 0.3 推荐运行顺序

第一次使用或大改代码后，建议先做轻量检查：

```bash
python smoke_test.py
python data_quality_report.py
python leakage_audit.py
python -m py_compile config.py data_loader.py data_quality_report.py factors.py factor_taxonomy.py runtime_utils.py project_fingerprint.py cli.py factor_library.py factor_metadata.py leakage_audit.py single_factor_backtest.py composite_factor_backtest.py multi_symbol_backtest.py experiment_utils.py hard_prune_factors.py smoke_test.py
```

日常单品种研究推荐顺序：

```bash
# 1. 测试新增单因子，并更新 active/rejected/all 因子库
python data_quality_report.py --symbol C.DCE
python cli.py single --symbol C.DCE --scope new --start-index 126978

# 2. 导出因子元数据，便于解释、治理和聚类
python factor_metadata.py

# 3. 只使用 active 因子池做综合模型滚动预测与回测
python cli.py composite --symbol C.DCE --models xgboost,logistic_regression
```

多品种研究推荐顺序：

```bash
# 1. 覆盖中国商品期货中流动性较好的主力连续品种，并生成多品种组合汇总
python cli.py multi --symbols liquid_commodity

# 也可以手工指定少量品种做快速实验
python cli.py multi --symbols C.DCE,M.DCE,Y.DCE,P.DCE

# 2. 预演低质量因子硬删除，不修改源码
python hard_prune_factors.py

# 3. 检查 factor_hard_delete_report.csv 后，如确认无误再真正删除
python hard_prune_factors.py --apply
```

如果不想使用 CLI，也可以直接运行旧入口：

```bash
python single_factor_backtest.py
python composite_factor_backtest.py
python multi_symbol_backtest.py
```

直接运行旧入口时会读取 `config.py` 中的默认参数；使用 `cli.py` 时可以用命令行参数覆盖常用配置，更适合频繁实验。

### 0.4 常见输出位置

```text
wind_hf_multifactor_output/
  logs/                              # 每次运行的控制台日志
  runs/                              # 每次运行的配置、manifest 和关键结果快照
  factor_library/                    # 单品种 active/rejected/all 因子库
  single_factor/                     # 单因子汇总、分组、图表
  composite_factor/                  # 综合因子明细、摘要、模型对比、诊断图表
  by_symbol/                         # 多品种独立结果和组合汇总
```

## 1. 项目结构

```text
高频/
  config.py
  data_loader.py
  factors.py
  runtime_utils.py
  cli.py
  factor_builders/
    __init__.py
    basic.py
    parametric.py
    calendar.py
    cross_asset.py
    macro_state.py
    non_cross.py
    common.py
  factor_library.py
  single_factor_backtest.py
  composite_factor_backtest.py
  experiment_utils.py
  hard_prune_factors.py
  smoke_test.py
  AI_FACTOR_GENERATION_PROMPT.md
  PROJECT_DOCUMENTATION.md
  wind_hf_multifactor_output/
```

主要文件职责如下：

| 文件 | 作用 |
| --- | --- |
| `config.py` | 全局配置中心，控制数据、因子库、单因子回测、XGBoost、图表、实验目录等参数。 |
| `data_loader.py` | 数据读取层，负责本地行情缓存、Wind 分钟 K 线、相关品种行情、Wind 日频宏观/利率/指数数据读取与缓存。 |
| `factors.py` | 因子编号、按需构建、因子总装配入口。为了兼容旧脚本，仍转导 `fetch_intraday_data`、`stop_wind` 等数据函数。具体数据读取已拆到 `data_loader.py`，具体因子公式已拆到 `factor_builders/`。 |
| `runtime_utils.py` | 统一执行追踪工具，将控制台输出同步写入日志，并生成运行状态、耗时、配置哈希、错误堆栈和更新文件清单。 |
| `cli.py` | 推荐的统一命令行入口，可在不修改 `config.py` 的情况下运行单因子、综合因子和多品种流程。 |
| `factor_builders/basic.py` | 基础量价因子，例如 momentum、reversal、breakout、volume_confirm。 |
| `factor_builders/parametric.py` | 主品种自身的参数化量价因子，包括收益、波动、量价、K线结构、流动性 proxy 等。 |
| `factor_builders/calendar.py` | 交易日历、日内时段、周/月/季度/年度季节性及其与量价状态交互的因子。 |
| `factor_builders/cross_asset.py` | 跨品种联动因子，包括相关品种收益、价差、beta、相关性、滞后联动和成交活跃度差异等。 |
| `factor_builders/macro_state.py` | 资金利率、指数、汇率、债券等 Wind 日频宏观状态代理因子。 |
| `factor_builders/non_cross.py` | 更复杂的非跨品种高阶因子，包括 `ultra_`、`hyper_`、`omega_` 系列。 |
| `factor_builders/common.py` | 因子构造共享工具，例如滚动 z-score、跨品种对齐、宏观日频对齐、代码名清洗等。 |
| `factor_library.py` | 因子库管理，负责 active/rejected/all 三类因子的入库、排序、门槛过滤和相关性去重。 |
| `single_factor_backtest.py` | 单因子批量回测，输出训练集/验证集/测试集表现，并调用因子库逻辑更新 active 因子。 |
| `composite_factor_backtest.py` | 综合因子回测，使用 active 因子池内的因子做 XGBoost 滚动训练、滚动选因、滚动预测。 |
| `multi_symbol_backtest.py` | 多品种批量入口，为每个品种创建独立输出目录，依次运行单因子和综合因子流程，并生成跨品种汇总。 |
| `hard_prune_factors.py` | 因子硬删除工具，把多品种淘汰池中的低质量因子结构真正从因子构造源码中删除。模块化后重点面向 `factor_builders/`。 |
| `experiment_utils.py` | 实验运行目录、配置快照、输出快照、因子数量快照等工程辅助函数。 |
| `smoke_test.py` | 轻量冒烟测试，快速检查数据读取、因子构建、active 因子池和小规模 XGBoost 是否能跑通。 |
| `AI_FACTOR_GENERATION_PROMPT.md` | 给后续 AI 自动生成新因子的工作提示词和约束说明。 |

## 2. 当前默认配置概览

默认配置集中在 `BacktestConfig`。

| 参数 | 当前值 | 含义 |
| --- | --- | --- |
| `symbol` | `"C.DCE"` | 当前主预测品种。 |
| `symbols` | `LIQUID_COMMODITY_MAIN_SYMBOLS` | 多品种批量回测时依次运行的主预测品种。默认覆盖中国商品期货中流动性较好的主力连续品种，可用 `--symbols liquid_commodity` 显式指定同一品种池。 |
| `bar_size` | `30` | 30 分钟 K 线。 |
| `prefer_local_data` | `True` | 优先读取本地缓存行情，失败后再尝试 Wind。 |
| `commission_bps` | `0.5` | 单边手续费，单位 bps。 |
| `slippage_bps` | `0` | 当前不计滑点。 |
| `cost_stress_bps_list` | `0, 0.5, 1, 2, 3` | 综合回测成本压力测试档位，表示手续费+滑点的单边总成本。 |
| `signal_threshold` | `0.7` | 因子 z-score 转多空信号阈值。 |
| `zscore_window` | `120` | 因子滚动标准化窗口。 |
| `auto_select_train_ratio` | `0.7` | 前 70% 样本作为训练段，用于判断因子方向。 |
| `auto_select_validation_ratio` | `0.15` | 训练段之后的 15% 样本作为验证段，用于因子入库；最后 15% 作为最终测试段。 |
| `enable_cross_asset_factors` | `True` | 启用跨品种因子。 |
| `related_symbols` | `CS.DCE, M.DCE, Y.DCE, P.DCE` | 默认相关品种数据源。 |
| `enable_macro_state_factors` | `True` | 启用资金利率、指数、汇率、债券等 Wind 日频宏观状态因子。 |
| `macro_state_symbols` | `000300.SH, 000001.SH, 399006.SZ, USDCNY.IB, CBA00101.CS` | 默认宏观/市场状态代理代码，可按 Wind 权限和研究方向调整。 |
| `macro_state_lag_daily_bars` | `1` | 宏观日频数据对齐到分钟线前整体滞后 1 个日频数据点，避免盘中使用当天收盘后才知道的数据。 |
| `single_factor_scope` | `"new"` | 默认只测试新增因子。 |
| `single_factor_keep_top_n` | `200` | active 因子库最多保留 200 个因子。 |
| `factor_library_enable_family_quota` | `True` | 是否启用 active 因子家族配额，防止同质因子过度集中。 |
| `factor_library_family_max_counts` | 见 `config.py` | 各因子家族的 active 数量上限，例如 parametric、cross_asset、calendar、macro_state 等。 |
| `use_frozen_active_library` | `False` | 综合回测是否读取冻结版 active 因子库快照。 |
| `frozen_active_library_path` | `None` | 冻结版 active 因子库路径，可用于严格样本外检验。 |
| `single_factor_new_factor_start_index` | `126978` | 当前增量测试从第 126978 个因子开始，对应最新追加的 `macro_` 宏观状态因子。 |
| `single_factor_new_factor_start_index_by_symbol` | `C.DCE/M.DCE/Y.DCE/P.DCE/CS.DCE -> 126978` | 多品种模式下可为不同期货品种单独设置新增因子起始编号。 |
| `enable_factor_pruning` | `True` | 是否启用多品种因子淘汰池机制。 |
| `factor_prune_list_path` | `factor_prune_list.csv` | 软淘汰清单路径；相对路径默认放在 `output_dir` 下。 |
| `factor_pruning_min_tested_symbols` | `2` | 一个因子至少在多少个品种上完成测试后，才允许进入淘汰判断。 |
| `factor_pruning_keep_if_active_any_symbol` | `True` | 如果因子在任一品种进入 active，则默认保护，不纳入淘汰池。 |
| `factor_pruning_keep_min_sharpe` | `0.0` | 只要任一品种初筛夏普达到该值，就暂不淘汰。 |
| `factor_pruning_keep_min_total_return` | `0.0` | 只要任一品种初筛累计收益达到该值，就暂不淘汰。 |
| `factor_pruning_apply_to_build` | `True` | 构建因子矩阵时是否自动过滤软淘汰清单中的因子。 |
| `factor_library_min_sharpe` | `1.0` | 入库初筛夏普至少大于等于 1。当前初筛会参考训练集和验证集的较弱表现。 |
| `factor_library_min_train_sharpe` | `1.0` | 训练集夏普也至少大于等于 1 才能入库。 |
| `factor_library_min_train_win_rate` | `0.5` | 训练集胜率必须严格大于 50% 才能入库。 |
| `factor_library_min_test_win_rate` | `0.5` | 验证集存在时验证胜率必须严格大于 50%；否则测试胜率必须严格大于 50%。 |
| `factor_library_max_corr` | `0.80` | 入库因子与已入库因子的最大相关性阈值。 |
| `multi_symbol_use_opportunity_selection` | `True` | 多品种组合层是否启用横截面机会选择，只保留当前机会评分较强的品种参与组合。 |
| `multi_symbol_opportunity_top_n` | `8` | 每根 K 线最多交易机会评分最高的 8 个品种；设为 0 表示不限制数量。 |
| `multi_symbol_opportunity_score_mode` | `"edge_probability"` | 机会评分使用上一根综合模型概率差绝对值乘方向概率，避免组合层偷看当前收益。 |
| `multi_symbol_opportunity_weight_power` | `1.0` | 机会评分对原组合权重的放大幂次，越大权重越集中到强机会品种。 |
| `multi_symbol_use_group_risk_budget` | `True` | 是否启用多品种板块/产业链风险预算，限制同一板块权重过度集中。 |
| `multi_symbol_max_group_weight` | `0.35` | 单一板块最大组合权重，默认任一板块最多 35%。 |
| `composite_model_names` | `["xgboost", "logistic_regression", "random_forest", "extra_trees"]` | 综合因子滚动训练时依次对比的模型；如果某个 sklearn 模型依赖缺失，会自动跳过并在模型比较表中记录。 |
| `composite_sklearn_n_estimators` | `120` | RandomForest 和 ExtraTrees 使用的树数量。 |
| `composite_logistic_max_iter` | `1000` | 逻辑回归最大迭代次数。 |
| `composite_logistic_c` | `1.0` | 逻辑回归 L2 正则强度倒数，越小正则越强。 |
| `composite_enable_validation_test_gap_report` | `True` | 是否输出验证集到最终测试集的表现衰减诊断，用于识别样本外失效和过拟合风险。 |
| `composite_gap_warn_sharpe_retention` | `0.5` | 测试夏普低于验证夏普该比例时触发衰减预警。 |
| `composite_gap_warn_return_retention` | `0.5` | 测试累计收益低于验证累计收益该比例时触发衰减预警。 |
| `xgboost_feature_scope` | `"best"` | 综合模型在 active 池内滚动选择表现较好的因子。 |
| `xgboost_best_top_n` | `50` | 每次重训最多选 50 个基础因子。 |
| `xgboost_train_window` | `1200` | 每次 XGBoost 训练最多使用过去 1200 根 K 线。 |
| `xgboost_min_train_samples` | `600` | 训练样本少于 600 时跳过预测。 |
| `xgboost_retrain_every` | `25` | 每 25 根 K 线重新选因并训练一次。 |
| `xgboost_train_use_time_decay_weight` | `True` | 是否启用训练样本时间衰减加权，让更靠近预测时点的样本在模型训练中占更高权重。 |
| `xgboost_train_time_decay_half_life` | `400` | 时间衰减半衰期，单位是 K 线根数；距离训练窗口尾部 400 根 K 线的样本时间权重大约减半。 |
| `xgboost_train_time_decay_min_weight` | `0.25` | 时间衰减最低权重下限，避免早期样本几乎完全失效。 |
| `xgboost_train_time_decay_normalize` | `True` | 是否把时间衰减权重均值归一到 1，建议开启以保持模型整体学习强度稳定。 |
| `xgboost_feature_mode` | `"both"` | 同时使用因子连续值和因子多空信号。 |
| `xgboost_include_factor_state_features` | `False` | 当前默认不额外加入因子状态特征。 |
| `market_state_regime_window` | `120` | 市场状态分层诊断的滚动窗口，用于计算趋势强度、波动分位和流动性分位。 |
| `market_state_trend_strength_threshold` | `0.25` | 趋势/震荡状态划分阈值，趋势强度等于窗口累计收益绝对值除以窗口逐 K 线绝对收益和。 |
| `xgboost_trade_use_regime_filter` | `False` | 是否按市场状态标签过滤交易。默认关闭，只输出诊断报告，不直接改变策略。 |
| `allowed_market_state_regimes` | `[]` | 开启状态标签过滤后允许交易的状态标签，例如 `趋势上涨`、`趋势下跌`、`高波动`、`高流动性`。 |
| `xgboost_use_position_rules` | `True` | 是否启用综合策略持仓规则优化。 |
| `xgboost_min_holding_bars` | `2` | 新开仓后最少持有 2 根 K 线，减少短期噪声反向。 |
| `xgboost_reentry_cooldown_bars` | `1` | 清仓或反转后冷却 1 根 K 线再允许重新开仓。 |
| `xgboost_min_position_change` | `0.10` | 同方向目标仓位变化小于 10% 时忽略，降低微调换手。 |
| `xgboost_position_smoothing_alpha` | `1.0` | 仓位平滑系数，1 表示不平滑。 |
| `xgboost_walk_forward_feature_selection` | `False` | 当前默认不在每个滚动窗口内重新选因。 |
| `xgboost_target_horizon` | `1` | 默认预测下一根 K 线方向。 |

## 3. 数据读取与因子生成

`data_loader.py` 是数据入口；`factors.py` 是因子总装配入口；具体因子公式放在 `factor_builders/` 子文件夹中。

当前拆分原则：

```text
data_loader.py
  负责本地 CSV、Wind 分钟线、相关品种行情、Wind 日频宏观/利率/指数数据读取与缓存。

factors.py
  负责因子编号、按需构建、软淘汰过滤和 build_factors 总装配。
  同时保留旧的数据函数导入入口，避免其他脚本大量改 import。

factor_builders/
  每一类因子一个文件，避免单个 factors.py 无限膨胀。
```

数据读取逻辑：

```text
如果 prefer_local_data=True
  先读取 wind_hf_multifactor_output/{symbol}_{bar_size}min_data.csv
  如果本地数据缺失、字段不合法、周期不匹配或时间范围不覆盖，再尝试 Wind
否则
  直接从 Wind 获取数据
```

本地数据会经过统一清洗：

```text
时间索引转 DatetimeIndex
字段名转小写
删除重复时间
检查 open/high/low/close
补齐 volume/amt/amount 可选字段
删除核心价格缺失行
校验本地数据 K 线周期是否等于 config.bar_size
```

当前因子结构：

```text
4 个基础因子
+ 16400 个参数化因子
+ 10000 个 ultra_ 非跨品种复杂因子
+ 10000 个 hyper_ 非跨品种高阶因子
+ 20000 个 omega_ 非跨品种终极因子
+ 20093 个 calendar_ 交易日历/季节性因子
+ 390 个 macro_ 资金利率/宏观状态因子
+ 可选跨品种因子，默认相关品种配置下约为 50480 个
```

不启用跨品种时，当前主品种自身因子约为 `76887` 个：`4` 个基础因子、`16400` 个参数化因子、`10000` 个 `ultra_`、`10000` 个 `hyper_`、`20000` 个 `omega_`、`20093` 个 `calendar_` 和约 `390` 个 `macro_`。启用跨品种时，会按相关品种和窗口额外生成约 `50480` 个联动类因子：旧版跨品种结构约 `480` 个，`crossmega_`、`crossultra_`、`crosshyper_` 各 `10000` 个，`crossomega_` 约 `20000` 个；默认全量因子规模约为 `127367` 个。

基础因子包括：

```text
momentum
reversal
breakout
volume_confirm
```

参数化因子覆盖的主要方向包括：

```text
收益动量
均线偏离
波动率
上下行波动差异
量价相关
成交量冲击
K 线实体和影线
突破位置
价格/成交量/波动区间 rank
流动性和冲击成本 proxy
资金流和 OBV
gap 特征
多种价格结构交互项
```

跨品种因子覆盖的主要方向包括：

```text
主品种相对相关品种收益
相对强弱
价格比率动量和 z-score
收益相关性
滚动 beta
beta 残差
相关品种滞后收益
跨品种成交量比例
跨品种波动差
跨品种区间差
```

跨品种数据只允许有限 forward fill，不允许 backfill，避免把未来相关品种数据填到过去。

交易日历/季节性因子覆盖的主要方向包括：

```text
日内时间 sin/cos
周内、月内、季度内、年内周期位置
夜盘、早盘、午盘、下午盘、开盘、收盘等时段状态
月初/月末、季初/季末、年初/年末状态
时间状态与收益、波动、成交量、K线实体、缺口等量价状态的交互
```

资金利率与宏观状态因子覆盖的主要方向包括：

```text
Wind 日频指数、汇率、债券或利率代理序列
宏观代理自身收益、变化、动量、波动和 z-score
主品种收益与宏观代理收益的差值、乘积、滚动相关和滚动 beta
宏观波动冲击、风险状态与主品种区间波动的交互
```

宏观日频数据默认会先滞后 `macro_state_lag_daily_bars=1`，再 forward fill 对齐到 30 分钟 K 线。这样可以避免在盘中使用当天收盘后才知道的宏观/指数日频数据。

## 4. 因子编号与标签

项目中每个因子都会按当前 `factors` DataFrame 的列顺序获得一个从 1 开始的编号。

```text
factor_name -> factor_id
momentum -> 1
reversal -> 2
...
```

当前几个重要的新增起点：

```text
calendar_ 日历/季节性因子起点：106885
macro_ 资金利率/宏观状态因子起点：126978
当前默认 single_factor_new_factor_start_index：126978
```

输出里常见的“因子标签”格式是：

```text
127_return_volume_beta_3
```

这样便于在报表、图表和代码之间快速定位因子。

注意：因子编号依赖列顺序。为了保证历史回测可追踪，新增因子应尽量追加到对应模块的尾部，并确保 `build_factors` 总装配顺序不把新因子插入到旧因子之前。

## 5. 单因子回测

运行命令：

```bash
python single_factor_backtest.py
```

单因子回测流程：

```text
读取行情
-> 构建所有因子
-> 根据 single_factor_scope 选择本轮要测试的因子
-> 按 auto_select_train_ratio 和 auto_select_validation_ratio 切分训练集/验证集/最终测试集
-> 每个因子在训练集上自动判断正向/反向
-> 固定该方向后分别回测训练集、验证集和最终测试集
-> 计算 qcut 分组表现
-> 汇总单因子结果
-> 同时参考训练集和验证集表现更新 factor_library
-> 为 Top 因子生成图表
```

三段式样本用途：

```text
训练集：只用于判断单因子方向和观察训练内表现。
验证集：和训练集一起用于因子入库排序、收益门槛过滤和相关性去重。
最终测试集：只用于留存评估，不直接参与 active 因子入库。
```

这样做的目的是降低“用最终测试集筛因子”的研究污染风险，同时避免只看训练集或只看验证集导致单段偶然表现过度影响入库。旧版历史结果如果没有验证列，因子库逻辑会回退使用训练列；如果训练列也不存在，再兼容旧测试列。

单因子信号规则：

```text
factor_score >  signal_threshold ->  1
factor_score < -signal_threshold -> -1
其他 -> 0
```

实际持仓使用：

```text
position = raw_signal.shift(1)
```

原因是当前 K 线结束后才能得到因子信号，下一根 K 线才能执行交易。

单因子回测收益使用每根 K 线的 open-to-close 收益：

```text
bar_return_oc = close / open - 1
strategy_gross_return = position * bar_return_oc
trading_cost = turnover * (commission_bps + slippage_bps) / 10000
strategy_net_return = strategy_gross_return - trading_cost
```

单因子输出包括训练集、验证集和最终测试集指标：

```text
累计收益
年化收益
年化波动
夏普比率
最大回撤
胜率
交易次数
样本 K 线数
信号覆盖率
持仓覆盖率
月度样本数
盈利月份占比
月度平均收益
月度收益波动
最差月收益
最佳月收益
月度收益集中度
IC
RankIC
ICIR
RankICIR
IC胜率
分组单调性
分组收益差
qcut 分组表现
```

其中“月度收益集中度”用于观察收益是否过度依赖少数月份。该值越高，说明正收益越集中，因子稳定性通常越值得谨慎复核。

其中预测能力指标的含义是：

```text
IC：方向调整后的因子值与下一根 K 线 open-to-close 收益的 Pearson 相关。
RankIC：方向调整后的因子值与下一根 K 线收益的 Spearman 秩相关，更关注排序能力。
ICIR / RankICIR：按月计算 IC 后，用月度均值除以月度波动，衡量预测相关性的稳定性。
IC胜率：月度 IC 大于 0 的月份占比。
分组单调性：Q1-Q5 分组平均未来收益与分组编号的 Spearman 相关，越接近 1 越单调。
分组收益差：最高组平均收益 - 最低组平均收益，用于观察强弱组是否真正拉开。
```

单因子报告图：

```text
*_report.png 当前是 5 行 x 3 列。

第 1 列：训练集
第 2 列：验证集
第 3 列：最终测试集

第 1 行：策略净值 vs 基准净值
第 2 行：回撤
第 3 行：因子分数 / 实际仓位
第 4 行：qcut 分组净值
第 5 行：qcut 分组平均收益
```

## 6. 因子库机制

因子库由 `factor_library.py` 管理。

主要输出目录：

```text
wind_hf_multifactor_output/factor_library/
```

主要文件：

```text
active_factors.csv
factor_library_all.csv
rejected_factors.csv
```

三类文件含义：

| 文件 | 含义 |
| --- | --- |
| `active_factors.csv` | 当前正式进入综合模型候选池的因子。 |
| `factor_library_all.csv` | 历史上评估过的全部因子及其最新表现。 |
| `rejected_factors.csv` | 未通过入库条件、被相关性去重淘汰或排名靠后的因子。 |
| `active_factor_family_summary.csv` | active 因子库按因子家族统计的数量、平均初筛夏普、平均 RankIC 和平均分组单调性。 |

当前入库逻辑：

```text
训练集和验证集的较弱夏普 >= factor_library_min_sharpe
训练集和验证集的较弱累计收益 >= factor_library_min_total_return
训练集夏普 >= factor_library_min_train_sharpe
训练集累计收益 >= factor_library_min_train_total_return
验证集和训练集满足可选交易次数、信号覆盖率和最大回撤约束
与已入库因子的最大相关性 < factor_library_max_corr
若启用家族配额，则同一因子家族 active 数量不能超过 factor_library_family_max_counts
按训练集和验证集的较弱夏普、较弱 RankIC、较弱分组单调性、较弱累计收益排序
active 因子最多保留 single_factor_keep_top_n 个
```

默认情况下，`factor_library_min_selection_rank_ic` 和 `factor_library_min_selection_monotonicity` 为 `None`，表示预测指标先参与排序和诊断，但不强制过滤。若希望更严格，可以在 `config.py` 中设置最低 RankIC 或最低分组单调性门槛。

active 因子库默认启用家族配额控制：

```text
factor_library_enable_family_quota = True

basic <= 10
parametric <= 60
non_cross_complex <= 45
cross_asset <= 45
calendar <= 25
macro_state <= 15
```

这一步的目的不是判断哪个家族一定更好，而是防止 active 因子库被同一类参数变体占满。最终综合模型仍然只会在 active 因子池内继续做 best/all/selected 选择。

当前默认设置比较严格：

```text
入库初筛夏普 >= 1.0
训练夏普 >= 1.0
入库初筛累计收益 >= 0
训练累计收益 >= 0
active 上限 = 200
最大相关性阈值 = 0.80
```

最终测试集指标仍会写入结果表，但不直接参与入库筛选。这样可以把最终测试集保留为更干净的样本外评估区间。

相关性去重支持两种口径：

```text
因子连续值相关性
因子 -1/0/1 信号相关性
```

如果任一口径相关性过高，该候选因子会被拒绝入库。

冻结 active 因子库：

```text
use_frozen_active_library = True
frozen_active_library_path = "runs/某次实验/active_factors_snapshot.csv"
```

开启后，综合因子回测不会读取当前最新的 `factor_library/active_factors.csv`，而是读取指定的历史快照。这样可以把“因子发现/入库阶段”和“后续综合模型样本外检验阶段”分开，减少反复更新 active 因子库带来的研究污染。

## 6.1 因子淘汰池与硬删除脚本

随着 AI 自动生成因子和参数化扩展持续增加，因子构造源码很容易越来越臃肿。当前项目已经把公式拆到 `factor_builders/`，并支持两层淘汰：

```text
第一层：软淘汰清单
多品种回测后生成 factor_prune_list.csv，后续 build_factors 时可以自动过滤这些因子。

第二层：硬删除脚本
hard_prune_factors.py 读取淘汰池，备份相关源码文件，然后把可安全定位的公式行从因子构造源码中删除。
```

### 6.1.1 软淘汰池如何生成

`multi_symbol_backtest.py` 在多品种批量回测结束后，会汇总每个品种的：

```text
single_factor/single_factor_all_summary.csv
factor_library/active_factors.csv
```

然后生成或更新：

```text
wind_hf_multifactor_output/factor_prune_list.csv
wind_hf_multifactor_output/by_symbol/factor_pruning_candidates.csv
```

默认淘汰逻辑比较保守：

```text
至少在 factor_pruning_min_tested_symbols 个品种上被测试过
并且没有在任何品种进入 active
并且没有任何品种达到 factor_pruning_keep_min_sharpe
并且没有任何品种达到 factor_pruning_keep_min_total_return
```

这意味着“不要求每个好因子都对所有品种有效”。只要某个因子在任一品种有明显价值，默认就会保留，不进入淘汰池。

### 6.1.2 软淘汰和硬删除的区别

软淘汰：

```text
不改 factors.py 源码
不改 factor_builders/ 中的因子公式源码
只是在 build_factors 阶段读取 factor_prune_list.csv 并过滤
适合短期观察、可回滚、低风险
```

硬删除：

```text
会真正修改因子构造源码
会从源码中删除可安全识别的公式行
会自动生成源码备份
适合确认某批因子结构长期无效后清理代码体积
```

### 6.1.3 推荐使用流程

第一步，运行多品种回测，生成各品种表现和软淘汰池：

```bash
python multi_symbol_backtest.py
```

第二步，预演硬删除，不修改源码：

```bash
python hard_prune_factors.py
```

预演会生成：

```text
wind_hf_multifactor_output/factor_hard_delete_pool.csv
wind_hf_multifactor_output/factor_hard_delete_report.csv
```

第三步，查看 `factor_hard_delete_report.csv`。重点检查：

```text
硬删状态 = matched_formula_line
源码行号
匹配模板
```

只有 `matched_formula_line` 表示脚本找到了可以安全删除的源码公式行。

第四步，确认无误后执行真正硬删除：

```bash
python hard_prune_factors.py --apply
```

执行后会自动备份：

```text
源文件名.bak_YYYYMMDD_HHMMSS
```

如果需要回滚，可以用备份文件恢复。

### 6.1.4 为什么有些因子不会被硬删

因子构造源码中有两类因子：

```text
显式公式行：
(f"ret_{window}", close.pct_change(window))

动态组合因子：
f"omega_{transform_name}_{input_name}_{window}"
```

显式公式行可以安全删除，因为脚本能精确定位到某一条公式模板。

动态组合因子不会被脚本盲目删除，因为删除一个 `transform` 或一个 `input` 可能一次性误伤大量其他因子结构。对于这类因子，报告中通常会标记为：

```text
unsupported_dynamic_template
```

如果后续确实要清理动态组合因子，建议先人工判断是删除整个 input、整个 transform，还是仅调整生成数量上限，然后再改对应的 `factor_builders/*.py`。

### 6.1.5 安全注意事项

第一，硬删除会改变因子列顺序和后续因子编号。若需要严格复现实验，请先冻结 active 因子库或保留相关源码备份。

第二，硬删除前建议先看 `factor_hard_delete_report.csv`，不要直接盲目 `--apply`。

第三，如果当前正在跑历史对比实验，不建议中途硬删因子结构；更稳的做法是等一轮实验闭环结束后再清理。

第四，硬删除脚本只删除可安全定位的公式行，不会自动改复杂循环结构，避免“一刀下去世界安静了但 alpha 也没了”的惨案。

## 7. 综合因子 / XGBoost 回测

运行命令：

```bash
python composite_factor_backtest.py
```

综合模型的一个关键原则是：

```text
active_factors.csv 是硬边界。
无论 xgboost_feature_scope 是 all、best 还是 selected，
综合模型都只能在 active 因子池内继续选择。
```

这意味着：

```text
xgboost_feature_scope="all"
  使用全部可生成的 active 因子

xgboost_feature_scope="best"
  在 active 因子池内选 best

xgboost_feature_scope="selected"
  selected_factors 必须是 active 因子池的子集
```

这样设计的目的是避免综合模型直接从几千个未经入库筛选的原始因子里寻找噪声关系。

## 8. XGBoost 自变量、目标和滚动方式

XGBoost 的原始候选自变量来自 active 因子池。

每个入选因子可以形成以下特征：

```text
factor_value
factor_signal
factor_value_diff_1
factor_value_lag_1
factor_signal_lag_1
factor_signal_change_1
factor_signal_streak
```

其中：

```text
factor_value 是标准化后的连续因子值
factor_signal 是 -1/0/1 多空信号
state features 用于描述因子信号是否刚翻转、是否持续、是否增强或衰减
```

预测目标是未来 horizon 根 K 线收益方向：

```text
future_horizon_return = close[t+horizon] / open[t+1] - 1
```

分类标签：

```text
future_horizon_return >  neutral_threshold ->  1
future_horizon_return < -neutral_threshold -> -1
否则 -> 0
```

当前默认：

```text
xgboost_target_horizon = 1
xgboost_target_neutral_bps = 3.0
neutral_threshold = 3.0 bps
```

如果 `xgboost_target_neutral_bps=None`，才会退回使用 `commission_bps + slippage_bps` 作为中性阈值。

XGBoost 使用三分类模型：

```text
objective = multi:softprob
类别 -1/0/1 映射为 0/1/2
输出 prob_down / prob_flat / prob_up
```

交易信号由概率优势生成：

```text
edge = prob_up - prob_down

edge >= xgboost_trade_min_edge  -> 做多
edge <= -xgboost_trade_min_edge -> 做空
其他 -> 空仓
```

当前默认 `xgboost_trade_min_edge = 0.01`，即多空概率差至少 1% 才开仓。

模型原始信号生成目标仓位后，还会经过交易执行层持仓规则：

```text
target_position_before_rules = raw_signal * position_size
target_position = apply_position_rules(target_position_before_rules)
position = target_position.shift(1)
```

当前持仓规则包括：

```text
xgboost_min_holding_bars：新开仓后至少持有若干根 K 线，减少刚开仓就反向。
xgboost_reentry_cooldown_bars：清仓或反转后冷却若干根 K 线，减少 1 -> -1 -> 1 高频抖动。
xgboost_min_position_change：同方向仓位变化太小时忽略，降低小幅调仓换手。
xgboost_position_smoothing_alpha：可选仓位平滑，1 表示不平滑。
```

这些规则只使用当前和历史目标仓位，不使用未来收益。`composite_detail.csv` 会同时保留 `target_position_before_rules` 和 `target_position`，便于对比规则前后的交易变化。

滚动训练方式：

```text
从第 xgboost_min_train_samples 根附近开始尝试预测
每个预测时点只使用它之前的历史样本
每次训练最多取过去 xgboost_train_window 根 K 线
每 xgboost_retrain_every 根 K 线重新训练一次
中间预测点复用上一次模型
如果启用 walk-forward feature selection，每次重训前都只在当前历史窗口内重新选因
```

多周期目标的防泄露处理：

```text
预测第 t 根 K 线时，
训练标签必须已经在 t 时刻之前完全落地。

训练结束位置 = t - xgboost_target_horizon

例如 xgboost_target_horizon = 3：
目标收益使用 open[t+1] -> close[t+3]
预测 t 时，训练集最多只能使用到 t-3 的标签
不能使用 t-2 / t-1 的标签
```

这相当于在每个滚动训练窗口尾部加入 `xgboost_target_horizon - 1` 根 K 线的标签隔离，避免模型训练时看到预测时点之后才会完整出现的未来收益。

当前默认：

```text
train_window = 1200
min_train_samples = 600
retrain_every = 25
每次滚动重训最多选择 Top 50 个 active 因子
xgboost_walk_forward_feature_selection = False
xgboost_include_factor_state_features = False
```

控制台里的滚动进度分母是“预测 K 线时点数量”，不是因子数量。例如：

```text
XGBoost滚动预测进度: 576/3204 个K线时点；active候选因子=139；本轮最多选因=50
```

这里的 `3204` 是需要滚动预测的 K 线数量，不是参与选因的因子数量。

## 9. 综合模型输出

主要输出目录：

```text
wind_hf_multifactor_output/composite_factor/
```

主要文件：

| 文件 | 含义 |
| --- | --- |
| `composite_detail.csv` | 最终测试集逐 K 线明细，包含预测、概率、信号、持仓、收益和净值。 |
| `composite_summary.csv` | XGBoost 测试集绩效与关键配置摘要，包含预测目标跨度、训练标签隔离K线数、阈值和仓位配置。 |
| `composite_model_comparison.csv` | XGBoost、LogisticRegression、RandomForest、ExtraTrees 的训练/验证/最终测试表现，以及等权投票基准表现对比。 |
| `composite_validation_test_gap.csv` | 各模型验证集到最终测试集的收益、夏普、回撤、胜率等指标衰减诊断，用于发现验证有效但最终测试明显失效的模型。 |
| `composite_report.png` | XGBoost 训练集、验证集和最终测试集三栏图；其他模型会输出 `composite_report_{model_name}.png`。 |
| `benchmark_vote_report.png` | 等权投票基准训练集、验证集和最终测试集三栏图。 |
| `composite_xgboost_feature_importance.csv` | XGBoost 特征重要性。 |
| `composite_factor_contribution.csv` | 把模型特征重要性聚合回基础因子后的贡献表，可用于判断 active 因子是否真的被模型使用。 |
| `composite_xgboost_feature_selection.csv` | 每次滚动重训时的选因明细。 |
| `composite_prediction_diagnostics.csv` | 预测准确率、方向准确率、概率差相关性等诊断。 |
| `composite_prediction_confusion_matrix.csv` | 校准后预测方向与真实方向的混淆矩阵。 |
| `composite_xgboost_edge_diagnostics.csv` | 按概率优势分桶统计未来收益。 |
| `composite_robustness_report.csv` | 最终测试集按月度、季度、波动状态和模型置信度拆分后的稳健性表现。 |
| `composite_market_state_report.csv` | 最终测试集按趋势、波动、流动性、交易时段以及组合状态拆分后的策略表现和预测诊断。 |
| `composite_cost_stress_report.csv` | 基于同一组持仓重算不同交易成本档位下的策略表现。 |
| `related_data_coverage.csv` | 跨品种数据对齐审计，包含覆盖率、前向填充占比、缺失段和对齐滞后。 |

`composite_report.png` 当前是 4 行 x 3 列：

```text
第 1 列：训练集滚动预测
第 2 列：验证集滚动预测
第 3 列：最终测试集滚动预测

第 1 行：策略净值 vs 基准净值
第 2 行：回撤
第 3 行：综合分数 / 实际仓位
第 4 行：校准概率差 vs 未来 horizon 收益散点
```

## 10. 多品种批量回测

多品种批量入口是：

```bash
python multi_symbol_backtest.py
```

也推荐使用 CLI 入口：

```bash
python cli.py multi --symbols liquid_commodity
```

它的定位是“多品种独立建模 + 统一汇总”，而不是把所有品种混在一个模型里训练。每个品种仍然会独立读取行情、构造因子、维护 active 因子库、训练综合模型和输出图表。`liquid_commodity` 是内置品种池别名，等价于 `config.py` 中的 `LIQUID_COMMODITY_MAIN_SYMBOLS`。

核心配置：

| 参数 | 含义 |
| --- | --- |
| `symbols` | 批量运行的主预测品种列表。默认使用中国商品期货中流动性较好的主力连续品种池，也可以手工传入逗号分隔列表。 |
| `multi_symbol_run_single_factor` | 是否先为每个品种运行单因子流程并更新该品种自己的 active 因子库。 |
| `multi_symbol_run_composite` | 是否为每个品种运行 XGBoost 综合因子流程。 |
| `multi_symbol_separate_output_dirs` | 是否为每个品种使用独立输出目录。建议保持开启。 |
| `multi_symbol_output_subdir` | 多品种结果放在 `output_dir` 下的哪个子目录，默认是 `by_symbol`。 |
| `multi_symbol_skip_existing` | 是否复用已经存在的单因子库和综合回测明细，用于长任务断点续跑。 |
| `multi_symbol_use_rolling_portfolio_weights` | 多品种组合是否使用滚动历史权重；建议保持开启，避免组合层用完整测试集计算权重。 |
| `multi_symbol_portfolio_weight_window` | 滚动组合权重的历史窗口长度，单位为 K 线根数。 |
| `multi_symbol_portfolio_min_weight_samples` | 估计滚动组合权重所需的最少历史样本数，样本不足时退化为等权。 |
| `multi_symbol_portfolio_max_symbol_weight` | 单一品种最大组合权重上限，用于避免组合过度集中。 |
| `multi_symbol_use_opportunity_selection` | 是否启用横截面机会选择。开启后会在基础组合权重之上，只保留当前机会评分最高的一部分品种。 |
| `multi_symbol_opportunity_top_n` | 每根 K 线最多交易多少个机会评分最高的品种；0 表示不限制数量。 |
| `multi_symbol_opportunity_min_score` | 机会评分最低阈值，低于该阈值的品种不参与当期组合。 |
| `multi_symbol_opportunity_score_mode` | 机会评分模式，支持 `edge`、`edge_probability`、`edge_rank`、`position`。默认用上一根概率差绝对值乘方向概率。 |
| `multi_symbol_opportunity_weight_power` | 机会评分对基础权重的放大幂次，越大越偏向强机会品种。 |
| `multi_symbol_use_group_risk_budget` | 是否启用板块/产业链风险预算。开启后会在机会选择之后限制同一板块总权重。 |
| `multi_symbol_max_group_weight` | 单个板块最大权重上限，默认 0.35。若活跃板块数量太少导致数学上无法满足，程序会自动使用可行上限。 |
| `multi_symbol_group_map` | 品种到板块/产业链的映射，例如黑色、有色、化工、油脂油料、能源、农产品等。 |
| `multi_symbol_use_vol_target` | 是否启用组合层波动率目标，根据历史已实现波动率缩放风险暴露。 |
| `multi_symbol_target_annual_vol` | 多品种组合目标年化波动率。 |
| `multi_symbol_max_portfolio_leverage` | 组合风控允许的最大风险杠杆倍数。 |
| `multi_symbol_use_drawdown_control` | 是否启用组合层回撤降仓。 |
| `multi_symbol_drawdown_reduce_start` | 组合回撤达到该水平后开始线性降仓。 |
| `multi_symbol_drawdown_stop` | 组合回撤达到该水平后降到 0 风险暴露。 |

默认输出结构：

```text
wind_hf_multifactor_output/
  by_symbol/
    C_DCE/
      run_config.json
      factor_library/
        active_factors.csv
        factor_library_all.csv
        rejected_factors.csv
      single_factor/
        single_factor_summary.csv
        single_factor_all_summary.csv
        qcut_group_summary.csv
      composite_factor/
        composite_detail.csv
        composite_summary.csv
        composite_model_comparison.csv
        composite_report.png
        benchmark_vote_report.png
    M_DCE/
      ...
    multi_symbol_summary.csv
    multi_symbol_portfolio_detail.csv
    multi_symbol_portfolio_summary.csv
    multi_symbol_portfolio_weights.csv
    multi_symbol_opportunity_scores.csv
    multi_symbol_opportunity_selection.csv
    multi_symbol_group_weights.csv
    multi_symbol_group_contribution.csv
    multi_symbol_portfolio_contribution.csv
    multi_symbol_strategy_return_corr.csv
    multi_symbol_run_manifest.json
    multi_symbol_portfolio_report.png
    factor_pruning_candidates.csv
  factor_prune_list.csv
  factor_hard_delete_pool.csv
  factor_hard_delete_report.csv
```

`multi_symbol_summary.csv` 是跨品种汇总表，主要包含：

```text
品种
输出目录
单因子状态
综合因子状态
active因子数
active平均入库夏普
active最高入库夏普
综合_累计收益
综合_夏普比率
综合_最大回撤
综合_胜率
综合_交易次数
错误
```

如果多个品种都成功生成 `composite_detail.csv`，脚本还会自动生成组合层结果：

| 文件 | 含义 |
| --- | --- |
| `multi_symbol_portfolio_detail.csv` | 各品种最终测试集收益、仓位，不同组合方法的收益、净值、回撤和平均仓位。 |
| `multi_symbol_portfolio_summary.csv` | 不同组合方法的累计收益、夏普、最大回撤、胜率、参与品种数、平均权重和权重参数。 |
| `multi_symbol_portfolio_weights.csv` | 滚动组合权重明细，每一行是该时点各品种权重。 |
| `multi_symbol_opportunity_scores.csv` | 每根 K 线每个品种的横截面机会评分。默认使用上一根综合模型概率优势，避免当前收益信息泄露。 |
| `multi_symbol_opportunity_selection.csv` | 每个组合方法在每根 K 线上实际选中的品种，1 表示入选，0 表示未入选。 |
| `multi_symbol_group_weights.csv` | 每个组合方法在每根 K 线上的板块/产业链权重。 |
| `multi_symbol_group_contribution.csv` | 每个组合方法下，各板块/产业链对组合收益的贡献。 |
| `multi_symbol_portfolio_contribution.csv` | 各品种按权重计算后的组合收益贡献。 |
| `multi_symbol_strategy_return_corr.csv` | 各品种综合策略最终测试集收益相关矩阵。 |
| `multi_symbol_run_manifest.json` | 多品种批量运行清单，包含配置快照、品种状态、错误信息和输出文件索引。 |
| `multi_symbol_portfolio_report.png` | 多组合方法图表，对比净值、回撤、累计收益和平均绝对仓位。 |
| `factor_pruning_candidates.csv` | 本轮多品种回测识别出的新增淘汰候选因子。 |
| `factor_prune_list.csv` | 全局软淘汰清单，后续构建因子时可自动过滤。 |
| `factor_hard_delete_pool.csv` | 硬删除池，由 `hard_prune_factors.py` 从软淘汰清单合并生成。 |
| `factor_hard_delete_report.csv` | 硬删除预演/执行报告，记录每个因子是否能匹配到可安全删除的源码公式行。 |

当前组合层支持三种方法：

```text
equal_weight：等权组合
inverse_vol：波动率倒数加权
positive_sharpe：夏普正向加权
```

默认情况下，组合层使用滚动历史窗口估计权重：每根 K 线的权重只使用它之前的历史收益，不使用当前及未来测试集表现，因此比全样本静态权重更接近真实样本外组合。在基础权重生成后，横截面机会选择会使用上一根 K 线已经产生的模型概率优势，对当前 K 线组合权重做 TopN 过滤和机会强度加权。随后板块/产业链风险预算会限制同一板块总权重，避免强机会集中在高度相关的一组品种中。若关闭 `multi_symbol_use_rolling_portfolio_weights`，静态基础权重也会被转换成逐时点权重矩阵，再经过同一套横截面机会选择和风险预算逻辑，仅建议用于诊断对比。

使用建议：

```text
第一，默认保持每个品种独立输出，避免不同品种的 active 因子库互相覆盖。
第二，如果只想批量跑综合模型，可以先手动准备好各品种的 active_factors.csv，再把 multi_symbol_run_single_factor 设为 False。
第三，如果只想批量刷新单因子库，可以把 multi_symbol_run_composite 设为 False。
第四，related_symbols 仍然是跨品种因子的数据源；symbols 是要被预测和回测的主品种列表，两者含义不同。全市场批量跑时，不建议无脑把 `related_symbols` 也扩成全市场，否则跨品种因子计算量会明显膨胀。
第五，多品种批量目前是“逐品种独立模型 + 多方法组合汇总”，后续总风险预算和跨品种仓位约束可以在此基础上继续扩展。
```

## 11. 等权投票基准

综合回测中同时生成一个等权投票基准。

逻辑：

```text
vote_score = 入选因子的 -1/0/1 信号均值
vote_score 达到阈值后转为多空信号
```

它的作用是提供一个简单但重要的对照组。

如果 XGBoost 长期无法跑赢等权投票，通常说明：

```text
特征本身预测力不足
目标噪声太高
模型过拟合窗口内噪声
交易规则过于粗糙
XGBoost 复杂度并没有带来增量价值
```

## 12. 实验目录和快照

当 `enable_experiment_run_dirs=True` 时，每次运行会在：

```text
wind_hf_multifactor_output/runs/
```

下生成独立实验目录，并保存：

```text
run_config.json
factor_count.json
active_factors_snapshot.csv
related_data_coverage.csv
关键输出 CSV 和图表快照
```

这样可以避免后续运行覆盖最新目录后，无法复盘某一次实验结果。

### 12.1 统一日志与执行清单

直接运行 `single_factor_backtest.py`、`composite_factor_backtest.py`、`multi_symbol_backtest.py`，或通过 `cli.py` 运行时，均会使用 `runtime_utils.py` 做执行追踪。

新增产物：

```text
wind_hf_multifactor_output/
  logs/
    {run_id}_single.log
    {run_id}_composite.log
    {run_id}_multi.log
  runs/
    {run_id}_{run_type}/
      run_config.json
      execution_manifest.json
```

`execution_manifest.json` 记录：

```text
运行类型与成功/失败状态
开始时间、结束时间和运行耗时
Python 与操作系统版本
命令行参数
完整配置的 SHA256 摘要
日志文件路径
返回结果摘要
失败时的异常与堆栈
本次运行期间更新的输出文件清单
```

这样即使长时间批量任务中途失败，也能定位失败阶段、复用相同参数重新运行，并保留控制台输出作为审计轨迹。

### 12.2 因子元数据、泄露审计与缓存指纹

当前项目新增了三类工程审计能力：

```text
factor_metadata.py
  生成 wind_hf_multifactor_output/factor_metadata.csv
  生成 wind_hf_multifactor_output/factor_metadata_family_summary.csv
  记录因子编号、标签、家族、来源文件、是否跨品种、是否宏观、是否日历、估计窗口和复杂度。

data_quality_report.py
  生成 wind_hf_multifactor_output/data_quality_report.csv
  检查主品种、相关品种和宏观日频数据的行数、起止时间、重复时间戳、缺失率、K线间隔异常、OHLC异常、零成交量和极端收益。
  注意：K线间隔异常会包含夜盘/午休/节假日等自然断点，主要用于提示时间轴断点，不等同于数据错误。

leakage_audit.py
  生成 wind_hf_multifactor_output/leakage_audit_report.csv
  扫描 shift(-n)、bfill、expanding、全样本统计、fit_transform 等高风险写法。
  报告中的“白名单说明”不是自动放行，只表示该处有已知用途，仍建议人工复核。

project_fingerprint.py
  计算 config.py、data_loader.py、factors.py、factor_builders/*.py 的源码哈希。
  composite_factor_backtest.py 的 active 因子矩阵缓存会记录该哈希；
  一旦因子公式或关键构建代码变化，旧缓存会自动失效，避免读到过期因子矩阵。
```

推荐在每次大规模新增因子或修改因子构造逻辑后运行：

```bash
python leakage_audit.py
python factor_metadata.py
python data_quality_report.py
```

## 13. 冒烟测试

运行：

```bash
python smoke_test.py
```

冒烟测试会使用较小窗口和较少树快速检查：

```text
本地数据读取是否正常
因子矩阵能否构建
单因子小样本回测是否能跑通
active 因子池是否可读取
小规模 XGBoost 滚动预测是否能跑通
```

这是每次大改代码后建议优先跑的轻量验证。

## 14. 推荐运行顺序

推荐优先使用统一命令行入口，以减少为了切换品种或窗口而反复修改 `config.py`：

```bash
python cli.py single --symbol C.DCE --scope new --start-index 126978
python cli.py composite --symbol C.DCE --models xgboost,logistic_regression --train-window 1200
python cli.py multi --symbols liquid_commodity --no-skip-existing
```

可使用 JSON 文件覆盖任意已存在的 `BacktestConfig` 参数：

```bash
python cli.py composite --config-json my_experiment_config.json --run-id macro_compare_01
```

如果命令行参数和 JSON 中同时提供同一参数，命令行参数优先。

完整研究流程：

```bash
python single_factor_backtest.py
python composite_factor_backtest.py
```

建议先跑单因子，因为综合模型依赖 `active_factors.csv`。

多品种批量研究流程：

```bash
python multi_symbol_backtest.py
python hard_prune_factors.py
# 确认 factor_hard_delete_report.csv 后再执行：
python hard_prune_factors.py --apply
```

该脚本会读取 `config.symbols`，为每个品种生成独立配置。默认输出结构为：

```text
wind_hf_multifactor_output/
  by_symbol/
    C_DCE/
      factor_library/
      single_factor/
      composite_factor/
    M_DCE/
      factor_library/
      single_factor/
      composite_factor/
    multi_symbol_summary.csv
```

默认情况下，每个品种都会先运行单因子流程，更新该品种自己的 active 因子库，然后再运行综合因子流程。这样不同品种的因子库、模型结果和图表不会互相覆盖。

如果启用了因子淘汰机制，多品种流程结束后还会更新全局 `factor_prune_list.csv`。建议先运行 `hard_prune_factors.py` 预演并检查报告，再决定是否用 `--apply` 真正清理因子构造源码。

新增一批 AI 因子后的流程：

```text
把新因子追加到 factor_builders/ 中对应类别文件的末尾
确认 single_factor_new_factor_start_index 指向新一批因子的起始编号
运行 python single_factor_backtest.py
检查 active_factors.csv / rejected_factors.csv
运行 python composite_factor_backtest.py
查看 composite_model_comparison.csv 和诊断文件
```

## 15. AI 自动生成因子的工作流

推荐用 `AI_FACTOR_GENERATION_PROMPT.md` 作为后续 AI 生成新因子的固定脚本。

建议流程：

```text
AI 只生成候选因子公式
人工或程序检查是否存在未来函数
追加到 factor_builders/ 对应类别文件的末尾，保持旧编号稳定
只对新增因子跑单因子回测
因子库自动根据训练/验证表现和相关性判断是否入库
只允许 active 因子进入综合模型
```

不建议 AI 直接做的事情：

```text
直接删除旧因子
直接修改历史回测结果
根据测试集结果反复微调同一批因子
使用 shift(-1)、未来 high/low/close 或全样本统计构造因子
跳过单因子入库，直接进入综合模型
```

## 16. 研究原则和风险点

当前框架已经具备比较完整的研究闭环，但仍需要注意以下原则性风险。

第一，海量因子测试一定会产生幸运因子。因子越多，随机噪声里看起来“很好”的因子越多，因此不能只看某一次累计收益和夏普。

第二，active 因子库只是第一层过滤，不代表因子已经被证明有效。后续仍建议加入月份稳定性、分市场阶段稳定性、IC/ICIR、分组单调性、换手惩罚和收益集中度检查。

第三，XGBoost 是非线性合成工具，不是预测力来源本身。真正的预测力仍然来自高质量、稳定、低冗余的因子。

第四，当前成本默认是 0。最终判断策略可用性时必须纳入手续费、滑点、冲击成本、交易限制和真实可成交性。

第五，训练窗口非常关键。你之前观察到“拉长 XGBoost 训练窗口后效果明显改善”，这通常说明短窗口噪声太大，模型需要更多样本才能稳定识别弱信号。

第六，XGBoost 多周期预测目标已经做了训练标签隔离。预测第 t 根 K 线时，训练集不会使用尚未完全落地的尾部标签；隔离长度为 `xgboost_target_horizon - 1`。例如 `horizon=3` 时，训练集最多用到 `t-3` 的标签，避免 `t-2`、`t-1` 标签中包含预测时点之后的未来收益。

第七，active 因子库仍然需要注意时间冻结。如果 active 因子库是根据同一段最终测试集表现筛出来的，再用于综合模型测试，会形成研究流程层面的样本外污染。更严格的做法是在某个时间点冻结 active 因子库，再测试冻结之后的未来区间。

## 17. 后续优化方向

工程层面：

```text
继续细化 factor_builders/，例如把候选 AI 因子单独放入 candidate 或 experimental 模块
增加候选因子暂存层，避免 AI 因子直接污染正式因子文件
增加更完整的单元测试
增加命令行参数，减少频繁修改 config.py
保存每个 active 因子的公式、来源、生成时间、类别和复杂度
```

研究层面：

```text
加入 train/validation/test 三段式验证
增加月份和市场状态稳定性检验
增加 IC、RankIC、ICIR、分组单调性
加入换手率和交易成本惩罚
对因子做家族聚类，避免同质因子过多
比较 LogisticRegression、Ridge、LightGBM、等权、加权投票等多种模型
```

多品种层面：

```text
当前已经支持 multi_symbol_backtest.py 做多品种独立建模和统一汇总
当前已经输出多品种最终测试集多方法组合净值和图表
增加组合层风险预算和品种相关性约束
增加跨品种 active 因子稳定性比较
增加任务级并行，加快多品种批量回测
```

## 18. 常用命令

```bash
python cli.py --help
python cli.py single --symbol C.DCE --scope new --start-index 126978
python cli.py composite --symbol C.DCE --models xgboost,logistic_regression
python cli.py multi --symbols liquid_commodity --no-skip-existing
python smoke_test.py
python single_factor_backtest.py
python composite_factor_backtest.py
python multi_symbol_backtest.py
python -m py_compile config.py data_loader.py factors.py factor_taxonomy.py runtime_utils.py cli.py factor_builders/__init__.py factor_builders/basic.py factor_builders/common.py factor_builders/parametric.py factor_builders/calendar.py factor_builders/cross_asset.py factor_builders/macro_state.py factor_builders/non_cross.py factor_library.py single_factor_backtest.py composite_factor_backtest.py multi_symbol_backtest.py experiment_utils.py hard_prune_factors.py smoke_test.py
```

## 19. 总结

当前项目已经形成了一个比较完整的高频多因子研究闭环：

```text
数据 -> 因子 -> 单因子验证 -> 因子库 -> active 硬边界 -> XGBoost 滚动合成 -> 训练/测试诊断 -> 多品种汇总
-> 多品种组合评估
```

现在最重要的方向不是单纯继续堆更多因子，而是让因子库变得更稳、更少冗余、更可解释、更可复现。AI 可以显著提高因子生成速度，但真正决定框架质量的是严格的入库规则、稳定的样本外验证、合理的对照基准和对过拟合的持续克制。

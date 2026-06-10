# AGENT_CONTEXT

本文件用于让新的 AI 助手或新电脑上的 Codex 快速接手本项目。开始工作前，请优先阅读本文件和 `PROJECT_DOCUMENTATION.md`。

## 项目定位

这是一个面向中国商品期货 30 分钟 K 线的多因子研究框架，目标是形成可持续扩展的因子研究、因子入库、综合模型预测、多品种组合和风险控制闭环。

当前研究流程是：

```text
Wind / 本地 CSV 行情数据
-> 因子批量生成
-> 单因子训练 / 验证 / 测试回测
-> 因子库入库、去相关、家族配额治理
-> 综合模型滚动训练预测
-> 持仓规则优化和策略回测
-> 多品种组合与横截面机会选择
-> 输出诊断、图表、文档和审计报告
```

## 推荐运行顺序

轻量检查：

```bash
python smoke_test.py
python data_quality_report.py
python leakage_audit.py
python -m py_compile config.py data_loader.py factors.py factor_taxonomy.py runtime_utils.py cli.py factor_library.py factor_metadata.py single_factor_backtest.py composite_factor_backtest.py multi_symbol_backtest.py
```

单品种研究：

```bash
python cli.py single --symbol C.DCE --scope new --start-index 126978
python cli.py composite --symbol C.DCE --models xgboost,logistic_regression
```

多品种研究：

```bash
python cli.py multi --symbols liquid_commodity
```

`liquid_commodity` 是内置商品期货流动性较好主力连续品种池。

## 主要文件职责

```text
config.py                       全局配置中心。
data_loader.py                  Wind / 本地 CSV 数据读取与缓存。
factors.py                      因子总装配入口，负责按需构建因子矩阵。
factor_builders/                各类因子构造模块。
factor_taxonomy.py              因子家族分类和复杂度标签。
factor_metadata.py              因子元数据导出。
factor_library.py               active/rejected/all 因子库管理。
single_factor_backtest.py       单因子训练、验证、测试回测和入库。
composite_factor_backtest.py    综合模型滚动训练、预测和回测。
multi_symbol_backtest.py        多品种批量回测和组合层汇总。
experiment_utils.py             实验快照。
runtime_utils.py                日志和运行 manifest。
project_fingerprint.py          因子源码哈希。
leakage_audit.py                未来函数 / 泄露静态审计。
data_quality_report.py          数据质量检查。
hard_prune_factors.py           因子硬删除工具。
cli.py                          推荐命令行入口。
PROJECT_DOCUMENTATION.md        详细项目文档。
AI_FACTOR_GENERATION_PROMPT.md  AI 自动生成新因子的提示词脚本。
```

## 已完成的重要机制

1. 因子模块已经拆分到 `factor_builders/`，每类因子独立维护。
2. 数据读取已经拆到 `data_loader.py`，支持本地缓存和 Wind 拉取。
3. 单因子回测有训练集、验证集、最终测试集三段。
4. 单因子汇总加入了 `IC / RankIC / ICIR / RankICIR / 分组单调性 / 分组收益差`。
5. 因子库入库同时参考收益、夏普、胜率、训练/验证表现、相关性和家族配额。
6. active 因子库默认上限为 200，并启用因子家族配额。
7. 综合模型只允许从 active 因子池里选因子。
8. 综合模型支持 XGBoost、逻辑回归、随机森林、ExtraTrees 等模型对比。
9. XGBoost 是滚动训练预测，且多周期目标已做训练标签隔离，避免标签泄露。
10. 综合策略加入了持仓规则优化，包括最小持仓期、反转冷却、小变化忽略和可选平滑。
11. 多品种组合层加入了横截面机会选择，默认使用上一根模型概率优势做 TopN 品种筛选。
12. 多品种组合层支持等权、波动率倒数、正夏普加权、波动率目标和回撤降仓。
13. 项目有 `leakage_audit.py`、`data_quality_report.py`、`project_fingerprint.py` 等工程审计工具。

## 重要设计原则

1. 不要让综合模型直接使用全部因子，必须以 active 因子库为硬边界。
2. 不要只看最终测试集调参，最终测试集应尽量作为留存样本外评估。
3. 新增因子后优先跑单因子增量测试，不要直接塞进综合模型。
4. 因子入库不仅看收益和夏普，还要看训练/验证稳定性、预测指标和相关性。
5. 多品种组合层的权重和机会选择不能使用当前或未来收益。
6. 大规模输出、缓存、图表和 CSV 不应纳入 Git。
7. 修改交易逻辑后，至少运行 `py_compile` 和小样本测试。
8. 如发现未来函数风险，优先修复，不要为了收益保留可疑逻辑。

## 当前重点可继续优化方向

1. 多 horizon 自动对比：比较 `xgboost_target_horizon = 1,2,3,5` 的验证集和测试集表现。
2. 训练样本时间衰减权重：滚动窗口内近期样本权重更高。
3. 模型一致性过滤：多个模型方向一致时才交易，分歧时降仓或空仓。
4. 市场状态分层：趋势/震荡、高波动/低波动、日盘/夜盘分别诊断和校准阈值。
5. 更真实的手续费、滑点、冲击成本和容量模型。
6. active 因子库版本冻结和严格样本外跟踪。

## Git 同步建议

建议提交代码、文档和轻量配置，不提交以下内容：

```text
wind_hf_multifactor_output/
__pycache__/
*.pyc
*.png
*.pkl
*.log
```

如果需要跨电脑复现实验，应单独同步必要的原始行情 CSV 或通过 Wind 重新拉取缓存。

## 给后续 AI 的工作方式

1. 先读 `AGENT_CONTEXT.md` 和 `PROJECT_DOCUMENTATION.md`。
2. 再根据用户当前问题读取相关源码。
3. 修改代码前先给用户一句简短说明。
4. 手工改文件请用补丁方式，避免误删用户已有改动。
5. 修改后运行必要验证，并在最终回复里说明改了什么、如何验证。

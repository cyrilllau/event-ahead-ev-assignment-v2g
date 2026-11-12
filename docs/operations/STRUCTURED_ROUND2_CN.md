# 结构化合成 EV 实验 Round2：入口、身份与安全续跑

登记日期：2026-11-12。本次登记时 validation 仍在执行，尚未完成整轮最终验收。
本文是私有开发分支的操作说明，不是实时状态页、结果表或论文采用决定。
实时状态及数值证据只在下述外部 store；不得据本文推断任务已结束或全部通过。

本分支是新的有限套餐合成 P3/P2 实验，**不是原冻结 L08 丰富矩模型的恢复运行**。
通用边界见 [WORKFLOW_CN.md](WORKFLOW_CN.md)；原工程状态见
[IMPLEMENTATION_STATUS.md](../architecture/IMPLEMENTATION_STATUS.md)。

## 1. 实现入口与职责

| 入口 | 职责 |
|---|---|
| [l08/catalogue.py](../../src/ev_assignment/l08/catalogue.py) | `build_first` / `build_recourse`；计数、逐车套餐与原逐车矩阵的映射和审计适配 |
| [methods/catalogue.py](../../src/ev_assignment/methods/catalogue.py) | `solve_structured(case, ambiguity, control, ...)`；连续预分离、稀疏分解、完整支持整数 UB、逐场景原矩阵展开审计 |
| [methods/catalogue_direct.py](../../src/ev_assignment/methods/catalogue_direct.py) | `solve_direct(...)`；同一有限支持问题的直接扩展模型对照；native incumbent 本身不是已审计 UB |
| [catalogue_probability.py](../../src/ev_assignment/methods/catalogue_probability.py) | `calibrated` / `finite_domain`；显式支持、矩边界及合法概率集合身份 |
| [structured_round2.py](../../src/ev_assignment_experiments/datasets/structured_round2.py) | 固定物理负荷、扩充可选同类车池；输入根必须显式指定 |
| [stage.py](../../scripts/structured/stage.py) | 使用公共 `BudgetLedger` 准入、限定单个 worker、保存父子身份并结算整个 wrapper 墙钟 |
| [validation_campaign.py](../../scripts/structured/validation_campaign.py) | 按外部 `FREEZE.json` 的顺序调度；核对已存在 Run 的身份与结算，不重复已结算工作 |
| [validation_worker.py](../../scripts/structured/validation_worker.py) | 六方法、计数/逐车表示及预分离开关的单 job 入口；保留进度、见证、模型大小与服务口径 |
| [oos_worker.py](../../scripts/structured/oos_worker.py) | 对冻结固定政策作配对 OOS 评价；核对政策、样本、方法和物理身份 |
| [validation_audit.py](../../scripts/structured/validation_audit.py) | 最终证据核对、同问题区间检查、调用预注册统计模块、输出证据哈希清单 |
| [report_round2.py](../../scripts/structured/report_round2.py) | 只读格式化外部报告；不重算模型、物理、风险或 bootstrap |

核心接收显式 case、概率域与 `Control`，不推断数据目录，也不负责累计预算。
实验脚本复用现有公共账本；不是另建优化平台，也不接管原 PowerAI DET 等历史进程。
worker 通常由 wrapper 传入 `OUT SECONDS ...`，不要绕过 wrapper 手工执行科学 worker。

计数表示要求同一类型、同一套餐的车辆系数一致；不同出发套餐的 SOC 可以不同。
整数计数可按固定合同规范展开到逐车套餐，再映射原矩阵，包括未选套餐的 Big-M 行。
这不证明原 LP 松弛等价，也不证明有限套餐限制对旧完整任务模型无损。

## 2. 冻结科学范围与六方法

核心冻结族为固定 N6 物理需求、N30 可选车辆、G6、96 时段、72 套餐；
TRAIN400、独立 CAL2000、每 seed 的 OOS1000；四项费用乘数均为 1。
这些是设计参数，不是运行结果。扩充候选池不等于增加实际服务需求。
网络、能量、路线重放及资格内核沿用原实现；类型/SOC、套餐、天气、负荷、预测与校准
属于新合成假设。所有细节以冻结科学身份及源清单为准。

| 方法 | 优化支持与目标 |
|---|---|
| DET | 自身预测 singleton，支持数 1；不能将它的子问题身份伪装为 TRAIN400 |
| SAA | 同一 TRAIN400，使用声明的分层权重；精确 dyadic singleton 的舍入差异另行记录 |
| ARO | 同一 TRAIN400 上的完整概率 simplex |
| P3 | 同一 TRAIN400，保留 L、R、LR 三个校准矩 |
| P2 | 相同 case/support，只删除 P3 的联合矩，保留两个边际矩 |
| P3-noadvance | 相同 P3 风险域，只允许冻结的最晚出发套餐；不是灾后重新决策 |

P3 的三行不能称为恢复原 322311 行 ST-DRO。CAL 的逐坐标 Wilson 区间不提供同时覆盖保证。
不同方法的训练目标不可直接作为共同政策排名；共同 OOS 使用冻结样本和名义权重。
同问题算法时间对照是计数与逐车套餐表示；P2、无提前和改变类型/SOC 的 G 压力不是
纯表示对照。已有健康 DET 任务同机运行，不声称独占服务器、CPU 绑核或受控后台负载。

冻结计划包含 V0 准备、18 个核心 job、3 个逐车对照、1 个无预分离、3 个 G 压力、
3 个 OOS 和 1 个 VREPORT，共 30 个 job。种子及确切参数、顺序、预算见 `FREEZE.json`，
不因 validation/OOS 结果更换 seed、调整费用、模型或生成规则。

## 3. 外部位置与身份链

以下为定位入口；配置、源码清单和实际执行主机的账本才是权威。
本地镜像可以滞后，不能用镜像证明远端进程死亡。

| 位置 | 路径 |
|---|---|
| 远端原开发根 | `/home/Shixin/ev-assignment-synthetic/structured-fleet-design-20261112` |
| 远端 Round2 根（下文 `ROUND2_ROOT`） | `/home/Shixin/ev-assignment-synthetic/structured-fleet-design-20261112/round2` |
| 远端执行 store | `/home/Shixin/ev-assignment-synthetic/structured-fleet-design-20261112/store` |
| 本地 Round2 镜像 | `/Users/shixinliu/Desktop/Research/with Lusha/event-ahead-ev-assignment-v2g-store/validation/structured-fleet-design-20261112/round2` |

私有源码分支为 `codex/structured-case-round2-20261112`。
validation 冻结计算源码为 `1bd6cf7dd27a9ddfef8f732bab3e84fd93b56924`，
执行 release 为 `ROUND2_ROOT/releases/r08`。以其 `RELEASE.json`、wheel/脚本 SHA256、
环境和 `validation/FREEZE.json.source_release` 相互核对，不只检查 Git HEAD。
报告 formatter 最初提交为 `bf37879`；后续格式化修订以最终报告记录的脚本哈希为准，
它是独立格式化工具版本，**不替换冻结计算源码**。
开发 R0/R1/R2/R5 保留各自历史 release；不得用 r08 重新标记旧结果。
资源中断后的调度恢复采用独立控制器 `controllers/52933fe/validation_campaign.py`，
仍调用原 r08 worker/wrapper；绑定记录为 `validation/CONTROLLER_RECOVERY_MANIFEST.json`。
该控制器只认可已核实的保守结算链，不修改冻结计算源码或重复中断的科学 Run。

| 身份层 | 证据及含义 |
|---|---|
| Source | 不可变 release、commit、实际 wheel/脚本哈希、执行环境；新源码不覆盖旧 release |
| Freeze | `validation/FREEZE.json`：完整 job 列表、方法/seed、资源、统计规则、源码和输入入口 |
| Case | 原始输入 `source-manifest.json`；每个已准备 case 的 ordered IDs、权重、物理/套餐/校准身份与 pickle 哈希 |
| Run | `ROOT/STAGE/RUN_ID`，含 `EXECUTION_CONFIG.json`、`CLAIM.json`、`CHILD.json`、`WAIT.json`、`RESULT.json`、日志/见证 |
| Attempt | `CLAIM.attempt` 及 host/boot/PID/create-time/argv 身份；不是可随意复用的 PID |
| Budget | `DESIGN_CONFIG.json` 的显式 `ledger_store` / `budget_group`；公共账本定义、hash chain、active claim、settlement receipt |

开发配置在 `ROUND2_ROOT/DESIGN_CONFIG.json`，组为 `structured-design-development-v1`，
总上限 7200 秒，保留原轮次已经发生的全部费用。
验证配置在 `ROUND2_ROOT/validation/DESIGN_CONFIG.json`，组为
`structured-design-validation-round2-v1`，总上限 28800 秒。
已用/剩余额度只读实际账本，不在 Git 文档复制易过期的数值。
失败、重试、构建、准备、审计及 wrapper 开销均累计；换 Run 或格式化报告不清除旧成本。

原始私有输入在原开发根的 `inputs/`，入口清单为 `inputs/source-manifest.json`。
V0 在任何 validation 优化前生成并封定所有 TRAIN/OOS case；位置由冻结 job 指定，
当前为 `validation/V0/prepare-r08/` 下的 `validation_SEED.pkl`、`oos_SEED.pkl` 及对应
`*_MANIFEST.json`。后续方法读取同一已保存 case，不重新生成各自样本。
pickle 只限可信、清单绑定的私有输入；报告脚本不打开 pickle。

## 4. 安全观察与续跑

**当前执行中的 validation 保持原状；不要再启动一个 campaign。** 下列流程用于后续
确需恢复调度时，不是新计算授权，也不是重启旧任务的快捷方式。

1. 在实际执行主机只读检查两个 `DESIGN_CONFIG.json`、`FREEZE.json`、相应 release 清单，
   以及 `CASE_DESIGN_STATUS.md`、`CAMPAIGN_PROGRESS.jsonl`、当前 Run 的进度/结果。
2. 使用公共 `BudgetLedger(config["ledger_store"], config["budget_group"]).read()`
   读取并校验两条账本；检查 active claim、父/子身份及同机已有 controller。
   active 为空不单独证明 controller 已退出，它可能处于两个 job 之间。
3. 若父/子或 controller 健康，继续观察，不重复提交。离线、未知或 PID 不匹配不能当作死亡。
   失联/死亡场景先按既有身份与账本恢复规则核实并留证；本脚本没有手工清锁、删 Run、
   删除 ledger、重建零费用预算或接管原生产 PID 的恢复入口。
4. 仅当原 controller 确已退出、相关 claim 均已合法结算、冻结身份一致且有尚未执行的
   已授权 job 时，才可恢复登记的 controller。有 WAIT 须匹配 CLAIM 的
   argv/release/FREEZE/配置及账本 settlement receipt。缺 WAIT 默认拒绝；唯一受支持的
   例外是 `RECOVERY.json` 严格绑定原 claim、原进度、父子死亡/OOM观测及公共账本的
   全预留保守结算事件。它不创建虚假 WAIT/RESULT，不将资源中断视为科学通过。
5. 已结算的失败、超时也保留并复用；controller 不自动重算这些 job。
   它恢复的是剩余 job 的调度，**不承诺恢复已终止求解器树或从逐场景中断处继续同一 worker**。
   确需实现修复时按既有授权登记新 release/Run、沿用累计预算，保留原失败；不按验证结果改模型。

满足上述前提后，冻结 controller 的调用形式如下（本说明不执行）：

```bash
ROUND2_ROOT=/home/Shixin/ev-assignment-synthetic/structured-fleet-design-20261112/round2
ROUND2_RELEASE="$ROUND2_ROOT/releases/r08"
ROUND2_CONTROLLER="$ROUND2_ROOT/controllers/52933fe/validation_campaign.py"
"$ROUND2_RELEASE/.venv/bin/python" "$ROUND2_CONTROLLER" \
  "$ROUND2_ROOT/validation" "$ROUND2_RELEASE"
```

预算不足不启动后续 job；不得提高上限或新建预算组绕过累计成本。
准备门失败或 `IMPLEMENTATION_OR_IDENTITY_ERROR` / 无效证书 / 不相容区间等状态必须先处理，
不能以缺失结果为零或删除失败恢复“全通过”。资源以冻结配置为准；r08 primitive 调用
上限 60 秒、最多一个新增 solver job，历史 release 的实际设置不能事后改写。

## 5. 不同验收状态必须分开

| 字段或状态 | 可以说明什么 | 不能推出什么 |
|---|---|---|
| attempted | 有匹配 CLAIM 的执行尝试 | 不保证实际发生 native 求解 |
| settled / `CAMPAIGN_COMPLETE.json` | 对应费用已结算；campaign 标记所有冻结 job 已结算 | 不等于全部科学任务通过 |
| `gap_passed` | 保存的 LB/UB 满足冻结项目阈值 | 不等于原矩阵审计完成 |
| `audit_complete` | 原审计匹配完整有序支持，残差满足项目要求 | 不单独证明优化 gap 或论文机制成立 |
| `acceptance_supported` | 正式接受记录、合法界差、完整原审计及结算条件同时支持 | 不要求把每个 native 状态重写成 OPTIMAL，也不是有理数全局证明 |
| `GAP_PASSED_AUDIT_INCOMPLETE` | gap 已闭合，原逐车展开审计仍不完整 | 不能称为优化器 gap 未收敛，也不能当作完整验收通过 |
| VREPORT passed | 最终证据整理/检查按其自身门完成 | 不会使失败或超时 job 变成通过 |

原生状态、界来源与缺失值保留。原审计覆盖使用 `len(original_audit)`；
`audit_completed` 是可能滞后的 progress 计数，不替代实际已保存行数。
gap 采用 `UB-LB <= max(1, 0.01*max(1, abs(UB)))`，原行容差 1e-9、目标分项容差 1e-5。
native/有限盒 LB 与数值审计 recourse 加风险对偶 UB 的证据级别须明确，不称为
精确有理数物理可行性证明。固定政策风险下界不能冒充全局优化 LB。

OOS 仅使用具有完整原审计的固定整数政策；其训练 gap 接受状态仍单列。
只闭合 gap 但审计不全的政策不能进入冻结 OOS 评价。
正式配对 CI 要求完整 1000 样本及六方法齐全，否则为 `NOT_ELIGIBLE`；共同前缀只能描述。
VREPORT 调用 [structured_statistics.py](../../src/ev_assignment_experiments/analysis/structured_statistics.py)：
四层共享重采样索引、保留层样本数/质量、500 次、seed 为 `910000 + validation_seed`，
报告 P3 减各对照的成本均值/CVaR95及关键 ENS 均值差。
这是条件于固定训练政策的 OOS 抽样不确定性，不是概率集覆盖；不在最终 formatter 重算。

## 6. 报告、复现与私有数据边界

最终报告只写到外部 `ROUND2_ROOT`：

- `CASE_DESIGN_ROUND2_REPORT_CN.md`
- `CASE_DESIGN_ROUND2_RESULTS.json`
- `CASE_DESIGN_STATUS.md`
- `REPRODUCE_CN.md`

formatter 接口为 `report_round2.py ROOT [--stopped REASON] [--display-root LOCAL_ROOT]`。
正常收尾要求两账本无 active、campaign 完成且冻结 job 的结算收据匹配；当前 validation
进行中不要生成最终报告。`--stopped` 仅允许有明确理由的不完整快照，不会停止/接管进程，
也不证明仍登记 active 的任务已死。`--display-root` 只映射 Markdown 本地链接，
不改证据中的远端路径、哈希或复现命令。已有最终报告拒绝覆盖；状态页原内容另行保留。

复现先核对每个 Run 的实际 release、环境、输入清单、case、支持顺序/权重和概率身份，
再阅读外部 `REPRODUCE_CN.md`。历史 CLAIM 没有明文 argv 时，接口重建必须标为推定，
不能冒充实际执行命令。源代码 checkout 的新 HEAD 或报告版本不是替换历史计算版本的依据。
只读格式化/哈希核对、科学审计/统计、真正重跑是不同操作；后两者仍受各自执行与预算规则约束。

Git 只存源码、测试、协议和本操作文档。不得加入私有数据、结果表、CASE.pkl、样本清单内容、
日志、见证、检查点、账本、许可证或凭据；本文不复制任何运行的 LB/UB、部署、耗时或 CI。
原 Round1 报告和失败账本保持原件。完整输入/源码/结果/证据闭包保存在外部私有 store，
本地镜像以清单和哈希校验归档，不能自动视为可执行恢复环境。
公开复现须在研究者明确选择论文结果后另立公开仓库，本任务没有发布或新增求解授权。

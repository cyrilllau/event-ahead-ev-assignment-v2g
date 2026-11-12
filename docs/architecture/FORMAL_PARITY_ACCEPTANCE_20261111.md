# 正式规模历史方案对账验收

日期：2026-11-11 America/Los_Angeles（证据时间为 2026-11-12 UTC）。
本次完成的是**精确历史方案的数值评价一致性**。801 个单元通过；原始 gap 元数据有一项
不一致，已单列。没有重新求解完整整数规划，没有证明重构后的优化轨迹或全局界相同。

## 实际覆盖

| 项目 | 真实验证与结论 |
|---|---|
| 输入 | 2,047 份冻结输入逐文件校验；30 EV、forecast44/train400；OOS 未打开 |
| 原方案 | DET RECOVERY_01/START；DRO ATTEMPT_05/BEST 指向 ATTEMPT_04；ARO 原 outer_00027 的最佳已评方案；每项精确 SHA 和向量核对 |
| DET | 1 个正式 forecast44 固定方案 LP；原第一阶段、原调度/P/Q/SOC 重审；目标差约 3.3e-11 |
| DRO | 全部 400 个固定方案 LP；256 份原调度重审，144 个原 analytic-zero 声明通过新的固定方案 LP 独立核实 |
| ARO | 全部 400 个固定方案 LP，400 份原调度重审；完整 max 聚合区间相交 |
| DRO 风险 | 原 322,311 条概率约束/45,139 条等价压缩行、400 项顺序及原概率证书重审；新旧风险区间相交；概率求解调用 0 |
| 证书读回 | 801 项全部重建模型、读回新的 primal/dual/有限界；额外求解调用 0 |
| 自动检查 | 90 项本地单元/集成/native 测试通过（包含新增 10 项和原 12 项真实本地 Gurobi 测试）；原 portable scientific 检查通过；ruff/wheel 通过 |

新旧逐场景目标最大绝对差 `5.820766091346741e-11`；recourse 成本组件最大差
`1.1641532182693481e-10`。ENS/V2G/shortfall 指标最大差约 `3.3e-12`。
新旧物理残差最大 `2.6943780540023e-11`，保持原阈值。原已审方案的路线、窗口、
采购量及所有第一阶段辅助列保持；未借机更改模型、样本、物理阈值或接受精度。

数值上界的保护量会随新 LP 见证变化：DRO 对偶平面显式平移约 `5.17e-6`，重新核对
全部场景后，新旧风险区间相交。不能将“目标数值一致”写成“所有 guarded LB/UB 字节相同”。
此外有一个先行 DET 资源探针，证据为 `det-probe.json`；不计入 801 项正式验收单元。

## 原状态与未证明事项

- DET 原 `STOPPED_NOT_CERTIFIED`、DRO 原用户停止状态及 ARO 原中断/待评价状态保持。
  此处使用已保存的最佳已评方案，未将未审 pending incumbent 混入。
- DET/ARO 原 native LB 只做来源读回及 gap 算术核对，不在本次重新证明全局 MIP LB。
  DRO 的采用 LB 来自原 93-scenario continuous relaxation，不能改称完整整数问题的新 LB。
  该下界证书及上游指针哈希、原 LB 值已核对；未导入/重审整个下界依赖证明闭包。
- ARO 原 `external_gap.relative_gap=0.5026055208069328` 是过时摘要；按同一状态最新
  LB/UB 重算是 `0.5020178057059171`。报告 `all_metadata_consistent=false`，保留旧字节，
  不回写 PowerAI，也不把它当成重构引入的数学差异。
- 本次未重跑原全局规划搜索，不证明重新优化得到相同最优解或收敛轨迹；不授予方案接受、
  正式 OOS 准入、论文结果选择或公开发布。

## 可追溯证据

输出均在代码仓库外的 research store：`validation/formal-parity-20261111/`。
`report.json` 汇总逐方法指标/区间/原生界；`accepted/{DET,DRO,ARO}/` 是单元、数值见证
及独立 readback；`old/` 为 1,330 份选定历史文件。原始结果、表格、日志、输入和许可证
不进入 Git。私有 Git 只保存验证代码、测试、协议和此验收索引。

| 对象 | SHA256 |
|---|---|
| report.json | `ba8dcbec937ebc2c63a16d9e95191e3749b8bacc9a2c03ac66ad18750deb4b69` |
| 执行源码 source.zip | `2d11fe13f264da7b17583d5fc4d5fec8f486a0f5558b52b879044a4aa76a68be` |
| 固定输入清单 | `310d39bcafa28a095460aa81d593cfef83e7dcb6f6a4009bdef06ce074258d7a` |
| 历史证据清单 | `4c05b1e262b4f41b6db7682bd6acba9076aa42b2797a6d5b628b1a4b1266dd25` |
| 对账 spec | `a7cab1f114b22897e3ba54d2eb374ca4cd21f940bf36c7440cbd3fd7bd159e82` |

一次刻意改变 checker 源码的续跑被来源门在任何求解之前拒绝，记录为
`source-change-negative.json`。缺场景、错序、混合方案、坏哈希、不合法物理见证和不相交
目标区间的负例保持拒绝；未删除科学断言或放宽检查以得到通过。

PowerAI 全程只读。末次原父/子 `2850511/2850521` 和 Nov 3 起始时间保持，入口 SHA
`8778427bfefd9548e939e6fd17716618b0cd00ea58672c73b0b608321a224fe8` 未变。
没有写远端源码/current/状态/预算，没有发送信号或启动远端求解。源工作区未修改。

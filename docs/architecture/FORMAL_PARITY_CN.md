# 正式历史结果对账

2026-11-11 用户授权。D/E/F/G 工程验收不等于正式历史结果等价；本项是追加科学验证。

## 锁定范围

- 只读 PowerAI 已保存的 L08 DET/DRO/ARO 方案和证据，记录观察时点、原路径和完整文件 SHA。
- 使用已冻结的 30 EV、forecast44 / train400 数据清单；不读取或解锁 OOS。
- 用活动代码重建同一方案的物理模型，重新计算第一阶段成本及固定方案 recourse；
  比较旧可行见证、成本分量、场景顺序、风险聚合和原 LB/UB/gap 算术。
- 原生全局 LB、求解状态及 gap 保留其原证据等级。固定方案 LP 通过不能证明新的整数
  规划求解获得相同最优解、相同树或相同全局 LB，也不能把未收敛结果改为收敛。
- 多重最优 recourse 可导致不同调度/成本分量；先重审原见证，再判断目标区间是否一致。
  物理阈值、目标口径、固定路线及随机流不修改。
- PowerAI 不写入、不启动求解、不操作进程/预算/current。结果与输入保存在外部 research
  store；Git 只保存检查代码、测试、协议和验收索引。

## 顺序与当前状态

1. 已完成：读取旧 campaign 状态及 L08 结果指针；找到 DET RECOVERY_01 原方案审计，
   DRO ATTEMPT_05/BEST 指向 ATTEMPT_04 的完整 400 场景记录。ARO 指针需继续追踪。
2. 已完成：冻结 1,330 份所选原始文件（含逐场景 primal/cut、方案和状态），核对哈希。
3. 已完成：DET forecast44 实际固定方案重算及原见证审计；目标差约 3.3e-11。
4. 已完成：DRO/ARO 各 400 项重算及风险聚合通过，所有单元逐项落盘。
5. 已完成本地验收：801 项零求解独立读回，90 项测试及坏来源/缺场景等负例通过。
   [实际覆盖和限制](FORMAL_PARITY_ACCEPTANCE_20261111.md)；私有 Git 交付仍须 CI 通过，
   实际 commit/CI/main 读回另存外部 `git-delivery.json`，不能由本清单代替。

每个对账结论分别列出：输入/方案身份、原见证物理审计、固定方案目标、风险聚合、
原生界来源。报告只对精确历史快照有效，不声称与持续运行中的最新 incumbent 同步。

初始证据索引：外部 `validation/formal-parity-20261111/inventory.json`，
观察时点 `2026-11-12T03:19:33.355590+00:00`。此索引不是验收结果。

## 可重复执行的入口

`scripts/validation/check_formal_parity.py` 只读取明确传入的本地证据和输入；每工作片
最多 60 秒，完成单元逐项封存。重复命令只继续同一 source/spec，不改变原规划预算。
每次新增固定方案 LP 限制为一个线程；此检查不产生新的整数规划接受决定。

```bash
python scripts/validation/check_formal_parity.py \
  --spec /ABS/parity/spec.json --output /ABS/parity/accepted --seconds 55
python scripts/validation/check_formal_parity.py \
  --spec /ABS/parity/spec.json --output /ABS/parity/accepted --seconds 55 --reaudit
python scripts/validation/summarize_formal_parity.py \
  --spec /ABS/parity/spec.json --bounds /ABS/parity/bounds-spec.json \
  --validation /ABS/parity/accepted \
  --checker /ABS/repo/scripts/validation/check_formal_parity.py \
  --output /ABS/parity/report.json
```

两个逐项命令重复到 `COMPLETE`。`--reaudit` 重建物理模型并读回已存 primal/dual/
有限界证书，求解器调用为零。汇总要求全部单元与读回齐全，不能用部分记录出完整结论。
历史解析只接受清单文件，拒绝混合方案、缺场景、错序、坏哈希、非法路径及非有限区间。

私有 spec 包含 `inputs`、`input_manifest`、`evidence_root`、`evidence_manifest`、
`evidence_manifest_sha` 和 `cases`。每个 case 锁定 `method`、`point`、`point_key`、
`point_sha`；DET 另需 `audit`/`names`，DRO/ARO 另需 `records`。证据清单映射相对路径
到 `bytes`/`sha256`；bounds-spec 显式选择 det_result/det_native、dro_best/dro_final/
dro_sources、aro_state/aro_native。源文件、spec 和输入清单的 SHA 绑定在 `RUN.json`。

本项使用原生区间相交判定数值目标一致，不扩大已有 LP 证书阈值；旧 primal 与原记录
目标的交叉核对使用绝对 1e-7。DRO 读回原概率证书，并将同一对偶平面按新 LP 数值
保护量作显式平移，重新检查全部场景和原概率约束，不重新优化概率或整数主问题。
ARO 重新聚合完整 400 项 max；不同风险目标不作跨方法的优劣比较。

新发现的原始问题：ARO 旧状态的 `external_gap` 与最新 LB/UB 不一致。保留原字节，
在汇总中单列 stale metadata 和按原 LB/UB 重算的值；不把它归咎于重构，也不写回服务器。

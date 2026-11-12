# 研究工作流与操作边界

代码、研究记录、计算位置分别管理。代码仓库不保存 results/artifacts；下文 `ABS`
是仓库之外的绝对研究存储路径。新流程不接管旧 PowerAI campaign。

## 安装与入口

Python 3.12：`uv sync --locked --group dev`。实际求解加 `--extra gurobi`，使用
Gurobi 12.0.3 和用户已有许可；导入包不初始化许可证。所有 CLI 的 `--store ABS`
放在子命令之前。可选 `evcs` 入口仅为保留的历史工具，不能据此启动旧工作流。

```bash
ev-assignment --store ABS plan --experiment experiments/l08/smoke-det.toml
ev-assignment --store ABS job launch --id RUN_SHA
ev-assignment --store ABS job status --attempt LAUNCH_TOKEN
ev-assignment --store ABS audit --id ATTEMPT_SHA
ev-assignment --store ABS context --study l08-smoke-det --refresh --markdown
```

`RUN_SHA` 是计划记录，`LAUNCH_TOKEN` 是父进程启动身份，`ATTEMPT_SHA` 是完成后封存
记录，三者不同。DRO/ARO 小例分别使用同目录配置。小例明确为 development。
实际 frozen400 输入使用 `input_kind="frozen_l08"`、显式 `input_root` 和
`input_manifest` SHA256；资源/预算必须来自该科学任务自己的准入决定。
`L08Dataset` 仅允许 forecast44 和已锁定 train/0000..0399，拒绝 OOS。

Plan 固定源码、依赖环境、输入、方法、随机种子、顺序和资源参数。源代码有改动时
不能继续原 Run；新 Run 可沿用同一个 budget_group。继承费用用
`inherited_seconds` 与 `inherited_evidence`（已封存 Ref 列表）明确登记。
同 budget_group 的上限/继承定义一经建立不能修改；实际父进程从 claim 至 wait
的完整单调时钟费用累加，包含构建、审计与子进程启动开销。

## 恢复与状态

```bash
ev-assignment --store ABS job launch --id RUN_SHA --resume CHECKPOINT_SHA
ev-assignment --store ABS job recover --budget-group GROUP
```

续跑要求精确相同源码/环境、配方、数据和方法。DRO/ARO 恢复已封存切面、概率行、
未完成轮次、逐场景结果及有效界；不会重复已完成场景。DET 恢复原方案 warm start
和原界，不声称保存了 Gurobi branch-and-bound tree。缺失值为 null，不填零。
预算不足即停止；不会自动升级内存、换算法或创建新预算。

一个 study 同时只允许一个父任务。身份包含 host/boot/PID/create-time/argv。
父进程核验子进程 READY 后才允许求解。未知/远程身份不等于死亡。
恢复未知费用的死亡任务会保守扣除原预留余额，并保留原因。健康或未知进程拒绝恢复。
只有 supervisor 可发送终止信号给它自己创建且身份仍匹配的 worker。

执行、物理检查、收敛与研究选择分别记录。中断有 checkpoint 不等于已完成科学审计。
`audit` 使用封存原始物理 primal/dual 证据重新组装并检查，无新优化调用；native MIP LB
的证据级别仍为 native_solver，不能宣称已独立重证全局有理数界。

## 同步、离线与 AI 上下文

```bash
ev-assignment --store ABS sync --study STUDY --mode results --destination LOCAL_ABS
ev-assignment --store ABS sync --study STUDY --mode full --destination LOCAL_ABS
ev-assignment --store LOCAL_ABS pull --host powerai --remote-store REMOTE_ABS --attempt ATTEMPT_SHA
ev-assignment --store ABS watch --study STUDY --destination LOCAL_ABS --seconds 60
ev-assignment --store ABS context --study STUDY --limit 10 --offset 0
```

- metadata：带观察时间的摘要/外部引用；明确 `archived=false`，不冒充可复现归档。
- results：选定 attempt/audit/decision/publication 的完整引用闭包，包含重审和重绘所需
  输入、源码、方案、证据。恢复所需 checkpoint 可能在闭包中，不能为减小体积删依赖。
- full：该 study 的全部记录与依赖，含失败、事件及未被最终结果选中的 checkpoint。

SSH pull 是只读、显式 sealed-store 传输，不读取活动 campaign 文件。单次传输可续传、
校验前缀/完整 SHA，永不覆盖完整坏文件。watch 监视本地或挂载的 sealed store，并在
给定时间到期后退出；SSH 持续追平用 watch 的 `--host powerai --remote-store REMOTE_ABS`，只扫描新流程的已封存记录；断线观察保留原归档并记录 unavailable，不创建系统定时任务。

`context --refresh` 仅查询执行该命令主机的已登记进程；远程状态需在相应主机运行后
保存/同步 observation。输出列明目标/协议、四类状态、精确采纳引用、失败路线、
具名 next-action 与依赖，以及省略数量/分页入口。无决定时显示未登记，不猜下一步。
SQLite 是可重建索引，不参与在线复制：`ev-assignment --store ABS index`。

## 决策、共同评价与论文包

```bash
ev-assignment --store ABS decide --attempt ATTEMPT_SHA --outcome approximate --reason '用户精确接受记录' --actor 'recorded decision maker'
ev-assignment --store ABS next --study STUDY --action '下一项明确任务' --reason '已有依据' --actor 'researcher' --depends-on decisions/DECISION_SHA
ev-assignment --store ABS select --definition /absolute/selection.json
ev-assignment --store ABS bundle --attempt ATTEMPT_SHA --publication PUBLICATION_SHA --destination /absolute/new-bundle
ev-assignment --store ABS verify-bundle --destination /absolute/new-bundle
```

selection JSON 包含 study、title、items；每项固定 label、attempt Ref、decision Ref、
generator=`objective_table`。扩展图表生成器时增加具名实现和测试，不执行任意 shell。
显式决定不得改写原 gap/native status/convergence；更正通过 supersedes 追加。
共同评价用 `analysis.evaluation.lock` 锁定所有方法、计划 SHA、配对样本 ID、随机流和
协议后，才可 `open_sample`。数据读取者显式注入，不会擅自打开未授权 OOS。
`paired_summary` 拒绝错序、缺失和非有限值。

导出包内有完整 archive、实际 source release、环境、结果 CSV/SVG 表及 `reproduce.py`。
`figures` 从封存结果重绘，`verify` 重审，均不求解；`smoke` 显式新 store、小例和许可证。
完整复跑用 `python reproduce.py full --attempt ATTEMPT_SHA --store NEW_ABS --budget-seconds SECONDS`；要求安装精确选定源码、显式全新存储/预算，自动绑定包内输入，不恢复旧 campaign 的预算。
工程重构只构建本地示例包；最终论文选择、完整规模求解验收和公开发布属于后续研究工作。

## 不可变代码同步

Git 同步源码：本地 commit/push 私有分支；计算机 clone/fetch 后固定 commit，在一个
新 release 路径安装。`release --run RUN_SHA --destination NEW_ABS` 导出实际源码及 SHA
清单；校验 `runtime.release.verify` 后在新环境安装。不得覆盖活动 release、current
链接、任务 cwd 或输入。服务端临时开发必须在独立分支提交后由本地拉取验证，不将运行
目录 rsync 到主代码。当前重构没有在 PowerAI 新部署、启动或重启求解器。

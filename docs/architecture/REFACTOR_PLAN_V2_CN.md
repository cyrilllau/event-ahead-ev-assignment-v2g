# EV Assignment 完整重构计划 v2

批准日期：2026-11-11（America/Los_Angeles）。本文件保存用户批准的设计；
实现状态另见 `docs/architecture/IMPLEMENTATION_STATUS.md`。计划不是运行成功证据。

## 1. 目标与范围

围绕 IEEE33 event-ahead EV assignment / V2G reserve procurement，迁移最新
L08 DET/DRO/ARO 主线。核心代码、实验工作流、研究记录分别管理。旧 Case0/S01–S09
作为来源和历史，不能将其他 EVCS 选址模型引入本项目。冻结数学、数据语义、单位、
随机流、样本配对、固定 H/路线和 OOS 锁定规则。重构不新增算法或放宽科学阈值。

完成后：实验增多仍能追溯来源；新 agent 能从文件恢复研究进度；论文交付后全部必要
资料可脱离 PowerAI 使用。原始工作区和当前服务器任务保持不变。

## 2. 环境职责与权威

| 位置 | 职责 | 保存内容 |
|---|---|---|
| 本地 | 开发、决策、分析、长期归档 | 代码工作区、记录、完整必要产物、发布包 |
| 私有 GitHub | 协作、代码历史、CI | 源码、测试、协议、实验定义、文档 |
| PowerAI | 指定版本执行、暂存产物 | 不可变 release、活动状态、日志、检查点 |
| 新的公开 GitHub | 论文复现 | 最终代码、配置、输入入口、结果和复现命令 |

用户已选择另建公开复现仓库，保留私有开发仓库。
代码身份以 Git commit 和实际源码包哈希为准。活动状态以服务器实时观测为准。
完结资料经过清单校验归档本地，PowerAI 不承担唯一长期存储职责。
离线快照必须显示观察时间，不推断任务死亡或自动重启。

## 3. 参考与采用范围

- prime-rl configuration：最终生效配置、启动命令、每次 attempt 独立留存。
  https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/configuration.md
- open-instruct AGENTS：真实测试入口、重试任务身份和 agent 操作约定。
  https://github.com/allenai/open-instruct/blob/main/AGENTS.md
- AI-Scientist-v2 journal：计划、代码、结果、失败和父子实验关系。
  https://github.com/SakanaAI/AI-Scientist-v2/blob/main/ai_scientist/treesearch/journal.py
- autoresearch program：明确修改范围、评价规则、预算和尝试记录；不照搬单指标
  自动选优、无限重试或缺失指标填零。
  https://github.com/karpathy/autoresearch/blob/master/program.md
- OpenAI harness engineering：短入口、按需文档、自动检查依赖边界。
  https://openai.com/index/harness-engineering/
- Anthropic scientific computing：失败路线记忆、可信数值参考实现。
  https://www.anthropic.com/research/long-running-claude

## 4. 包结构与接口

一个 Python distribution，两个命名空间；Python 3.12、uv.lock、pytest、Ruff。
现有适用 unittest 由 pytest 执行，保留科学断言；数值依赖不顺便升级。
Gurobi 为可选后端，只有实际求解时初始化。

```text
src/ev_assignment/
  domain/ physics/ models/ methods/ solvers/ validation/
src/ev_assignment_experiments/
  config/ datasets/ workflows/ runtime/ tracking/ analysis/ publication/ cli.py
experiments/examples/
experiments/l08/
reproduction/
tests/unit/ integration/ scientific/ fixtures/
docs/architecture/ science/ operations/ decisions/
skills/
scripts/
AGENTS.md README.md pyproject.toml uv.lock
```

CLI/实验层 → 科学核心 → 数值库和求解器适配。核心禁止导入实验管理模块，禁止依赖
实验目录名、服务器路径或 GitHub。通过结构测试检查；不使用 sys.path 修改或按文件
父目录层数猜数据根。路径显式配置，数据/场景可逐项读取。

接口职责：PlanningProblem（数据、顺序、变量、科学协议）；SolverOptions（参数）；
DecisionPlan（第一阶段决策）；SolveResult（原生状态、方案、界及来源）；
AuditReport（独立检查证据）；RunPlan（冻结执行配方）。核心接收明确的停止信号、
进度回调、恢复状态；实验层负责日志、传输、运行登记。

## 5. 实验与研究记录

Study → ExperimentRevision → Run → Attempt。Decision 引用研究结果；Publication
引用图表所选结果、审计和决定。所有记录有 schema_version 和稳定 ID。

ExperimentRevision 固定假设、baseline、固定/变化因素、矩阵、指标、预算、停止标准。
Run 固定代码、实际配置、环境、数据哈希、场景顺序和随机流。
同代码同配置恢复为新 Attempt；代码、科学设置或资源策略改变为关联的新 Run。
检查点复用须验证兼容性。预算组独立于 Run，保留继承费用，换目录不能清零。
保留失败、负结果、后继解决关系；论文选择精确引用，不能使用“最新目录”。

状态分开：执行（queued/running/exited/failed/cancelled/unknown）、科学检查
（unchecked/pass/fail）、收敛（met/unmet/unknown）、研究决定
（pending/adopted/approximate/rejected/superseded）。
LB/UB 绑定适用模型、整数/连续语义、场景域、来源和审计；不兼容界禁止组合。
缺失界为空；近似接受不改原 gap、原生状态或 gap_passed。不同方法训练目标分别记录，
比较使用共同评价协议。独立审计重新检查输入/方案，退出零不能代替科学通过。
更正采用追加记录，旧证据不可覆盖。OOS 按协议锁定，不因工作流重构提前读取。

## 6. 运行和同步

代码使用 Git：本地提交私有仓库，PowerAI 安装指定 commit 的不可变代码包；已有任务
固定其启动版本。服务器临时开发进入独立工作分支，提交后本地获取、验证、合并。
不从运行目录 rsync 覆盖主代码，不热替换活动 release。

产物同步模式：metadata（状态/配置/摘要/决定/清单）、results（已完成方案、审计及
分析依赖）、full（本研究范围完整日志、检查点、输入和源码闭包）。
同步按 manifest 和 SHA256，支持续传、去重、临时文件验证后登记。活动文件为带时间的
快照；可靠检查点必须封存。元数据导出一致性快照，不复制活动 SQLite 数据库。
中断不破坏完整文件，默认不删除任一端。历史 archive 不整盘复制；被证据引用的历史
依赖纳入闭包。随时可主动同步，在线 watch 可追平；离线后下次连接继续。

每个 study 运行准入独占；使用进程身份，不能仅凭 PID 判断。恢复验证真实终态和费用，
不得重启健康任务、扩大旧预算或操作其他项目进程。

## 7. 研究存储与归档

存储位于代码仓库之外：studies/、experiments/、runs/<id>/{run.json,attempts/,
artifacts.json,summary.json}、decisions/、publications/、objects/、cache/、index.sqlite。
objects 内容寻址去重；索引和摘要可重建。定义/决定由研究工作流登记；每个 attempt
执行事件由单个 supervisor 写入；审计是独立不可变报告。归档改变位置不改变身份。

最终归档门：断开 PowerAI 后，本地能追溯所有论文结果、执行审计并重绘图表。
清单和全部必要依赖哈希齐全才标记已归档。删除/远端清理仅生成候选清单，另行执行；
已采纳证据、活动/恢复材料及引用闭包受保护。缓存可再生成，不能混作原始证据。

## 8. AI 接手

AGENTS.md 仅保留科学边界、代码地图、关键命令、权威入口。skills 按需提供开发验证、
定义/启动、监控/恢复、审计比较、发布操作；调用同一程序，不复制业务逻辑。
`ev-assignment context --study l08 --refresh` 输出有界 Markdown 和 JSON：目标/协议、
代码/同步时间、活动任务/有效进展、采纳结果/证据、失败路线/阻塞、下一步/依赖、引用。
事实由记录生成；解释/假设为具名记录，不提升为事实。必须报告省略数量和分页入口。
无聊天历史验收：新 agent 从入口找到上述答案并引用证据。

## 9. 论文发布

Publication 为图、表、实验逐项绑定配置、Run、方案、审计、决定、源码/输入和生成命令。
公开包包括可安装代码、环境锁、论文配置、可发布输入或获取校验方法、最终结果、图表、
必要审计、来源清单、README/引用和发布说明。中间研究产物不进入开发 Git 历史。
小型论文表格/必要图表可进入公开仓库；大型复现数据/证据为版本化 GitHub Release assets。
Figures 使用结果重绘，无需重新求解；Smoke 小例验证；Full 明确资源及许可证，按协议重跑。
结果对应的实际源码版本必须可获得。统一代码改变后需等价验证或重跑受影响项；不改旧
manifest 冒充新代码结果。公开包从本地归档生成，在干净环境验证，不依赖 PowerAI。
创建公开仓库及发布在论文材料选择/审计完成后进行，不在工程重构阶段自动公开。

## 10. 分阶段实施

| 阶段 | 交付 | 完成门 |
|---|---|---|
| A 基线 | tag、源码/依赖清单、迁移映射 | 来源闭包明确，缺失项显式记录并补齐 |
| B 基础 | 包结构、安装、路径、接口、CI | 仓库外可导入，无计算副作用 |
| C 小流程 | 定义→求解→审计→登记→同步→摘要 | 完整真实小例通过 |
| D 方法 | DET/DRO/ARO 与公共内核迁移 | 适用科学断言保留、各方法验证 |
| E 运行 | 隔离部署、恢复、预算、回传 | 中断/离线/归档场景通过 |
| F 发布 | 清单、绘图、导出、复现入口 | 示例包独立复现 |
| G 交付 | 映射、验证报告、文档 | 全部验收后合并私有 main 并 push |

不将组件通过、合成小例或 mock 通过称为正式 L08 通过；记录真实验证层级。
旧工作区保持原样，独立 checkout/分支实施。每阶段提交可审查成果；未通过总验收前不
合并私有 main。原始来源保留于基线 tag，活动源码不能仅改目录而继续依赖旧目录。

## 11. 验证矩阵

- 工程：安装导入、单向依赖、配置边界、无硬编码机器路径或导入副作用。
- 科学：手算小例、模型系数/变量类型、物理约束、直接/算法求解交叉核对。
- 方法：DET/DRO/ARO 目标、界证明、原 gap、共同评价、配对和 OOS。
- 恢复：进程身份、不可覆盖历史、继承费用、checkpoint 兼容和重复提交防护。
- 同步：断线、坏哈希、缺失引用、活动快照、索引重建、断开服务器的归档验证。
- AI：无聊天历史定位任务、结果、失败及下一步；一万条合成记录验证查询和有界摘要。
- 发布：干净安装、结果校验、重绘、小例、全部引用可访问。

CI 仅本地测试和 manifest/结构检查，不连接 PowerAI、不使用商业许可证、不跑正式批次。
PowerAI 重构小例隔离运行，单次求解≤60 秒，累计求解≤10 分钟；不覆盖正式预算。
保留测试断言，不删除失败用例、跳过依赖或放宽物理阈值来制造通过。

## 12. 完成交付

可安装的新代码、可执行实验流程、代码版本同步与产物回传、可恢复研究记录、
公开论文包构建流程、测试及来源报告。正式实验完成后用真实论文结果填充并发布公开版。

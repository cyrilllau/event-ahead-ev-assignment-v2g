# D/E/F/G 收尾验收（2026-11-11）

本报告替代此前“全部实现”的结论；`ACCEPTANCE_20261111.md` 保留为原时点、原范围证据。
当前工程代码完成 D/E/F，本地总验收通过；G 的最终 Git 门以当前分支的 CI 和实际 main
提交为准，见 IMPLEMENTATION_STATUS。任何正式求解成功都不从工程验收推出。

## 当前来源与证据位置

实际源码包 SHA256：
`b1dc6588bb4a79a2b0b8d1aa9564826da4e223970dafd1fcdfe89034c21746a2`。
源码包包含实际执行的两个命名空间和依赖环境；修改文件之后生成新记录，旧包/收据未覆盖。
Git 提交、源码包和历史源的身份分别保留，不把旧运行重新绑定到新提交。

本地证据根是代码仓库旁 `event-ahead-ev-assignment-v2g-store/validation/defg-20261111/`。
`delivery/` 为最终发布实现的实际验收；`accepted/`、`final/` 及顶层首次运行记录保留各自
原来源，不能冒充最终执行。下文未受后续纯发布改动影响的科学/结构/归档门按影响范围复用。
结果、输入、PDF/SVG、日志和环境均不进入开发 Git。文件 SHA 记录在
`defg-evidence-manifest.json`；该文件只含验收定位信息。

## D：结构收敛与完整实验定义

- 默认 wheel 只包含 `ev_assignment`、`ev_assignment_experiments`，只有 `ev-assignment`
  入口；旧模块不是默认依赖。691个基线Python源、2个后补原始科学源和旧配置分别移至
  `reference/baseline`，原字节保留，reference-locations.json 提供精确路径/SHA映射。
- 正式冻结域 DET/DRO/ARO 配方均可从 CLI 绑定输入根、study、资源及预算；固定/变化因素、
  指标、预测/支持域和设计元数据进入不可变 Experiment。登记不自动求解或接管旧campaign。
- 三个完整配方实际注册，完整输入离线闭包2059成员、20100路线，未启动正式优化。
  `accepted/full-recipes-receipt.json`。原2047输入/401场景/322311概率行检查通过；
  首阶段60780列/90768行/25770二元、DET R38 99234列/140611行、13104路线筛除及84切面
  与冻结数学来源一致，`frozen-structure.json`。优化调用0，未读取正式OOS。
- 三方法真实小Gurobi规划与直接模型/独立原物理审计交叉核对。控制器不声称保留旧搜索轨迹、
  basis/node tree、S07同源benchmark，或已获得替换当前生产任务的准入。

## E：迁移、隔离执行与完整回传

- `import-legacy` 按显式文件清单只读流式导入，不执行历史源码，不修改远端。
  真实读取 PowerAI 的旧 DET RESULT/WAIT/NATIVE_END 和受保护入口，共4文件/6闭包成员，
  再同步离线验证。记录是未验收历史观察，不是新方法接受或实时任务状态。
  `legacy-import-receipt.json` 范围为 selected_reference_files，不宣称全历史导入。
- supervisor 的原身份、READY/GO、单study独占、累计费用和恢复门保持；评价也共享study锁，
  保留逐单元checkpoint，健康/未知身份不可恢复。实际进程死亡恢复测试通过，未知开销
  保守收费并归档原账本，而非免费重开。
- 修复单独拉取 attempt 漏带父结算的问题：results 同步纳入随后封存的父receipt；
  审计声明其费用依赖。真实 parent/worker→同步→离线结算读回通过。
- clean source release 安装、真实开发任务、同步/完整离线包、故障/坏哈希/中断回归通过。
  PowerAI 原父/子2850511/2850521及起始时间保持；入口SHA仍为
  `8778427bfefd9548e939e6fd17716618b0cd00ea58672c73b0b608321a224fe8`。
  没有远端写入、安装、发送进程信号或启动新求解器。

## F：实际共同评价与论文包

- `evaluate lock/run/audit/recover` 已实现。接受决定、方案SHA、协议、随机流、样本清单、
  方法/样本顺序和源码全部绑定；清单锁定前不打开样本。每项原固定方案LP、物理/电池/
  AC采购约束检查和完整primal/dual见证单独封存；每片最多60秒，同Run续算缺失项。
- 真实三规划→三方法×两样本共同评价，首片1单元中断，下一片补其余5单元，旧单元字节与
  累计费用保持。方案/流/来源/成本篡改拒绝；全部单元需对应结算才能接受。
- 共同比较包含总成本/第一阶段/recourse/关键与普通ENS/实际AC V2G/shortfall，训练目标
  单列。采购表/图、共同指标表、配对差异图生成PDF/SVG；明确标准误与单位，微小数值噪声
  不通过自动缩放放大。保留原CSV数值，图表生成源码也进manifest/release闭包。
- 导出包在原store/样本删除后仍可审计和确定性重绘。干净环境仅安装精确导出源码，确认
  无旧包/旧CLI；无Gurobi执行审计、图表和配对LP重评价；随后安装Gurobi执行smoke和
  单项完整开发规划复跑。`delivery/clean-receipt.json`。
- 这些是完整开发示例，**不是正式研究OOS结果**。正式清单生成来源/独立性、已授权接受
  决定和论文解释仍按实际研究spec核验；未解锁正式OOS，未创建公开仓库。

## G：总验收门

- 79项本地测试通过，其中12项真实Gurobi测试；111项科学回归通过，私有fixture根显式指定。
  原断言未删改、原数值阈值未放宽；源移动造成的测试定位错误已修复，首次失败日志保留。
- 原1887项历史全套的4失败/49错误是此前基线与旧环境的记录，未重跑或改称PASS；当前
  活动套件和交叉版本科学套件是本次范围，原所有测试源码作为参考完整保留。
- 一万条记录 context 8976bytes/约1.27秒，分页和省略计数保留；未完成计划、评价审计、
  历史观察和具名下一步均有精确引用。索引重建约1.20秒。
- README、短AGENTS、两命名空间地图、正式配方、评价schema、历史导入和任务skill已一致。
- 私有分支CI必须通过，确认无研究产物/许可证/凭据进入Git之后才推进main；最终状态见
  IMPLEMENTATION_STATUS。已有研究任务的终态不由本次工程验收改写。

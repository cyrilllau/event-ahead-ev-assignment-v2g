# 结构化车队实验交接

在私有 `event-ahead-ev-assignment-v2g` 仓库里，从 `codex/structured-round2-core` 查看本次提交审查的模型、求解器、实例生成器和测试。合并后从 `main` 开始下一轮开发。历史执行、恢复和报告脚本保存在同仓库的归档分支 `codex/structured-case-round2-20261112`，分支尖端 `4db2c57` 对应最终报告工具。

## PowerAI 上已有的结果

2026 年 11 月 12 日的 Round 2 采用固定 N6 物理需求、N30 可选车池和 72 个套餐。P3 使用 L、R、LR 三个校准矩。全部 30 个冻结任务已结算，其中 18 个通过各自的验收门，开发和验证账本的活动 claim 均为空。P3 在三个验证种子上通过 400 场景的 gap 门及完整逐车审计，ARO 完整验收为 0/3。结算记录标记为 `SETTLED_WITH_LIMITATIONS`。本轮已经收尾，失败任务、既有费用和 OOS 记录原样保留。下一轮单独登记实验目标和执行范围。

历史报告中的 `qL`、衍生的 `risk_L` 和 OOS 下端按数值证据使用。普通舍入的限制见同目录的 `IMPLEMENTATION_CRITIC_CN.md`，需要严格向下包围的研究结论应先通过独立的舍入方向验证。

用你获分配的 SSH 别名和个人账号登录 PowerAI，再执行以下只读命令。若 `/home/Shixin` 下的归档不可读，请项目负责人提供团队可读的私有镜像或授权访问。

```bash
ROUND2=/home/Shixin/ev-assignment-synthetic/structured-fleet-design-20261112/round2
cat "$ROUND2/CASE_DESIGN_STATUS.md"
cat "$ROUND2/validation/CAMPAIGN_COMPLETE.json"
sed -n '1,24p' "$ROUND2/REPRODUCE_CN.md"
sha256sum "$ROUND2/validation/FREEZE.json" "$ROUND2/releases/r08/RELEASE.json"
```

历史验证运行固定在 `round2/releases/r08`，科学源码提交为 `1bd6cf7dd27a9ddfef8f732bab3e84fd93b56924`。按 `validation/FREEZE.json` 里的 `source_release` 核对 `RELEASE.json` 的 SHA-256，再按 `RELEASE.json` 核对 wheel 和各脚本。历史 OOM 恢复使用 `round2/controllers/52933fe/validation_campaign.py` 调度余下任务，worker 仍来自 r08。需要读历史计算源码时，在私有 Git 仓库内另开只读审查用的工作树。

```bash
git fetch origin codex/structured-case-round2-20261112
git worktree add --detach ../ev-assignment-r08-source 1bd6cf7dd27a9ddfef8f732bab3e84fd93b56924
```

交接资料还包括实验根下 `inputs/source-manifest.json` 及其五个输入文件，Round 2 开发与验证的两份 `DESIGN_CONFIG.json`，`validation/V0/prepare-r08` 的冻结样本与配对清单，原预算账本和逐任务收据，以及 `CASE_DESIGN_ROUND2_REPORT_CN.md` 和 `CASE_DESIGN_ROUND2_RESULTS.json`。经授权通过私有研究存储共享这些资料。把许可证、原始数据、样本、日志、检查点和结果留在 Git 仓库之外。仅在确有需要且文件来自可信归档时打开 pickle，并先核对锁定清单中的文件大小和 SHA-256。

## 在本地验证代码

以下命令在包含本次核心代码的工作树中执行，使用 Python 3.12 与仓库锁定依赖。选列的单元测试使用测试夹具，无需 PowerAI 私有输入或 Gurobi 许可证。

```bash
uv sync --locked --group dev
uv run pytest -q tests/unit/test_catalogue.py tests/unit/test_catalogue_direct.py \
  tests/unit/test_catalogue_probability.py tests/unit/test_structured_generator.py \
  tests/unit/test_structured_round2.py tests/unit/test_structured_statistics.py
```

拿到私有输入权限后，用显式路径预检输入文件的大小、哈希及物理数据结构。此步骤只读，输出清单和内容快照的 SHA-256。把这两个摘要纳入新计划。实例生成包括校准样本构造，放进下一轮有预算的开发运行。

```bash
STRUCTURED_INPUTS=/absolute/path/to/authorized/inputs
uv run python - "$STRUCTURED_INPUTS" <<'PY'
import sys

from ev_assignment_experiments.datasets.structured import read_source

source = read_source(sys.argv[1])
print("manifest_sha256", source["source_refs"]["manifest_sha256"])
print("snapshot_sha256", source["source_refs"]["snapshot_sha256"])
PY
```

## 在 PowerAI 上发起下一轮

用 `datasets.structured_round2.make_fixed_load_case` 构造固定需求实例，
用 `methods.catalogue_probability` 准备对应的概率域，并通过
`methods.catalogue.solve_structured` 或 `methods.catalogue_direct.solve_direct`
运行有界求解。`solve_direct` 要求同一有序支持及三个 P3 校准矩。
下一轮的批次启动器和执行清单随其独立实验计划准备。

1. **登记实验。** 先写清要回答的问题、沿用或改变的模型假设、对照方法、共享样本、审计和通过标准。固定训练、校准和 OOS 的随机流、样本顺序和权重。新建独立的 study、Run、私有输出目录及经批准的累计预算，单独保留 Round 2 的结果和费用。
2. **构建发布并预检。** 从评审通过的提交构建可校验的 wheel，用新目录和独立环境部署。实际优化使用 Gurobi 12.0.3 和执行账号的许可证。在首次 job 准入前核对 Git 提交、wheel/worker/启动脚本的 SHA-256、Python 和依赖环境、输入清单及文件摘要、执行配置、预算组与活动任务。如果复用封存 case，启动器必须在反序列化之前把其字节数和 SHA-256 与预先批准的清单做比较。
3. **先跑一个有界开发小例。** 例如用 `split="development"`、`seed=101`、`scenario_count=8`、`fleet_size=6`、`groups=6` 构造实例，按有序支持及 `case["weights"]` 调用 `finite_domain`，再显式以 `method_name="SAA"` 调用 `solve_structured`。遇到 `PHYSICAL_GATE_FAILED` 时停止，遇到 `NEED_ORIGINAL_FEASIBILITY_LP` 时先在该运行预算内补原矩阵可行性检查。只提交一个 job，使用新登记的路径和时限，保存完整父进程耗时、退出结果、候选、可用的界和原矩阵审计。完整支持上的原行检查、目标分项一致及合法上下界都通过后，再根据新登记的门推进同问题对照和规模。失败保留进度、原因与费用。
4. **正式批次按冻结计划启动。** 先绑定各方法的固定输入与发布身份，由单一控制器按计划准入。完成已锁定政策与物理审计后，再按登记协议打开配对 OOS。新的容量假设、概率矩或费用都要绑定新的身份和 Run。

历史 `validation_campaign.py` 读取冻结配置后可以直接启动缺失任务，启动前没有强制核对 `FREEZE.source_release`。历史训练 worker 在反序列化后记录 pickle 哈希，新的运行入口需预先核对。归档 controller 专用于那次执行，查看进度用上面的只读命令。下一轮先通过新入口的发布与输入身份预检，再启动开发小例或正式任务。

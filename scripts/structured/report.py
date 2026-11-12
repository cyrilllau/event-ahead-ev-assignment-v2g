"""Assemble the closed-round report from immutable receipts; no solver calls."""

from datetime import datetime, timezone
from pathlib import Path
import json
import subprocess
import sys

from ev_assignment_experiments.analysis.diagnostic import read, reference, save
from ev_assignment_experiments.runtime.ledger import BudgetLedger


root = Path(sys.argv[1]).resolve()
ledger = BudgetLedger(root / "store", "structured-design-development-v1").read()
if ledger["active"] is not None:
    raise ValueError("Do not close a round with an unsettled worker")
paths = ["A0/RESULT.json", "A1/r01/RESULT.json", "A1/r02/RESULT.json",
         "A2/r02/RESULT.json", "A2/r03/RESULT.json", "A3/r03/RESULT.json", "A5/r03/RESULT.json"]
a0, a1failed, a1, old, revised, medium, closeout = [read(root / p) for p in paths]
assert a0["passed"] and a1["passed"] and revised["passed"] and not medium["passed"]
assert not closeout["full400_solved"] and len(closeout["all_shedding_audits"]) == 130
assert all(r["native_status"] == 2 and r["original_LP_feasible"] for r in closeout["zero_EV_LPs"])
claims = {e["payload"]["attempt"]: e["payload"] for e in ledger["events"]
          if e["payload"]["kind"] == "claim"}
stage_cost = {k: 0. for k in ["A0", "A1", "A2", "A3", "A4", "A5"]}
for event in ledger["events"]:
    e = event["payload"]
    if e["kind"] == "settlement":
        stage_cost[claims[e["attempt"]]["run"]["stage"]] += e["charged_seconds"]
    elif e["kind"] == "external_charge":
        stage_cost[e["run"]["stage"]] += e["charged_seconds"]
assert abs(sum(stage_cost.values()) - ledger["spent_seconds"]) < 1e-7
assert all(stage_cost[k] <= read(root / "DESIGN_CONFIG.json")["stage_caps"][k] for k in stage_cost)
now = datetime.now(timezone.utc)
start = datetime.fromisoformat(read(root / "WORK_SESSION.json")["started_utc"])
wall = (now - start).total_seconds()
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
status = "NO_COMPUTABLE_FAMILY_FOUND_WITHIN_BUDGET"
budget = dict(development_charged_seconds=ledger["spent_seconds"], development_limit=7200,
              development_remaining=ledger["remaining_seconds"], validation_charged_seconds=0,
              validation_limit=28800, stage_charged_seconds=stage_cost,
              remote_wrapper_wall_seconds=ledger["spent_seconds"] - 120,
              local_preflight_measured_seconds=11.507, unmeasured_local_overhead_reserve=108.493,
              external_debit_seconds=120, new_solver_job_concurrency=1, threads_max=4,
              new_paid_resources=0, coding_reading_deployment_observation=dict(
                  session_wall_seconds=wall,
                  outside_remote_wrappers_wall_seconds=wall - (ledger["spent_seconds"] - 120),
                  exact_active_coding_seconds=None,
                  note="Wall outside charged remote wrappers includes coding/reading/deployment/reporting and local tests. "
                       "Not CPU time; coding overlapped some workers, so not an additive active-work measurement."))
frozen = dict(status="DEVELOPMENT_SNAPSHOT_ONLY_NOT_VALIDATION_ADMISSION", selected_family=None,
    source_commit_at_report=commit, deployed_releases={k: reference(root / "releases" / k /
        "ev_assignment-0.2.0.dev0-py3-none-any.whl") for k in ["r01", "r02", "r03"]},
    source_inputs=reference(root / "inputs/source-manifest.json"), solver=read(root / "SOLVER_ENVIRONMENT.json"),
    design=read(root / "DESIGN_CONFIG.json"), sole_revision=read(root / "REVISION_1_PLAN.json"),
    candidates=read(root / "PREREGISTERED_CANDIDATES.json"),
    validation_seeds_unused=[1101, 1102, 1103], stress_seed_unused=2101,
    protocol=reference(root / "EV_Assignment_Inverse_Benchmark_Design_Agent_Playbook_CN.md"))
save(root / "FROZEN_DEVELOPMENT_MANIFEST.json", frozen)
candidates = []
for version, rows in [(0, old["candidates"]), (1, revised["candidates"])]:
    for row in rows:
        c = dict(row, revision=version, split="development", selected_using_validation=False,
                 global_LB=None, global_UB=None, global_gap=None,
                 OOS_mean=None, OOS_CVaR95=None, OOS_critical_ENS=None, OOS_shortfall=None,
                 paired_OOS_ids=[], OOS_status="NOT_OPENED")
        if version == 0 and "policies" in c:
            c["reported_solver_calls_original_field"] = c["solver_calls"]
            c["solver_calls"] = 38
            c["solver_call_accounting_correction"] = "Original field27 counted only recourse; add2 feasibility+2joint+6risk+1inverse. Wall cost unchanged."
        candidates.append(c)
full = dict(nominal_target=dict(vehicles=30, groups=6, periods=96, scenarios=400),
            attempted=False, LB=None, UB=None, absolute_gap=None, relative_gap=None,
            elapsed_seconds=None, peak_memory_bytes=None, original_expansion_passed=None,
            reason="A3 frozen design suitability gate failed before decomposition; no A4 admission")
results = dict(schema_version=1, status=status, completed_utc=now.isoformat(),
    found_admitted_full400_family=False, failure_category="PHYSICAL_SCALING_DESIGN_GATE_NOT_SOLVER_NONCONVERGENCE",
    stop_gate="A3 constructed connected-full-supply witness; stronger than original LP feasibility",
    protocol_caveat="This additional constructive design screen must not be called a proof of original infeasibility. "
                    "Original LP allows shedding and was shown feasible; the scaled benchmark has severe healthy-grid ENS.",
    budget=budget, full400=full, candidate_identity_count=12, revision_rounds=1,
    revised_candidate_ids=["C11", "C12"], finalist_count=2, fallback_count=0,
    fallback_reason="Only A4 failure authorizes catalogue fallback; A4 was not entered",
    stages=dict(A0="PASS_ANALYTICAL_ONLY", A1="PASS_TINY_INTEGER_EQUIVALENCE",
                A2="V0_FAILED_REVISION1_FINITE_REFERENCE_PASS", A3="DESIGN_GATE_FAILED_BEFORE_OPTIMIZATION",
                A4="NOT_RUN", A5="FAILURE_EVIDENCE_AND_REPORT_COMPLETE", validation="NOT_OPENED"),
    implementations=dict(exact_aggregation="implemented_and_tiny_native_validated",
                         inverse_LP="implemented_and_remote_exercised",
                         continuous_prepass_sparse_Benders="implemented_but_NOT_exercised_on_real_optimization_in_this_round",
                         full400_expansion="implemented_but_NOT_executed",
                         frozen_validation_controller="not_completed_after_stop_gate"),
    evidence=[reference(root / p) for p in paths], A0=a0, A1=a1,
    failed_A1_attempt=a1failed, candidates=candidates, A3=medium, A5=closeout,
    frozen=reference(root / "FROZEN_DEVELOPMENT_MANIFEST.json"),
    software_validation=reference(root / "SOFTWARE_VALIDATION.txt"),
    scientific_claims=dict(H1="finite_reference_policies_supported_development_only",
                           H2="fixed_policy_risks_and_three_policy_ranking_supported_development_only",
                           H3="small_integer_equivalence_supported_speedup_unidentified",
                           unseen_seed_mechanism=None, full_scale_convergence=None,
                           structural_speedup=None, OOS_advantage=None),
    next_single_improvement="Replace blind fleet-proportional load amplification with a feeder-capacity-constrained multi-area load generator; preserve original network limits and separate healthy-grid congestion from disaster loss before any new screening.")
save(root / "CASE_DESIGN_RESULTS.json", results)

def table(headers, rows):
    return "| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |\n" + "\n".join(
        "| " + " | ".join(map(str, row)) + " |" for row in rows)

policy_tables = []
for c in revised["candidates"]:
    F = [p["policy"]["first_stage_cost"] for p in c["policies"]]
    rows = []
    for method, Q in c["risk_intervals"].items():
        J = [F[i] + q[1] for i, q in enumerate(Q)]
        rows.append([method, *[f"{v:.6f}" for v in J], "z" + str(J.index(min(J)))])
    policy_tables.append(f"### {c['id']}，6车／8支持场景\n\n" + table(
        ["评价规则", "z0目标上端($)", "z1目标上端($)", "z2目标上端($)", "三者内最佳"], rows))
failed_table = table(["候选", "gap / 缺口时长 / joint", "首轮结果", "数值证据"], [
    [c["id"], f"{c['parameters']['gap_h']}h / {c['parameters']['early_duration_h']}h / {c['parameters']['joint_nominal']}",
     "时序门失败" if "inverse" not in c else "机制区间为空",
     f"最小裕度{c['physical_audit']['min_block_margin_h']:.6f}h" if "inverse" not in c else f"最大margin {c['inverse']['max_margin']:.6f}$"]
    for c in old["candidates"]])
a1table = table(["A1小例", "计数整数最优", "逐车套餐最优", "原逐车矩阵最优", "活跃约束"], [
    [c["name"], *[f"{c['formulations'][m]['objective']:.9f}" for m in ["count", "individual", "original"]],
     "端口slack=0，选1/3车" if c["name"] == "port_binding" else "能量slack=0，失败场景短缺163.4kWh"] for c in a1["cases"]])
zero_table = table(["原LP", "目标($)", "全部ENS(kWh)", "断网ENS", "全网健康时ENS", "原行残差"], [
    [f"{r['candidate']}/{'forecast' if r['scenario']==0 else 'scene'+str(r['scenario'])}",
     f"{r['objective']:.6f}", f"{r['total_ENS_kwh']:.6f}", f"{r['disconnected_ENS_kwh']:.6f}",
     f"{r['all_lines_healthy_ENS_kwh']:.6f}", f"{r['original_row_residual']:.3g}"] for r in closeout["zero_EV_LPs"]])
stage_table = table(["阶段", "计费秒", "上限秒", "实际状态"], [
    [k, f"{v:.6f}", frozen["design"]["stage_caps"][k], results["stages"][k]] for k, v in stage_cost.items()])
report = f"""# EV Assignment 合成实例设计报告

**结论：本轮没有找到验收通过的30车、96时段、完整400场景实例族。状态：`{status}`。**

- **有效gap与耗时：** full400未运行，LB、UB、gap及全规模耗时均为`null`，不是0。开发累计计费 **{ledger['spent_seconds']:.6f}秒／7200秒**；验证计费0。停止原因是A3的缩放设计门，预算尚余{ledger['remaining_seconds']:.6f}秒。
- **假设改动：** 新的同类型车队和有限套餐目录、新的灾害生成与三行概率矩、新负荷及代表预测；唯一修订又改变了预测边际区停电和负荷随车数的缩放。原电网约束、电池、功率、短缺和运营费用内核保留。绝不是旧全任务模型的无损压缩。
- **机制：** H1/H2在开发小支持的三个固定政策上成立；没有独立seed、全局最优政策或OOS验证。H3的整数等价在两个小例成立，算法加速尚未辨识。
- **结构收益：** A1确认计数到逐车的可行映射和目标一致；30车同目录360二元→72整数是设计规模，未取得该规模的运行性能。不能声称25,770→72无损压缩或已加速。
- **论文可支持内容：** 精确聚合条件、小例数值核对、可复现的有限政策机制，以及容量约束导致实例缩放失败的负结果。当前不足以进入30车/full400主结果表。

## 1. 执行范围、来源与真实完成状态

协议已完整读取并保留原文及SHA256。所有正式科学求解都在PowerAI新目录执行，最多4线程，recourse和概率LP为1线程，同时至多1个新增solver job。原DET父/子PID2850511/2850521仍属原release；收尾只读观察到子进程仍为运行态，未修改旧源、链接、campaign、预算或进程。

外部证据目录：`{root}`。远端：`/home/Shixin/ev-assignment-synthetic/structured-fleet-design-20261112`。源码分支`codex/structured-case-design-20261112`，报告时commit `{commit}`；真实运行代码以各release wheel的SHA256为准。

完成A0、A1、12候选首轮A2及唯一修订的C11/C12复查。A3构建两个12车/64支持案例后在设计门停止，**没有调用分解优化器**。A4未进入，不能使用仅限A4失败时的套餐回退。A5完成失败定位和证据核对。验证、OOS、个体化压力和消融均未打开。稀疏Benders适配器已实现并静态审查，但本轮没有实际运行验证；不称为完整平台或算法验收完成。

## 2. A0与A1：解析保证和原矩阵核对

A0只是解析原型：C=(0,80,200)，R=(3000,800,200)，交点2/55与1/5；四状态小LP上界0.10/0.25。它们不是EV模型实测结果。

{a1table}

三种整数表示均由Gurobi返回OPTIMAL。原矩阵最大残差1.4211e-14，最大记录映射目标差7.276e-12；沿用1e-9行容差和1e-5目标容差。两例保留未选择套餐的Big-M行，能量例包含成功／失败路线两个场景。首轮6次优化后发生NumPy标量提取错误；修复只使用`.item()`，第二轮重审并复用端口见证，新增3次能量例优化。失败记录和开销均保留。

等价条件：每类相同套餐在所有场景中的系数完全相同、没有车辆标签约束、约束及成本线性。计数整数解把E/P/S等分给互不重复的车辆，合约分配跨场景固定；反方向按套餐求和。不同出发套餐的SOC可以不同，但同一套餐的类型内车辆完全相同。**不声称连续松弛相同；不声称任意不均匀合约政策压缩后Q不变。**

## 3. 概率、物理和费用设定

采用原IEEE33及L08八站路网资产，实际入弧时刻重放，不直接填写arrival或Q。固定line20、corridor1、t=66定义G/B/GB；对应原注册表联合行294937。Full保留3个L/R/LR坐标，Marginal只删GB；两者均使用同一联合支持。**三行新校准集不是原322,311行概率模型**，其余风场、线线、路路、负荷及其他时空矩被明确省略。

新生成机制为共同天气类别、线路首故障/恢复、持续路段封锁和独立旅行／负荷扰动。TRAIN采用四状态比例分层并保留正确权重；SAA用这些权重，不能用1/N。CAL2000为独立IID，采用预先声明的逐坐标95% Wilson区间，不宣称同时覆盖。所有实际概率上界通过原行认证的LP取得，未用原型0.1/0.25替代。

类型来源于原正常业务轨迹；初始电量偏移不在出发／到达时重置SOC。实测候选路线长度约1.67–11.53km、出发SOC约0.570–0.729，部分超出任务书给出的10–40km、0.6–0.8建议起点；这属于明确披露的设计差异，不是放宽物理能量约束。目的站只用L08的两个站；72个套餐，不制造逐车噪声。费用为原四项成本的无量纲乘数，允许[0.8,1.2]；原提前项实现是`30*(r+1-d)/4`，包含到预约结束的时长，不能另称单纯相对常规出发的提前小时。

## 4. 有限候选筛选与唯一修订

{failed_table}

最短档严格要求路段进入两侧各>0.25h，实际仅约0.101–0.102h，直接失败。其他8个候选原预测下Q(z0)=Q(z1)≈473.090403、Q(z2)≈121.540169，增加边际保护节约351.550233美元，超过费用盒允许的补偿。逆向LP最大共同margin为负，未扩大费用范围。

唯一修订在再评价前写入`REVISION_1_PLAN.json`，只复查已有C11/C12，不新增seed或候选ID：代表预测保留line20的17–17.5h停电，取消低频边际line23预测停电；受影响四个bus负荷按N/6缩放，其他负荷和所有容量不变；参考政策保护车辆数按ceil(N/6)缩放。原三参数随机流地址保持不变。该修订是观察首轮失败后进行的允许开发修订，**不是最初预注册或独立验证**。

下表是固定政策的F+风险上端，仅代表三者之间的排序，不能作为全局优化LB。区间完整数值及p/对偶见JSON与RISK文件。

{chr(10).join(policy_tables)}

C11实际joint上界区间为[0.0937760571,0.0937762571]（Full）、[0.2504896430,0.2504898430]（Marginal）；C12分别为[0.0305412071,0.0305414071]、[0.2673975092,0.2673977092]。C11最大margin52.597326，最小费用改动解保持w=(1,1,1,1)，实际共同margin37.249941；C12对应21.412555与9.412555，w也未改变。强邻点SAA选z1、低频邻点SAA选z0，没有强迫SAA失败。

这支持开发域内真实取舍和联合信息的作用。C11/C12是不同科学身份的小支持邻点，不是未见seed结果或正式paired OOS检验。全局可能存在更好的第四种策略，本轮未计算，不能排除。

## 5. A3失败的确切原因：容量缩放，不是“不收敛”

N12时，盲目将四个受影响bus需求翻倍后，line20在预测18.25h需要206.703968kW，原限额144.9648kW，超出61.739168kW；相应无功需求68.212309kvar也超过64.428kvar。训练最大有功超限C11为67.693674kW、C12为67.355597kW。

**必须区分两个命题。** 本轮额外设计门检查“根连通负荷全部供给、孤岛负荷全部削减”的构造见证；它比原LP可行性更强，失败不能推出原模型不可行。A5在原矩阵上重审了两例64支持加预测，共130个全削减见证，均可行；另对共享预测及各例最坏超限场景求解原无EV LP，全部OPTIMAL：

{zero_table}

C12被选中的场景全天96时段都无线故障，却仍有537.974433kWh ENS。预测全网健康的94个时段有494.392441kWh ENS，而断网ENS仅102.496450kWh。由此确认，新增需求缩放引入了显著非灾害拥塞，混淆原拟研究的提前部署机制。

因此本轮按冻结的**设计适用性门**停止，并非把这些案例认定为数学不可行，也不是证明结构算法无法收敛。原LP允许削减负荷；若放弃健康网全供电设计要求，可以研究另一类拥塞实例，但那需要另行声明科学问题。本轮不在看见失败后删除该门、上调线路容量或使用第二轮物理修订。A4未执行，不能推断30车/full400不可计算。

## 6. 名义规模与实际计算规模

- 正式A2：6车、6类型、72套餐、96时段、8个互异物理支持及1个预测；CAL2000。计数一阶段72整数+72能量变量、918行、792非零；此时每类仅1辆，**没有车辆标签压缩收益**。
- A2单场景运营块实测19,656变量、20,544行、W非零85,104；T非零C11为528、C12为720。完整33bus/32line/96时段物理模型参与了固定政策评价。
- 正式A3：12车、6类型、每类2辆、72套餐、64个互异物理轨迹。只有实例构建和物理检查；全局master尚未建立，不填优化行数、节点数或gap。
- 目标A4：30车、6类型、每类5辆、400支持，未建立／未求解。整数规模360→72是同目录的理论计数。逐车perspective实现还有360个套餐能量变量，原逐车合约实现只有30个能量变量，不能混淆总连续变量压缩比。
- 运营全域等价类数量未证明，填`null`；未据单点Q相同合并场景。0/1灾害类别不等于运营等价类。

## 7. 预算、失败、软件验证与复现

{stage_table}

合计{ledger['spent_seconds']:.6f}秒包含远端wrapper {budget['remote_wrapper_wall_seconds']:.6f}秒，以及本地真实来源预检保守计费120秒。后者是13次生成、0次求解：首次C05配置N30/16支持，随后12候选N6/8支持；累计测得11.507秒，另留108.493秒覆盖未测启动／导入／打印开销。原始执行日志只存在聊天工具记录，不能伪称有完整墙钟收据；这些预检不作为冻结版本正式验收证据。详细来源在`LOCAL_PREFLIGHT_DETAIL.json`。本轮有12个候选ID，revision1复查其中2个；没有额外候选或新seed。

远端正式优化调用共396次：A0=2、A1=9、A2=382、A5=3。初版A2的`solver_calls=27`只计recourse，漏掉11个概率／逆向LP；汇总JSON另保留原字段并更正为每个完成候选38次，修订成功候选39次。累计wall预算从未漏掉这些调用。

软件回归：127项unit/integration测试通过，11条既有HiGHS选项提示；其中部分旧测试使用小型软件夹具LP，与本轮真实数据科学求解分开。相关新物理／概率身份测试18项通过。**测试通过不替代全规模求解**。新增稀疏分解适配器未经本轮真实优化执行，后续使用前仍需端到端验收。

会话至报告wall约{wall/60:.1f}分钟，远端wrapper以外约{budget['coding_reading_deployment_observation']['outside_remote_wrappers_wall_seconds']/60:.1f}分钟，包含编码、阅读、部署、报告和本地软件测试；未精确计量主动编码CPU时间，不能把并行编码与求解时间重复相加。验证28,800秒未使用，未因剩余预算充足绕过阶段门。

运行版本、完整配置、来源、solver hash及wheel hash见`FROZEN_DEVELOPMENT_MANIFEST.json`。它是失败开发快照，**不是验证冻结族**。所有结果/数据/日志/wheel在Git外，源码只保存在任务分支，未合并main或发布GitHub。复现及只读检查命令见仓库`experiments/structured/README.md`；同一已关闭预算不能靠重启清零。

## 8. 下一轮最值得做的一项改进

**实现容量约束下的多区域负荷—车队联合生成器，替代直接按N/6放大少数bus需求。** 保持原线路限额，利用同一个原运营矩阵先验证健康网供电，再分配2–3个灾害区域及类型目的站，使30车的有效参与来自空间需求结构，而非健康网先天过载。先固定这条生成规则与可行性证据，再重新进行有限候选筛选；不要仅调整罚金或放宽gap。

仍待辨识：修正后的30车/full400有效gap及wall、是否存在更优第四种政策、未见seed机制稳定性、逐车对照和预分离的真实收益、G=3/15/30压力，以及完整paired OOS。上述指标全部留空，本轮不支持任何全规模收敛、OOS优越或算法加速的肯定论文主张。
"""
(root / "CASE_DESIGN_REPORT_CN.md").write_text(report)
save(root / "FINAL_BUDGET.json", budget)
with (root / "CASE_SEARCH_LEDGER.jsonl").open("a") as f:
    f.write(json.dumps(dict(stage="CLOSEOUT", status=status, observed_utc=now.isoformat(),
                           report=reference(root / "CASE_DESIGN_REPORT_CN.md"),
                           results=reference(root / "CASE_DESIGN_RESULTS.json"), budget=budget), ensure_ascii=False) + "\n")
(root / "CASE_DESIGN_STATUS.md").write_text(f"""# 合成实例设计状态（本轮已关闭）

状态：{status}。A0解析PASS；A1小例原矩阵PASS；A2唯一修订的固定政策机制PASS；A3负荷缩放设计门失败；A4未进入；A5失败核对完成。验证/OOS未开放。

开发计费{ledger['spent_seconds']:.6f}/7200秒，余{ledger['remaining_seconds']:.6f}秒；验证0/28800秒。唯一revision已使用，A4专属fallback未使用且当前不适用。未因剩余额度开启新搜索。

当前配置、代码、输入、solver及wheel身份见FROZEN_DEVELOPMENT_MANIFEST.json。它是失败开发快照，不是通过验证的实例族。源码commit：{commit}。

活动新增任务：无。全部wrapper已结算；日志在A0.log、A1-r01/r02.log、A2-r02/r03.log、A3-r03.log、A5-r03.log；每Run目录保留CHILD/WAIT/RESULT。不要复用旧PID或重新提交已完成实验。

确证：同目录整数映射小例正确；C11/C12有限政策margin37.249941/9.412555美元；line20限额144.9648kW而N12预测需求206.703968kW。130个原矩阵全削减见证及3个最优LP表明原模型可行；失败是冻结的健康网全供电设计要求及严重非灾害拥塞，不是求解不收敛。

下一步具体操作：先阅读CASE_DESIGN_REPORT_CN.md、CASE_DESIGN_RESULTS.json和CASE_SEARCH_LEDGER.jsonl。只读校验命令：
`.venv/bin/python scripts/structured/check_bundle.py '{root}'`

本轮没有合法的继续求解命令。后续应另立容量约束的多区域负荷生成设计，明确新的修订/预算授权；保留本轮失败与成本，不复用为验收通过族。
""")
print(json.dumps(dict(status=status, report=str(root / "CASE_DESIGN_REPORT_CN.md"), budget=budget), ensure_ascii=False))

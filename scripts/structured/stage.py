"""Bound a single remote science worker with the existing cumulative ledger.

Usage: python stage.py ROOT STAGE RUN_ID WORKER.py PYTHON [worker extra args...]
Worker arguments: ROOT/ STAGE/RUN_ID, seconds, then extra args. No running
production process is inspected/adopted/killed; only this admitted child session.
"""

from datetime import datetime, timezone
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
import traceback

from ev_assignment_experiments.runtime.ledger import BudgetLedger
from ev_assignment_experiments.runtime.identity import wait_identity
from ev_assignment_experiments.analysis.diagnostic import read, reference, save


root, stage, run_id, worker, python, *extra = sys.argv[1:]
root = Path(root).resolve()
config = read(root / "DESIGN_CONFIG.json")
ledger = BudgetLedger(config.get("ledger_store", str(root / "store")),
                      config.get("budget_group", "structured-design-development-v1"))
ledger.create(config.get("study", "structured-fleet-design-v1"), config["development_budget"])
snapshot = ledger.read()
claims = {e["payload"]["attempt"]: e["payload"] for e in snapshot["events"]
          if e["payload"]["kind"] == "claim"}
used = sum(e["payload"]["charged_seconds"] for e in snapshot["events"]
           if e["payload"]["kind"] == "settlement" and
           claims[e["payload"]["attempt"]]["run"].get("stage") == stage)
used += sum(e["payload"]["charged_seconds"] for e in snapshot["events"]
            if e["payload"]["kind"] == "external_charge" and e["payload"]["run"].get("stage") == stage)
available = min(config["stage_caps"][stage] - used, snapshot["remaining_seconds"])
available = min(available, config.get("run_caps", {}).get(run_id, available))
if stage not in config.get("report_stages", ["R5"]):
    available = min(available, snapshot["remaining_seconds"] - config.get("reserve_report_seconds", 0))
if available < 5:
    raise RuntimeError(f"Stage {stage} has only {available}s remaining; no new admission")
out = root / stage / run_id
out.mkdir(parents=True, exist_ok=False)
(out / "EXECUTION_CONFIG.json").write_bytes((root / "DESIGN_CONFIG.json").read_bytes())
claim = ledger.claim(dict(stage=stage, run_id=run_id, source=reference(worker),
                          argv=[python, worker, *extra],
                          release=reference(Path(worker).parent / "RELEASE.json") if (Path(worker).parent / "RELEASE.json").exists() else None,
                          freeze=reference(root / "FREEZE.json") if (root / "FREEZE.json").exists() else None,
                          wrapper=reference(__file__), config=reference(out / "EXECUTION_CONFIG.json")),
                     available, charge_process_start=True)
save(out / "CLAIM.json", claim)
rc = 1
child = None
start = claim["started_monotonic"]
def status(message):
    (root / "CASE_DESIGN_STATUS.md").write_text(
        f"# 合成实例设计状态\n\n{message}\n\n"
        f"Run: {stage}/{run_id}; source: {worker}\n"
        f"预算组 {config.get('budget_group', 'structured-design-development-v1')}："
        f"已结算 {snapshot['spent_seconds']:.6f}/{config['development_budget']}秒；本次预留 {available:.6f}秒，"
        "所有重试累计；验证准入以冻结文件与证据为准。\n"
        f"PID/身份: {out}/CHILD.json；日志 {out}/WORKER.log；配置 {root}/DESIGN_CONFIG.json。\n"
        "恢复时先读取预算 active、CHILD.json 并核对远端进程；未知/离线不当作已死，勿重复提交。\n"
        f"下一步：读取 {out}/RESULT.json 与 WAIT.json，按阶段门决定后续；未验收阶段不得宣称通过。\n")
try:
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    remaining = available - (time.monotonic() - start) - 2
    if remaining <= 0:
        raise TimeoutError("No worker allocation after wrapper startup")
    with (out / "WORKER.log").open("x") as log:
        child = subprocess.Popen([python, worker, str(out), str(remaining), *extra],
                                 env=env, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        identity = wait_identity(child.pid)
        ledger.child(claim["attempt"], identity)
        save(out / "CHILD.json", identity)
        status(f"{stage}运行中，其结果尚未验收；之前阶段见追加式 CASE_SEARCH_LEDGER.jsonl。")
        try:
            rc = child.wait(timeout=max(.01, available - (time.monotonic() - start) - 1))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=.3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            rc = 124
except BaseException:
    (out / "WRAPPER_ERROR.txt").write_text(traceback.format_exc())
    if child is not None and child.poll() is None:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    raise
finally:
    save(out / "WAIT.json", dict(returncode=rc, elapsed_seconds=time.monotonic() - start))
    ledger.settle(claim["attempt"], rc, reference(out / "WAIT.json"))
    snapshot = ledger.read()
    save(out / "BUDGET_AFTER.json", snapshot)
    result = read(out / "RESULT.json") if (out / "RESULT.json").exists() else {}
    event = dict(observed_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                 run_id=run_id, status=result.get("status", "INCOMPLETE_OR_WORKER_ERROR"),
                 result=reference(out / "RESULT.json") if result else None,
                 wait=reference(out / "WAIT.json"), budget_used=snapshot["spent_seconds"],
                 budget_remaining=snapshot["remaining_seconds"], returncode=rc)
    with (root / "CASE_SEARCH_LEDGER.jsonl").open("a") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")
    status(f"{stage}/{run_id} 已结算，返回码 {rc}；阶段结果 {event['status']}；活动新任务：无。")
    print(json.dumps(event, ensure_ascii=False))

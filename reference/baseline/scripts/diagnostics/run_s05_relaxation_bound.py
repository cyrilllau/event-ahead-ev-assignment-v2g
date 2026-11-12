"""One source-bound LP relaxation of an accepted, saved formal RMP.

The LP is a separate numerical bound component. It cannot modify a C&CG
checkpoint, supply an integer plan, authorize OOS, or reset planning time.
"""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import sys
import time

import numpy as np
from scipy.sparse import load_npz

from evcs_experiments.config import get_paths
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import observation
from evcs_experiments.relaxation_bound import solve_relaxation_bound

KIND = 'S05_SAVED_RMP_RELAXATION_PROBE_V1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))


def reference(path):
    p = Path(path).resolve(strict=True)
    h = hashlib.sha256()
    with p.open('rb') as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return dict(path=str(p), sha256=h.hexdigest(), size_bytes=p.stat().st_size)


def verify(ref):
    need(set(ref) == {'path', 'sha256', 'size_bytes'} and Path(ref['path']).is_absolute(), 'Malformed reference')
    need(reference(ref['path']) == ref, 'Evidence bytes changed: ' + ref['path'])
    return Path(ref['path'])


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')
    return reference(path)


def budget_check(request, settlement):
    need(request['kind'] == KIND and request['plan_id'] == settlement['plan_id'], 'Wrong plan or probe kind')
    need(settlement['status'] == 'SETTLED_UB_ONLY_PENDING_INDEPENDENT_REVIEW'
         and settlement['cumulative_seconds_provisional'] is False
         and settlement['nested_seconds_added'] is False, 'Unsettled predecessor cost')
    total = settlement['cumulative_accounted_seconds']
    opening = settlement['opening_budget']['opening_cumulative_seconds']
    parent = settlement['terminal_parent_seconds']
    need(all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in (total, opening, parent)), 'Invalid cost')
    need(total == opening + parent and settlement['remaining_seconds'] == 7200 - total
         and settlement['cumulative_cap_seconds'] == 7200, 'Reset or double-counted budget')
    need(request['wall_seconds'] == 590 and request['solver_seconds'] <= 550
         and request['solver_seconds'] > 0 and request['parent_job']['timeout_seconds'] == 600,
         'Unknown-resource probe must retain its fixed shared wall')
    need(600 + 1790 + 10 <= settlement['remaining_seconds'], 'Probe consumes reserved final evidence time')
    return total


def bind_parent(req, rr):
    p = req['parent_job']; paths = get_paths(); directory = Path(p['directory']).resolve()
    need(directory == paths.campaign_root.resolve() / 'jobs' / p['job_id'], 'Wrong parent path')
    need(os.environ.get('EVCS_JOB_ID') == p['job_id']
         and os.environ.get('EVCS_JOB_DIRECTORY') == str(directory), 'Not a supervised job')
    mr = reference(directory / 'manifest.json'); m = read(mr['path']); s = read(directory / 'status.json')
    expected = [sys.executable, '-B', str(Path(__file__).resolve()), '--request', rr['path'],
                '--expected-request-sha256', rr['sha256']]
    need(m['argv'] == expected and m['spec_id'] == 'S05' and m['job_id'] == p['job_id']
         and m['cwd'] == str(paths.project_root.resolve()) and m['outroot'] == str(directory.parent), 'Parent command differs')
    need(m['threads'] == p['threads'] == 1 and m['timeout_seconds'] == p['timeout_seconds'] == 600
         and m['grace_seconds'] == p['grace_seconds'] == 5, 'Parent resources differ')
    need(verify(p['controller_reference']) == Path(m['controller_path']).resolve()
         == paths.project_root / 'src/evcs_experiments/runtime/jobctl.py', 'Wrong parent controller')
    need(s['state'] == 'RUNNING' and os.getpid() == s['command_pid']
         and os.getsid(0) == s['supervisor_sid'] and observation(s)['supervisor_alive'], 'Wrong live parent')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        need(os.environ.get(key) == '1', 'Thread policy differs')
    created = datetime.fromisoformat(m['created_at_utc'])
    age = (datetime.now(timezone.utc) - created).total_seconds()
    need(created.tzinfo is not None and 0 <= age < 590, 'Expired parent clock')
    return dict(manifest_reference=mr, deadline=time.monotonic() + 590 - age,
                manifest_age_seconds=age, job_id=p['job_id'])


def source_check(ref, expected_root):
    manifest = read(verify(ref)); need(Path(ref['path']).parent == expected_root, 'Source root differs')
    files = {}
    for row in manifest['files']:
        p = (expected_root / row['path']).resolve()
        need(p.is_relative_to(expected_root) and str(p) not in files, 'Source path escape or duplicate')
        item = dict(path=str(p), sha256=row['sha256'], size_bytes=row['size_bytes'])
        verify(item); files[str(p)] = item
    return files


def execute(request_path, expected_sha):
    started = time.monotonic(); rr = reference(request_path)
    need(rr['sha256'] == expected_sha, 'Request SHA differs')
    req = read(request_path); refs = req['references']; paths = get_paths()
    parent = bind_parent(req, rr)
    # Reserve before expensive evidence loading. A terminal failure still owns
    # this tail and must have its actual parent cost settled before any retry.
    attempt = paths.campaign_root / 'bound_attempts' / canonical_hash(refs['settlement'])
    attempt.mkdir(parents=True, exist_ok=False)
    write(attempt / 'reservation.json', dict(request_reference=rr, parent=parent,
          predecessor_settlement=refs['settlement'], requires_terminal_parent_settlement=True))
    out = Path(req['output_root']).resolve()
    need(out == paths.campaign_root.resolve() / 'bound_probes' / req['parent_job']['job_id'], 'Wrong fresh output path')
    out.mkdir(parents=True, exist_ok=False)
    try:
        for ref in refs.values(): verify(ref)
        settlement = read(refs['settlement']['path']); opening = budget_check(req, settlement)
        card = read(refs['probability_acceptance']['path'])
        need(card['status'] == 'PASS_PROBABILITY_RECOVERY_COMPONENT_REVIEW', 'Probability component not independently accepted')
        need(refs['settlement'] in card['references'].values(), 'Acceptance does not bind exact settled parent')
        saved = read(refs['saved160_acceptance']['path'])
        need(saved['status'] == 'PASS_SAVED160_COMPONENT_REVIEW' and saved['complete_lp_count'] == 160, 'Original RMP/160 audit missing')
        need(card['references']['saved160_acceptance'] == refs['saved160_acceptance'], 'Different saved160 acceptance')
        need(refs['checkpoint'] in saved['references'].values() and refs['history'] in saved['references'].values()
             and refs['old_source_manifest'] in saved['references'].values(), 'Saved audit source or origin differs')
        root = paths.project_root.resolve(); files = source_check(refs['source_manifest'], root)
        need(files[str(Path(__file__).resolve())] == refs['script'], 'Unpinned executable')
        for name, module in tuple(sys.modules.items()):
            if name == 'evcs_experiments' or name.startswith('evcs_experiments.'):
                p = getattr(module, '__file__', None)
                if p is not None: need(reference(p) == files.get(str(Path(p).resolve())), 'Mixed package source')
        cp = read(refs['checkpoint']['path']); state = cp['state']; pending = state['pending']
        need(card['problem_sha256'] == saved['problem_sha256'] == state['problem_sha256'], 'Accepted formal problem differs')
        need(cp['sha256'] == canonical_hash(state) and state['problem_sha256'] == canonical_hash(state['identity']), 'Checkpoint content identity differs')
        need(state['status'] == 'PROBABILITY_FAILED' and state['lower_bound_usd'] is None
             and pending['iteration'] == 1 and state['active_indices'] == pending['active_indices'] == [67, 87, 98, 137], 'Unexpected original state')
        raw = read(refs['raw_request']['path'])
        need(raw['domain'] == 'formal' and raw['formal_OOS'] is False and raw['problem_identity'] == state['identity'], 'Not original formal problem')
        history = read(refs['history']['path']); need(history[1] == refs['rmp_record'], 'Wrong original RMP')
        event = read(refs['rmp_record']['path']); need(event['event'] == 'rmp' and event['iteration'] == 1, 'Wrong event')
        used = {Path(ref['path']).name: ref for ref in event['artifacts']}
        for ref in event['artifacts']: verify(ref)
        for key, name in (('matrix', 'matrix.npz'), ('model_arrays', 'model_arrays.npz'), ('model', 'model.json')):
            need(refs[key] == used[name], 'RMP input not bound to original event')
        matrix = load_npz(refs['matrix']['path']).tocsr()
        with np.load(refs['model_arrays']['path'], allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in z.files}
        model = read(refs['model']['path'])
        auditor = verify(refs['digest_script']); need(files.get(str(auditor)) == refs['digest_script'], 'Unpinned semantic digest')
        spec = importlib.util.spec_from_file_location('_saved_rmp_digest', auditor)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        need(module.model_digest(model, matrix, arrays) == model['model_sha256'] == pending['rmp_model_sha256'], 'Original RMP coefficient digest differs')
        need(model['risk_model'] == 'DRO' and model['objective_constant_usd'] == float(arrays['objective_constant_usd'])
             and model['scenario_ids'] == [state['identity']['scenario_ids'][i] for i in state['active_indices']], 'RMP support or constant differs')
        problem = dict(arrays, matrix=matrix, source_model_sha256=model['model_sha256'], formal_problem_sha256=state['problem_sha256'])
        remaining = min(req['solver_seconds'], parent['deadline'] - time.monotonic() - 30)
        need(remaining > 0, 'Verification consumed probe wall')
        write(out / 'execution.json', dict(kind=KIND, request_reference=rr, parent=parent,
              opening_cumulative_seconds=opening, original_model_sha256=model['model_sha256'],
              formal_problem_sha256=state['problem_sha256'], active_indices=state['active_indices'],
              matrix_shape=list(matrix.shape), nnz=matrix.nnz, verification_seconds=time.monotonic()-started,
              integrality_only_relaxed=True, solver_grant_seconds=remaining, formal_OOS=False))
        record, values = solve_relaxation_bound(problem, time_limit_seconds=remaining, threads=1,
                                                log_path=out / 'solver_summary.json')
        np.savez_compressed(out / 'certificate_vectors.npz', **values)
        record.update(request_reference=rr, original_native_status='TIME_LIMIT', original_native_lower_bound_usd=None,
                      original_checkpoint_modified=False, scientific_acceptance=False, lockable=False,
                      formal_OOS=False, automatic_resume_supported=False, recourse_solver_calls=0, mip_solver_calls=0,
                      requires_terminal_parent_settlement=True, cumulative_seconds_provisional=True,
                      opening_cumulative_seconds=opening, attempt_elapsed_seconds=time.monotonic()-started,
                      parent_job_id=parent['job_id'], peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(out / 'result.json', record)
        write(out / 'artifact_manifest.json', dict(files=[reference(p) for p in sorted(out.iterdir()) if p.is_file()]))
        print(json.dumps({'output':str(out), 'status':record.get('status'), 'scientific_acceptance':False}))
        return 0
    except Exception as exc:
        write(out / 'failure.json', dict(error=type(exc).__name__, message=str(exc), request_reference=rr,
              elapsed_seconds=time.monotonic()-started, requires_terminal_parent_settlement=True,
              scientific_acceptance=False, formal_OOS=False))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--expected-request-sha256', required=True)
    a = p.parse_args()
    raise SystemExit(execute(a.request, a.expected_request_sha256))

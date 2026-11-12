"""Append-only terminal-parent accounting for the probability-only bridge.

Settlement reads evidence; it does not solve, launch, resume, or authorize OOS.
The parent elapsed time replaces nested provisional clocks, including failures.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from evcs_experiments.config import get_paths
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import observation

KIND = 'PROBABILITY_RECOVERY_PARENT_SETTLEMENT_V1'
BRIDGE_KIND = 'PROBABILITY_CERTIFICATE_REPAIR_FULL160_V1'
CAP_SECONDS = 7200.


def need(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path):
    p = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with p.open('rb') as handle:
        while chunk := handle.read(4 * 1024 * 1024):
            digest.update(chunk)
    return dict(path=str(p), sha256=digest.hexdigest(), size_bytes=p.stat().st_size)


def verify(ref):
    need(isinstance(ref, dict) and set(ref) == {'path', 'sha256', 'size_bytes'}, 'Malformed settlement reference')
    need(Path(ref['path']).is_absolute() and reference(ref['path']) == ref, 'Settlement evidence bytes changed')
    return Path(ref['path'])


def load(ref):
    return json.loads(verify(ref).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def number(value, name):
    need(type(value) in (int, float) and math.isfinite(value) and value >= 0, 'Invalid ' + name)
    return float(value)


def _bridge(request):
    """Only import the request-pinned bridge beside this release's helper."""
    refs = request['references']
    script = verify(refs['recovery_script'])
    helper = verify(refs['settlement_script'])
    root = get_paths().project_root.resolve()
    need(script == root/'scripts/recovery/prepare_s05_probability_recovery.py'
         and helper == Path(__file__).resolve()
         and helper == root/'scripts/recovery/settle_s05_probability_recovery.py', 'Wrong installed recovery scripts')
    manifest = load(refs['new_source_manifest'])
    need(Path(refs['new_source_manifest']['path']).parent == root, 'Wrong settlement source release')
    rows = {row['path']: row for row in manifest['files']}
    need(len(rows) == len(manifest['files']), 'Duplicate release source paths')
    for key in ('recovery_script', 'settlement_script'):
        ref = refs[key]; relative = str(Path(ref['path']).relative_to(root))
        need(rows.get(relative) == dict(path=relative, sha256=ref['sha256'], size_bytes=ref['size_bytes']),
             'Recovery script differs from source manifest')
    spec = importlib.util.spec_from_file_location('_bound_probability_settlement_bridge', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    verified_root, files, _, _ = module.release(refs['new_source_manifest'])
    need(verified_root == root, 'Settlement runtime release changed')
    for name, imported in tuple(sys.modules.items()):
        if name == 'evcs_experiments' or name.startswith('evcs_experiments.'):
            filename = getattr(imported, '__file__', None)
            if filename is not None:
                path = str(Path(filename).resolve())
                need(path in files, 'Mixed or unbound settlement package source')
                verify(files[path])
    return module


def _parent(request_ref, request, parent_result_ref, campaign):
    expected = request['parent_job']; directory = Path(expected['directory']).resolve()
    need(directory == campaign/'jobs'/expected['job_id'], 'Wrong successor parent directory')
    need(verify(parent_result_ref) == directory/'result.json', 'Wrong terminal parent receipt')
    result = load(parent_result_ref)
    manifest_ref = reference(directory/'manifest.json'); manifest = load(manifest_ref)
    need(result.get('complete') is True and result.get('state') in ('SUCCEEDED', 'FAILED', 'TIMED_OUT', 'CANCELLED'),
         'Parent is not a completed execution')
    need(not observation(result)['blocks_new_launch'], 'Parent still has live processes')
    need(result['job_id'] == manifest['job_id'] == expected['job_id'], 'Parent job identity differs')
    need(type(result.get('returncode')) is int, 'Parent has no terminal command return code')
    # A timed-out/cancelled command may handle SIGTERM and exit zero. Its parent
    # is still a failed attempt whose actual elapsed time must be settled.
    need(result['state'] != 'SUCCEEDED' or result['returncode'] == 0, 'Parent success and return code differ')
    wall = number(request['wall_seconds'], 'program wall')
    need(0 < wall <= 120. and expected['timeout_seconds'] == wall+10.
         and expected['grace_seconds'] == 5 and expected['threads'] == 1, 'Changed parent resource budget')
    for key in ('timeout_seconds', 'grace_seconds', 'threads'):
        need(manifest[key] == expected[key], 'Parent manifest resources differ: ' + key)
    root = get_paths().project_root.resolve()
    need(manifest['cwd'] == str(root) and manifest['outroot'] == str(campaign/'jobs')
         and manifest['spec_id'] == 'S05', 'Parent campaign/source scope differs')
    controller = verify(expected['controller_reference'])
    need(manifest['controller_path'] == str(controller), 'Parent controller source differs')
    argv = manifest['argv']
    need(len(argv) == 9 and Path(argv[0]).is_absolute()
         and argv[1:] == ['-B', request['references']['recovery_script']['path'], '--request', request_ref['path'],
                          '--expected-request-sha256', request_ref['sha256'], '--action', 'run'],
         'Parent command is not the exact atomic bridge request')
    need(str(Path(argv[0]).parent.parent) == str(root/'.venv'), 'Parent Python is outside the pinned virtual environment')
    return result, manifest_ref


def checked_settlement(request_reference, parent_result_reference):
    """Compute an authoritative accounting record without writing any file."""
    request = load(request_reference)
    need(request['kind'] == BRIDGE_KIND and request['domain'] == 'formal' and request['spec_id'] == 'S05',
         'Unknown probability recovery request')
    refs = request['references']; predecessor = verify(refs['job_result'])
    need(predecessor.name == 'result.json' and predecessor.parents[1].name == 'jobs', 'Wrong predecessor receipt root')
    campaign = predecessor.parents[2]
    need(campaign == get_paths().campaign_root.resolve(), 'Settlement campaign differs from configured authority')
    out = Path(request['output_root']).resolve()
    need(out.is_relative_to(campaign/'recoveries'), 'Recovery output is outside campaign')
    attempt = campaign/'recovery_attempts'/canonical_hash(refs['job_result'])
    reservation_ref = reference(attempt/'reservation.json'); reservation = load(reservation_ref)
    need(reservation['kind'] == BRIDGE_KIND and reservation['request_reference'] == request_reference
         and reservation['predecessor_job_result_reference'] == refs['job_result']
         and reservation['output_root'] == str(out) and reservation['automatic_retry_allowed'] is False,
         'Attempt reservation does not bind this chain tail')
    parent, manifest_ref = _parent(request_reference, request, parent_result_reference, campaign)
    parent_binding = dict(job_id=request['parent_job']['job_id'], directory=request['parent_job']['directory'],
                          manifest_reference=manifest_ref)
    need(all(reservation['parent_job'].get(key) == value for key, value in parent_binding.items()),
         'Reserved parent source differs')
    bridge = _bridge(request)
    worker = load(refs['worker_request']); invocation = load(refs['invocation'])
    opening = bridge.opening_budget(refs, worker, invocation)
    amount = number(opening['opening_cumulative_seconds'], 'opening cumulative time')
    need(opening['cap_seconds'] == CAP_SECONDS and 0 < amount < CAP_SECONDS, 'Opening plan budget differs')
    elapsed = number(parent['elapsed_seconds'], 'terminal parent elapsed')
    need(elapsed > 0, 'Terminal parent elapsed must be positive')
    evidence = dict(request=request_reference, parent_manifest=manifest_ref, parent_result=parent_result_reference,
                    reservation=reservation_ref, original_job_result=refs['job_result'],
                    worker_request=refs['worker_request'], invocation=refs['invocation'],
                    prior_carryover=worker['failure_carryover_reference'],
                    recovery_script=refs['recovery_script'], settlement_script=refs['settlement_script'],
                    source_manifest=refs['new_source_manifest'])
    nested = None; bridge_status = None; derived_ref = None
    result_path = out/'result.json'
    if result_path.exists():
        result_ref = reference(result_path); result = load(result_ref); evidence['bridge_result'] = result_ref
        need(result['kind'] == BRIDGE_KIND and result['request_reference'] == request_reference
             and result['original_job_result_reference'] == refs['job_result'] and result['opening_budget'] == opening,
             'Bridge result changed its origin or opening budget')
        need(all(result['parent_job'].get(key) == value for key, value in parent_binding.items()), 'Bridge parent differs')
        need(result['scientific_acceptance'] is False and result['lockable'] is False and result['formal_OOS'] is False
             and result['lower_bound_usd'] is None and result['gap_relative'] is None
             and result['cumulative_seconds_provisional'] is True, 'Bridge result promoted unsupported scientific scope')
        nested = number(result['attempt_elapsed_seconds'], 'nested provisional time')
        need(nested <= elapsed+1e-5, 'Terminal parent does not cover all nested runtime')
        need(abs(result['cumulative_accounted_seconds']-(amount+nested)) <= 1e-8, 'Bridge provisional cumulative time differs')
        claim_ref = result['claim_reference']; claim = load(claim_ref)
        need(verify(claim_ref) == campaign/'recovery_claims'/(canonical_hash(refs['job_result'])+'.json')
             and claim['request_reference'] == request_reference and claim['predecessor_job_result_reference'] == refs['job_result']
             and claim['successor_output_root'] == str(out) and claim['plan_id'] == request['plan_id'], 'Wrong successor claim')
        evidence['claim'] = claim_ref
        names = set()
        for ref in result['artifacts']:
            path = verify(ref); need(path.parent == out and path.name not in names, 'Bridge artifact scope/duplicate')
            names.add(path.name); evidence['artifact:'+path.name] = ref
            if path.name == 'derived_state.json': derived_ref = ref
        bridge_status = result['status']
    for label, path in (('attempt_failure', attempt/'failure.json'), ('bridge_failure', out/'failure.json')):
        if path.exists(): evidence[label] = reference(path)
    success = parent['state'] == 'SUCCEEDED'
    need(not success or bridge_status == 'VALID_INCUMBENT_BOUND_UNAVAILABLE', 'Successful parent lacks the declared UB-only result')
    if success:
        need(derived_ref is not None, 'Successful bridge omitted the derived evidence')
        derived = load(derived_ref)
        need(derived['status'] == bridge_status and derived['lower_bound_usd'] is None and derived['gap_relative'] is None
             and derived['lockable'] is False and derived['formal_OOS'] is False and derived['scientific_acceptance'] is False,
             'Derived result is outside the UB-only scope')
        need(math.isfinite(number(derived['upper_bound_usd'], 'derived upper bound')), 'Invalid derived upper bound')
    total = amount+elapsed
    status = ('SETTLED_BUDGET_EXCEEDED' if total > CAP_SECONDS else
              'SETTLED_UB_ONLY_PENDING_INDEPENDENT_REVIEW' if success else 'SETTLED_FAILED_ATTEMPT_REQUIRES_REVIEW')
    record = dict(schema_version=1, kind=KIND, status=status, plan_id=request['plan_id'], references=evidence,
                  parent_state=parent['state'], parent_returncode=parent['returncode'], bridge_status=bridge_status,
                  opening_budget=opening, terminal_parent_seconds=elapsed, nested_provisional_seconds=nested,
                  nested_seconds_added=False, cumulative_accounted_seconds=total,
                  remaining_seconds=max(0., CAP_SECONDS-total), cumulative_cap_seconds=CAP_SECONDS,
                  cumulative_seconds_provisional=False, accounting_authority='terminal jobctl parent elapsed once',
                  planning_budget_charge_includes='preparation, bridge execution, export and parent supervisor runtime',
                  settlement_verification_scope='independent post-run accounting audit; no planning computation',
                  lower_bound_usd=None, gap_relative=None, lockable=False, scientific_acceptance=False, formal_OOS=False,
                  automatic_retry_allowed=False, normal_ccg_checkpoint_compatible=False)
    return record, attempt/'settlement.json'


def settle(request_reference, parent_result_reference, output_path=None):
    record, path = checked_settlement(request_reference, parent_result_reference)
    need(output_path is None or Path(output_path).resolve() == path, 'Settlement output must be the unique attempt receipt')
    encoded = (json.dumps(record, indent=2, allow_nan=False)+'\n').encode()
    with path.open('xb') as handle: handle.write(encoded)
    return dict(record=record, settlement_reference=reference(path))


def verify_settlement(settlement_reference):
    record = load(settlement_reference)
    need(record['kind'] == KIND, 'Unknown terminal settlement')
    expected, path = checked_settlement(record['references']['request'], record['references']['parent_result'])
    need(verify(settlement_reference) == path and record == expected, 'Terminal settlement evidence or accounting changed')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True, type=Path)
    parser.add_argument('--expected-request-sha256', required=True)
    parser.add_argument('--parent-result', required=True, type=Path)
    parser.add_argument('--expected-parent-result-sha256', required=True)
    args = parser.parse_args()
    rr = reference(args.request); pr = reference(args.parent_result)
    need(rr['sha256'] == args.expected_request_sha256 and pr['sha256'] == args.expected_parent_result_sha256,
         'Explicit settlement input SHA differs')
    print(json.dumps(settle(rr, pr), allow_nan=False)); return 0


if __name__ == '__main__': raise SystemExit(main())

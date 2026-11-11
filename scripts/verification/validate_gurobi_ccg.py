"""Explicit licensed development gate; never invoked by CI or opens OOS.

Use the accepted I02 reference matrices and hand optima. Save fresh Gurobi
C&CG/resume evidence, then replay original physical matrices and certificates.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, load_npz

from evcs_experiments.config import get_paths
from evcs_experiments.evidence import sha256, source_provenance
from evcs_experiments.native_gurobi import solve_milp
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.validation.validate_s03_ccg import export_ccg, load_i02, I02_SHA
from evcs_experiments.validation.s02_toys import _json_file


def primitive_gate(out):
    fixtures = [
        (csr_matrix([[2.]]), [1.], [1], [0.], [1.], [1.], [np.inf], [1.], 1.),
        (csr_matrix([[1.,1.],[1.,-1.]]), [2.,-1.], [0,0], [-np.inf,0.], [np.inf,3.],
         [1.,-2.], [1.,2.], None, -2.5),
        (csr_matrix([[1.]]), [1.], [1], [0.], [1.], [2.], [np.inf], None, None),
    ]
    results = []
    for i, (matrix,c,integers,lo,hi,rl,ru,start,expected) in enumerate(fixtures):
        opts = dict(time_limit=5., mip_rel_gap=1e-9, threads=1)
        result = solve_milp(c,integers,lo,hi,matrix,rl,ru,start=start,options=opts,log_path=out/f'primitive_{i}.log')
        reference = milp(c,integrality=integers,bounds=Bounds(lo,hi),
                         constraints=LinearConstraint(matrix,rl,ru),options={'time_limit':5.,'threads':1})
        if expected is None:
            assert reference.status == result.status == 2 and result.x is None
        else:
            assert reference.status == result.status == 0
            assert abs(reference.fun-expected) <= 1e-8 and abs(result.fun-expected) <= 1e-8
            assert result.mip_dual_bound is not None and abs(result.mip_dual_bound-expected) <= 1e-8
        results.append(dict(expected=expected,status=int(result.status),native=result.native_record))
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--s03-i02-acceptance',type=Path,required=True)
    args = ap.parse_args(); began = time.monotonic(); paths = get_paths()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    card=load_i02(args.s03_i02_acceptance,I02_SHA)
    reference=Path(card['evidence'])
    module_path=paths.project_root/'scripts/audits/audit_s03_i03.py'
    spec=importlib.util.spec_from_file_location('independent_saved_ccg_audit',module_path)
    auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)
    reference_files=auditor.old.old.artifact_manifest(reference,card['artifact_manifest_sha256'])
    reference_release=auditor.old.old.release_manifest(Path(card['source_root']),card['current_source_manifest']['sha256'])
    auditor.old.old.source_manifest(Path(card['source_root']),
        auditor.read(card['code_manifest']['path'],card['code_manifest']['sha256']))
    auditor.old.old.verify_refs(card)
    manifest=paths.project_root/'RELEASE_MANIFEST.json'
    for row in json.loads(manifest.read_text())['files']:
        if sha256(paths.project_root/row['path']) != row['sha256']:
            raise ValueError('New release changed: '+row['path'])
    primitive=primitive_gate(out)
    ccg=out/'ccg';ccg.mkdir()
    records=export_ccg(ccg,10.,reference,lazy=True,native_start=True,
                       master_backend='gurobi',optimizer_threads=4)
    names=('ccg.py','extensive.py','first_stage.py','recourse.py','probability.py',
           'probability_phase1.py','random_streams.py','lazy_recourses.py',
           'zero_presolve.py','native_highs.py','native_gurobi.py')
    engine=canonical_hash({n:sha256(paths.project_root/'src/evcs_experiments'/n) for n in names})
    rows=[auditor.audit_run(ccg/'toys'/name,risk,mode,reference,engine,
                           expected_master_backend='gurobi',expected_optimizer_threads=4)
          for name in auditor.old.GOLDENS for risk in ('DRO','ARO') for mode in ('continuous','resumed')]
    hint_spec=importlib.util.spec_from_file_location('independent_hint_audit',paths.project_root/'scripts/audits/audit_mip_start.py')
    hints=importlib.util.module_from_spec(hint_spec);hint_spec.loader.exec_module(hints)
    starts=[]
    for path in sorted(ccg.rglob('mip_start.npz')):
        d=path.parent
        with np.load(d/'model_arrays.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        with np.load(path,allow_pickle=False) as z:vector=z['vector']
        starts.append(hints.audit_mip_start(load_npz(d/'matrix.npz'),arrays,vector,json.loads((d/'solver.json').read_text())))
    for name in auditor.old.GOLDENS:
        for risk in ('DRO','ARO'):
            pair=[r for r in rows if r['case_id']==name and r['risk_model']==risk]
            assert len(pair)==2 and abs(pair[0]['audit_upper_bound_usd']-pair[1]['audit_upper_bound_usd']) <= 1e-7
    report=dict(status='PASS_GUROBI_CCG_DEVELOPMENT_GATE',full_S03_pass=False,OOS=False,
        primitive=primitive,records=records,runs=rows,hint_checks=starts,
        RMP_matrices=sum(v['RMP_matrices'] for v in rows),
        fixed_LP_certificates=sum(v['fixed_LP_certificates'] for v in rows),
        engine_sha256=engine,source_manifest_sha256=sha256(manifest),
        source_provenance=source_provenance(),audit_program_sha256=sha256(module_path),
        reference_acceptance={'path':str(args.s03_i02_acceptance),'sha256':I02_SHA},
        reference_artifact_files=reference_files,reference_release_files=reference_release,
        wall_seconds=time.monotonic()-began,independent_scientific_review_required=True)
    _json_file(out/'audit.json',report)
    _json_file(out/'hashes.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()})
    print(json.dumps({k:report[k] for k in ('status','RMP_matrices','fixed_LP_certificates','wall_seconds')}))


if __name__=='__main__':
    main()

#!/usr/bin/env python3
"""Independent raw-array feature/identity audit; no scientific kernel imports; path/hash infrastructure only."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,csv,hashlib,json,math
import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_file, historical_path, legacy_transfer_manifest, LEGACY_CASE_NAME, source_provenance, verify_scientific_kernel_identity)


def file_sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def ah(value):
    value=np.ascontiguousarray(value)
    return hashlib.sha256(str((value.shape,value.dtype.str)).encode()+value.tobytes()).hexdigest()
def same(left,right):assert np.array_equal(left,right)
def close(left,right,tol=1e-12):
    left=np.asarray(left);right=np.asarray(right);assert left.shape==right.shape
    error=float(np.max(np.abs(left-right),initial=0.));assert error<=tol,error
    return error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['case-root','joint-output','route-evidence','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    root=a.joint_output.resolve();case=a.case_root.resolve();routes=a.route_evidence.resolve()
    summary=json.loads((root/'increment_validation.json').read_text())
    assert summary['status'] in ['PASS_PREFLIGHT','PASS_INCREMENT_AUTOMATED']
    assert not summary['formal_OOS_outcomes_generated'] and not summary['production_experiments_executed']
    assert summary['test_count']>=112 and summary['test_returncode']==0
    test_log=(root/'tests.stderr.log').read_text();assert '\nOK\n' in test_log and 'skip' not in test_log.lower()
    for name,digest in summary['source_hashes'].items():assert file_sha(frozen_source_file(name))==digest,name
    artifacts=json.loads((root/'artifact_sha256.json').read_text())
    for name,digest in artifacts.items():assert file_sha(root/name)==digest,name
    upstream=json.loads((root/'upstream_validation.json').read_text())
    assert upstream==summary['upstream_validation']
    for name,digest in upstream['input_hashes'].items():assert file_sha(case/name)==digest,name
    # Bind current inputs to the S00-accepted release manifest, not merely
    # to the generator's freshly computed hashes of those same files.
    state=json.loads((get_paths().campaign_root/'state.json').read_text())
    s00_path=historical_path(state['specs']['S00']['acceptance']);s00=json.loads(s00_path.read_text())
    assert s00['status']=='PASS'
    transfer=legacy_transfer_manifest()
    transfer_check=next(v for v in s00['checks']if v['check']=='TRANSFER_MANIFEST_SHA256.csv')
    assert transfer_check['status']=='PASS' and file_sha(transfer)==transfer_check['manifest_sha256']
    frozen_inputs={row['path']:row['sha256']for row in csv.DictReader(transfer.open())}
    for name,digest in upstream['input_hashes'].items():assert frozen_inputs[LEGACY_CASE_NAME+'/'+name]==digest,name
    historical_input_evidence={'S00_acceptance':str(s00_path),'S00_acceptance_sha256':file_sha(s00_path),
        'transfer_manifest_sha256':file_sha(transfer),'current_inputs_equal_frozen_release':len(upstream['input_hashes'])}
    for key in ['last_acceptance','route_acceptance']:assert file_sha(historical_path(upstream[key]))==upstream[key+'_sha256']
    route_accept=json.loads(historical_path(upstream['route_acceptance']).read_text());assert route_accept['status']=='PASS_INCREMENT'
    for name,item in route_accept['raw_artifacts'].items():assert file_sha(routes/name)==item['sha256'],name
    base_events=sorted(p.name.removesuffix('.tasks.json') for p in routes.glob('*.tasks.json'));assert len(base_events)==8
    expected_targets=base_events[:1] if upstream['preflight_only'] else base_events
    assert [v['event']for v in summary['targets']]==expected_targets
    axes=dict(np.load(root/'static_axes.npz'))
    for key,reference in [('point_ids',np.arange(441)),('line_ids',np.arange(1,33)),('road_ids',np.arange(1,89)),('bus_ids',np.arange(1,34))]:same(axes[key],reference)
    assert axes['points_xy_km'].shape==(441,2) and len(np.unique(axes['points_xy_km'],axis=0))==441
    baseline=axes['baseline_p_kw'];assert baseline.shape==(24,33) and baseline.sum()>0
    feature_csv=list(csv.DictReader((root/'features.csv').open()));ledger_csv=list(csv.DictReader((root/'scenario_ledger.csv').open()))
    csv_by_id={row['scenario_id']:row for row in feature_csv};ledger_by_id={row['scenario_id']:row for row in ledger_csv}
    assert len(csv_by_id)==len(feature_csv)==len(ledger_csv)==len(ledger_by_id)==summary['total_scenarios']
    seen=set();primitives={};details=[];max_error=0.;total_task_rows=0;projection_nonzero=0
    names=['Phi_W','Phi_L','Phi_R','Phi_D','Phi_load']
    for event in expected_targets:
        bank=json.loads((routes/f'{event}.tasks.json').read_text());assert bank['certified'] and bank['status']=='COMPLETE' and not bank['tasks_provisional']
        assert canonical({k:bank[k]for k in ['dependency_sha256','config','certified','tasks']})==bank['task_sha256']
        index=json.loads((root/f'{event}.task_index.json').read_text());tasks=bank['tasks']
        task_ids=[t['task_id']for t in tasks];vehicles=[t['vehicle_id']for t in tasks];energy=np.array([t['usable_energy_kwh']for t in tasks])
        assert index['task_ids']==task_ids and index['task_vehicle_ids']==vehicles and index['task_sha256']==bank['task_sha256']
        same(index['forecast_usable_dc_kwh'],energy);assert np.all(energy>0)
        groups={v:np.array([i for i,x in enumerate(vehicles)if x==v])for v in sorted(set(vehicles))}
        weights=np.array([1/(len(groups)*len(groups[v]))for v in vehicles]);same(weights,index['task_weights'])
        event_values=[]
        for split,nvar in [('planning',20),('calibration',10)]:
            nvar=1 if upstream['preflight_only'] else nvar
            data=dict(np.load(root/f'{event}.{split}.npz'))
            metadata=json.loads((root/f'{event}.{split}.ledger.json').read_text());rows=metadata['scenarios'];diagnostics=metadata['diagnostics']
            assert len(rows)==len(diagnostics)==8*nvar
            assert [(x['variant_index'],x['base_weather_event'])for x in rows]==[(v,b)for v in range(nvar)for b in base_events]
            same(data['task_ids'],task_ids);same(data['times_h'],np.arange(-4,96)/4);same(data['times_h'],index['times_h'])
            for name,value in data.items():assert ah(value)==metadata['array_payload_hashes'][name]==summary['array_payload_hashes'][event+'/'+split][name],name
            for i,(row,diag) in enumerate(zip(rows,diagnostics)):
                sid=row['scenario_id'];assert sid not in seen;seen.add(sid)
                assert sid==f"{event}/development/{split}/{row['base_weather_event']}/v{row['variant_index']+1:03d}/R1"==str(data['scenario_ids'][i])
                assert row['target_event']==event and row['split']==split and row['development'] is True and row['regime']=='R1' and row['lead_h']==1
                assert row['variant']==row['variant_index']+1 and row['residual_donor_event']!=event
                assert row['residual_donor_event'] in base_events
                assert row['task_sha256']==bank['task_sha256'] and row['assignment_state_sha256']==index['assignment_state_sha256']
                assert row['grid_sha256']==upstream['grid_sha256']
                assert canonical({k:v for k,v in row.items()if k!='scenario_sha256'})==row['scenario_sha256']
                for name,digest in row['array_hashes'].items():assert ah(data[name][i])==digest,(sid,name)
                for key,value in ledger_by_id[sid].items():assert str(row[key])==value,(sid,key)
                for field in ['line_standard_normals','line_threshold_uniforms','road_corridor_uniforms','load_epsilon','load_standard_innovation','load_p_kw','load_q_kvar']:
                    key=(split,row['base_weather_event'],row['variant_index'],field);digest=ah(data[field][i])
                    if key in primitives:assert primitives[key]==digest,(sid,field)
                    else:primitives[key]=digest
                g=data['grid_hazard'][i];l=data['line_outages'][i];rb=data['road_blocked'][i];e=data['task_usable_energy_dc_kwh'][i];lp=data['load_p_kw'][i]
                assert g.shape==(100,441) and l.shape==(100,32) and rb.shape==(100,88) and e.shape==(len(tasks),) and lp.shape==(24,33)
                assert np.all((g>=0)&(g<=1)) and l.dtype==bool and rb.dtype==bool and np.all(e>=0)
                # Count/total formulas and math.fsum deliberately differ from
                # the feature module's numpy mean implementation.
                delivery=math.fsum(math.fsum(1-min(1,float(e[j]/energy[j]))for j in ix)/len(ix)for ix in groups.values())/len(groups) if groups else 0.
                expected=np.array([math.fsum(g.ravel())/g.size,np.count_nonzero(l)/l.size,np.count_nonzero(rb[:,:80])/(100*80),delivery,math.fsum(lp.ravel())/math.fsum(baseline.ravel())])
                max_error=max(max_error,close(data['features'][i],expected))
                max_error=max(max_error,close([float(csv_by_id[sid][n])for n in names],expected))
                for field in ['target_event','split','base_weather_event']:assert csv_by_id[sid][field]==row[field]
                assert int(csv_by_id[sid]['variant'])==row['variant']
                fd=diag['feature_diagnostics']
                assert fd['time_count']==100 and fd['weather_grid_count']==441 and fd['closed_line_count']==32
                assert fd['road_feature_arc_count']==80 and fd['excluded_connector_count']==8 and fd['candidate_vehicle_count']==len(groups)
                assert fd['empty_H']==(not tasks) and fd['task_sha256']==bank['task_sha256'] and fd['phi_load_exceeds_one']==(data['features'][i,4]>1)
                assert diag['sampled_points']==561 and diag['weather_grid_points']==441
                projection_nonzero+=diag['spatial_projection_points_all_domains']>0
                # Persistent damage and hourly worst-slot mapping, independent
                # of the upstream replay implementations.
                assert np.all(l[1:]>=l[:-1]) and np.all(rb[1:]>=rb[:-1])
                same(data['line_operating_outages'][i],l[4:].reshape(24,4,32).any(axis=1))
                assert not np.any(rb&data['road_movable'][i])
                eps=data['load_epsilon'][i];assert np.all(np.abs(eps)<=.09)
                close(lp,baseline*(1+eps[:,None]))
                close(data['load_q_kvar'][i],axes['baseline_q_kvar']*(1+eps[:,None]))
                close(data['load_archived_q_kvar'][i],axes['baseline_archived_q_kvar']*(1+eps[:,None]))
                total_task_rows+=len(tasks);event_values.append(expected)
        values=np.array(event_values)
        details.append({'event':event,'scenarios':len(values),'task_count':len(tasks),'vehicle_count':len(groups),'feature_min':values.min(axis=0).tolist(),'feature_max':values.max(axis=0).tolist()})
    assert len(seen)==summary['total_scenarios'] and seen==set(csv_by_id)
    # Split labels must correspond to separate random draws, while a complete
    # primitive address must be shared across targets.
    paired=0
    for key,digest in primitives.items():
        if key[0]=='planning':
            other=('calibration',*key[1:])
            if other in primitives:
                assert digest!=primitives[other],key;paired+=1
    result={'status':'PASS_INDEPENDENT_FEATURE_IDENTITY_AUDIT','scope':'preflight'if upstream['preflight_only']else'full_development_bank','full_S01_pass':False,'created_at_utc':datetime.now(timezone.utc).isoformat(),'joint_output':str(root),'scenarios':len(seen),'task_scenario_rows':total_task_rows,'max_independent_feature_difference':max_error,'feature_comparison_tolerance':1e-12,'targets':details,'scenarios_with_spatial_projection':projection_nonzero,'split_primitive_difference_checks':paired,'unique_primitive_addresses':len(primitives),'historical_input_evidence':historical_input_evidence,'all_source_and_raw_artifact_hashes_verified':True,'all_frozen_task_semantic_hashes_verified':True,'all_ledger_and_payload_hashes_verified':True,'script_sha256':file_sha(Path(__file__)),'input_summary_sha256':file_sha(root/'increment_validation.json'),'formal_OOS_outcomes_generated':False,'physical_kernel_reconstruction':'Separate chain audit required; this file independently checks raw feature formulas, IDs, persistence, hourly mapping and load factors.'}
    result.update(source_provenance())
    (out/'independent_feature_identity_audit.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()

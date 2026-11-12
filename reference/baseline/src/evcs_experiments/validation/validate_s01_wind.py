#!/usr/bin/env python3
"""Bounded development-bank wind validation; not whole-S01 acceptance."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys

import numpy as np
from evcs_experiments.config import get_paths
from evcs_experiments.evidence import (frozen_runner_root, frozen_source_paths, frozen_control_root, source_provenance, acceptance_for, historical_path, verify_scientific_kernel_identity)
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.wind_scenarios import load_wind_bank, draw_parameters, generator_contract


def write_json(path, value):
    if Path(path).name == "increment_validation.json":
        value = {**value, **source_provenance()}
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def array_hash(array):
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(json.dumps({'shape':array.shape,'dtype':str(array.dtype)},sort_keys=True).encode())
    h.update(array.tobytes())
    return h.hexdigest()


def main():
    verify_scientific_kernel_identity()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();case=args.case_root.resolve();out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    runner=frozen_runner_root()
    command=[sys.executable,'-m','unittest','discover','-s',str(get_paths().project_root / 'tests'),'-p','test_*.py','-v']
    test=subprocess.run(command,cwd=case,capture_output=True,text=True,timeout=60)
    (out/'tests.stdout.log').write_text(test.stdout);(out/'tests.stderr.log').write_text(test.stderr)
    if test.returncode:
        write_json(out/'increment_validation.json',{'increment':'S01-I02','status':'FAIL','test_returncode':test.returncode,'test_command':command})
        return test.returncode
    bank=load_wind_bank(case)
    points_path=case/'07_weather/loc_L_loc_A__weather_sampling_points.csv'
    with points_path.open(newline='') as f:all_points=list(csv.DictReader(f))
    # The archive includes 37 lines (five fixed-open) and 80 directed road arcs.
    # Preserve arc IDs for downstream replay, but exclude fixed-open ties.
    points_table=[row for row in all_points if row['index_type']=='a' or
                  (row['index_type']=='ell' and row['hazard_eligible_ell']=='1')]
    asset_ids=[row['index_type']+':'+row['index_id'] for row in points_table]
    assert len(asset_ids)==112 and len(set(asset_ids))==112
    points=np.array([[float(row['loc_x_km']),float(row['loc_y_km'])] for row in points_table])
    times=np.arange(-8,96)/4
    unique_locations=len({tuple(point) for point in points.tolist()})
    physical_corridors={tuple(sorted((int(row['from_index']),int(row['to_index']))))
                        for row in points_table if row['index_type']=='a'}
    assert len(physical_corridors)==40 and unique_locations==67
    context={'input_hashes':dict(bank.input_hashes),'sampling_points_sha256':hashlib.sha256(points_path.read_bytes()).hexdigest(),
             'sampling_records':112,'physical_assets':72,'unique_sampling_locations':unique_locations,'fixed_open_ties_excluded':5,
             'times_h':times.tolist(),'asset_ids':asset_ids,'points_xy_km':points.tolist()}
    rows=[];parameters=[];r1_payload=[];r2_payload=[];repeat_ok=True
    for variant in range(2):
        for base in bank.event_ids:
            for target in bank.event_ids:
                p=draw_parameters(bank.event_ids,target,base,variant,development=True)
                r1=bank.sample(p,times,points,regime='R1')
                r2=bank.sample(p,times,points,regime='R2')
                r3=bank.sample(p,times,points,regime='R3')
                assert p.donor_event!=target and r1.spatial_projection_count==0
                assert np.array_equal(r1.vectors,r3.vectors)
                np.testing.assert_allclose(r2.speed,np.minimum(1.15*r1.speed,r1.cap_m_s),atol=1e-10,rtol=1e-12)
                assert float(r1.speed.max())<=r1.cap_m_s+1e-10 and np.all((r1.hazard>=0)&(r1.hazard<=1))
                ids={reg:canonical_hash({'parameters':p.as_dict(),'regime':reg,'kappa_W':1.,'context':context})
                     for reg in ('R1','R2','R3')}
                parameters.append(p)
                rows.append({'target_event':target,'base_weather_event':base,'variant':variant+1,
                             'error_donor_event':p.donor_event,'parameters_sha256':r1.parameters_sha256,
                             'wind_realization_ids':ids,'parameters':p.as_dict(),
                             'min_speed_m_s':float(r1.speed.min()),'mean_speed_m_s':float(r1.speed.mean()),
                             'max_speed_m_s':float(r1.speed.max()),'cap_m_s':r1.cap_m_s,
                             'at_cap_fraction':float(np.mean(np.isclose(r1.speed,r1.cap_m_s,atol=1e-10,rtol=0))),
                             'temporal_projection_count':r1.temporal_projection_count,
                             'spatial_projection_count':r1.spatial_projection_count})
                r1_payload.append(r1.vectors);r2_payload.append(r2.vectors)
    # Regenerate in reverse order: same IDs cannot depend on execution order.
    for i in reversed(range(len(parameters))):
        p=parameters[i]
        again=draw_parameters(bank.event_ids,p.target_event,p.base_event,p.variant_index,development=True)
        repeat_ok &= p==again and np.array_equal(bank.sample(again,times,points).vectors,r1_payload[i])
    assert repeat_ok
    arrays={'wind_R1_vectors_m_s':np.stack(r1_payload),'wind_R2_vectors_m_s':np.stack(r2_payload),
            'times_h':times,'points_xy_km':points,'asset_ids':np.array(asset_ids)}
    np.savez_compressed(out/'development_wind.npz',**arrays)
    write_json(out/'development_wind_ledger.json',rows)
    simple=[{k:v for k,v in row.items() if k not in ('parameters','wind_realization_ids')} for row in rows]
    with (out/'development_wind_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(simple[0]));writer.writeheader();writer.writerows(simple)
    payload_hashes={name:array_hash(values) for name,values in arrays.items()}
    write_json(out/'array_payload_hashes.json',payload_hashes)
    contract=generator_contract();write_json(out/'generator_contract.json',contract)
    write_json(out/'context.json',context);write_json(out/'native_wind_diagnostics.json',dict(bank.diagnostics))
    source=frozen_source_paths()
    summary={'increment':'S01-I02','status':'PASS_INCREMENT_AUTOMATED','full_S01_pass':False,
             'created_at_utc':datetime.now(timezone.utc).isoformat(),'development_wind_realizations':len(rows),
             'regimes_checked':['R1','R2','R3'],'stored_regimes':['R1','R2'],'R3_wind_identical_to_R1':True,
             'time_slots':len(times),'asset_sampling_records':len(points),'unique_sampling_locations':unique_locations,
             'physical_assets':72,
             'fixed_open_ties_excluded':5,'reverse_order_reproduction':bool(repeat_ok),
             'all_spatial_projections_zero':all(row['spatial_projection_count']==0 for row in rows),
             'OOS_outcomes_generated':False,'production_experiments_executed':False,
             'test_returncode':test.returncode,'test_command':command,'array_payload_hashes':payload_hashes,
             'source_hashes':{str(p.relative_to(runner)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source},
             'contract_sha256':canonical_hash(contract),'runtime':{'host':platform.node(),'python':platform.python_version(),'numpy':np.__version__},
             'remaining_S01':['line/road/load generation','forecast routes/H','fixed-route replay','features/moment feasibility']}
    write_json(out/'increment_validation.json',summary)
    write_json(out/'artifact_sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.iterdir()) if p.is_file()})
    print(json.dumps(summary,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())

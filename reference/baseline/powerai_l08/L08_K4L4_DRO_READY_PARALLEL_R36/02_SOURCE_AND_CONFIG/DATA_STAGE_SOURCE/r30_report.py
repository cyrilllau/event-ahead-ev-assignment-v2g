from r3_common import *
import subprocess,zipfile

def report():
 from r30_control import inventory
 inv=inventory('REPORT_STATE');mapping={'INPUTS_VERIFIED':'01_REPORTS/PREFLIGHT_COMPLETE.json','DOMAIN_VALIDATED':'04_R10_COMPACT_DOMAIN/COMPLETE.json','REPLAY400_COMPLETE':'05_SUPPORT400_ROUTE_REPLAY/COMPLETE.json','TESTS_PASSED':'06_TESTS/COMPLETE.json','DET_CERTIFICATE':'05_SELECTED_PLAN_AND_CERTIFICATES/AUDIT_COMPLETE.json','FIXED400_COMPLETE':'07_FIXED400/COMPLETE.json'}
 records={k:read(R/v) if (R/v).exists() else None for k,v in mapping.items()}
 out=R/'09_DELIVERY_REPORT';out.mkdir(exist_ok=True);dump(out/'STATUS.json',records)
 audit=records['DET_CERTIFICATE'];fixed=records['FIXED400_COMPLETE'];txt=[R.name,'Forecast R29_FORECAST512_K4L4_V1/source44; R28 frozen400 nominal weighted support.','Science: R10 full dwell auxiliary, R15 occupancy fee. No scenario regeneration, calibration or other procurement method.',
 'Parameter precedence: source station mapping CSV retains historical P_int_bar_c_kW=300/S_int_bar=375/reactive_support_mode fields. They are not active R30 limits. R15 and explicit R30 policy use 1800 AC-kW per station,20 ports and fixed Q/P=.33, as independently checked in the active matrices. Source CSV is preserved unchanged.','']
 for k,v in records.items():txt.append(k+': '+('PENDING' if v is None else str(v.get('absolute_gap_met',v.get('passed',False)))))
 if audit:
  txt+=['',f"DET selected {audit['selected_count']}/30; F={audit['F']:.12f}; LB={audit['solver_LB']:.12f}; safe UB={audit['audited_UB']:.12f}; signed gap={audit['signed_gap']:.12f} USD.",f"Forecast critical ENS={audit['critical_ENS']:.12f}, ordinary ENS={audit['other_ENS']:.12f} kWh; actual V2G={audit['actual_V2G_ac']:.12f} AC-kWh; shortfall={audit['shortfall_ac']:.12f}."]
  a=loadnp(R/'05_SELECTED_PLAN_AND_CERTIFICATES/FORECAST44/PRIMAL.npz');net=layout()['net'];critical=np.array(net['shed_cost'])==50;events=[]
  for start,end in [(13.75,20.25),(22.5,22.75),(23.25,24.)]:
   sl=slice(int(start*4),int(end*4));events.append(dict(start_h=start,end_h=end,total_ENS=.25*float(a['shed'][sl].sum()),critical_ENS=.25*float(a['shed'][sl][:,critical].sum()),actual_V2G_AC=.25*float(a['p_dis'][:,:,sl].sum()),qualification_latest_h=23.))
  csvout(out/'EVENT_RESULTS.csv',events);dump(out/'EVENT_RESULTS.json',events)
  vv=read(R/'05_SELECTED_PLAN_AND_CERTIFICATES/FORECAST44/VEHICLES.json');txt+=['',f"Stations/buses and all departure, arrival, reserve windows: ../05_SELECTED_PLAN_AND_CERTIFICATES/ALL_30_EV_DECISIONS.csv",f"Total onboard auxiliary from arrival to J: {sum(v['aux_dc_kwh'] or 0 for v in vv):.12f} DC-kWh.",f"Per-event actual results: EVENT_RESULTS.csv; capacities are not added as proof of joint supply."]
 if fixed:txt+=['','FIXED400 weighted diagnostic (not OOS/DRO):',json.dumps(fixed,indent=2)]
 txt+=['','Protected old sources unchanged: '+str(not inv['source_changes']),'Signals sent by R30: none. Existing owner identities and reservations: ../08_SCOPE/REPORT_STATE.json.','Completed LPs and certificate caches carry input, plan, physical, recovery, replay, matrix and M identity.','Native branch tree is not inherited. No old bound or candidate20 route reused.','Numerical certificates use frozen R15 residual thresholds and finite-bound Lagrangian guards; not exact rational proofs.']
 (out/'REPORT.txt').write_text('\n'.join(txt)+'\n')
 if audit:
  p=subprocess.Popen(['/usr/bin/python3',str(R/'tools/r30_plots.py')],stdout=subprocess.PIPE,stderr=subprocess.PIPE);so,se=p.communicate();dump(out/'PLOT_PARENT_WAIT.json',dict(returncode=p.returncode,stdout=so.decode(),stderr=se.decode(),actual_parent_wait=True));assert p.returncode==0
 dump(out/'COMPLETE.json',dict(passed=True,DET_certified=bool(audit and audit['absolute_gap_met']),fixed400_complete=bool(fixed),epoch=time.time()))

def package():
 assert (R/'09_DELIVERY_REPORT/COMPLETE.json').exists(),'report must be explicitly created before package'
 out=R/'delivery';out.mkdir(exist_ok=True);index=out/'PACKAGE_INDEX.json'
 if index.exists():
  idx=read(index)
  for row in idx['packages']:assert sha(out/row['name'])==row['sha256']
  print('PACKAGE_ALREADY_COMPLETE',flush=True);return
 files=[p for p in sorted(R.rglob('*')) if p.is_file() and 'delivery' not in p.relative_to(R).parts and '__pycache__' not in p.parts and p.suffix not in ['.lock','.tmp'] and not p.name.endswith('.inprogress')]
 largeprefix=['03_FORECAST44_ROUTES','04_R10_COMPACT_DOMAIN','05_SUPPORT400_ROUTE_REPLAY','04_MODEL_AND_NATIVE_SOLVE','07_FIXED400']
 data=[p for p in files if p.relative_to(R).parts[0] in largeprefix];analysis=[p for p in files if p not in data];groups=[(R.name+'_PRO_ANALYSIS.zip',analysis)]
 bucket=[];size=0;i=1
 for p in data:
  if bucket and size+p.stat().st_size>350*1024**2:
   groups.append((R.name+f'_DATA_{i:02d}.zip',bucket));i+=1;bucket=[];size=0
  bucket.append(p);size+=p.stat().st_size
 if bucket:groups.append((R.name+f'_DATA_{i:02d}.zip',bucket))
 packages=[]
 for name,group in groups:
  manifest=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in group]
  with zipfile.ZipFile(out/name,'x',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
   for p in group:z.write(p,str(p.relative_to(R)))
   z.writestr('FILE_MANIFEST.json',json.dumps(manifest,indent=2))
  with zipfile.ZipFile(out/name) as z:assert z.testzip() is None
  packages.append(dict(name=name,bytes=(out/name).stat().st_size,sha256=sha(out/name),members=len(group)));print('PACKAGED',name,flush=True)
 dump(index,dict(task=R.name,packages=packages,restore='Extract all archives into one empty directory; shared FILE_MANIFEST.json is per-package metadata. tools/restore_and_verify.py verifies each archive and all payload hashes.',physical_dependencies='Original frozen physical inputs reside in R28/R29 already delivered archives; exact source-file paths/hashes and source bank/id retained in 08_SCOPE/INPUT_MANIFEST.json and 02_SOURCE_AND_CONFIG/SCENE_IDENTITIES.json. No new physical data generated.',epoch=time.time()))

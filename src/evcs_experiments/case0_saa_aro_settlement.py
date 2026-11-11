"""Exact-once terminal accounting for the remaining Case0 OR/SAA/ARO parent."""
import copy,json,math,subprocess,os
from datetime import datetime
from pathlib import Path
from evcs_experiments.campaign_batch import file_reference,verify_reference
from evcs_experiments.random_streams import canonical_hash
from evcs_experiments.runtime.jobctl import atomic_json,observation,TERMINAL

METHODS=('SAA','ARO');EA='size0/EA_DRO';CAMPAIGN='ieee33-v54-q1-case0-expanded'

def need(ok,msg):
    if not bool(ok):raise ValueError(msg)
def read(ref):verify_reference(ref);return json.loads(Path(ref['path']).read_text())
def local(path):return read(file_reference(path))
def number(value,label):
    need(type(value) in (int,float) and math.isfinite(value) and value>=0,'Invalid '+label);return float(value)
def elapsed(start,stop):return (datetime.fromisoformat(stop)-datetime.fromisoformat(start)).total_seconds()

def owned_unit_dead(unit):
    """Read only the explicitly admitted Case0 unit; never stop another unit."""
    need(isinstance(unit,str) and unit.startswith('case0-') and '/' not in unit and not any(x.isspace() for x in unit),'Foreign owned unit')
    command=['systemctl','--user','show',unit,'--property=Id,LoadState,ActiveState,SubState,MainPID,ControlGroup']
    out=subprocess.run(command,capture_output=True,text=True,timeout=5)
    fields=dict(line.split('=',1) for line in out.stdout.splitlines() if '=' in line)
    need(fields.get('LoadState')=='not-found' or fields.get('Id') in (unit,unit+'.service'),'Cannot verify exact owned unit')
    need(fields.get('ActiveState') in ('inactive','failed') and fields.get('MainPID','0')=='0','Owned unit remains live or unknown')
    group=fields.get('ControlGroup','');members=[]
    if group:
        p=Path('/sys/fs/cgroup')/group.lstrip('/');need(p.resolve().is_relative_to('/sys/fs/cgroup'),'Foreign cgroup')
        if p.exists():
            for procs in p.rglob('cgroup.procs'):members.extend(procs.read_text().split())
    need(not members,'Owned cgroup has live members')
    return dict(unit=unit,properties=fields,live_members=members)

def _dead(parent,unit):
    need(parent['complete'] is True and parent['state'] in TERMINAL,'Owned parent is not terminal')
    obs=observation(parent)
    need(not obs['supervisor_alive'] and not obs['command_alive'] and not obs['live_group_members'] and not obs['blocks_new_launch'],'Owned parent/group still alive')
    return dict(parent_observation=obs,unit_observation=owned_unit_dead(unit))

def prepare_remaining_settlement(root,admission_reference,parent_reference,*,external_cost_references=()):
    """Read and derive a reviewable proposal; never alter a campaign ledger."""
    root=Path(root).resolve();need(root.name==CAMPAIGN,'Remaining settlement is expanded Case0 only')
    ad=read(admission_reference);operation=Path(admission_reference['path']).parent
    need(operation.is_relative_to(root/'operations') and operation!=root/'operations','Foreign operation')
    need(ad['kind']=='CASE0_REMAINING_METHODS_ADMISSION_V1' and ad['passed'] is True,'Remaining parent admission required')
    need(ad['settler_reference']==file_reference(__file__),'Wrong admitted settlement source')
    for k in ('driver_reference','guard_reference','request_reference','job_request_reference','producer_source_reference','settler_reference'):verify_reference(ad[k])
    admission_review=read(ad['review_reference'])
    need(admission_review['kind']=='CASE0_REMAINING_METHODS_PRODUCTION_REVIEW_V1' and admission_review['passed'] is True and
         admission_review['settler_reference']==ad['settler_reference'] and admission_review['driver_sha256']==ad['driver_reference']['sha256'] and
         admission_review['guard_reference']==ad['guard_reference'] and admission_review['job_request_reference']==ad['job_request_reference'], 'Admitted source/request review differs')
    need(list(external_cost_references)==ad['external_cost_references']==admission_review['external_cost_references'],'Admitted external metadata fees omitted or changed')
    oldref=ad['state_before_reference'];opening=read(oldref);state_ref=file_reference(root/'state.json');state=read(state_ref)
    job=read(ad['job_request_reference']);request=read(ad['request_reference']);parent=read(parent_reference)
    unit=ad['unit'];jobid=job['job_id']
    need(unit==jobid and Path(parent_reference['path'])==root/'jobs'/jobid/'result.json' and parent['job_id']==jobid,'Foreign actual parent/unit')
    manifest_ref=file_reference(root/'jobs'/jobid/'manifest.json');manifest=read(manifest_ref)
    need(all(manifest[k]==job[k] for k in ('job_id','cwd','outroot','argv','timeout_seconds','grace_seconds','threads')) and manifest['threads']==1,'Actual parent manifest differs')
    need(state['active_job_id']==jobid and state['pending_remaining_methods']==admission_reference,'Wrong pending remaining parent')
    need(request['kind']=='CASE0_REMAINING_METHODS_REQUEST_V1' and request['methods']==list(METHODS) and Path(request['operation_directory'])==operation and request['references']['state']==oldref and request['review_reference']==ad['review_reference'] and admission_review['request_without_review']=={k:v for k,v in request.items() if k!='review_reference'},'Foreign remaining driver request')
    need(state['accepted_approximate_plans']==opening['accepted_approximate_plans']==ad['accepted_approximate_plans'] and state['source_release_reference']==opening['source_release_reference']==ad['producer_source_reference'] and
         state['active_config_reference']==opening['active_config_reference']==request['references']['config'],'EA acceptance or producer config/source changed')
    cfg=ad['opening_configuration_spent_seconds'];masters=ad['opening_configuration_master_spent_seconds']
    need(cfg==opening['configuration_spent_seconds'] and masters==opening['configuration_master_spent_seconds'] and
         ad['opening_compute_seconds']==opening['compute_seconds'],'Admitted opening ledger snapshots differ')
    need(state['configuration_spent_seconds'][EA]==cfg[EA] and state['configuration_master_spent_seconds'][EA]==masters[EA],'Accepted EA costs changed')
    need(state['configuration_spent_seconds']['size0/OR_DRO']==cfg['size0/OR_DRO'] and state['configuration_master_spent_seconds']['size0/OR_DRO']==masters['size0/OR_DRO'] and state['deferred_quality_plans']==opening['deferred_quality_plans'],'Deferred OR evidence or costs changed')
    dead=_dead(parent,unit);whole=number(parent['elapsed_seconds'],'actual parent elapsed')
    charged=number(state['compute_seconds']-opening['compute_seconds'],'already charged inner cost')
    need(charged<=whole+1e-5,'Inner charged amount exceeds actual parent; independent diagnosis required')
    references=dict(admission=admission_reference,parent=parent_reference,manifest=manifest_ref,opening_state=oldref,observed_state=state_ref,
        request=ad['request_reference'],job_request=ad['job_request_reference'])
    inner_path=operation/'inner_accounting.json'
    if inner_path.exists():
        ir=file_reference(inner_path);inner=read(ir);references['inner_accounting']=ir
        need(inner['kind']=='CASE0_REMAINING_METHODS_INNER_ACCOUNTING_V1' and inner['opening_compute_seconds']==opening['compute_seconds'] and
             inner['closing_inner_compute_seconds']==state['compute_seconds'] and inner['inner_charged_seconds']==charged and inner['parent_settlement_required'] is True,
             'Inner receipt and actual cumulative ledger differ')
        for r in inner['method_accounting_references']:verify_reference(r)
    updates=copy.deepcopy(state);rows=[];interrupted=[]
    for method in METHODS:
        key='size0/'+method;folder=root/'planning'/key;start_path=folder/'start.json';start=None
        if start_path.exists():
            sr=file_reference(start_path);start=read(sr)
            owned=(Path(start.get('accepted_ea_record_reference',{}).get('path',''))==operation/'accepted_ea_comparison.json')
        else:owned=state.get('active_configuration')==key
        if not owned:
            need(state.get('configuration_spent_seconds',{}).get(key,0)==cfg.get(key,0) and state.get('configuration_master_spent_seconds',{}).get(key,0)==masters.get(key,0),'Unowned method fees changed')
            continue
        need(cfg.get(key,0)==masters.get(key,0)==0 and key not in opening['completed'],'Owned new method had prior work')
        finer=whole;method_refs={}
        if start is not None:
            need(start['method']==method and start['source_release_reference']==ad['producer_source_reference'] and start['active_config_reference']==opening['active_config_reference'] and start['prior_configuration_seconds']==start['prior_master_seconds']==0.,'Owned method start provenance changed')
            verify_reference(start['accepted_ea_record_reference']);method_refs['start']=sr
            finer=elapsed(start['started_at_utc'],parent['finished_at_utc']);need(0<=finer<=whole+.01,'Method UTC start outside parent interval')
        ap=folder/'accounting.json';rp=folder/'solve/result.json';result=None
        if ap.exists():
            ar=file_reference(ap);receipt=read(ar);method_refs['accounting']=ar
            need(receipt['method']==method and receipt['global_charge_nested'] is True,'Method receipt identity differs')
            spent=number(receipt['configuration_wall_seconds'],'actual method wall');master=number(receipt['master_seconds'],'actual/conservative method master')
            need(spent<=finer+.01 and master<=spent+.01,'Method/master receipt exceeds parent time envelope')
            if receipt['result_reference'] is not None:
                need(Path(receipt['result_reference']['path'])==rp,'Foreign result receipt');result=read(receipt['result_reference']);method_refs['result']=receipt['result_reference']
            complete=not receipt['failed'] and result is not None
            basis=receipt['master_charge_scope']
        else:
            spent=finer;complete=False;master=None;basis='Interrupted method: UTC start to terminal parent upper bound; no saved master receipt, no clamp.'
            if rp.exists():
                rr=file_reference(rp);saved=read(rr);method_refs['unsettled_result']=rr
                if saved.get('method_id')==method and type(saved.get('master_wall_seconds')) in (int,float) and math.isfinite(saved['master_wall_seconds']) and saved['master_wall_seconds']>=0:
                    master=float(saved['master_wall_seconds']);basis='Interrupted accounting: saved matching result actual master wall; configuration remains conservative UTC upper bound.'
            mp=folder/'solve/master.json'
            if master is None and mp.exists():
                mr=file_reference(mp);method_refs['unsettled_master']=mr
                try:
                    value=read(mr).get('wall_seconds')
                    if type(value) in (int,float) and math.isfinite(value) and value>=0:
                        master=float(value);basis='Interrupted accounting: saved owned master actual wall; configuration remains conservative UTC upper bound.'
                except (ValueError,TypeError,AttributeError):
                    basis='Interrupted accounting: malformed retained master; full configuration is conservative master upper bound, never clamped.'
            if master is None:master=spent
            need(master<=spent+.01,'Saved interrupted master exceeds configuration upper bound')
            interrupted.append(key)
        need(state.get('configuration_spent_seconds',{}).get(key,0) in (0,spent) and state.get('configuration_master_spent_seconds',{}).get(key,0) in (0,master),'Method live ledger conflicts with retained receipt')
        if result is not None:
            need(result['method_id']==method and result['backend']=='gurobi' and result['formal_OOS'] is False,'Method result provenance differs')
            if complete:
                need(result.get('numerical_valid') is True and type(result.get('gap')) in (int,float) and math.isfinite(result['gap']) and result['gap']>=0 and result.get('gap_passed') is (result['gap']<=.005),'Completed remaining result lacks unchanged numerical/gap semantics')
                if key in state['completed']:
                    need(state['completed'][key]['path']==method_refs['result']['path'] and state['completed'][key]['sha256']==method_refs['result']['sha256'],'Completed method pointer differs')
                else:
                    updates['completed'][key]=dict(path=method_refs['result']['path'],sha256=method_refs['result']['sha256'],gap=result['gap'],numerical_valid=True,gap_passed=result['gap_passed'])
        updates.setdefault('configuration_spent_seconds',{})[key]=spent;updates.setdefault('configuration_master_spent_seconds',{})[key]=master
        rows.append(dict(method=method,configuration_seconds=spent,master_seconds=master,complete_receipt=complete,normal_gap_passed=bool(result is not None and result.get('gap_passed') is True),references=method_refs,master_charge_basis=basis))
    extras=[]
    for r in external_cost_references:
        d=read(r);amount=number(d['actual_seconds'],'external metadata cost');extras.append(dict(reference=r,seconds=amount))
    need(len({e['reference']['path'] for e in extras})==len(extras),'Duplicate external metadata fees')
    extra=sum(e['seconds'] for e in extras);residual=max(0.,whole-charged)
    updates['compute_seconds']=state['compute_seconds']+residual+extra
    over=[r['method'] for r in rows if r['configuration_seconds']>1800 or r['master_seconds']>1200]
    failed=parent['state']!='SUCCEEDED' or bool(interrupted) or bool(over) or state['status']=='FAILURE_REQUIRES_REVIEW'
    if failed:status='REMAINING_METHODS_STOPPED_FAILURE'
    elif state['status']=='DIAGNOSIS_REQUIRED' or any(not r['normal_gap_passed'] for r in rows):status='REMAINING_METHODS_STOPPED_DIAGNOSIS'
    elif all('size0/'+m in updates['completed'] for m in METHODS):status='SAA_ARO_RESULTS_PENDING_OR_QUALITY_AND_ALL_PLAN_AUDIT'
    else:status='REMAINING_METHODS_PARENT_BOUNDARY_SETTLED'
    updates.update(active_job_id=None,active_configuration=None,pending_remaining_methods=None,remaining_methods_parent_pending_settlement=False,status=status)
    updates['next_action']='Remaining-method parent terminally settled. Accepted EA is immutable; no EA solve or generic gap waiver. '+('Preserve interruption/failed evidence and diagnose before any further launch.' if failed else 'Independently audit remaining methods and adapt exact-EA comparison acceptance before OOS; any next parent needs a new source/budget admission.')
    record=dict(kind='CASE0_REMAINING_METHODS_SETTLEMENT_V1',passed=True,scientific_acceptance=False,formal_OOS=False,references=references,owned_terminal_observation=dead,
        opening_compute_seconds=opening['compute_seconds'],actual_parent_seconds=whole,already_charged_inner_seconds=charged,parent_residual_seconds=residual,
        external_costs=extras,closing_compute_seconds=updates['compute_seconds'],method_accounting=rows,interrupted_methods=interrupted,budget_exceeded_methods=over,
        accepted_ea_configuration_seconds=cfg[EA],accepted_ea_master_seconds=masters[EA],status=status,
        accounting_scope='Whole parent once global minus previously charged inner; each actual completed method/master retained. Unreceipted interrupted methods use actual start-to-terminal upper bound and missing master uses full conservative configuration, never clamped. EA unchanged.')
    proposal=dict(kind='CASE0_REMAINING_METHODS_SETTLEMENT_PROPOSAL_V1',record=record,updated_state=updates,output_path=str(operation/'terminal_settlement.json'))
    proposal['proposal_sha256']=canonical_hash(proposal);return proposal

def commit_remaining_settlement(root,proposal,independent_review_reference):
    """Commit only an independently source/proposal-bound terminal proposal."""
    root=Path(root).resolve();body=dict(proposal);digest=body.pop('proposal_sha256');need(canonical_hash(body)==digest,'Changed settlement proposal')
    review=read(independent_review_reference)
    need(review['passed'] is True and review['kind']=='CASE0_REMAINING_SETTLEMENT_REVIEW_V1' and review['proposal_sha256']==digest and review['settlement_source_reference']==file_reference(__file__),'Independent exact settlement source/proposal review required')
    record=copy.deepcopy(proposal['record']);refs=record['references'];read(refs['observed_state']);parent=read(refs['parent']);ad=read(refs['admission']);_dead(parent,ad['unit'])
    # Recheck every concrete file used by the reviewed calculation, including
    # receipts and externally measured fees, before publishing its commit marker.
    def reverify(value):
        if isinstance(value,dict):
            if {'path','sha256','size_bytes'}<=value.keys():verify_reference(value)
            else:
                for v in value.values():reverify(v)
        elif isinstance(value,list):
            for v in value:reverify(v)
    reverify(record)
    output=Path(proposal['output_path']);need(root.name==CAMPAIGN and output==Path(refs['admission']['path']).parent/'terminal_settlement.json' and output.is_relative_to(root/'operations'),'Foreign settlement output')
    # Exclusive receipt is the commit marker. A crash afterward must reconcile
    # this receipt; it cannot charge the same parent a second time.
    record.update(independent_review_reference=independent_review_reference,proposal_sha256=digest,settlement_source_reference=file_reference(__file__))
    with output.open('x') as f:json.dump(record,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    state=copy.deepcopy(proposal['updated_state']);state['remaining_methods_settlement_reference']=file_reference(output)
    from evcs_experiments.case0_campaign import save_state,decision
    save_state(root,state);decision(root,'CASE0_REMAINING_METHODS_PARENT_SETTLED',settlement_reference=file_reference(output))
    return record

"""Read-only checkpoint DAG validation with explicit release inventory schema.

May be loaded as a separately source-bound validator; it performs no model
construction, source migration, checkpoint rewrite, or solver call.
"""
from pathlib import Path
from datetime import datetime
import math
from .campaign_batch import file_reference, verify_reference, read_json


def _need(value,message):
    if not bool(value):raise ValueError(message)


def _read(ref):
    verify_reference(ref)
    return read_json(ref['path'])


def _release_inventory_closure(ref, manifest):
    """Validate prepare_release's relative inventory, not a generic ref DAG."""
    expected_keys={'schema_version','release_name','created_at_utc','git_head','working_tree_snapshot','files'}
    root=Path(ref['path']).parent
    _need(isinstance(manifest,dict) and set(manifest) in (expected_keys,expected_keys|{'inherited_manifest','preflight_reference'}) and
          type(manifest['schema_version']) is int and manifest['schema_version']==1 and
          manifest['working_tree_snapshot'] is True and manifest['release_name']==root.name and
          isinstance(manifest['created_at_utc'],str) and isinstance(manifest['git_head'],str) and
          len(manifest['git_head'])==40 and all(c in '0123456789abcdef' for c in manifest['git_head']) and
          isinstance(manifest['files'],list) and bool(manifest['files']), 'Malformed release manifest schema')
    created=datetime.fromisoformat(manifest['created_at_utc'].replace('Z','+00:00'))
    _need(created.tzinfo is not None,'Release creation timestamp requires timezone')
    # Exact producer filters from ops/powerai/prepare_release.py. Runtime venv,
    # machine configuration and caches are deliberately outside source inventory.
    excluded={'.git','.venv','artifacts','__pycache__','.pytest_cache','.ruff_cache','build','dist'}
    def included(relative):
        return not (set(relative.parts)&excluded or relative.name=='.DS_Store' or relative.suffix=='.pyc' or
                    any(part.endswith('.egg-info') for part in relative.parts) or
                    relative.as_posix() in {'configs/local.toml','configs/powerai.toml','RELEASE_MANIFEST.json'})
    seen=set()
    for row in manifest['files']:
        _need(isinstance(row,dict) and set(row)=={'path','sha256','size_bytes'} and
              isinstance(row['path'],str),'Malformed release inventory member')
        relative=Path(row['path'])
        _need(not relative.is_absolute() and relative.parts and '..' not in relative.parts and
              relative.as_posix()==row['path'] and included(relative) and row['path'] not in seen,
              'Release inventory duplicate/escape/excluded path')
        target=root/relative
        _need(target.resolve().is_relative_to(root) and
              not any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)),
              'Release inventory symlink/escape')
        _need(type(row['size_bytes']) is int and row['size_bytes']>=0 and
              isinstance(row['sha256'],str) and len(row['sha256'])==64 and
              all(c in '0123456789abcdef' for c in row['sha256']), 'Invalid release member digest/size')
        verify_reference(dict(path=str(target),sha256=row['sha256'],size_bytes=row['size_bytes']))
        seen.add(row['path'])
    actual=set()
    for path in root.rglob('*'):
        relative=path.relative_to(root)
        if not included(relative):continue
        _need(not path.is_symlink(),'Unlisted release symlink')
        if path.is_file():actual.add(relative.as_posix())
    _need(seen==actual,'Release inventory does not match source files')


_R20_PREFLIGHT_STATUS = 'PASS_EXACT_INHERITANCE_AND_REVIEWED_OVERLAY_BYTES'
_ASSOCIATION_PREPARATION_FILES = {
    'src/evcs_experiments/formal_planning.py','src/evcs_experiments/raw_template_audit.py',
    'src/evcs_experiments/raw_plan_audit.py','scripts/audits/audit_s05_candidate_recourse.py',
    'tests/test_recourse_template_index.py','tests/test_raw_template_audit.py','tests/test_raw_plan_audit.py',
    'docs/experiments/reviews/template_association_review_20260908.md',
    'docs/experiments/reviews/raw_plan_audit_development_20260908.md',
    'docs/experiments/reviews/template_audit_integration_preparation_20260908.md'}


def _relative_source_member(root, row):
    """Interpret one inventory row only against its explicitly supplied root."""
    _need(isinstance(row,dict) and set(row)=={'path','sha256','size_bytes'} and
          isinstance(row['path'],str),'Malformed relative source inventory member')
    relative=Path(row['path'])
    _need(not relative.is_absolute() and relative.parts and '..' not in relative.parts and
          relative.as_posix()==row['path'],'Relative source inventory path escapes root')
    target=root/relative
    _need(root.is_absolute() and root.resolve()==root and target.resolve().is_relative_to(root) and
          not any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)),
          'Relative source inventory symlink/escape')
    bound=dict(row,path=str(target));_reference_shape(bound);verify_reference(bound)
    return bound


def _association_preparation_sources(ref, value, base_ref, base):
    """Exact pack_development.py inventory and historical test-log metadata.

    Its source bytes are live dependencies, including superseded overlay files.
    The three workstation test-log addresses are recorded outcomes of this
    already-hashed preparation; they are not paths on the consuming server.
    Their role never enters the general dependency cache.
    """
    keys={'status','created_at_utc','python_version','base_release_manifest_sha256','files','tests','test_logs',
          'independent_review','source_preparation_only','installed_on_server','new_formal_plan_executed',
          'production_source_gate_accepted','master_bound_audited','whole_S05_PASS','formal_OOS','next_step'}
    _need(isinstance(value,dict) and set(value)==keys and
          value['status']=='PASS_LOCAL_DEVELOPMENT_PENDING_FORMAL_WORKER_INTEGRATION' and
          value['python_version']=='3.12.0' and value['base_release_manifest_sha256']==base_ref['sha256'] and
          value['source_preparation_only'] is True and all(value[k] is False for k in
          ('installed_on_server','new_formal_plan_executed','production_source_gate_accepted',
           'master_bound_audited','whole_S05_PASS','formal_OOS')),
          'Malformed historical association development preparation')
    created=datetime.fromisoformat(value['created_at_utc'].replace('Z','+00:00'))
    _need(created.tzinfo is not None and all(isinstance(value[k],str) and value[k]
          for k in ('independent_review','next_step')),'Invalid association preparation metadata')
    counts={'pointer_envelope_existing_count':40,'selected_plan_numerics_probability_count':11,'legacy_kernel_count':9}
    _need(value['tests']==counts and all(type(v) is int for v in value['tests'].values()),
          'Historical association test counts differ')
    logs=value['test_logs'];names=['local_pointer_envelope_tests.log','local_raw_plan_audit_tests.log','legacy_kernel_regression.log']
    _need(isinstance(logs,list) and len(logs)==3,'Historical association log inventory differs')
    parents=set()
    for log,name in zip(logs,names):
        _reference_shape(log);path=Path(log['path'])
        _need(path.name==name and str(path)==log['path'] and '..' not in path.parts,
              'Historical association log identity differs')
        parents.add(path.parent)
    _need(len(parents)==1 and next(iter(parents)).name==Path(ref['path']).parent.name,
          'Historical association log operation differs')
    rows=value['files'];_need(isinstance(rows,list) and len(rows)==len(_ASSOCIATION_PREPARATION_FILES),
                            'Historical association source inventory differs')
    old={r['path']:r for r in base['files']};result={};root=Path(ref['path']).parent/'prepared_source'
    for row in rows:
        _need(isinstance(row,dict) and set(row)=={'relative_path','sha256','size_bytes','change','inherited_reference'} and
              row['relative_path'] in _ASSOCIATION_PREPARATION_FILES and row['relative_path'] not in result,
              'Historical association missing/duplicate/foreign source')
        name=row['relative_path'];inherited=old.get(name)
        _need(row['inherited_reference']==inherited and row['change']==('modified' if inherited else 'new'),
              'Association source/base inventory binding differs')
        if inherited is not None:_relative_source_member(Path(base_ref['path']).parent,inherited)
        member=dict(path=name,sha256=row['sha256'],size_bytes=row['size_bytes'])
        _relative_source_member(root,member);result[name]=member
    return result


def _release_preflight_closure(ref, manifest_ref, manifest, seen):
    """Interpret the exact r20 producer; unknown preflights retain generic DAG rules."""
    value=_read(ref)
    if not isinstance(value,dict) or value.get('status')!=_R20_PREFLIGHT_STATUS:
        verify_checkpoint_closure(ref,seen);return
    keys={'status','base_manifest','preparations','patches','files','inherited_unchanged',
          'source_file_count','release_root','source_admission_scope'}
    root=Path(manifest_ref['path']).parent
    _need(set(value)==keys and value['base_manifest']==manifest['inherited_manifest'] and
          value['release_root']==str(root) and
          value['source_admission_scope']=='OR input generation only; SAA fullQ/queue and OOS gates pending',
          'Malformed inherited release preflight identity')
    base_ref=value['base_manifest'];base=_read(base_ref)
    verify_checkpoint_closure(base_ref,seen)
    old={r['path']:r for r in base['files']};merged=dict(old)
    preparations=value['preparations']
    _need(isinstance(preparations,list) and len(preparations)==3 and
          len({p['path'] for p in preparations})==3,'Exact three inherited preparations required')
    for index,prep_ref in enumerate(preparations):
        _reference_shape(prep_ref);prep=_read(prep_ref)
        if index==0:
            merged.update(_association_preparation_sources(prep_ref,prep,base_ref,base));continue
        status=('PASS_DEVELOPMENT_GENERIC_PIPELINE','PASS_SCOPED_SAA_OR_COMPONENT_DEVELOPMENT')[index-1]
        common={'status','source_references','test_logs','production_deployed','scientific_spec_pass','formal_OOS','remaining'}
        extra=({'unchanged_budgets','validation','new_allocation','independent_review'} if index==1 else
               {'inherited_preparation','unchanged','runtime_limits','independently_reviewed_files','implementations',
                'total_recorded_tests','next_efficiency','real_development_evidence','tests','inherited_modified_whitelist'})
        _need(isinstance(prep,dict) and set(prep)==common|extra and prep.get('status')==status and
              isinstance(prep.get('source_references'),list) and bool(prep['source_references']) and
              all(prep.get(k) is False for k in ('production_deployed','scientific_spec_pass','formal_OOS')),
              'Inherited generic preparation schema differs')
        if index==2:_need(prep['inherited_preparation']==preparations[1],'Method preparation predecessor differs')
        sources=set();prepared_root=Path(prep_ref['path']).parent/'prepared_source'
        for source in prep['source_references']:
            _reference_shape(source);path=Path(source['path'])
            _need(path.is_relative_to(prepared_root),'Prepared source belongs to another operation')
            name=path.relative_to(prepared_root).as_posix()
            _need(name not in sources,'Duplicate inherited prepared source');sources.add(name)
            row=dict(source,path=name);_relative_source_member(prepared_root,row);merged[name]=row
        verify_checkpoint_closure(prep_ref,seen)
    patch_names={'src/evcs_experiments/formal_bank.py','tests/test_bank_worker_deadline.py'}
    patches=value['patches'];_need(isinstance(patches,list) and len(patches)==2,'Exact r20 patches required')
    found=set();patch_root=Path(ref['path']).parent/'patches'
    for patch in patches:
        _reference_shape(patch);path=Path(patch['path'])
        _need(path.is_relative_to(patch_root),'Patch belongs to another operation')
        name=path.relative_to(patch_root).as_posix()
        _need(name in patch_names and name not in found,'Missing/duplicate/foreign r20 patch');found.add(name)
        row=dict(patch,path=name);_relative_source_member(patch_root,row);merged[name]=row
    final={r['path']:r for r in manifest['files']}
    _need(merged==final,'Inherited preparations do not reproduce final release inventory')
    changed={name:row for name,row in final.items() if old.get(name)!=row}
    rows=value['files'];_need(isinstance(rows,list) and len(rows)==len(changed),'Preflight changed inventory differs')
    seen_paths=set()
    for row in rows:
        _relative_source_member(root,row)
        _need(row['path'] not in seen_paths and changed.get(row['path'])==row,'Preflight source delta differs')
        seen_paths.add(row['path'])
    _need(seen_paths==set(changed) and type(value['source_file_count']) is int and
          value['source_file_count']==len(final) and type(value['inherited_unchanged']) is int and
          value['inherited_unchanged']==len(final)-len(changed),'Preflight source counts differ')


_AUDIT_REPAIR_SCOPE = 'Repaired standalone independent auditor and reviewed launch controls; no current release or scientific input mutation'
_AUDIT_REPAIR_FILES = {'scripts/audits/audit_formal_bank.py','tests/test_formal_bank_audit.py',
    'independent_review.md','prepare_retry.py','launch_retry.py','verify_baseline.py','audit_test_config.json'}


def _audit_repair_inventory_closure(ref, manifest, seen):
    """Exact legacy prepare_retry.py inventory: paths relative to its operation."""
    root=Path(ref['path']).parent
    _need(set(manifest)=={'scope','files'} and manifest['scope']==_AUDIT_REPAIR_SCOPE and
          isinstance(manifest['files'],list) and len(manifest['files'])==len(_AUDIT_REPAIR_FILES),
          'Malformed standalone audit repair inventory')
    paths=set(); references=[]
    for row in manifest['files']:
        _need(isinstance(row,dict) and set(row)=={'path','sha256','size_bytes'} and
              isinstance(row['path'],str) and row['path'] in _AUDIT_REPAIR_FILES and row['path'] not in paths,
              'Audit repair inventory missing/duplicate/foreign path')
        relative=Path(row['path']);target=root/relative
        _need(target.resolve().is_relative_to(root) and
              not any((root/Path(*relative.parts[:i])).is_symlink() for i in range(1,len(relative.parts)+1)),
              'Audit repair inventory symlink/escape')
        _need(type(row['size_bytes']) is int and row['size_bytes']>=0 and
              isinstance(row['sha256'],str) and len(row['sha256'])==64 and
              all(c in '0123456789abcdef' for c in row['sha256']), 'Invalid audit repair digest/size')
        bound=dict(row,path=str(target));verify_reference(bound);references.append(bound);paths.add(row['path'])
    _need(paths==_AUDIT_REPAIR_FILES,'Incomplete audit repair inventory')
    config=read_json(root/'audit_test_config.json')
    _need(isinstance(config,dict) and isinstance(config.get('paths'),dict) and
          config['paths'].get('project_root')==str(root),'Audit repair declared source root differs')
    for bound in references:verify_checkpoint_closure(bound,seen)


_ASSOCIATION_CHECK_KEYS = {'status','passed','index_reference','proof_references','acceptance_reference',
    'prior_review_reference','source_manifest','rows_reference','checked_references','verified_template_count',
    'verified_template_artifact_count','hashed_unique_files','hashed_bytes_including_repeats','elapsed_seconds',
    'solver_calls','recourse_reassembly_calls','full_numeric_audit_repeated','old_Q_imported',
    'production_envelope_audit_accepted','formal_OOS','formal_plan_budget_changed'}
_ASSOCIATION_CHECK_STATUS = 'PASS_FULL160_ASSOCIATION_CHECKS_PENDING_REVIEW'


def _historical_association_index(ref, certificate):
    """Verify a prior check's immutable index, not re-run its historical checks.

    This role applies only to association_checks.checked_references. It never
    adds the indexed paths to the dependency cache: a directly referenced current
    artifact must still pass the ordinary source/model validation path.
    """
    _need(set(certificate)==_ASSOCIATION_CHECK_KEYS and certificate['status']==_ASSOCIATION_CHECK_STATUS and
          certificate['passed'] is True and certificate['verified_template_count']==160 and
          certificate['verified_template_artifact_count']==1280 and certificate['hashed_unique_files']==3204 and
          certificate['solver_calls']==0 and certificate['recourse_reassembly_calls']==0 and
          all(certificate[k] is False for k in ('full_numeric_audit_repeated','old_Q_imported',
              'production_envelope_audit_accepted','formal_OOS','formal_plan_budget_changed')),
          'Malformed historical full-support association certificate')
    integer_fields=('verified_template_count','verified_template_artifact_count','hashed_unique_files',
                    'solver_calls','recourse_reassembly_calls','hashed_bytes_including_repeats')
    _need(all(type(certificate[k]) is int and certificate[k]>=0 for k in integer_fields) and
          type(certificate['elapsed_seconds']) in (int,float) and
          math.isfinite(certificate['elapsed_seconds']) and certificate['elapsed_seconds']>=0,
          'Invalid historical association measurement type/range')
    def strong_shape(value):
        _need(isinstance(value,dict) and set(value)=={'path','sha256','size_bytes'} and
              isinstance(value['path'],str) and Path(value['path']).is_absolute() and
              str(Path(value['path']))==value['path'] and '..' not in Path(value['path']).parts and
              type(value['size_bytes']) is int and value['size_bytes']>=0 and
              isinstance(value['sha256'],str) and len(value['sha256'])==64 and
              all(c in '0123456789abcdef' for c in value['sha256']),
              'Malformed required association strong reference')
    for field in ('index_reference','acceptance_reference','prior_review_reference','source_manifest',
                  'rows_reference','checked_references'):
        strong_shape(certificate[field])
    proofs=certificate['proof_references']
    _need(isinstance(proofs,dict) and set(proofs)=={'bank_audit','numeric_audit','reader_acceptance'},
          'Historical association proof reference scope differs')
    for proof in proofs.values():strong_shape(proof)
    index_ref=certificate['checked_references']
    _need(isinstance(index_ref,dict) and set(index_ref)=={'path','sha256','size_bytes'} and
          Path(index_ref['path'])==Path(ref['path']).parent/'checked_references.json',
          'Historical association index identity differs')
    rows=_read(index_ref)
    _need(isinstance(rows,list) and len(rows)==certificate['hashed_unique_files'],
          'Historical association index count differs')
    paths=set()
    for row in rows:
        _need(isinstance(row,dict) and set(row)=={'path','sha256','size_bytes'} and
              isinstance(row['path'],str) and Path(row['path']).is_absolute() and
              str(Path(row['path']))==row['path'] and '..' not in Path(row['path']).parts and
              row['path'] not in paths and type(row['size_bytes']) is int and row['size_bytes']>=0 and
              isinstance(row['sha256'],str) and len(row['sha256'])==64 and
              all(c in '0123456789abcdef' for c in row['sha256']),
              'Malformed historical association index member')
        paths.add(row['path'])
    # No indexed payload is read here: these are the recorded outcomes/scope of
    # an old check, not a new declaration of current numerical dependencies.
    return index_ref


def _reference_shape(ref):
    """Exact absolute reference shape; callers separately verify bytes."""
    _need(isinstance(ref,dict) and set(ref)=={'path','sha256','size_bytes'} and
          isinstance(ref['path'],str) and Path(ref['path']).is_absolute() and
          type(ref['size_bytes']) is int and ref['size_bytes']>=0 and isinstance(ref['sha256'],str) and
          len(ref['sha256'])==64 and all(c in '0123456789abcdef' for c in ref['sha256']),
          'Malformed role-bound proof reference')


def _proof_reference(ref, header):
    """Hash a typed proof artifact; its historical verification inputs are data."""
    _reference_shape(ref)
    proof=_read(ref)
    _need(isinstance(proof,dict) and all(k in proof and type(proof[k]) is type(v) and proof[k]==v
          for k,v in header.items()),'Role-bound proof schema differs')
    return proof


_SAA_REQUEST_KEYS={'schema_version','domain','spec_id','model_id','method_id','target_id','plan_id',
    'budget','model_input_mode','max_coefficient_bytes','output_root','helper_reference','planning_source_references',
    'template_references','protocol_reference','config_reference','bank_acceptance_reference','summary_reference',
    'audit_reference','template_index_reference','template_association_review_reference','model_checkpoint_reserve_seconds',
    'model_reload_reserve_seconds','release_reference','failure_carryover_reference'}
_TEMPLATE_INDEX_KEYS={'kind','status','domain','first_stage_sha256','scenario_ids','scenario_sha256s','rows',
    'original_request_reference','original_snapshot_reference','proof_references','formal_OOS','scientific_acceptance','old_q_reused'}
_RAW_REQUEST_KEYS={'schema_version','domain','problem_identity','input_binding','exporter_sha256','formal_OOS','independent_scientific_audit_required'}
_BANK_PROOF={'status':'PASS_INDEPENDENT_FORMAL_BANK_AUDIT','formal_OOS':False}


_STREAMING_KEYS={'schema_version','algorithm_id','risk_model','domain','input_binding','first_stage_sha256',
    'scenario_ids','scenario_sha256s','empirical_probabilities','scenario_cost_constants_usd','objective_constant_usd',
    'scenarios','shape','nnz','coefficient_bytes','max_coefficient_bytes','preparation_seconds','numeric_artifacts',
    'status','solver_calls','scientific_spec_pass','formal_OOS','solver_peak_rss_measured',
    'shared_first_stage_unchanged','manifest_sha256'}
_COMPLETION_KEYS={'kind','request_reference','model_manifest_reference','model_input_mode','planning_source_references',
    'input_binding','model_checkpoint_reserve_seconds','model_reload_reserve_seconds','preparation_elapsed_seconds',
    'prior_cumulative_seconds','accounting_authority','numerical_audit_complete','scientific_acceptance','formal_OOS'}
_STREAM_BINDING_KEYS={'summary','audit','target','task_source','formal_protocol_lock','bank_source_manifest',
    'planning_source','physical_case_sha256','task_sha256','config_hash','assignment_state_sha256','method_id','rho',
    'formal_OOS','template_provider','objective_spec','algorithm_id'}
_STREAM_ALGORITHM='SAA_EXACT_STREAMING_EXTENSIVE_ZERO_V1'


def _streaming_input_proofs(binding,walk):
    # The producer copies this exact binding into the complete-model checkpoint.
    # The loader additionally compares it with freshly prepared physical inputs;
    # this walker only gives its two explicit certificates their evidence role.
    _need(isinstance(binding,dict) and set(binding)==_STREAM_BINDING_KEYS and
          binding['method_id']=='SAA' and binding['algorithm_id']==_STREAM_ALGORITHM and
          binding['formal_OOS'] is False,'Malformed streaming SAA input binding')
    for field in ('summary','formal_protocol_lock','bank_source_manifest'):_reference_shape(binding[field])
    task=binding['task_source']
    _need(isinstance(task,dict) and set(task)=={'path','sha256','bytes'},'Streaming task source shape differs')
    _reference_shape(dict(path=task['path'],sha256=task['sha256'],size_bytes=task['bytes']))
    # This legacy spelling is an actual dependency, not a proof leaf.
    walk(dict(path=task['path'],sha256=task['sha256'],size_bytes=task['bytes']))
    target=binding['target']
    _need(isinstance(target,dict) and set(target)=={'target_event','H','planning_count','calibration_count',
          'planning_index','calibration_index','moments','bootstrap'} and
          isinstance(target['target_event'],str) and bool(target['target_event']) and
          all(type(target[k]) is int and target[k]>0 for k in ('H','planning_count','calibration_count')) and
          target['planning_count']==160,'Streaming target binding differs')
    for field in ('planning_index','calibration_index','moments','bootstrap'):_reference_shape(target[field])
    sources=binding['planning_source']
    _need(isinstance(sources,dict) and bool(sources) and all(isinstance(k,str) and k.endswith('.py') and
          isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for k,v in sources.items()),
          'Streaming planning source map differs')
    provider=binding['template_provider']
    _need(isinstance(provider,dict) and set(provider)=={'template_index_reference','template_association_review_reference'},
          'Streaming template provider scope differs')
    index_ref=provider['template_index_reference'];_reference_shape(index_ref)
    review=_proof_reference(provider['template_association_review_reference'],
        {'status':'PASS_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW','passed':True,'domain':'formal','formal_OOS':False})
    _need(review.get('index_reference')==index_ref,'Streaming review/current index differs')
    _proof_reference(binding['audit'],_BANK_PROOF)
    # Current index, sources, scenario inputs and all coefficient/shard refs keep
    # ordinary verification. No proof leaf is inserted into the shared cache.
    for key,child in binding.items():
        if key=='template_provider':walk(index_ref)
        elif key!='audit':walk(child)


def _streaming_numeric_shapes(value):
    names={'data','indices','indptr','row_lower','row_upper','lower_bounds','upper_bounds','c','integrality'}
    refs=value['numeric_artifacts']
    _need(isinstance(refs,list) and len(refs)==9,'Streaming numeric inventory missing')
    for ref in refs:_reference_shape(ref)
    _need({Path(r['path']).name for r in refs}=={n+'.npy' for n in names} and
          len({str(Path(r['path']).parent) for r in refs})==1,'Streaming numeric inventory names/root differ')
    ids=value['scenario_ids'];hashes=value['scenario_sha256s'];rows=value['scenarios']
    _need(isinstance(ids,list) and len(ids)==160 and all(isinstance(i,str) and i for i in ids) and len(set(ids))==160 and
          isinstance(hashes,list) and len(hashes)==160 and all(isinstance(h,str) and len(h)==64 and
          all(c in '0123456789abcdef' for c in h) for h in hashes) and isinstance(rows,list) and len(rows)==160,
          'Streaming full support inventory differs')
    keys={'index','scenario_id','scenario_sha256','template_sha256','empirical_probability','original_recourse_columns',
          'column_slice','row_slice','nnz','mapping_file','mapping_reference','artifacts'}
    root=Path(refs[0]['path']).parent
    for i,row in enumerate(rows):
        _need(isinstance(row,dict) and set(row)==keys and type(row['index']) is int and row['index']==i and
              row['scenario_id']==ids[i] and row['scenario_sha256']==hashes[i], 'Streaming ordered shard differs')
        _reference_shape(row['mapping_reference'])
        folder=root/'shards'/f'{i:03d}'
        _need(row['mapping_file']==f'shards/{i:03d}/recourse_columns.npy' and
              Path(row['mapping_reference']['path'])==folder/'recourse_columns.npy', 'Streaming mapping path differs')
        artifacts=row['artifacts']
        _need(isinstance(artifacts,list) and len(artifacts)==4,'Streaming shard inventory missing')
        for ref in artifacts:_reference_shape(ref)
        _need({r['path'] for r in artifacts}=={str(folder/name) for name in
              ('arrays.npz','matrix.npz','recourse_columns.npy','zero_proof.json')} and
              row['mapping_reference'] in artifacts,'Streaming shard inventory identity differs')


def _typed_proof_fields(value,walk):
    """Only current consumers' explicit certificate fields have proof roles.

    Original worker prerequisites, TemplateProvider and failure-carryover gates
    retain their semantic/current-source checks. This does not replace them.
    """
    if value.get('kind')=='P2_RELEASE_GENERIC_PIPELINE_COMPONENT_REVIEW_V1':
        _p2_pipeline_development_proofs(value,walk)
        return True
    if value.get('algorithm_id')==_STREAM_ALGORITHM and value.get('domain')=='formal' and 'numeric_artifacts' in value:
        from .random_streams import canonical_hash
        _need(set(value)==_STREAMING_KEYS and type(value['schema_version']) is int and value['schema_version']==1 and
              value['risk_model']=='SAA' and value['status']=='ASSEMBLED_PENDING_RESOURCE_AND_INDEPENDENT_NUMERICAL_AUDIT' and
              type(value['solver_calls']) is int and value['solver_calls']==0 and
              all(value[k] is False for k in ('scientific_spec_pass','formal_OOS','solver_peak_rss_measured')) and
              value['shared_first_stage_unchanged'] is True,'Malformed formal streaming manifest proof consumer')
        _need(value['manifest_sha256']==canonical_hash({k:v for k,v in value.items() if k!='manifest_sha256'}),
              'Streaming manifest semantic digest differs')
        _streaming_numeric_shapes(value)
        _streaming_input_proofs(value['input_binding'],walk)
        for key,child in value.items():
            if key!='input_binding':walk(child)
        return True
    if value.get('kind')=='FORMAL_SAA_NUMERICAL_PHASES_V1':
        keys={'kind','model_manifest_reference','sources','helper_reference','input_binding','scenario_ids',
              'scenario_sha256s','template_references','case_sha256','task_sha256','cumulative_budget_seconds',
              'master_seconds','audit_reserve_seconds','seal_reserve_seconds','formal_admission_reference',
              'final_certificate_reserve_seconds','master_threads','recourse_workers','master_mip_gap'}
        _need(set(value)==keys,'Malformed formal SAA phase proof consumer')
        for key in ('model_manifest_reference','helper_reference','formal_admission_reference'):_reference_shape(value[key])
        manifest=_read(value['model_manifest_reference'])
        _need(manifest.get('domain')=='formal' and manifest.get('algorithm_id')==_STREAM_ALGORITHM and
              value['input_binding']==manifest.get('input_binding') and
              value['scenario_ids']==manifest.get('scenario_ids') and
              value['scenario_sha256s']==manifest.get('scenario_sha256s'),
              'Formal phase/current model input differs')
        admission=_read(value['formal_admission_reference'])
        _need(admission.get('kind')=='SAA_FORMAL_PHASE_ADMISSION_V1' and admission.get('domain')=='formal' and
              admission.get('model_manifest_reference')==value['model_manifest_reference'],
              'Formal phase admission/current model differs')
        for key in ('sources','template_references'):
            _need(isinstance(value[key],list) and bool(value[key]),'Formal phase current references missing')
            for reference in value[key]:_reference_shape(reference)
        _need(len(value['template_references'])==len(value['scenario_ids'])==160 and
              value['sources']==admission['planning_source_references'] and
              value['task_sha256']==value['input_binding']['task_sha256'] and
              value['case_sha256']==value['input_binding']['physical_case_sha256'],
              'Formal phase support/source/task differs')
        fixed={'cumulative_budget_seconds':7200,'master_seconds':600,'audit_reserve_seconds':600,
               'seal_reserve_seconds':10,'final_certificate_reserve_seconds':1790,'master_threads':4,
               'recourse_workers':1,'master_mip_gap':.001}
        _need(all(type(value[k]) in (int,float) and value[k]==v for k,v in fixed.items()),
              'Formal phase budget/solver contract differs')
        _streaming_input_proofs(value['input_binding'],walk)
        for key,child in value.items():
            if key!='input_binding':walk(child)
        return True
    if value.get('kind')=='SAA_COMPLETE_MODEL_CHECKPOINT_V1':
        _reference_shape(value.get('model_manifest_reference'))
        manifest=_read(value['model_manifest_reference'])
        # Development/preassembled completion retains ordinary traversal.
        if manifest.get('domain')!='formal':return False
        _need(set(value)==_COMPLETION_KEYS and value['model_input_mode']=='assemble' and
              all(value[k] is False for k in ('numerical_audit_complete','scientific_acceptance','formal_OOS')),
              'Malformed complete-model proof consumer')
        _reference_shape(value['request_reference'])
        _need(manifest.get('algorithm_id')==_STREAM_ALGORITHM and manifest.get('input_binding')==value['input_binding'],
              'Completion/current model input binding differs')
        _need(isinstance(value['planning_source_references'],list) and bool(value['planning_source_references']),
              'Completion source references missing')
        for reference in value['planning_source_references']:_reference_shape(reference)
        _streaming_input_proofs(value['input_binding'],walk)
        for key,child in value.items():
            if key!='input_binding':walk(child)
        return True
    if set(value)==_SAA_REQUEST_KEYS and value.get('method_id')=='SAA' and value.get('domain')=='formal':
        _need(type(value['schema_version']) is int and value['schema_version']==1 and value['domain']=='formal' and
              value['spec_id']=='S05' and value['model_id']=='IEEE33-v54-Q1' and value['model_input_mode']=='assemble' and
              isinstance(value['planning_source_references'],list) and bool(value['planning_source_references']) and
              isinstance(value['template_references'],list) and len(value['template_references'])==160,
              'Malformed formal SAA request proof consumer')
        for field,reference in value.items():
            if field.endswith('_reference'):_reference_shape(reference)
        for reference in value['planning_source_references']+value['template_references']:_reference_shape(reference)
        _proof_reference(value['audit_reference'],_BANK_PROOF)
        _proof_reference(value['bank_acceptance_reference'],{'status':'PASS_COMPONENT','formal_OOS':False})
        review=_proof_reference(value['template_association_review_reference'],
            {'status':'PASS_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW','passed':True,'domain':'formal','formal_OOS':False})
        _need(review.get('index_reference')==value['template_index_reference'],'Review/current template index differs')
        # The original worker verifies the exact carryover claim/request/release,
        # opening cost and cumulative budget before invoking checkpoint closure.
        _proof_reference(value['failure_carryover_reference'],{'kind':'SAA_PRE_ASSEMBLY_FAILED_PARENT_CARRYOVER_V1'})
        proof_fields={'audit_reference','bank_acceptance_reference','template_association_review_reference','failure_carryover_reference'}
        for k,v in value.items():
            if k not in proof_fields:walk(v)
        return True
    if value.get('kind')=='ORIGINAL_RECOURSE_TEMPLATE_INDEX_V1' and value.get('domain')=='formal':
        _need(set(value)==_TEMPLATE_INDEX_KEYS and value['status']=='PENDING_INDEPENDENT_TEMPLATE_ASSOCIATION_REVIEW' and
              value['domain']=='formal' and all(value[k] is False for k in ('formal_OOS','scientific_acceptance','old_q_reused')) and
              isinstance(value['rows'],list) and len(value['rows'])==160 and
              isinstance(value['scenario_ids'],list) and len(value['scenario_ids'])==160 and
              isinstance(value['scenario_sha256s'],list) and len(value['scenario_sha256s'])==160,
              'Malformed current full-support template index')
        _need(len(set(value['scenario_ids']))==160,'Duplicate template support ID')
        for field in ('original_request_reference','original_snapshot_reference'):_reference_shape(value[field])
        for i,row in enumerate(value['rows']):
            _need(isinstance(row,dict) and set(row)=={'scenario_index','scenario_id','scenario_sha256','template_sha256','binding_reference','original_template_reference'} and
                  type(row['scenario_index']) is int and row['scenario_index']==i and row['scenario_id']==value['scenario_ids'][i] and
                  row['scenario_sha256']==value['scenario_sha256s'][i],'Template ordered row identity differs')
            _reference_shape(row['binding_reference']);_reference_shape(row['original_template_reference'])
        proofs=value['proof_references']
        _need(isinstance(proofs,dict) and set(proofs)=={'bank_audit','numeric_audit','reader_acceptance'},'Template proof scope differs')
        _proof_reference(proofs['bank_audit'],_BANK_PROOF)
        _proof_reference(proofs['numeric_audit'],{'status':'PASS_INDEPENDENT_CANDIDATE_160_RECOURSE_AUDIT','passed':True,'formal_OOS':False,'scientific_acceptance':False})
        _proof_reference(proofs['reader_acceptance'],{'status':'PASS_COMPONENT'})
        for k,v in value.items():
            if k!='proof_references':walk(v)
        return True
    if set(value)==_RAW_REQUEST_KEYS and value.get('domain')=='formal':
        _need(type(value['schema_version']) is int and value['schema_version']==1 and value['formal_OOS'] is False and
              value['independent_scientific_audit_required'] is True and isinstance(value['input_binding'],dict),
              'Malformed original raw request proof consumer')
        binding=value['input_binding']
        _need('audit' in binding,'Raw request bank proof missing')
        _proof_reference(binding['audit'],_BANK_PROOF)
        for k,v in value.items():
            if k=='input_binding':
                for field,child in v.items():
                    if field!='audit':walk(child)
            else:walk(v)
        return True
    return False


def _p2_pipeline_development_proofs(value,walk):
    """Verify the publisher's explicit P2 mirror without rebasing live inputs.

    Development artifacts are hash-checked at their published evidence location;
    their original workstation addresses remain historical identities. Neither
    original nor mirrored proof paths are inserted into the dependency cache.
    """
    keys={'status','passed','kind','created_at_utc','planning_sources','numerical_sources','release_manifest_reference',
          'independent_review_reference','p2_acceptance_reference','original_p2_canonical_evidence_reference',
          'p2_test_log_reference','publication_script_reference','scope','original_numerical_sources_unchanged',
          'full_p2_package_source_identity_verified','reviewed_worker_methods','postprocessing_reserve_seconds',
          'cumulative_wall_seconds','single_master_seconds','final_certificate_reserve_seconds','shutdown_reserve_seconds',
          'formal_OOS','scientific_spec_pass','deployment_is_not_numerical_acceptance','metadata_seconds_before_write'}
    _need(set(value)==keys and value['status']=='PASS_INDEPENDENT_GENERIC_PLANNING_PIPELINE_REVIEW' and
          all(value[k] is True for k in ('passed','original_numerical_sources_unchanged',
              'full_p2_package_source_identity_verified','deployment_is_not_numerical_acceptance')) and
          value['formal_OOS'] is False and value['scientific_spec_pass'] is False and
          value['reviewed_worker_methods']==['EA','OR'],'Malformed published P2 pipeline proof consumer')
    fixed={'postprocessing_reserve_seconds':600,'cumulative_wall_seconds':7200,'single_master_seconds':600,
           'final_certificate_reserve_seconds':1790,'shutdown_reserve_seconds':10}
    _need(all(type(value[k]) in (int,float) and value[k]==v for k,v in fixed.items()),'P2 review budget differs')
    acceptance_ref=value['p2_acceptance_reference'];_reference_shape(acceptance_ref)
    acceptance=_read(acceptance_ref);mirror_root=Path(acceptance_ref['path']).parent
    akeys={'kind','status','accepted_at_utc','production_ready','formal_OOS','scientific_spec_pass','scope','tests',
           'previous_failed_test_log_reference','positive_summary_reference','plans','positive_reference_closure_reference',
           'disclosure_reference','source_and_test_snapshots','independent_checks','limitations','independent_metadata_seconds'}
    _need(isinstance(acceptance,dict) and set(acceptance)==akeys and
          acceptance['kind']=='P2_CANONICAL_IN_ATTEMPT_DEVELOPMENT_ACCEPTANCE_V1' and
          acceptance['status']=='PASS_DEVELOPMENT_ACCOUNTING_COMPONENT' and
          all(acceptance[k] is False for k in ('production_ready','formal_OOS','scientific_spec_pass')) and
          set(acceptance['tests'])=={'count','wall_seconds','log_reference'} and
          type(acceptance['tests']['count']) is int and acceptance['tests']['count']==71,
          'Malformed original P2 development acceptance')
    original_log=acceptance['tests']['log_reference'];_reference_shape(original_log)
    original_root=Path(original_log['path']).parent
    _need(Path(original_log['path']).name=='accepted_tests.log','P2 original log identity differs')
    def mirror(reference):
        _reference_shape(reference);path=Path(reference['path'])
        _need(path.is_relative_to(original_root),'Development proof escapes original operation')
        return _relative_source_member(mirror_root,dict(reference,path=path.relative_to(original_root).as_posix()))
    _need(mirror(original_log)==value['p2_test_log_reference'],'P2 mirror test log differs')
    for field in ('previous_failed_test_log_reference','positive_summary_reference',
                  'positive_reference_closure_reference','disclosure_reference'):mirror(acceptance[field])
    plans=acceptance['plans'];_need(isinstance(plans,list) and len(plans)==2 and
        [p.get('method_id') for p in plans]==['EA_DRO','SAA'],'P2 development plan inventory differs')
    plan_keys={'method_id','objective_usd','selected_count','numerical_status','completion_reference',
               'canonical_record_reference','solver_record_reference','budget_reference','cutoff_seconds',
               'charged_attempt_seconds','publication_actual_seconds'}
    original=None
    for row in plans:
        _need(set(row)==plan_keys and row['numerical_status']=='CONVERGED','P2 development plan metadata differs')
        for field in ('completion_reference','canonical_record_reference','solver_record_reference','budget_reference'):
            bound=mirror(row[field])
            if row['method_id']=='EA_DRO' and field=='canonical_record_reference':
                _need(bound==value['original_p2_canonical_evidence_reference'],'P2 canonical mirror differs')
                original=read_json(bound['path'])
    _need(isinstance(original,dict) and set(original)=={'plan_id','target_id','method_id','setting_id','plan_sha256','references'}
          and original['method_id']=='EA_DRO' and set(original['references'])==
          {'tasks','routes','initial_state','planning_support','planning_source','plan','solver_record','budget','budget_seal'},
          'P2 original canonical proof schema differs')
    for field,reference in original['references'].items():
        if field!='planning_source':mirror(reference)
    release_ref=value['release_manifest_reference'];_reference_shape(release_ref);release=Path(release_ref['path']).parent
    sources=value['planning_sources'];old_sources=original['references']['planning_source']
    _need(isinstance(sources,list) and isinstance(old_sources,list) and len(sources)==len(old_sources)==157,
          'P2 full package source inventory differs')
    expected={}
    for reference in old_sources:
        _reference_shape(reference);marker='/src/evcs_experiments/'
        _need(reference['path'].count(marker)==1,'P2 original source identity differs')
        name='src/evcs_experiments/'+reference['path'].split(marker)[1]
        _need(name not in expected,'Duplicate P2 original source')
        expected[name]=(reference['sha256'],reference['size_bytes'])
    actual={}
    for reference in sources:
        _reference_shape(reference);path=Path(reference['path'])
        _need(path.is_relative_to(release),'P2 planning source outside release')
        name=path.relative_to(release).as_posix();_need(name not in actual,'Duplicate P2 planning source')
        _relative_source_member(release,dict(reference,path=name));actual[name]=(reference['sha256'],reference['size_bytes'])
    _need(actual==expected and set(actual)=={p.relative_to(release).as_posix() for p in (release/'src/evcs_experiments').rglob('*.py')},
          'P2 original/current package source binding differs')
    snapshots=acceptance['source_and_test_snapshots'];_need(isinstance(snapshots,list) and len(snapshots)==16,'P2 snapshot inventory differs')
    seen_paths=set()
    for row in snapshots:
        _need(isinstance(row,dict) and set(row)=={'original_reference','package_relative_path'} and
              isinstance(row['package_relative_path'],str) and row['package_relative_path'].startswith('source_snapshots/'),
              'Malformed P2 source/test snapshot')
        original_ref=row['original_reference'];_reference_shape(original_ref)
        name=row['package_relative_path'].removeprefix('source_snapshots/')
        _need(name not in seen_paths and (name.startswith('src/evcs_experiments/') or name.startswith('tests/'))
              and original_ref['path'].endswith('/'+name),'P2 source/test snapshot identity differs')
        seen_paths.add(name);_relative_source_member(release,dict(original_ref,path=name))
    proof_fields={'p2_acceptance_reference','original_p2_canonical_evidence_reference','p2_test_log_reference'}
    for key,child in value.items():
        if key not in proof_fields:walk(child)


def verify_checkpoint_closure(ref,seen=None):
    """Verify immutable checkpoint DAG, including nested raw artifact refs."""
    seen=set() if seen is None else seen
    verify_reference(ref);key=(ref['path'],ref['sha256'])
    if key in seen:return
    seen.add(key)
    if not ref['path'].endswith('.json'):return
    def walk(value):
        if isinstance(value,dict):
            if set(value)=={'path','sha256','size_bytes'}:verify_checkpoint_closure(value,seen)
            else:
                for child in value.values():walk(child)
        elif isinstance(value,list):
            for child in value:walk(child)
    value=_read(ref)
    if isinstance(value,dict) and _typed_proof_fields(value,walk):return
    if Path(ref['path']).name=='RELEASE_MANIFEST.json':
        _release_inventory_closure(ref,value)
        for field in ('inherited_manifest','preflight_reference'):
            if field in value:
                _need(isinstance(value[field],dict) and set(value[field])=={'path','sha256','size_bytes'},'Malformed inherited release evidence')
                if field=='preflight_reference':_release_preflight_closure(value[field],ref,value,seen)
                else:verify_checkpoint_closure(value[field],seen)
    elif Path(ref['path']).name=='association_checks.json' and isinstance(value,dict) and value.get('status')==_ASSOCIATION_CHECK_STATUS:
        _historical_association_index(ref,value)
        for field,child in value.items():
            if field!='checked_references':walk(child)
    elif Path(ref['path']).name=='source_manifest.json' and isinstance(value,dict) and value.get('scope')==_AUDIT_REPAIR_SCOPE:
        _audit_repair_inventory_closure(ref,value,seen)
    else:walk(value)

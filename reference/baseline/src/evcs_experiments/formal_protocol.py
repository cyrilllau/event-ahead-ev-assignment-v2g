"""Freeze the complete matrix and future paired-plan scopes before formal OOS."""
import csv
import json
from pathlib import Path
from .config import get_paths
from .evaluation_lock import file_reference,create_protocol_lock
from .random_streams import event_order


def protocol_spec(project_root=None):
    paths=get_paths();root=paths.project_root if project_root is None else Path(project_root)
    config_path=root/'configs/formal_experiments.json';config=json.loads(config_path.read_text());targets=list(event_order(config['target_ids']))
    with (root/'docs/experiments/experiment_matrix.csv').open() as f:matrix=list(csv.DictReader(f))
    methods=sorted({m for row in matrix for m in row['planning_methods'].split('|')});batches=[]
    for row in matrix:
        group=row['group'];chosen=['michael_2018'] if row['target_scope']=='michael_2018' else targets
        # Methods/objectives themselves define settings for the first three groups.
        values=['baseline'] if group in ('mechanism','physical_chain','risk_model','certificate','anytime','AC_fidelity') else row['values'].split('|')
        expected=[dict(plan_id=f'{group}.{target}.{method}.{value}',target_id=target,method_id=method,setting_id=f"{row['parameter']}={value}") for target in chosen for method in row['planning_methods'].split('|') for value in values]
        regimes=['R1'] if row['oos_regimes']=='none' else row['oos_regimes'].split('|')
        # Non-OOS algorithm evidence has a reserved sample scope but must never
        # call OOS; its evaluator gateway is explicitly disabled below.
        samples={target:[f'{target}/formal/oos/{base}/v{v:03d}/{regime}' for regime in regimes for base in targets for v in range(1,51)] for target in chosen}
        batches.append(dict(batch_id=group,expected_plans=expected,sample_ids_by_target=samples,oos_enabled=row['oos_regimes']!='none',matrix_row=row))
    source=root/'src/evcs_experiments'
    refs=dict(source=[file_reference(source/name) for name in ('evaluation_lock.py','evaluation_access.py','evidence.py','formal_protocol.py')],
        evaluator=[file_reference(p) for p in sorted(source.glob('*.py'))],
        config=[file_reference(config_path),file_reference(root/'docs/experiments/PROTOCOL.md'),file_reference(root/'docs/experiments/runtime_policy.json'),file_reference(root/'configs/powerai.toml')],
        random_streams=[file_reference(source/'random_streams.py')],experiment_matrix=[file_reference(root/'docs/experiments/experiment_matrix.csv')],budgets=[file_reference(config_path)])
    data=root/'data/cases/ieee33-v54-q1'
    with (data/'MANIFEST_SHA256.csv').open() as f:physical=list(csv.DictReader(f))
    refs['physical_inputs']=[file_reference(data/'MANIFEST_SHA256.csv')]+[file_reference(data/row['relative_path']) for row in physical]
    manifest=root/'RELEASE_MANIFEST.json'
    if manifest.exists():refs['release_manifest']=[file_reference(manifest)]
    return dict(domain='formal',model_version=config['model_version'],target_ids=targets,method_ids=methods,references=refs,batches=batches,configuration=config,complete_matrix=matrix)


def freeze_formal_protocol(output,project_root=None):
    return create_protocol_lock(output,protocol_spec(project_root))

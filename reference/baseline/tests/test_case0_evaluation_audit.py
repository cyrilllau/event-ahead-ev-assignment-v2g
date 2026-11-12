"""Real physical saved LP and independently rebuilt coefficient/KKT negatives."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse import save_npz
from evcs_experiments import case0_evaluation as producer
from evcs_experiments import case0_evaluation_audit as audit
from evcs_experiments.recourse import assemble_recourse
from evcs_experiments.recourse_store import MATRICES,ARRAYS
from evcs_experiments.first_stage import FirstStagePlan
from evcs_experiments.validation.s02_toys import golden_single_bus


class Case0OOSAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve()
        self.f=golden_single_bus();f=self.f
        f.scenario.target_event='michael_2018';f.scenario.regime='R1';f.scenario.split='oos';f.scenario.metadata={'departure_h':-1}
        self.sources=[producer.file_reference(producer.__file__)]
        zero=FirstStagePlan(np.zeros_like(f.plan.x),np.zeros_like(f.plan.r),f.plan.task_sha256,f.plan.config_hash)
        plans={m:f.plan for m in producer.METHODS['R1']};plans['ZERO']=zero
        self.lock=dict(method_plan_sha256={m:producer.plan_identity(p) for m,p in plans.items()},
            sample_ids_by_regime={r:[f.scenario.scenario_id] for r in producer.METHODS})
        self.lockref=producer._write(self.root/'lock.json',self.lock)
        def provider(case,tasks,scene):
            folder=self.root/'template';folder.mkdir();t=assemble_recourse(case,tasks,scene)
            for name in MATRICES:save_npz(folder/(name+'.npz'),getattr(t,name))
            np.savez_compressed(folder/'arrays.npz',**{name:getattr(t,name) for name in ARRAYS})
            producer._write(folder/'identity.json',t.as_dict())
            ref=producer._write(folder/'record.json',dict(kind='recourse_template',template_sha256=t.template_sha256,
                scenario_index=0,scenario_id=scene.scenario_id,artifacts=[producer.file_reference(p) for p in sorted(folder.iterdir())]))
            return t,ref
        context=producer.Case0EvaluationContext(lock_reference=self.lockref,lock_validator=lambda r:self.lock,plans=plans,
            physical_factory=lambda m,i,r:(f.case,f.tasks,f.scenario),template_provider=provider,
            source_references=self.sources,samples_per_regime=1)
        self.item=producer.case0_items(1)[0]
        self.ref=producer.evaluate_case0_item(context,self.item,output_directory=self.root/'item',deadline=time.monotonic()+15)['record_reference']
        self.cache=audit.OriginalTemplateAuditCache()

    def run_audit(self):
        f=self.f
        with patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('Audit must not solve')):
            return audit.audit_case0_evaluation_item(record_reference=self.ref,item=self.item,lock_reference=self.lockref,
                lock=self.lock,plan=f.plan,case=f.case,tasks=f.tasks,scene=f.scenario,
                producer_source_references=self.sources,cache=self.cache,deadline=time.monotonic()+10)

    def reseal(self,name,edit):
        row=audit._read(self.ref);ref=next(r for r in row['artifacts'] if Path(r['path']).name==name)
        edit(Path(ref['path']));row['artifacts']=[producer.file_reference(r['path']) for r in row['artifacts']]
        Path(self.ref['path']).write_text(json.dumps(row));self.ref=producer.file_reference(self.ref['path'])

    def test_real_original_coefficient_kkt_physics_metrics_and_cache(self):
        report=self.run_audit();self.assertTrue(report['passed']);self.assertGreater(report['costs']['total_DSO_usd'],0.)
        self.assertEqual(self.cache.assemblies,1)
        self.run_audit();self.assertEqual(self.cache.assemblies,1);self.assertEqual(self.cache.hits,1)

    def test_resealed_fixed_rhs_tamper_is_rejected(self):
        def edit(path):
            with np.load(path) as z:arrays={k:z[k] for k in z.files}
            arrays['b_ub']=arrays['b_ub']+1.;np.savez_compressed(path,**arrays)
        self.reseal('dual_rhs.npz',edit)
        with self.assertRaisesRegex(ValueError,'RHS'):self.run_audit()

    def test_resealed_metrics_are_not_trusted(self):
        def edit(path):
            row=json.loads(path.read_text());row['costs']['total_DSO_usd']+=1.;path.write_text(json.dumps(row))
        self.reseal('metrics.json',edit)
        with self.assertRaisesRegex(ValueError,'metrics'):self.run_audit()

    def test_coherently_resealed_solver_and_raw_duals_fail_numerics(self):
        def solver(path):
            row=json.loads(path.read_text());row['certificate']['dual_lower_marginals'][0]+=1.;path.write_text(json.dumps(row))
        def raw(path):
            with np.load(path) as z:arrays={k:z[k] for k in z.files}
            arrays['lower'][0]+=1.;np.savez_compressed(path,**arrays)
        self.reseal('solver.json',solver);self.reseal('dual_rhs.npz',raw)
        with self.assertRaisesRegex(ValueError,'KKT'):self.run_audit()

    def test_wrong_departure_sample_and_missing_item_inventory_rejected(self):
        self.f.scenario.metadata['departure_h']=0
        with self.assertRaisesRegex(ValueError,'departure'):self.run_audit()
        self.f.scenario.metadata['departure_h']=-1;self.f.scenario.scenario_id='wrong'
        with self.assertRaisesRegex(ValueError,'sample'):self.run_audit()

    def test_template_record_scenario_label_tamper_is_rejected(self):
        folder=self.root/'template';record=folder/'record.json';row=json.loads(record.read_text())
        row['scenario_id']='wrong';record.write_text(json.dumps(row));item=audit._read(self.ref)
        item['template_reference']=producer.file_reference(record);Path(self.ref['path']).write_text(json.dumps(item))
        self.ref=producer.file_reference(self.ref['path'])
        with self.assertRaisesRegex(ValueError,'scenario label'):self.run_audit()

    def test_template_resealed_wrong_coefficient_is_rejected(self):
        folder=self.root/'template';p=folder/'A_ub.npz'
        from scipy.sparse import load_npz
        a=load_npz(p);a.data[0]+=1.;save_npz(p,a)
        row=json.loads((folder/'record.json').read_text());row['artifacts']=[producer.file_reference(r['path']) for r in row['artifacts']]
        (folder/'record.json').write_text(json.dumps(row));item=audit._read(self.ref);item['template_reference']=producer.file_reference(folder/'record.json')
        Path(self.ref['path']).write_text(json.dumps(item));self.ref=producer.file_reference(self.ref['path'])
        with self.assertRaisesRegex(ValueError,'semantic SHA'):self.run_audit()

class Case0OOSAxisTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name).resolve()
        self.lock=dict(item_count=768,method_plan_sha256={m:'fixed' for m in producer.METHODS['R1']},
            sample_ids_by_regime={r:[f'{r}:{i}' for i in range(64)] for r in ('R1','R2','R3')})
        self.progress=dict(completed={})
        for item in producer.case0_items():
            d=self.root/'evaluation/items'/item['item_id'];d.mkdir(parents=True)
            self.progress['completed'][item['item_id']]={'path':str(d/'record.json')}

    def test_exact768_six_mechanism_and_three_risk_methods(self):
        items=audit.validate_case0_oos_axis(self.root,self.lock,self.progress)
        self.assertEqual(len(items),768)
        self.assertEqual(sum(i['regime']=='R1' for i in items),384)
        self.assertEqual({i['method_id'] for i in items if i['regime']=='R2'},{'EA_DRO','SAA','ARO'})

    def test_missing_item_and_method_substitution_rejected(self):
        self.progress['completed'].pop('R1.000.ZERO')
        with self.assertRaisesRegex(ValueError,'coverage'):audit.validate_case0_oos_axis(self.root,self.lock,self.progress)

    def test_duplicate_sample_or_foreign_item_path_rejected(self):
        self.lock['sample_ids_by_regime']['R1'][1]=self.lock['sample_ids_by_regime']['R1'][0]
        with self.assertRaisesRegex(ValueError,'sample'):audit.validate_case0_oos_axis(self.root,self.lock,self.progress)
        self.lock['sample_ids_by_regime']['R1'][1]='R1:1'
        self.progress['completed']['R1.000.EA_DRO']['path']=str(self.root/'other/record.json')
        with self.assertRaisesRegex(ValueError,'Foreign'):audit.validate_case0_oos_axis(self.root,self.lock,self.progress)

class Case0OOSSettlementTests(unittest.TestCase):
    def test_exact_settled_parent_required_before_numerical_work(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d).resolve();original=producer._write(root/'original.json',dict(evidence=True))
            row=dict(kind='CASE0_EVALUATION_PARENT_SETTLEMENT_V1',cumulative_compute_seconds=3392.,
                terminal_result_reference=original,parent_manifest_reference=original,reservation_reference=original)
            ref=producer._write(root/'settlement.json',row)
            state=dict(last_evaluation_settlement=ref,compute_seconds=3400.)
            self.assertEqual(audit.verify_evaluation_parent_settlement(ref,state),ref)
            for modified in (dict(state,compute_seconds=3391.),dict(state,last_evaluation_settlement=original)):
                with self.assertRaisesRegex(ValueError,'settlement'):audit.verify_evaluation_parent_settlement(ref,modified)
            (root/'original.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'changed'):audit.verify_evaluation_parent_settlement(ref,state)

class PolicySourceCorrespondenceTests(unittest.TestCase):
    def test_only_typed_paths_accept_equal_manifest_bound_policy_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            refs=[]
            for name in ('producer','auditor'):
                root=Path(d).resolve()/name;policy=root/'docs/experiments/runtime_policy.json';policy.parent.mkdir(parents=True)
                policy.write_text('{"normalized":1e-8}')
                member=producer.file_reference(policy)
                refs.append(producer._write(root/'RELEASE_MANIFEST.json',dict(files=[dict(member,path='docs/experiments/runtime_policy.json')])))
            mapping=audit.policy_source_correspondence(*refs)
            a=dict(physical_check=dict(tolerances=dict(normalized=1e-8,source=mapping['producer_policy_reference']['path']),
                first_stage=dict(tolerances=dict(normalized=1e-8,source=mapping['producer_policy_reference']['path']))))
            b=json.loads(json.dumps(a))
            b['physical_check']['tolerances']['source']=mapping['auditor_policy_reference']['path']
            b['physical_check']['first_stage']['tolerances']['source']=mapping['auditor_policy_reference']['path']
            audit._same(a,b,'Original reported metrics',mapping)
            b['physical_check']['tolerances']['normalized']=1e-7
            with self.assertRaisesRegex(ValueError,'normalized'):audit._same(a,b,'Original reported metrics',mapping)
            with self.assertRaisesRegex(ValueError,'differs'):audit._same({'source':'old'},{'source':'new'},'Other metadata',mapping)
            Path(mapping['auditor_policy_reference']['path']).write_text('{"normalized":1e-7}')
            with self.assertRaisesRegex(ValueError,'policy differs'):audit.policy_source_correspondence(*refs)

if __name__=='__main__':unittest.main()




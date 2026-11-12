import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from evcs_experiments.core_method_registry import prepare_core_request, build_core_worklist, source_references
from evcs_experiments.evaluation_lock import file_reference
from evcs_experiments.random_streams import canonical_hash


class CoreMethodRegistryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.body = dict(domain='formal', target_ids=[f't{i}' for i in range(8)], batches=[])
        for batch, setting, methods in [('mechanism', 'lead_h=baseline', ['EA_DRO','OR','DSP_from_EA']),
                ('physical_chain','ablation=baseline',['EA_DRO','M6_static']),
                ('risk_model','risk_objective=baseline',['EA_DRO','SAA','ARO'])]:
            self.body['batches'].append(dict(batch_id=batch, expected_plans=[dict(
                plan_id=f'{batch}.{target}.{method}.baseline', target_id=target, method_id=method, setting_id=setting)
                for target in self.body['target_ids'] for method in methods]))
        self.ref = self.write('protocol',dict(kind='protocol_lock',payload=self.body,payload_sha256=canonical_hash(self.body)))
        self.sources = source_references()

    def write(self, name, value):
        p=self.root/(name+'.json');p.write_text(json.dumps(value));return file_reference(p)

    def test_all8_distinct_methods_and_no_jobs(self):
        with patch('evcs_experiments.core_method_registry.source_references', wraps=source_references) as sources:
            result=build_core_worklist(self.ref,self.sources)
        sources.assert_called_once_with()
        self.assertEqual(len(result['items']),32)
        self.assertEqual(len({i['identity']['plan_id'] for i in result['items']}),32)
        first=result['items'][:4]
        self.assertEqual([i['worker_method_id'] for i in first],['EA','OR','SAA','ARO'])
        self.assertIn('formal_onset.load_formal_onset_problem',first[1]['factory'])
        self.assertIn('saa_planning_worker',first[2]['worker_module'])
        self.assertEqual(len(first[0]['protocol_aliases']),3)
        self.assertFalse(result['queue_ready']);self.assertFalse(result['formal_OOS'])

    def test_reuse_accepted_ea_saa_never_prepare_again(self):
        items=build_core_worklist(self.ref,self.sources)['items']; accepted={}
        for index in (0,2):
            identity=items[index]['identity']; ref=self.write('record'+str(index),identity)
            card=self.write('card'+str(index),dict(status='PASS_COMPONENT',component_id='INDIVIDUAL_FORMAL_NUMERICAL_RECORD',plan_record_reference=ref))
            accepted[identity['plan_id']]=dict(plan_record_reference=ref,acceptance_reference=card)
        out=build_core_worklist(self.ref,self.sources,accepted)['items']
        self.assertEqual(sum(i['action']=='PREPARE' for i in out),30)
        self.assertEqual(out[0]['accepted_references'],accepted[out[0]['identity']['plan_id']])
        broken=copy.deepcopy(accepted);broken[out[0]['identity']['plan_id']]['acceptance_reference']=self.write('bad',dict(status='PREPARED'))
        with self.assertRaisesRegex(ValueError,'Unaccepted'):build_core_worklist(self.ref,self.sources,broken)

    def test_identity_method_and_source_rejections(self):
        base=self.body['batches'][0]['expected_plans'][0]
        for changes in [dict(method_id='EA'),dict(target_id='foreign'),dict(setting_id='lead_h=0'),
                        dict(method_id='M6_static'),dict(method_id='no_line_dependence'),
                        dict(method_id='no_line_road_coupling'),dict(method_id='independent_recombination')]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                prepare_core_request(self.ref,dict(base,**changes),self.sources)
        with self.assertRaisesRegex(ValueError,'source closure'):
            prepare_core_request(self.ref,base,self.sources[:-1])
        with self.assertRaisesRegex(ValueError,'Unknown accepted'):
            build_core_worklist(self.ref,self.sources,{'foreign':{}})
        Path(self.ref['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'changed'):prepare_core_request(self.ref,base,self.sources)

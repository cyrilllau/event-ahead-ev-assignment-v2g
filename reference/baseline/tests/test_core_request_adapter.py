"""Real worker/queue request contracts without launching formal computation."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path
import sys
import tempfile
import unittest

from evcs_experiments import core_request_adapter as adapter
from evcs_experiments.campaign_batch import checked_request, file_reference
from evcs_experiments.config import get_paths
from evcs_experiments.core_method_registry import source_references
from evcs_experiments.random_streams import canonical_hash


class CoreRequestAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.budget = dict(cumulative_wall_seconds=7200, single_master_seconds=600,
                           final_certificate_reserve_seconds=1790, shutdown_reserve_seconds=10,
                           master_threads=4, recourse_workers=1, blas_threads=1,
                           normal_gap_relative=.005, rmp_mip_gap=.001)
        self.config = dict(planning_budget=self.budget)
        self.config_ref = self.write('config', self.config)
        self.body = dict(domain='formal', configuration=self.config,
                         target_ids=[f't{i}' for i in range(8)], batches=[])
        for batch, setting, methods in [('mechanism', 'lead_h=baseline', ['EA_DRO', 'OR']),
                                        ('risk_model', 'risk_objective=baseline', ['SAA', 'ARO'])]:
            self.body['batches'].append(dict(batch_id=batch, expected_plans=[dict(
                plan_id=f'{batch}.{target}.{method}.baseline', target_id=target,
                method_id=method, setting_id=setting)
                for target in self.body['target_ids'] for method in methods]))
        self.protocol_ref = self.write('protocol', dict(kind='protocol_lock', payload=self.body,
                                                        payload_sha256=canonical_hash(self.body)))
        self.sources = source_references()
        self.context_args = dict(protocol_reference=self.protocol_ref, config_reference=self.config_ref,
                                 planning_source_references=self.sources)
        self.context = adapter.prepare_core_context(**self.context_args)
        self.bank = self.write('bank', {'fixture': 'contract-only'})
        self.audit = self.write('audit', {'fixture': 'contract-only'})
        self.card = self.write('card', dict(status='PASS_COMPONENT', summary_reference=self.bank,
                                           audit_reference=self.audit, protocol_reference=self.protocol_ref))
        project = get_paths().project_root
        numerical_sources = dict(helper=file_reference(project/'scripts/audits/audit_s05_pending_recovery.py'),
                                 numerical_kernel=file_reference(project/'scripts/audits/audit_s05_candidate_recourse.py'))
        review = self.write('pipeline-review', dict(status='PASS_INDEPENDENT_GENERIC_PLANNING_PIPELINE_REVIEW',
                                                   passed=True, planning_sources=self.sources,
                                                   numerical_sources=numerical_sources))
        self.numerical = dict(schema_version=1, postprocessing_reserve_seconds=600,
                              sources=numerical_sources, implementation_review_reference=review)
        self.saa = dict(model_input_mode='assemble', max_coefficient_bytes=50_000_000,
                        helper_reference=numerical_sources['helper'], template_references=[self.bank])
        self.inputs = {key: self.bank for key in ('tasks', 'routes', 'initial_state', 'planning_support')}
        self.onset_bank = self.write('onset-summary', dict(
            formal_protocol_lock=self.protocol_ref, source_manifest=self.bank,
            formal_plans_solved=False, atom_count=1920, formal_OOS=False,
            status='COMPLETE_FORMAL_ONSET_INPUTS_PENDING_INDEPENDENT_AUDIT'))
        self.onset_audit = self.write('onset-audit', dict(
            bank_kind='onset_v1', lead_h=0., audit_source_sha256=numerical_sources['helper']['sha256'],
            status='PASS_INDEPENDENT_FORMAL_BANK_AUDIT', source_summary_sha256=self.onset_bank['sha256'],
            source_output=str(self.root), formal_OOS=False, formal_plans_solved=False,
            scenario_count=1920, full_S05_pass=False, formal_protocol_lock=self.protocol_ref,
            source_manifest_sha256=self.bank['sha256']))
        self.onset_card = self.write('onset-card', dict(
            status='PASS_FORMAL_ONSET_INPUT_BANK_COMPONENT', formal_OOS=False,
            formal_plans_solved=False, whole_S05_PASS=False,
            bank_summary=self.onset_bank, numerical_audit=self.onset_audit,
            auditor_source=numerical_sources['helper'], development_review=review,
            production_review=review, operational_completion=review))

    def write(self, name, value):
        path = self.root/(name+'.json'); path.write_text(json.dumps(value))
        return file_reference(path)

    def identity(self, method):
        return next(p for b in self.body['batches'] for p in b['expected_plans']
                    if p['target_id'] == 't0' and p['method_id'] == method)

    def worker(self, method, **changes):
        batch = 'mechanism' if method in ('EA_DRO', 'OR') else 'risk_model'
        options = dict(identity=self.identity(method), canonical_batch_id=batch,
                       plan_input_references=self.inputs, bank_acceptance_reference=self.card,
                       summary_reference=self.bank, audit_reference=self.audit,
                       output_root=self.root/('output-'+method))
        if method == 'SAA': options['saa_options'] = self.saa
        else: options['numerical_audit'] = self.numerical
        if method == 'OR':
            options.update(onset_acceptance_reference=self.onset_card,
                           bank_acceptance_reference=self.onset_card,
                           summary_reference=self.onset_bank, audit_reference=self.onset_audit)
        options.update(changes)
        return adapter.build_core_worker_request(self.context, **options)

    def queue(self, items, **changes):
        args = dict(batch_id='core-first', queue_root=self.root/'queue', job_wall_seconds=7200,
                    budget_reference=self.write('budget', self.budget), items=items,
                    executable=sys.executable, cwd=self.root)
        args.update(changes)
        return adapter.build_core_queue(self.context, **args)

    def test_all_four_actual_worker_mappings_and_checked_queue_schema(self):
        items = []
        for method, native in [('EA_DRO', 'EA'), ('OR', 'OR'), ('SAA', 'SAA'), ('ARO', 'ARO')]:
            request = self.worker(method)
            self.assertEqual(request['method_id'], native)
            self.assertEqual(request['canonical_identity'], self.identity(method))
            self.assertEqual(request['plan_input_references'], self.inputs)
            self.assertNotIn('queue_ready', request)
            items.append(dict(item_id=method, identity=self.identity(method),
                              request_reference=self.write(method, request)))
        result = self.queue(items)
        self.assertEqual([item['argv'][2] for item in result['items']],
                         ['evcs_experiments.formal_planning_worker',
                          'evcs_experiments.formal_planning_worker',
                          'evcs_experiments.saa_planning_worker',
                          'evcs_experiments.formal_planning_worker'])
        checked, _ = checked_request(self.write('queue', result)['path'])
        self.assertEqual(checked, result)

    def test_no_unsupported_or_onset_fallback_or_option_overrides(self):
        for method in ('M6_static', 'EA', 'no_line_dependence'):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'supported core'):
                self.worker('EA_DRO', identity={**self.identity('EA_DRO'), 'method_id': method})
        with self.assertRaisesRegex(ValueError, 'onset acceptance'):
            self.worker('OR', onset_acceptance_reference=None)
        with self.assertRaisesRegex(ValueError, 'onset input component'):
            self.worker('OR', onset_acceptance_reference=self.card, bank_acceptance_reference=self.card,
                        summary_reference=self.bank, audit_reference=self.audit)
        with self.assertRaisesRegex(ValueError, 'Only OR'):
            self.worker('EA_DRO', onset_acceptance_reference=self.card)
        for changed in ({'domain': 'development'}, {'budget': {}}, {'canonical_identity': {}},
                        {'model_input_mode': 'preassembled'}, {'model_manifest_reference': self.bank}):
            with self.subTest(changed=changed), self.assertRaisesRegex(ValueError, 'fresh assembly'):
                self.worker('SAA', saa_options={**self.saa, **changed})
        with self.assertRaisesRegex(ValueError, 'supplied together'):
            self.worker('EA_DRO', template_index_reference=self.bank)

    def test_source_budget_and_domain_cannot_be_swapped(self):
        with self.assertRaisesRegex(ValueError, 'source closure'):
            adapter.prepare_core_context(**{**self.context_args, 'planning_source_references': self.sources[:-1]})
        with self.assertRaisesRegex(ValueError, 'domain/envelope'):
            adapter._development_context(**self.context_args)
        with self.assertRaises(FrozenInstanceError):
            self.context.domain = 'development'
        # Caller mutations do not alter the accepted context or other requests.
        result = self.worker('EA_DRO'); result['budget']['single_master_seconds'] = 700
        self.assertEqual(self.context.budget['single_master_seconds'], 600)
        self.budget['single_master_seconds'] = 800
        self.assertEqual(self.context.budget['single_master_seconds'], 600)
        self.budget['single_master_seconds'] = 600
        result = self.worker('EA_DRO'); result['budget']['single_master_seconds'] = 700
        item = dict(item_id='EA', identity=self.identity('EA_DRO'), request_reference=self.write('bad-request', result))
        with self.assertRaisesRegex(ValueError, 'binding differs: budget'):
            self.queue([item])
        result = self.worker('EA_DRO')
        item['request_reference'] = self.write('good-request', result)
        with self.assertRaisesRegex(ValueError, 'cannot replace a formal'):
            self.queue([item], _development_argv=[sys.executable, 'fixture.py'])

    def test_l_shaped_development_request_cannot_enter_formal_context(self):
        from evcs_experiments.saa_l_shaped import ALGORITHM_ID
        options = dict(model_input_mode='l_shaped_development', algorithm_id=ALGORITHM_ID,
                       helper_reference=self.saa['helper_reference'],
                       template_references=self.saa['template_references'])
        with self.assertRaisesRegex(ValueError, 'development only'):
            self.worker('SAA', saa_options=options)

    def test_referenced_bytes_duplicate_plans_and_bank_substitution_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Accepted bank'):
            self.worker('EA_DRO', audit_reference=self.bank)
        req = self.worker('EA_DRO')
        item = dict(item_id='EA', identity=self.identity('EA_DRO'), request_reference=self.write('EA', req))
        with self.assertRaisesRegex(ValueError, 'duplicate item/plan'):
            self.queue([item, {**item, 'item_id': 'EA-again'}])
        Path(self.audit['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Frozen reference changed'):
            self.worker('EA_DRO')


if __name__ == '__main__':
    unittest.main()

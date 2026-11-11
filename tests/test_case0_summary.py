"""Synthetic exact paired arithmetic; never consume unreviewed production OOS."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from evcs_experiments.config import get_paths

path=get_paths().project_root/'scripts/reports/case0_summary.py'
spec=importlib.util.spec_from_file_location('case0_summary_tested_script',path)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def rows():
    out=[]
    for base in range(8):
        for variant in range(8):
            value=8*base+variant
            for method in m.METHODS['R1']:
                x=value+(2 if method=='SAA' else 0)
                out.append(dict(item=dict(method_id=method,regime='R1'),
                    scenario_id=f'michael_2018/case0/development/oos/base{base}/v{10001+variant:05d}/R1',
                    costs={'total_DSO_usd':x},metrics={'first_reserve_h':1.},energy_kwh={'certified':x},na_reasons={}))
    return out[::-1]


class Case0SummaryTests(unittest.TestCase):
    def test_unsorted_sample_ids_pair_exact_mean_difference_and_fractional_tail(self):
        samples=m.paired_samples(rows(),'R1','costs.total_DSO_usd')
        with patch.object(m,'paired_bootstrap',wraps=m.paired_bootstrap) as call:
            result,vs_saa=m.bootstrap_references(samples)
            self.assertEqual(call.call_count,1)
        self.assertEqual(vs_saa['per_target'][0]['methods']['EA_DRO']['difference_from_reference']['ci95'],[-2.,-2.])
        self.assertEqual(result['bootstrap_indices_sha256'],vs_saa['bootstrap_indices_sha256'])
        self.assertEqual(result['layout_sha256'],vs_saa['layout_sha256'])
        saa=result['per_target'][0]['methods']['SAA']
        self.assertEqual(saa['estimate'],33.5)
        self.assertEqual(saa['difference_from_reference']['estimate'],2.)
        self.assertEqual(saa['difference_from_reference']['ci95'],[2.,2.])
        tail,tail_vs_saa=m.bootstrap_references(samples,'cvar95')
        self.assertEqual(tail_vs_saa['per_target'][0]['methods']['EA_DRO']['difference_from_reference']['ci95'],[-2.,-2.])
        self.assertAlmostEqual(tail['per_target'][0]['methods']['EA_DRO']['estimate'],61.875)
        self.assertEqual(tail['tail_definition']['boundary_observation_weight'],.2)
        self.assertEqual(tail['tail_definition']['tail_mass_in_samples'],3.2)

    def test_undefined_values_propagate_with_original_reason(self):
        data=rows();row=next(r for r in data if r['item']['method_id']=='SAA')
        row['metrics']['first_reserve_h']=None;row['na_reasons']={'first_reserve_h':'no certified reserve above reporting threshold'}
        samples=m.paired_samples(data,'R1','metrics.first_reserve_h')
        report,vs_saa=m.bootstrap_references(samples)
        self.assertIsNone(vs_saa['per_target'][0]['methods']['EA_DRO']['difference_from_reference']['estimate'])
        value=report['per_target'][0]['methods']['SAA']
        self.assertIsNone(value['estimate']);self.assertEqual(value['na_count'],1);self.assertEqual(value['na_rate'],1/64)
        row['na_reasons']={}
        with self.assertRaisesRegex(ValueError,'NA reason'):m.paired_samples(data,'R1','metrics.first_reserve_h')

    def test_missing_duplicate_or_foreign_variant_cannot_be_reshaped_into_pairs(self):
        data=rows();data.pop()
        with self.assertRaisesRegex(ValueError,'Missing paired'):m.paired_samples(data,'R1','costs.total_DSO_usd')
        data=rows();data.append(data[0])
        with self.assertRaisesRegex(ValueError,'Duplicate paired'):m.paired_samples(data,'R1','costs.total_DSO_usd')
        with self.assertRaisesRegex(ValueError,'variant'):m.sample_axis('michael_2018/case0/development/oos/base0/v20001/R1','R1')

    def test_pending_or_wrong_hash_audit_cannot_produce_tables(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'pending.json';p.write_text(json.dumps(dict(kind='COMPLETE_CASE0_ITEM_PENDING_INDEPENDENT_AUDIT',passed=True)))
            ref=m.file_reference(p)
            with self.assertRaisesRegex(ValueError,'Only independently audited'):m.load_audit(p,ref['sha256'])
            with self.assertRaisesRegex(ValueError,'SHA'):m.load_audit(p,'0'*64)

class ExpandedSummaryTests(unittest.TestCase):
    def config_fixture(self,root):
        from types import SimpleNamespace
        from evcs_experiments import case0_campaign as c
        campaign=root/'ieee33-v54-q1-case0-expanded';campaign.mkdir()
        (campaign/'MANIFEST_SHA256.csv').write_text('fixture')
        with patch.object(c,'get_paths',return_value=SimpleNamespace(campaign_root=campaign,data_root=campaign)),patch.object(c,'sources',return_value={'fixture':'source'}):
            c.prepare('gurobi','expanded48-v1')
        def write(name,value):
            path=root/name;path.write_text(json.dumps(value));return m.file_reference(path)
        config_ref=m.file_reference(campaign/'config.lock.json')
        acceptance_ref=write('acceptance.json',dict(config_reference=config_ref))
        lock=dict(kind='CASE0_PRE_OOS_LOCK_V1',formal_OOS=False,profile='expanded48-v1',target_event='zeta_2020',
                  config_reference=config_ref,acceptance_reference=acceptance_ref)
        lock_ref=write('lock.json',lock)
        return dict(lock_reference=lock_ref,planning_acceptance_reference=acceptance_ref),lock,write

    def test_expanded_exact_locked_target_stream_seed_and_tamper(self):
        with tempfile.TemporaryDirectory() as temp:
            audit,lock,write=self.config_fixture(Path(temp))
            scope=m.study_scope(audit)
            self.assertEqual((scope['target_event'],scope['variant_start'],scope['seed']),('zeta_2020',30000,20260912))
            self.assertEqual(m.study_scope(audit,20260912),scope)
            with self.assertRaisesRegex(ValueError,'statistics seed'):m.study_scope(audit,20260911)
            bad=dict(lock,target_event='michael_2018');audit['lock_reference']=write('bad_lock.json',bad)
            with self.assertRaises(ValueError):m.study_scope(audit)
            bad=dict(lock,profile='legacy');audit['lock_reference']=write('bad_profile.json',bad)
            with self.assertRaises(ValueError):m.study_scope(audit)
            audit['lock_reference']=write('changed_config_lock.json',dict(lock,config_reference=write('unbound_config.json',{})))
            with self.assertRaisesRegex(ValueError,'identity'):m.study_scope(audit)

    def test_zeta_explicit_axes_preserve_pairing_and_tail_arithmetic(self):
        data=rows()
        for row in data:
            row['scenario_id']=row['scenario_id'].replace('michael_2018','zeta_2020').replace('/v100','/v300')
        scope=dict(target_event='zeta_2020',variant_start=30000)
        samples=m.paired_samples(data,'R1','costs.total_DSO_usd',scope)
        self.assertEqual(samples.target_ids,('zeta_2020',))
        reports=m.bootstrap_references(samples,'cvar95',seed=20260912)
        self.assertAlmostEqual(reports[0]['per_target'][0]['methods']['EA_DRO']['estimate'],61.875)
        self.assertEqual(reports[0]['per_target'][0]['methods']['SAA']['difference_from_reference']['ci95'],[2.,2.])
        self.assertEqual(reports[0]['bootstrap_indices_sha256'],reports[1]['bootstrap_indices_sha256'])
        for sid in ['michael_2018/case0/development/oos/base0/v30001/R1','zeta_2020/case0/development/oos/base0/v10001/R1','zeta_2020/case0/development/oos/base0/v40001/R1']:
            with self.assertRaises(ValueError):m.sample_axis(sid,'R1',scope)
        data[0]['metrics']['first_reserve_h']=None;data[0]['na_reasons']={'first_reserve_h':'unavailable'}
        samples=m.paired_samples(data,'R1','metrics.first_reserve_h',scope)
        report=m.bootstrap_references(samples,seed=20260912)[0]
        self.assertIsNone(report['per_target'][0]['methods'][data[0]['item']['method_id']]['estimate'])

if __name__=='__main__':unittest.main()

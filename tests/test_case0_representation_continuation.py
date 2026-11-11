"""One settled failed-admission continuation, using the real preserved/mirror topology."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tests import test_case0_representation as fixtures
from evcs_experiments import case0_memory_repair as m,case0_campaign as c


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.RepresentationTests();self.f.setUp();self.addCleanup(self.f.doCleanups);f=self.f
        # Original card deliberately keeps an unavailable producer-local path.
        terminal=f.settlement['terminal_review_reference'];value=c.read(terminal['path'])
        value['observation_reference']['path']='/unavailable/producer/observation.json'
        f.settlement['terminal_review_reference']=f.write(terminal['path'],value);f.resettle()
        original_path=f.base/'original_request.json';original_path.write_bytes(Path(f.request['path']).read_bytes())
        self.original_request=f.ref(original_path)
        self.revision=f.root/'revisions'/m.REPRESENTATION
        self.reservation=f.write(self.revision/'reservation.json',dict(request_reference=self.original_request,
            admission_started_monotonic=1.,incomplete_admission_requires_manual_accounting=True,automatic_retry_allowed=False))
        before=f.base/'failed_state_before.json';before.write_bytes((f.root/'state.json').read_bytes())
        fail=f.write(f.base/'failed_prepare.json',dict(actual_seconds=.13020952977240086,requires_accounting_before_any_resume=True))
        pre=f.write(f.base/'preflight.json',dict(passed=True,development_verification_seconds=16.21209406852722))
        self.closed=dict(kind='CASE0_FAILED_REPRESENTATION_ADMISSION_SETTLEMENT_V1',preflight_reference=pre,
            failed_prepare_reference=fail,reservation_reference=self.reservation,state_before_reference=f.ref(before),
            source_release_reference=f.refs['source_manifest'],opening_compute_seconds=f.state['compute_seconds'],
            added_global_seconds=16.342303598299623,closing_compute_seconds=675.8872357382564,
            added_configuration_seconds=.13020952977240086,closing_configuration_seconds=613.7772281504361,
            closing_master_seconds=613.165128,solver_started=False,scientific_pass=False)
        self.refs={k:v for k,v in f.refs.items() if k!='review'}
        f.new_sources[m.MODULE]='1'*64
        self.resettle()
    def resettle(self):
        f=self.f;ref=f.write(f.base/'failed_settlement.json',self.closed)
        self.refs['failed_admission_settlement']=ref
        f.state.update(compute_seconds=self.closed['closing_compute_seconds'],configuration_spent_seconds={m.KEY:self.closed['closing_configuration_seconds']},
            configuration_master_spent_seconds={m.KEY:self.closed['closing_master_seconds']},
            failed_representation_admission_settlement_reference=ref,failed_representation_reservation_reference=self.reservation)
        c.save_state(f.root,f.state);self.refs['state']=f.ref(f.root/'state.json');self.reseal()
    def reseal(self):
        f=self.f;self.refs.pop('review',None)
        self.refs['review']=f.write(f.base/'continuation_review.json',dict(kind='CASE0_REPRESENTATION_CONTINUATION_REVIEW_V1',passed=True,
            revision=m.REPRESENTATION,references=dict(self.refs),new_source_sha256=f.new_sources))
        self.request=f.write(f.base/'continuation_request.json',dict(kind='CASE0_REPRESENTATION_CONTINUATION_REQUEST_V1',revision=m.REPRESENTATION,
            references=self.refs,new_source_sha256=f.new_sources))
    def check(self):return m.check_representation_continuation(self.request['path'],self.request['sha256'])
    def prepare(self):
        with patch.object(m.time,'monotonic',side_effect=[10.,10.125]):
            return m.prepare_representation_continuation(self.request['path'],self.request['sha256'])
    def test_real_mirror_readonly_all_gates_commit_and_live_validation(self):
        f=self.f;original=(self.revision/'reservation.json').read_bytes();statebytes=(f.root/'state.json').read_bytes()
        result=self.check();self.assertTrue(result['passed']);self.assertTrue(result['no_writes'])
        self.assertEqual({p.name for p in self.revision.iterdir()},{'reservation.json'})
        self.assertEqual((f.root/'state.json').read_bytes(),statebytes)
        state=self.prepare();config,t=m.validate_representation(f.root,state)
        self.assertTrue(config['solver_compact_recourses']);self.assertEqual(t['prior_master_seconds'],613.165128)
        self.assertEqual(t['prior_configuration_seconds'],613.7772281504361+.125)
        self.assertEqual(t['cumulative_compute_seconds'],675.8872357382564+.125)
        self.assertEqual((self.revision/'reservation.json').read_bytes(),original)
        self.assertTrue((self.revision/'continuation-1/reservation.json').exists())
        state['compute_seconds']+=2;c.save_state(f.root,state);m.validate_representation(f.root,state)
        with self.assertRaisesRegex(ValueError,'not available|already continued|Frozen reference changed'):self.prepare()
    def test_changed_mirror_bytes_fail_even_if_all_source_paths_are_external(self):
        p=Path(self.f.settlement['terminal_observation_reference']['path']);p.write_text('tampered')
        with self.assertRaisesRegex(ValueError,'Frozen reference changed'):self.check()
    def test_changed_mirror_size_with_resealed_settlement_fails(self):
        f=self.f;value=c.read(f.settlement['terminal_review_reference']['path']);value['observation_reference']['size_bytes']+=1
        f.write(f.settlement['terminal_review_reference']['path'],value)
        with self.assertRaisesRegex(ValueError,'Frozen reference changed'):self.check()
    def test_global_fee_reset_with_resealed_authority_rejected(self):
        self.closed['closing_compute_seconds']-=16.21209406852722;self.resettle()
        with self.assertRaisesRegex(ValueError,'closing reset'):self.check()
    def test_configuration_fee_reset_with_resealed_authority_rejected(self):
        self.closed['closing_configuration_seconds']-=.13020952977240086;self.resettle()
        with self.assertRaisesRegex(ValueError,'configuration reset'):self.check()
    def test_native_master_cannot_be_charged_or_reset(self):
        self.closed['closing_master_seconds']+=.13020952977240086;self.resettle()
        with self.assertRaisesRegex(ValueError,'master changed'):self.check()
    def test_new_source_change_outside_adapter_rejected(self):
        self.f.new_sources['src/evcs_experiments/case0_campaign.py']='0'*64;self.reseal()
        with self.assertRaisesRegex(ValueError,'more than'):self.check()
    def test_preexisting_continuation_claim_blocks_readonly_and_prepare(self):
        (self.revision/'continuation-1').mkdir()
        with self.assertRaisesRegex(ValueError,'already continued'):self.check()
        with self.assertRaisesRegex(ValueError,'already continued'):self.prepare()


if __name__=='__main__':unittest.main()

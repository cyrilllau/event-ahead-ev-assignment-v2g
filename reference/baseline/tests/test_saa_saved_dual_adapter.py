import json,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from evcs_experiments import saa_candidate_audit
from evcs_experiments.saa_saved_dual_adapter import SAASavedDualContext
from evcs_experiments.campaign_batch import file_reference
from evcs_experiments.saa_dual_bound import build_saa_dual_bound
from tests import test_saa_candidate_audit as fixtures

class SavedSAADualTests(unittest.TestCase):
 def setUp(self):
  self.f=fixtures.SAACandidateAuditTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
  a=saa_candidate_audit.audit_saa_candidate(**self.f.args)
  self.audit=self.f.root/'accepted.json';self.audit.write_text(json.dumps(a))
  self.kw=dict(candidate_audit_reference=file_reference(self.audit),request_reference=self.f.args['fixed_q_request_reference'],
   checkpoint_reference=self.f.args['checkpoint_reference'],template_envelope_reference=self.f.args['template_envelope_reference'],
   expected_first_stage_sha256=self.f.ts[0].first_stage.matrix_sha256,deadline=time.monotonic()+20)
 def test_real_original_saa_raw_duals_global_cuts_without_solver(self):
  with patch('evcs_experiments.recourse.solve_fixed_plan',side_effect=AssertionError('No new Q')):
   ctx=SAASavedDualContext(**self.kw);cuts={}
   for i in range(len(ctx.ids)):
    cut,g,e=ctx.derive(i);self.assertTrue(e['passed']);self.assertEqual(e['solver_calls'],0)
    self.assertAlmostEqual(cut['intercept_usd']+g@ctx.anchor,ctx.checkpoint['q_usd'][i],places=7);cuts[i]=(cut,g)
   fs=self.f.ts[0].first_stage
   first=dict(matrix=fs.A,**{k:getattr(fs,k) for k in ('c','integrality','lower_bounds','upper_bounds','row_lower','row_upper')},objective_constant_usd=0.)
   model,report=build_saa_dual_bound(first,ctx.ids,cuts,constants_usd=ctx.request['constants_usd'])
   self.assertAlmostEqual(model['c']@ctx.anchor+model['objective_constant_usd'],json.loads(self.audit.read_text())['upper_bound_usd'],places=6)
   self.assertEqual(report['risk_certificate_kind'],'fixed_empirical')
 def test_raw_dual_and_rhs_bytes_tamper_rejected(self):
  ctx=SAASavedDualContext(**self.kw);row=ctx.attempts[0][1]
  for name in ('dual_kkt.npz','fixed_rhs.npz'):
   ref=next(r for r in row['artifacts'] if Path(r['path']).name==name);p=Path(ref['path']);old=p.read_bytes()
   p.write_bytes(old+b'tamper')
   with self.assertRaises(ValueError):ctx.derive(0)
   p.write_bytes(old)
 def test_resealed_wrong_atom_identity_rejected(self):
  ctx=SAASavedDualContext(**self.kw);ctx.attempts[0][1]['scenario_id']='foreign'
  with self.assertRaises(ValueError):ctx.derive(0)
 def test_wrong_audit_binding_law_and_deadline_rejected(self):
  a=json.loads(self.audit.read_text())
  for mutation in ('law','first'):
   b=json.loads(json.dumps(a))
   if mutation=='law':b['fixed_q_audit']['empirical_probabilities'][0]+=.01
   else:b['fixed_q_audit']['checkpoint_reference']={}
   self.audit.write_text(json.dumps(b))
   with self.assertRaises(ValueError):SAASavedDualContext(**dict(self.kw,candidate_audit_reference=file_reference(self.audit)))
  with self.assertRaises(TimeoutError):SAASavedDualContext(**dict(self.kw,deadline=time.monotonic()-1))

 def test_independent_q_summation_roundoff_only(self):
  original=json.loads(self.audit.read_text())
  for changed,passes in ((np.nextafter(original['fixed_q_audit']['q_usd'][0],np.inf),True),
                          (original['fixed_q_audit']['q_usd'][0]+1e-5,False),(float('nan'),False)):
   value=json.loads(json.dumps(original));value['fixed_q_audit']['q_usd'][0]=changed
   self.audit.write_text(json.dumps(value))
   kw=dict(self.kw,candidate_audit_reference=file_reference(self.audit))
   if passes:
    ctx=SAASavedDualContext(**kw)
    self.assertEqual(ctx.checkpoint['q_usd'],json.loads(Path(self.kw['checkpoint_reference']['path']).read_text())['q_usd'])
   else:
    with self.assertRaisesRegex(ValueError,'full Q checkpoint differs'):SAASavedDualContext(**kw)

 def test_resealed_raw_dual_still_requires_original_numerical_certificate(self):
  cpref=self.kw['checkpoint_reference'];cp=json.loads(Path(cpref['path']).read_text())
  aref=cp['attempts'][0];atom=json.loads(Path(aref['path']).read_text())
  i=next(i for i,r in enumerate(atom['artifacts']) if Path(r['path']).name=='dual_kkt.npz')
  p=Path(atom['artifacts'][i]['path'])
  with np.load(p) as v:d={k:v[k].copy() for k in v.files}
  d['inequality'][0]+=1.
  np.savez_compressed(p,**d);atom['artifacts'][i]=file_reference(p)
  Path(aref['path']).write_text(json.dumps(atom));cp['attempts'][0]=file_reference(aref['path'])
  Path(cpref['path']).write_text(json.dumps(cp));newcp=file_reference(cpref['path'])
  audit=json.loads(self.audit.read_text());audit['fixed_q_audit']['checkpoint_reference']=newcp
  self.audit.write_text(json.dumps(audit))
  ctx=SAASavedDualContext(**dict(self.kw,checkpoint_reference=newcp,candidate_audit_reference=file_reference(self.audit)))
  with self.assertRaisesRegex(ValueError,'dual differs'):ctx.derive(0)
 def test_resealed_template_arrays_cannot_borrow_original_template_digest(self):
  envref=self.kw['template_envelope_reference'];env=json.loads(Path(envref['path']).read_text());trref=env['template_references'][0]
  tr=json.loads(Path(trref['path']).read_text());i=next(i for i,r in enumerate(tr['artifacts']) if Path(r['path']).name=='arrays.npz')
  p=Path(tr['artifacts'][i]['path'])
  with np.load(p) as v:d={k:v[k].copy() for k in v.files}
  d['b_ub'][0]+=1.
  np.savez_compressed(p,**d);tr['artifacts'][i]=file_reference(p)
  Path(trref['path']).write_text(json.dumps(tr));env['template_references'][0]=file_reference(trref['path'])
  Path(envref['path']).write_text(json.dumps(env));newenv=file_reference(envref['path'])
  audit=json.loads(self.audit.read_text());audit['fixed_q_audit']['template_envelope_reference']=newenv;self.audit.write_text(json.dumps(audit))
  ctx=SAASavedDualContext(**dict(self.kw,template_envelope_reference=newenv,candidate_audit_reference=file_reference(self.audit)))
  with self.assertRaisesRegex(ValueError,'semantic matrix'):ctx.derive(0)

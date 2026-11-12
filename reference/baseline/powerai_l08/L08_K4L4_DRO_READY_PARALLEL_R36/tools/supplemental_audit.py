from r36_probability import *
from r36_tests import negative
a=moments();cert=loadnp(R/'probability/ZERO/CERTIFICATE.npz');meta=read(R/'probability/ZERO/COMPLETE.json')
def verify(z,identity):
 assert identity==a['identity']
 assert z['q'].shape==z['p'].shape==(400,)
 assert np.array_equal(z['scale'],a['scale'])
 planes=float(z['beta0'])+np.asarray(a['phi'].T@(z['plus']-z['minus'])).ravel()
 assert min(z['plus'])>=0 and min(z['minus'])>=0
 assert np.max(z['q']-planes)<=1e-8
 r=raw_scan(z['p'],a);assert r['raw_residual']<=1e-8 and r['simplex_residual']<=1e-9 and r['min_p']>=0
verify(cert,meta['risk_identity'])
checks={}
checks['wrong_scale_rejected']=negative(lambda:verify(dict(cert,scale=cert['scale']*2),a['identity']))
checks['missing_support_rejected']=negative(lambda:verify(dict(cert,q=cert['q'][:-1]),a['identity']))
checks['old_risk_identity_rejected']=negative(lambda:verify(cert,'OLD_K1'))
# A feasible nominal probability is not an upper-bound certificate: simplex example.
w=np.ones(4)/4;q=np.array([1.,2.,3.,8.]);nominal=float(w@q);maximum=float(q.max())
assert nominal<maximum
checks['nominal_weight_not_worst_risk']=True
begin=read(R/'formal/NATIVE_BEGIN.json')['epoch']
dump(R/'tests/SUPPLEMENTAL_CERTIFICATE_AUDIT.json',dict(passed=all(checks.values()),checks=checks,epoch=time.time(),formal_begin=begin,timing='POST_START_SUPPLEMENTAL; not retroactively claimed as pre-start hard-test evidence',no_native_LP_or_MIP=True,nominal_counterexample=nominal,worst_counterexample=maximum))
print('SUPPLEMENTAL_AUDIT_PASS',flush=True)

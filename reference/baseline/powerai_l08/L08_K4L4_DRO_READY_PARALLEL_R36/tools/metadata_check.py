from r36_recourse import *
import r36_recourse as rr
import shutil
d=R/'tests/metadata_repair'
assert not d.exists()
src=R/'evaluations/ZERO/0000'
shutil.copytree(src,d)
(d/'COMPLETE.json').rename(d/'ORIGINAL_COMPLETE.json')
def forbidden(*a,**kw):raise AssertionError('COMPLETED_LP_REPEATED')
rr.evaluate_one=forbidden
r=evaluate([],0,d)
assert r['passed'] and read(d/'METADATA_REPAIR.json')['LP_repeated'] is False
dump(R/'tests/METADATA_REPAIR_TEST.json',dict(passed=True,real_completed_payload=True,native_LP_calls=0))
print('METADATA_REPAIR_PASS',flush=True)

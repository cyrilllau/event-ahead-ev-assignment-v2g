from r36_prepare import *
from r36_recourse import recourse_tests,zero400
from r36_probability import ambiguity_tests
from r36_tests import control_tests
from r36_model import resource_check,build_rmp
from r36_semantics import semantic_tests
from r36_integration import integration_tests
from r36_incident_test import incident_tests
from r36_delivery import delivery_tests
import argparse
STAGES={'inventory':inventory,'verify-payload':payload,'build-common-domain':common_domain,'replay400':replay400,'recourse-tests':recourse_tests,'zero400':zero400}
STAGES.update({'ambiguity-tests':ambiguity_tests,'control-tests':control_tests,'resource-check':resource_check,'semantic-tests':semantic_tests,'integration-tests':integration_tests,'build-rmp':build_rmp})
STAGES['incident-semantics-tests']=incident_tests
STAGES['delivery-tests']=delivery_tests
def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage');ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 if args.stage in ['status','report','package']:
  import r36_delivery as delivery
  result=getattr(delivery,args.stage)();print(json.dumps(result),flush=True);return
 if args.stage=='all':
  sequence=['inventory','verify-payload','build-common-domain','replay400','recourse-tests','zero400','control-tests','semantic-tests','ambiguity-tests','resource-check','integration-tests','incident-semantics-tests','build-rmp','delivery-tests','start-if-safe']
  p=subprocess.Popen([sys.executable,'-B',str(R/'tools/parent.py')]+sequence);rc=p.wait();dump(R/'operations/ALL_PARENT_WAIT.json',dict(actual_parent_wait=True,returncode=rc,epoch=time.time()));assert rc==0;return
 if args.stage in ['start-if-safe','resume']:
  from r36_master import stage_gates,run
  stage_gates()
  with ctrl.lock(R/'FORMAL_OWNER.lock'),admission('FORMAL_R36',read(R/'RESOURCE_PROFILE.json')['request_GiB'],4):run()
  return
 fn=STAGES[args.stage]
 with ctrl.lock(R/'PREPARE.lock'),admission(args.stage):
  result=committed(R/'gates'/f'{args.stage}.json',fn);print(json.dumps(dict(stage=args.stage,passed=result['passed'])),flush=True)
if __name__=='__main__':main()

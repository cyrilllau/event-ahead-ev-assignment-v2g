import json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from evcs_experiments import formal_bank as m
from evcs_experiments.evaluation_lock import file_reference

class BankDeadlineTests(unittest.TestCase):
    def test_source_time_and_export_reserve_subtracted_from_absolute_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/'bank';out.mkdir();clock=[0.]
            def write(p,v):p.write_text(json.dumps(v));return file_reference(p)
            code=root/'code.py';code.write_text('# code')
            manifest=write(root/'manifest.json',{'files':[{'path':'code.py','sha256':m.sha256(code)}]})
            request=write(root/'request.json',dict(source_manifest=manifest,outputroot=str(out),bank_export_reserve_seconds=10.))
            call=write(root/'invocation.json',dict(item_request_reference=request,domain='formal',previous_checkpoint=None,
                remaining_seconds=100.,worker_deadline_monotonic=100.,attempt_directory=str(root),result_path=str(root/'result.json')))
            original=m.sha256
            def sha(p):
                if Path(p).resolve()==code.resolve():clock[0]+=5
                return original(p)
            grants=[]
            def generate(request,**kwargs):
                grants.append(kwargs['wall_seconds']);write(out/'checkpoint.json',dict(completed_atoms={},completed_targets=[]))
                return {'status':'TIME_LIMIT'}
            with patch.dict(os.environ,{'EVCS_BATCH_ATTEMPT_REQUEST':call['path'],'EVCS_BATCH_INVOCATION_SHA256':call['sha256']}),patch.object(m,'sha256',side_effect=sha),patch.object(m.time,'monotonic',side_effect=lambda:clock[0]):
                self.assertEqual(m._worker_main(generate,'COMPLETE'),0)
            self.assertEqual(grants,[85.])
            record=json.loads((root/'result.json').read_text());self.assertEqual(record['source_verification_seconds'],5.)
            self.assertEqual(record['export_reserve_seconds'],10.)

if __name__=='__main__':unittest.main()

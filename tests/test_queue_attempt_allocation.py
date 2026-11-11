import tempfile,unittest
from pathlib import Path
from evcs_experiments.queue_attempt_allocation import allocate_attempt_directory

class AllocationTests(unittest.TestCase):
 def test_orphan_slot_is_preserved_and_not_charged_as_receipt(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();one=root/'m.SAA__attempt_0001';two=root/'m.SAA__attempt_0002';one.mkdir();two.mkdir()
   (one/'receipt.json').write_text('original receipt');(two/'invocation.json').write_text('failed invocation')
   before={str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()}
   new=allocate_attempt_directory(root,'m.SAA',minimum_ordinal=2)
   self.assertEqual(new.name,'m.SAA__attempt_0003');self.assertEqual(list(new.iterdir()),[])
   self.assertEqual(before,{str(p):p.read_bytes() for p in root.rglob('*') if p.is_file()})
   self.assertEqual(len(list(root.glob('*/receipt.json'))),1)
 def test_normal_ordinal_and_foreign_item_unchanged(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();(root/'other__attempt_9999').mkdir()
   self.assertEqual(allocate_attempt_directory(root,'mine',minimum_ordinal=1).name,'mine__attempt_0001')
 def test_malformed_symlink_and_unsafe_item_rejected(self):
  for mode in ('suffix','symlink','item','boolean'):
   with self.subTest(mode=mode),tempfile.TemporaryDirectory() as d:
    root=Path(d).resolve();item='mine';ordinal=1
    if mode=='suffix':(root/'mine__attempt_2').mkdir()
    if mode=='symlink':(root/'mine__attempt_0001').symlink_to(root,target_is_directory=True)
    if mode=='item':item='../escape'
    if mode=='boolean':ordinal=True
    with self.assertRaises(ValueError):allocate_attempt_directory(root,item,minimum_ordinal=ordinal)

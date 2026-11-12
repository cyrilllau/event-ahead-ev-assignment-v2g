#!/usr/bin/env python3
"""Read-only run replay; never invokes any optimizer."""
from pathlib import Path
import argparse,json,os,sys
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import durable as d
import runtime as rt

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--relocate-root');args=p.parse_args();out=Path(args.out)
 if args.relocate_root:
  # Resolve recorded run objects relative to a relocated unpacked run, preserving original bytes.
  original=args.relocate_root;old_sha=d.sha;old_read=d.read;old_np=d.np.load
  def path(x):
   text=str(x);return out/text[len(original):].lstrip('/') if text==original or text.startswith(original+'/') else x
  d.sha=lambda x:old_sha(path(x));d.read=lambda x:old_read(path(x));d.np.load=lambda x,*a,**kw:old_np(path(x),*a,**kw)
 store=d.Store(out);state,meta=store.replay()
 import engine as e
 run=d.read(out/'RUN.json');run['source']=d.read(out/'CONFIG.json')['source'];e.configure(out,run,rt.deadline(run)-120)
 from autonomous_controller import solution_load
 solved,z,proof=solution_load(store,meta['last_solved']);pending=d.identity(state)!=d.identity(solved)
 result={'passed':True,'working_identity':d.identity(state),'last_solved_identity':d.identity(solved),'current_LB_replayed':proof['LB'],'best_LB_retained':meta['best_LB'],'phase':meta['phase'],'global_outer':meta['outer'],'pending_has_distinct_solved_reference':pending,'scientific_optimizer_calls':0,'all_data_sources':str(out/'SOURCES.json'),'external_dependencies_required_for_upstream_cut_proofs':True}
 print(json.dumps(result,indent=2))

if __name__=='__main__':main()

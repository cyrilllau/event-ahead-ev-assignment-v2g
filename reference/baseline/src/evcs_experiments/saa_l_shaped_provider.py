"""Reviewed numeric-template and disk-witness adapter for development SAA only.

The original L-shaped master/recourse mathematics is unchanged. This component
is not a formal worker, whole-process memory bound, or production budget seal.
"""
from dataclasses import dataclass
import hashlib,json
from pathlib import Path
import numpy as np
from .saa_l_shaped import DevelopmentSAAProblem,ALGORITHM_ID,need
from .lazy_recourses import LazyRecourses
from .recourse_template_index import TemplateProvider
from .recourse_store import _verify_first_stage
from .campaign_batch import file_reference,verify_reference
from .random_streams import canonical_hash
from .config import get_paths

HANDLE='SAA_L_SHAPED_DISK_ATOM_REFERENCE_V1'
RECORD='SAA_L_SHAPED_DISK_ATOM_V1'


def _read(ref):
    verify_reference(ref);return json.loads(Path(ref['path']).read_text())


def _sources():
    names=('saa_l_shaped_provider.py','saa_l_shaped.py','lazy_recourses.py','recourse_template_index.py','recourse_store.py')
    return {n:file_reference(Path(__file__).with_name(n)) for n in names}


class DiskAtomStore:
    """Append-only per-Q witnesses; handles contain no primal/dual/gradient arrays."""
    def __init__(self,root,problem):
        self.root=Path(root);self.problem=problem
        need(self.root.is_absolute() and self.root.resolve()==self.root,'Explicit canonical evidence root required')
        paths=get_paths();protected=(Path(paths.data_root).resolve(),Path(paths.frozen_source_root).resolve(),Path(__file__).resolve().parent)
        need(not any(self.root.is_relative_to(p) for p in protected),'Evidence root overlaps frozen inputs or source')
        original_root=Path(problem.provider.index['original_request_reference']['path']).parent
        need(not self.root.is_relative_to(original_root),'Evidence root overlaps original raw evidence pack')
        need(self.root!=Path(problem.provider.index['original_snapshot_reference']['path']),'Evidence root overlaps original snapshot')
        for row in problem.provider.index['rows']:
            need(not self.root.is_relative_to(Path(row['original_template_reference']['path']).parent),'Evidence root overlaps original template')
        self.root.mkdir(parents=True,exist_ok=True);self.loads=0;self.writes=0

    def _binding(self,i):
        p=self.problem;row=p.provider.row(i)
        return dict(domain=p.domain,provider_identity=p.provider.identity,problem_identity_sha256=canonical_hash(p.identity),
            scenario_index=i,scenario_id=p.support_ids[i],scenario_sha256=p.recourses.scenario_sha256s[i],
            template_sha256=row['template_sha256'],first_stage_sha256=p.first_stage.matrix_sha256,
            scenario_constant_usd=float(p.constants_usd[i]),original_template_reference=row['original_template_reference'])

    def retain(self,i,atom):
        binding=self._binding(i);semantic=canonical_hash(dict(binding=binding,atom=atom))
        directory=self.root/semantic;path=directory/'record.json'
        if path.exists():
            old=_read(file_reference(path));handle=self._handle(file_reference(path),old)
            need(self.load(handle,i)==atom,'Existing atom differs');return handle
        directory.mkdir(exist_ok=False)
        raw=directory/'witness.npz'
        with raw.open('xb') as f:
            np.savez_compressed(f,primal=np.asarray(atom['primal']),
                **{'dual_'+key:np.asarray(value) for key,value in atom['dual'].items()})
        meta=dict(kind=RECORD,binding=binding,semantic_sha256=semantic,source_references=_sources(),
            anchor=atom['anchor'],anchor_sha256=canonical_hash(atom['anchor']),q_usd=atom['q_usd'],
            cut=atom['cut'],gradient=atom['gradient'],plane_sha256=canonical_hash(dict(gradient=atom['gradient'],intercept=atom['cut']['intercept_usd'])),
            witness_reference=file_reference(raw),scientific_acceptance=False,formal_OOS=False)
        with path.open('x') as f:json.dump(meta,f,indent=2,allow_nan=False);f.write('\n')
        self.writes+=1;return self._handle(file_reference(path),meta)

    @staticmethod
    def _handle(ref,meta):
        return dict(kind=HANDLE,reference=ref,anchor_sha256=meta['anchor_sha256'],q_usd=meta['q_usd'],plane_sha256=meta['plane_sha256'])

    def load(self,handle,i):
        need(set(handle)=={'kind','reference','anchor_sha256','q_usd','plane_sha256'} and handle['kind']==HANDLE,'Wrong disk atom handle')
        ref=handle['reference'];path=Path(ref['path'])
        need(path.resolve()==path and path.parent.parent==self.root and path.name=='record.json','Foreign atom evidence path')
        meta=_read(ref);need(meta['kind']==RECORD and meta['binding']==self._binding(i)
            and meta['scientific_acceptance'] is False and meta['formal_OOS'] is False,'Atom provider/scenario/constant binding differs')
        need(meta['source_references']==_sources(),'Atom implementation source differs')
        for r in meta['source_references'].values():verify_reference(r)
        raw=meta['witness_reference'];need(Path(raw['path'])==path.parent/'witness.npz','Foreign raw Q witness')
        need(Path(raw['path']).resolve()==Path(raw['path']),'Aliased raw Q witness')
        verify_reference(raw)
        need({p.name for p in path.parent.iterdir()}=={'record.json','witness.npz'},'Uncommitted atom evidence')
        with np.load(raw['path'],allow_pickle=False) as z:
            keys={'primal','dual_inequality','dual_equality','dual_lower','dual_upper'}
            need(set(z.files)==keys,'Incomplete original Q witness')
            need(all(z[k].dtype.kind=='f' and np.isfinite(z[k]).all() for k in keys),'Invalid Q witness arrays')
            atom=dict(anchor=meta['anchor'],q_usd=meta['q_usd'],cut=meta['cut'],gradient=meta['gradient'],
                primal=z['primal'].tolist(),dual={k:z['dual_'+k].tolist() for k in ('inequality','equality','lower','upper')})
        need(self._handle(ref,meta)==handle and meta['anchor_sha256']==canonical_hash(atom['anchor'])
            and meta['plane_sha256']==canonical_hash(dict(gradient=atom['gradient'],intercept=atom['cut']['intercept_usd']))
            and meta['semantic_sha256']==path.parent.name==canonical_hash(dict(binding=meta['binding'],atom=atom)),
            'Atom semantic/anchor/plane identity changed')
        self.loads+=1;return atom

    def retain_incomplete(self,i,plan,result):
        body=dict(kind='SAA_L_SHAPED_INCOMPLETE_Q_DIAGNOSTIC_V1',binding=self._binding(i),
            source_references=_sources(),anchor=self.problem.first_stage.vector(plan).tolist(),
            solver=result.as_dict(),used_for_upper_bound=False,scientific_acceptance=False,formal_OOS=False)
        directory=self.root/'diagnostics';directory.mkdir(exist_ok=True)
        path=directory/(canonical_hash(body)+'.json')
        if path.exists():need(_read(file_reference(path))==body,'Incomplete diagnostic changed')
        else:
            with path.open('x') as f:json.dump(body,f,indent=2,allow_nan=False);f.write('\n')
        return dict(kind='SAA_L_SHAPED_INCOMPLETE_Q_REFERENCE_V1',reference=file_reference(path))

    def verify_incomplete(self,handle):
        need(set(handle)=={'kind','reference'} and handle['kind']=='SAA_L_SHAPED_INCOMPLETE_Q_REFERENCE_V1','Wrong incomplete diagnostic reference')
        path=Path(handle['reference']['path']);need(path.resolve()==path and path.parent==self.root/'diagnostics','Foreign incomplete diagnostic')
        body=_read(handle['reference']);i=body['binding']['scenario_index']
        need(body['kind']=='SAA_L_SHAPED_INCOMPLETE_Q_DIAGNOSTIC_V1' and body['binding']==self._binding(i)
             and body['source_references']==_sources() and body['used_for_upper_bound'] is False
             and body['scientific_acceptance'] is False and body['formal_OOS'] is False
             and path.stem==canonical_hash(body),'Incomplete diagnostic binding changed')

    def statistics(self):return dict(loads=self.loads,writes=self.writes,resident_witness_arrays=0,root=str(self.root))


@dataclass(frozen=True)
class StoredDevelopmentSAAProblem(DevelopmentSAAProblem):
    provider: object=None
    evidence_root: object=None
    capacity: int=1

    def __post_init__(self):
        need(self.domain=='development','Formal/OOS provider solver admission is unsupported')
        need(isinstance(self.provider,TemplateProvider),'Reviewed TemplateProvider required')
        p=self.provider;index=p.verify();scenes=tuple(self.scenarios);ids=tuple(s.scenario_id for s in scenes)
        need(index['domain']=='development' and scenes and all(s.development is True for s in scenes)
             and self.tasks.config.get('synthetic_validation_only') is True,'Only synthetic development physical fixtures are admitted')
        need(p.case is self.case and p.tasks is self.tasks,'Provider uses different physical case/tasks objects')
        need(tuple(index['scenario_ids'])==ids and tuple(index['scenario_sha256s'])==tuple(s.scenario_sha256 for s in scenes),
             'Provider complete ordered support differs')
        first=p.first_stage;_verify_first_stage(first)
        need(type(self.capacity) is int and self.capacity in (1,2),'Bounded cache capacity must be1 or2')
        constants=np.asarray(self.constants_usd,dtype=float)
        need(constants.shape==(len(ids),) and np.isfinite(constants).all(),'Invalid scenario constants')
        def loader(i):
            template=p.load(i,scenes[i])
            need(np.all(template.c>=0) and np.all(template.lower_bounds[template.c>0]>=0),
                 'Initial theta lower bound requires nonnegative original operating costs')
            return template
        lazy=LazyRecourses(ids,[s.scenario_sha256 for s in scenes],first.matrix_sha256,
            dict(provider=p.identity,domain='development',sources=_sources()),loader,capacity=self.capacity)
        object.__setattr__(self,'scenarios',scenes);object.__setattr__(self,'first_stage',first)
        object.__setattr__(self,'support_ids',ids);object.__setattr__(self,'recourses',lazy)
        object.__setattr__(self,'constants_usd',np.frombuffer(constants.tobytes(),dtype=float))
        need(self.evidence_root is not None and Path(self.evidence_root).is_absolute()
             and Path(self.evidence_root).resolve()==Path(self.evidence_root),'Explicit canonical evidence root required')
        object.__setattr__(self,'evidence_root',str(Path(self.evidence_root)))
        # Admission scans existing templates sequentially, without compilation or
        # retaining the full tuple. It is bounded residency, not zero input I/O.
        def checked(template):return None
        for i in range(len(lazy)):checked(lazy[i])
        object.__setattr__(self,'atom_store',DiskAtomStore(Path(self.evidence_root),self))

    @property
    def identity(self):
        index=self.provider.verify()
        return dict(domain='development',algorithm_id=ALGORITHM_ID,first_stage_sha256=self.first_stage.matrix_sha256,
            support_ids=list(self.support_ids),scenario_sha256s=list(self.recourses.scenario_sha256s),
            template_sha256s=[r['template_sha256'] for r in index['rows']],constants_usd=self.constants_usd.tolist(),
            empirical_probabilities=[1/len(self.support_ids)]*len(self.support_ids),provider_identity=self.provider.identity,
            evidence_root=str(self.evidence_root),cache_capacity=self.capacity,adapter_sources=_sources())

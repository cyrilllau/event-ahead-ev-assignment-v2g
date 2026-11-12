"""Development LP-event export to the existing original-matrix SAA auditor.

This stores genuine solver events, not inferred OPTIMAL flags from cut atoms.
It never solves, certifies a master bound, settles time, or admits formal/OOS work.
"""
from pathlib import Path
import json
import numpy as np
from .campaign_batch import file_reference, verify_reference
from .random_streams import canonical_hash
from .config import get_paths
from . import recourse, recourse_check, first_stage_check, saa_l_shaped, saa_fixed_plan
from .saa_fixed_plan import _new, _serial


class DevelopmentFixedQEvidence:
    """Callable sink for one run's real L-shaped LP results; arrays stay on disk."""
    def __init__(self, *, case, tasks, scenario_ids, scenario_sha256s,
                 template_sha256s, constants_usd, scene_loader, input_binding,
                 source_references, output_root, domain='development',
                 resume_inventory_reference=None, kernel_checkpoint_reference=None):
        if domain=='formal':
            from .saa_l_shaped_formal import validate_formal_problem
            if type(self) is not FormalFixedQEvidence:raise ValueError('Use typed formal evidence factory, not a development domain toggle')
            validate_formal_problem(self.problem)
        elif domain != 'development' or tasks.config.get('synthetic_validation_only') is not True:
            raise ValueError('Only explicit synthetic development evidence is supported')
        self.domain=domain
        self.algorithm_id=(self.problem.identity['algorithm_id'] if domain=='formal' else saa_l_shaped.ALGORITHM_ID)
        self.producer=('L_SHAPED_REAL_LP_EVENT_FORMAL_V1' if domain=='formal' else 'L_SHAPED_REAL_LP_EVENT_DEVELOPMENT_V1')
        self.inventory_kind=('SAA_L_SHAPED_EVENT_INVENTORY_FORMAL_V1' if domain=='formal' else 'SAA_L_SHAPED_EVENT_INVENTORY_DEVELOPMENT_V1')
        self.case, self.tasks, self.scene_loader = case, tasks, scene_loader
        self.ids = list(scenario_ids); self.hashes = list(scenario_sha256s)
        self.templates = list(template_sha256s); self.constants = np.asarray(constants_usd, dtype=float)
        n = len(self.ids)
        if not n or len(set(self.ids)) != n or len(self.hashes) != n or len(self.templates) != n or self.constants.shape != (n,) or not np.isfinite(self.constants).all():
            raise ValueError('Invalid ordered physical support/constants')
        if not isinstance(input_binding, dict) or not input_binding:
            raise ValueError('Explicit physical input binding is required')
        self.binding = _serial(input_binding); self.sources = list(source_references)
        for ref in self.sources: verify_reference(ref)
        for path in (__file__, recourse.__file__, recourse_check.__file__, first_stage_check.__file__, saa_l_shaped.__file__, saa_fixed_plan.__file__):
            if file_reference(path) not in self.sources:
                raise ValueError('Actual evidence and physical checker source is missing')
        for i, sid in enumerate(self.ids):
            s = scene_loader(i)
            if s.development is not (self.domain=='development') or s.scenario_id != sid or s.scenario_sha256 != self.hashes[i] or s.physical_case_sha256 != case.case_sha256:
                raise ValueError('Scene order, source, domain or case differs')
        self.root = Path(output_root)
        if not self.root.is_absolute() or self.root != self.root.resolve():
            raise ValueError('Evidence output requires an explicit canonical absolute path')
        paths=get_paths()
        for protected in (paths.data_root, paths.frozen_source_root, paths.project_root/'src'):
            protected=Path(protected).resolve()
            if self.root == protected or protected in self.root.parents:
                raise ValueError('Evidence output must not modify frozen data or source')
        self.sweeps = {}; self._history = []; self._checkpoints = {}
        if (resume_inventory_reference is None) != (kernel_checkpoint_reference is None):
            raise ValueError('Resume requires paired inventory and kernel checkpoint references')
        if resume_inventory_reference is None:
            self.root.mkdir(parents=True, exist_ok=False)
        else:
            self._restore(resume_inventory_reference, kernel_checkpoint_reference)

    def __call__(self, *, iteration, scenario_index, template, plan, result):
        if type(iteration) is not int or iteration < 0 or type(scenario_index) is not int or not 0 <= scenario_index < len(self.ids):
            raise ValueError('Invalid iteration or support index')
        i = scenario_index
        for ref in self.sources: verify_reference(ref)
        if template.template_sha256 != self.templates[i] or template.scenario_sha256 != self.hashes[i] or template.first_stage.task_sha256 != self.tasks.task_sha256:
            raise ValueError('Actual LP template or physical source differs')
        pd = _serial(dict(x=plan.x, r=plan.r, task_sha256=plan.task_sha256, config_hash=plan.config_hash))
        if not first_stage_check.check_first_stage(self.case, self.tasks, plan)['passed']:
            raise ValueError('First-stage physical plan failed')
        digest = canonical_hash(pd)
        if iteration not in self.sweeps:
            folder = self.root / f'iteration_{iteration:06d}'; folder.mkdir()
            request = dict(kind='SAA_SERIAL_FIXED_PLAN_Q_V1', domain=self.domain, plan=pd,
                plan_sha256=digest, case_sha256=self.case.case_sha256, task_sha256=self.tasks.task_sha256,
                scenario_ids=self.ids, scenario_sha256s=self.hashes, template_sha256s=self.templates,
                constants_usd=self.constants.tolist(), sources=self.sources, input_binding=self.binding,
                evidence_producer=self.producer,
                original_solver_events_required=True, external_timekeeper_required=True)
            self.sweeps[iteration] = dict(root=folder, request=_new(folder/'request.json', request),
                                         plan_sha256=digest, rows=[], q={})
        sweep = self.sweeps[iteration]
        if sweep['plan_sha256'] != digest or i in sweep['q']:
            raise ValueError('Foreign plan or duplicate LP event within sweep')
        folder = sweep['root'] / f'atom_{len(sweep["rows"]):05d}'; folder.mkdir()
        artifacts = [_new(folder/'solver.json', result.as_dict()), _new(folder/'plan.json', pd)]
        bu, be = template.fixed_rhs(plan); np.savez_compressed(folder/'fixed_rhs.npz', b_ub=bu, b_eq=be)
        artifacts.append(file_reference(folder/'fixed_rhs.npz'))
        physical = None
        if result.solution is not None:
            np.save(folder/'primal.npy', template.pack(result.solution), allow_pickle=False)
            artifacts.append(file_reference(folder/'primal.npy'))
            physical = recourse_check.check_recourse(self.case, self.tasks, self.scene_loader(i), plan,
                                                      result.solution, check_objective_bound=result.optimal)
            artifacts.append(_new(folder/'physical_check.json', physical))
        cert = result.certificate
        valid = bool(result.status == 'OPTIMAL' and result.solver_status == 0 and result.optimal and
                     cert.get('optimal_certificate_valid') and result.solution is not None and
                     physical['passed'] and result.objective_usd is not None and np.isfinite(result.objective_usd))
        if valid:
            np.savez_compressed(folder/'dual_kkt.npz', **{
                key: np.asarray(cert['dual_'+key+'_marginals']) for key in ('inequality','equality','lower','upper')})
            artifacts.append(file_reference(folder/'dual_kkt.npz'))
        q = float(result.objective_usd + self.constants[i]) if valid else None
        row = dict(request_reference=sweep['request'], scenario_index=i, scenario_id=self.ids[i],
                   scenario_sha256=self.hashes[i], template_sha256=self.templates[i], plan_sha256=digest,
                   completed=valid, q_usd=q, native_status=result.status, artifacts=artifacts,
                   independent_numerical_audit_required=True)
        sweep['rows'].append(_new(folder/'record.json', row))
        if valid: sweep['q'][i] = q
        if not valid:
            if result.status == 'LIMIT_REACHED' and result.solver_status == 1 and not result.optimal and (physical is None or physical['passed']):
                return  # The kernel owns TIME_LIMIT and its partial-Q checkpoint.
            raise ValueError('Native LP or physical certificate is incomplete; raw failure retained')

    def checkpoint(self, iteration):
        sweep = self.sweeps[iteration]; n = len(self.ids)
        if sorted(sweep['q']) != list(range(n)):
            raise ValueError('Incomplete same-plan support cannot be audited as full Q')
        value = dict(status='COMPLETE_FIXED_Q_PENDING_INDEPENDENT_AUDIT',
            request_reference=sweep['request'], attempts=list(sweep['rows']),
            completed_indices=list(range(n)), q_usd=[sweep['q'][i] for i in range(n)],
            queue_receipt_still_required=True, external_timekeeper_required=True,
            full_support_verified=False, independent_numerical_audit_required=True,
            master_bound_audited=False, budget_settled=False, scientific_spec_pass=False, formal_OOS=False)
        ref = _new(sweep['root']/'checkpoint.json', value)
        self._checkpoints[iteration] = ref
        return ref

    def _identity(self):
        return dict(case_sha256=self.case.case_sha256, task_sha256=self.tasks.task_sha256,
            config_hash=self.tasks.config_hash, scenario_ids=self.ids, scenario_sha256s=self.hashes,
            template_sha256s=self.templates, constants_usd=self.constants.tolist(),
            input_binding=self.binding, source_references=self.sources)

    @staticmethod
    def _read(reference):
        verify_reference(reference)
        return json.loads(Path(reference['path']).read_text())

    def _inventory(self):
        files=[];directories=[]
        for path in sorted(self.root.rglob('*')):
            if path.is_symlink():raise ValueError('Symlink in event inventory')
            if path.is_dir():directories.append(str(path))
            elif path.is_file():files.append(file_reference(path))
            else:raise ValueError('Unsupported event inventory entry')
        return files,directories

    def _check_inventory(self):
        expected=list(self._history)+list(self._checkpoints.values())
        for sweep in self.sweeps.values():
            expected.append(sweep['request'])
            for ref in sweep['rows']:
                row=self._read(ref);expected.append(ref);expected.extend(row['artifacts'])
        actual,dirs=self._inventory()
        if len({r['path'] for r in expected})!=len(expected) or sorted(actual,key=lambda r:r['path'])!=sorted(expected,key=lambda r:r['path']):
            raise ValueError('Unregistered, partial or changed event inventory')
        expected_dirs={str(parent) for ref in expected for parent in Path(ref['path']).parents if parent!=self.root and self.root in parent.parents}
        if set(dirs)!=expected_dirs:raise ValueError('Unregistered event inventory directory')
        return actual,dirs

    def _check_kernel(self, reference):
        envelope=self._read(reference);state=envelope['state']
        if envelope['sha256'] != canonical_hash(state):raise ValueError('Kernel checksum changed')
        first=saa_l_shaped.assemble_first_stage(self.case,self.tasks)
        expected=dict(domain=self.domain,algorithm_id=self.algorithm_id,
            first_stage_sha256=first.matrix_sha256,support_ids=self.ids,
            scenario_sha256s=self.hashes,template_sha256s=self.templates,
            constants_usd=self.constants.tolist(),empirical_probabilities=[1/len(self.ids)]*len(self.ids))
        if any(state['identity'].get(k)!=v for k,v in expected.items()) or state['source_identity']!=saa_l_shaped.source_identity():
            raise ValueError('Kernel physical or source identity changed')
        if state['status'] not in ('TIME_LIMIT','ITERATION_LIMIT','CONVERGED'):
            raise ValueError('Failed or active kernel cannot snapshot resumable evidence')
        def plan_hash(anchor):
            h=len(self.tasks.task_ids);periods=len(self.case.periods)
            a=np.asarray(anchor)
            return canonical_hash(dict(x=a[:h].tolist(),r=a[h:].reshape(h,periods).tolist(),
                task_sha256=self.tasks.task_sha256,config_hash=self.tasks.config_hash))
        def match(iteration,anchor,q):
            sweep=self.sweeps.get(iteration)
            if sweep is None:
                if q:raise ValueError('Empty sink cannot inherit completed kernel Q')
                return
            if sweep['plan_sha256']!=plan_hash(anchor) or sorted(sweep['q'])!=list(range(len(q))) or list(sweep['q'].values())!=list(q):
                raise ValueError('Kernel completed Q and event inventory differ')
        for row in state['trace']:
            match(row['iteration'],row['master']['primal'][:len(first.c)],row['q_usd'])
        if state['pending'] is not None:
            pending=state['pending'];match(state['iterations']+1,pending['anchor'],[a['q_usd'] for a in pending['atoms']])
        permitted={row['iteration'] for row in state['trace']}
        if state['pending'] is not None:permitted.add(state['iterations']+1)
        if set(self.sweeps)-permitted:raise ValueError('Uncommitted sweep cannot be inherited')
        if state['best'] is not None:
            best=state['best'];digest=plan_hash(best['anchor'])
            if not any(s['plan_sha256']==digest and [s['q'].get(i) for i in range(len(self.ids))]==best['q_usd'] for s in self.sweeps.values()):
                raise ValueError('Best kernel plan lacks complete matching original events')
        return state

    def snapshot(self, kernel_checkpoint_reference):
        """Freeze events against one immutable copy of the kernel's saved state.

        The caller must retain this returned strong reference; recomputing a hash
        from modified inventory bytes is not recovery authorization. Kernel/queue
        clocks stay authoritative: this metadata export does not create fees.
        """
        state=self._check_kernel(kernel_checkpoint_reference)
        self._check_inventory()
        snapshots=self.root/'inventories';snapshots.mkdir(exist_ok=True)
        number=len(list(snapshots.glob('inventory_*.json')))
        kernel_copy=snapshots/f'kernel_{number:06d}.json'
        with kernel_copy.open('xb') as stream:stream.write(Path(kernel_checkpoint_reference['path']).read_bytes())
        # Reject mutations between the initial verification and immutable copy.
        copied=file_reference(kernel_copy)
        if any(copied[k]!=kernel_checkpoint_reference[k] for k in ('sha256','size_bytes')):
            raise ValueError('Kernel checkpoint changed during snapshot')
        self._history.append(copied)
        files,directories=self._check_inventory()
        value=dict(kind=self.inventory_kind,domain=self.domain,
            output_root=str(self.root),identity=self._identity(),kernel_checkpoint_reference=copied,
            kernel_accounting=dict(elapsed_seconds=state['elapsed_seconds'],sessions=state['sessions'],config=state['config']),
            sweeps=[dict(iteration=i,request_reference=s['request'],attempt_references=s['rows']) for i,s in sorted(self.sweeps.items())],
            files=files,directories=directories,history_references=list(self._history),
            complete_checkpoint_references={str(i):r for i,r in self._checkpoints.items()},external_timekeeper_required=True,
            budget_settled=False,master_bound_audited=False,production_ready=False,formal_OOS=False)
        ref=_new(snapshots/f'inventory_{number:06d}.json',value)
        self._history.append(ref)
        return ref

    def _restore(self, inventory_reference, kernel_reference):
        value=self._read(inventory_reference)
        path=Path(inventory_reference['path'])
        if path.parent!=self.root/'inventories' or path!=path.resolve() or value.get('kind')!=self.inventory_kind or value.get('domain')!=self.domain or value['output_root']!=str(self.root) or value['identity']!=self._identity():
            raise ValueError('Inventory source, identity, root or kind changed')
        files,dirs=self._inventory()
        expected=value['files']+[inventory_reference]
        if len({r['path'] for r in expected})!=len(expected) or sorted(files,key=lambda r:r['path'])!=sorted(expected,key=lambda r:r['path']) or dirs!=value['directories']:
            raise ValueError('Missing, changed or extra event inventory')
        saved=self._read(value['kernel_checkpoint_reference']);current=self._read(kernel_reference)
        if saved!=current or any(kernel_reference[k]!=value['kernel_checkpoint_reference'][k] for k in ('sha256','size_bytes')):
            raise ValueError('Different kernel checkpoint cannot resume event inventory')
        for row in value['sweeps']:
            iteration=row['iteration']
            if type(iteration) is not int or iteration<0 or iteration in self.sweeps:raise ValueError('Duplicate sweep inventory')
            folder=self.root/f'iteration_{iteration:06d}';request_ref=row['request_reference'];request=self._read(request_ref)
            if Path(request_ref['path'])!=folder/'request.json':raise ValueError('Foreign sweep request')
            for k,v in dict(domain=self.domain,case_sha256=self.case.case_sha256,task_sha256=self.tasks.task_sha256,
                scenario_ids=self.ids,scenario_sha256s=self.hashes,template_sha256s=self.templates,
                constants_usd=self.constants.tolist(),sources=self.sources,input_binding=self.binding).items():
                if request[k]!=v:raise ValueError('Saved sweep input identity changed')
            if request['plan_sha256']!=canonical_hash(request['plan']):raise ValueError('Saved plan identity changed')
            sweep=dict(root=folder,request=request_ref,plan_sha256=request['plan_sha256'],rows=[],q={})
            for j,ref in enumerate(row['attempt_references']):
                atom=self._read(ref);atom_dir=folder/f'atom_{j:05d}'
                if Path(ref['path'])!=atom_dir/'record.json' or atom['request_reference']!=request_ref or atom['plan_sha256']!=request['plan_sha256']:
                    raise ValueError('Foreign or reordered raw LP event')
                i=atom['scenario_index']
                if type(i) is not int or not 0<=i<len(self.ids) or atom['scenario_id']!=self.ids[i] or atom['scenario_sha256']!=self.hashes[i] or atom['template_sha256']!=self.templates[i]:
                    raise ValueError('Saved atom physical identity changed')
                for refa in atom['artifacts']:
                    if Path(refa['path']).parent!=atom_dir:raise ValueError('Escaped raw event artifact')
                    verify_reference(refa)
                if type(atom['completed']) is not bool:raise ValueError('Invalid LP completion flag')
                if atom['completed']:
                    if i in sweep['q'] or not np.isfinite(atom['q_usd']):raise ValueError('Duplicate or nonfinite completed Q')
                    sweep['q'][i]=atom['q_usd']
                sweep['rows'].append(ref)
            self.sweeps[iteration]=sweep
        self._history=value['history_references']+[inventory_reference]
        self._checkpoints={int(i):r for i,r in value['complete_checkpoint_references'].items()}
        for ref in self._history:
            if Path(ref['path']).parent!=self.root/'inventories':raise ValueError('Foreign inventory history')
        for i,ref in self._checkpoints.items():
            if i not in self.sweeps or Path(ref['path'])!=self.sweeps[i]['root']/'checkpoint.json':raise ValueError('Foreign complete Q checkpoint')
        self._check_inventory()
        state=self._check_kernel(kernel_reference)
        if value['kernel_accounting']!=dict(elapsed_seconds=state['elapsed_seconds'],sessions=state['sessions'],config=state['config']):
            raise ValueError('Kernel accounting history changed')


class FormalFixedQEvidence(DevelopmentFixedQEvidence):
    """Original LP event storage bound to a verified complete formal input pack.

    This admits input/evidence plumbing only. It cannot settle a parent budget
    or declare a production solver run accepted.
    """
    def __init__(self, *, problem, source_references, output_root,
                 resume_inventory_reference=None, kernel_checkpoint_reference=None):
        from .saa_l_shaped_formal import validate_formal_problem
        validate_formal_problem(problem);self.problem=problem
        identity=problem.identity
        super().__init__(case=problem.case,tasks=problem.tasks,
            scenario_ids=identity['support_ids'],scenario_sha256s=identity['scenario_sha256s'],
            template_sha256s=identity['template_sha256s'],constants_usd=identity['constants_usd'],
            scene_loader=problem.scene_loader,
            input_binding=problem.input_binding,
            source_references=source_references,output_root=output_root,domain='formal',
            resume_inventory_reference=resume_inventory_reference,kernel_checkpoint_reference=kernel_checkpoint_reference)

"""Resumable full onset inputs via the shared formal bank loop.

This produces pending-audit inputs, not accepted plans. No baseline H or
moment arrays are reused. The formal source admission gate is external.
"""
from .formal_bank import (_generate_bank, _worker_main, accepted_protocol_sources,
                          read_ref, write_new)
from .runtime.jobctl import atomic_json
from .evaluation_lock import file_reference
from .forecast_tasks import load_forecast_task_inputs, build_candidate_tasks
from .onset_planning import OnsetPlanningContext

COMPLETE = 'COMPLETE_FORMAL_ONSET_INPUTS_PENDING_INDEPENDENT_AUDIT'


class OnsetBankAdapter:
    context_type = OnsetPlanningContext
    complete_status = COMPLETE

    def sources(self, request):
        protocol, _ = accepted_protocol_sources(request)
        if (request.get('bank_kind') != 'onset_v1' or request.get('lead_h') != 0
                or isinstance(request.get('lead_h'),bool) or request.get('domain') != 'formal'
                or request.get('regime') != 'R1'
                or request.get('split_sizes') != {'planning':20,'calibration':10}):
            raise ValueError('Onset exporter requires explicit formal onset_v1/lead0/R1 full banks')
        return protocol, {}

    def prepare(self, context, target, taskdir, remaining, taskrefs):
        source_path=taskdir/'task_source.json'
        tasks_path=taskdir/'onset_tasks.json'
        if target in taskrefs and not source_path.exists():
            raise ValueError('Checkpointed onset task source is missing')
        if source_path.exists():
            import json
            ref=json.loads(source_path.read_text())
            if target in taskrefs and ref != taskrefs[target]:
                raise ValueError('Checkpointed onset task source changed')
            bank=read_ref(ref)
        elif tasks_path.exists():
            # A complete exclusive write before source-reference commit may be
            # adopted only after the normal forecast/H validation below.
            ref=file_reference(tasks_path);bank=read_ref(ref)
        else:
            inputs=load_forecast_task_inputs(context.case_root,target,lead_h=0.)
            bank=build_candidate_tasks(inputs,overall_time_limit_seconds=max(.001,remaining-5.)).as_dict()
            if not bank['certified']:
                return None
            prepared=context.prepare_target(target,bank)
            atomic_json(tasks_path,bank);ref=file_reference(tasks_path)
            taskrefs[target]=ref
            return prepared
        prepared=context.prepare_target(target,bank)
        taskrefs[target]=ref
        return prepared


def generate_onset_bank(request,*,wall_seconds,outputroot,resume=False):
    return _generate_bank(request,wall_seconds=wall_seconds,outputroot=outputroot,
        resume=resume,adapter=OnsetBankAdapter())


def worker_main():
    return _worker_main(generate_onset_bank,COMPLETE)


if __name__=='__main__':
    raise SystemExit(worker_main())

"""Baseline SAA physical factory and exact disk-backed extensive assembly.

This is preparation/assembly, not the production queue worker. A successful
assembly does not supply a solver record, independent numerical acceptance or
OOS permission. The empirical law is fixed uniformly on the ordered support;
the baseline moment set is retained only as input provenance.
"""
from dataclasses import dataclass
import time
import numpy as np
from .formal_planning import prepare_baseline_physics
from .lazy_recourses import LazyRecourses
from .random_streams import canonical_hash
from .saa_streaming import ALGORITHM_ID


@dataclass(frozen=True)
class PreparedFormalSAA:
    physical: object
    recourses: LazyRecourses
    empirical_probabilities: np.ndarray
    input_binding: dict

    @property
    def identity(self):
        return dict(algorithm_id=ALGORITHM_ID,risk_model='SAA',
            first_stage_sha256=self.physical.first_stage.matrix_sha256,
            scenario_ids=list(self.recourses.scenario_ids),
            scenario_sha256s=list(self.recourses.scenario_sha256s),
            empirical_probabilities=self.empirical_probabilities.tolist(),
            lazy_input_binding=self.recourses.identity,
            objective_spec_sha256=canonical_hash(self.input_binding['objective_spec']))


def prepare_formal_saa(*, summary_reference,audit_reference,target_id,case_root=None,
                       template_index_reference=None,template_association_review_reference=None):
    physical=prepare_baseline_physics(summary_reference=summary_reference,audit_reference=audit_reference,
        target_id=target_id,method_id='SAA',case_root=case_root,
        template_index_reference=template_index_reference,
        template_association_review_reference=template_association_review_reference)
    ids=physical.bank.scenario_ids
    if len(ids)!=160 or len(set(ids))!=160:
        raise ValueError('Formal SAA requires the complete ordered160 baseline support')
    weights=np.frombuffer(np.full(len(ids),1./len(ids),dtype=float).tobytes(),dtype=float)
    binding=dict(physical.input_binding,objective_spec=dict(risk_model='SAA',
        law='uniform_empirical',support_ids=list(ids),weights=weights.tolist(),
        moment_ambiguity_used_for_optimization=False),
        algorithm_id=ALGORITHM_ID)
    recourses=LazyRecourses(ids,physical.bank.scenario_sha256s,physical.first_stage.matrix_sha256,
        binding,physical.template_loader,capacity=2)
    return PreparedFormalSAA(physical,recourses,weights,binding)


def assemble_prepared_saa(prepared,*,output_directory,max_coefficient_bytes,deadline):
    """One bounded original extensive assembly; solver/export/audit stay pending.

    A resource failure preserves its partial disk artifacts for diagnosis.
    There is no automatic solver switch or reduced-support fallback.
    """
    from .saa_streaming import assemble_streaming_saa
    if not isinstance(prepared,PreparedFormalSAA):
        raise ValueError('Expected source-bound formal SAA preparation')
    if time.monotonic()>=deadline:
        raise ValueError('SAA assembly shared deadline expired')
    return assemble_streaming_saa(prepared.physical.first_stage,prepared.recourses,
        scenario_ids=prepared.recourses.scenario_ids,scenario_sha256s=prepared.recourses.scenario_sha256s,
        empirical_probabilities=prepared.empirical_probabilities,output_directory=output_directory,
        max_coefficient_bytes=max_coefficient_bytes,deadline=deadline,domain='formal',
        input_binding=prepared.input_binding)

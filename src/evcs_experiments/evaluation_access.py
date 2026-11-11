"""Sample-scoped authorization for the existing formal OOS physical streams.

The immutable protocol and complete paired-plan batch are verified at scope
entry. Primitive generators only check the active identity; they retain their
original random addresses and do not repeatedly hash the same source bank.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Integral
import re

from .case_inputs import InputValidationError
from .evaluation_lock import begin_evaluation, verify_plan_batch
from .random_streams import event_order


@dataclass(frozen=True)
class OOSAccess:
    target_event: str
    sample_id: str
    base_event: str
    base_index: int
    variant_index: int
    regime: str
    batch_id: str
    protocol_sha256: str
    batch_sha256: str
    access_sha256: str


_ACTIVE: ContextVar[OOSAccess | None] = ContextVar('evcs_locked_oos_access', default=None)


@contextmanager
def locked_oos_access(lock_dir, batch_id, target_id, sample_id, s04_acceptance_ref=None):
    """Authorize one declared sample only after the complete batch passes its gate.

    Sample IDs use the existing one-based ``vNNN`` presentation; primitive
    generators continue to use zero-based variant indices. A nested scope must
    refer to the same sample and the same verified protocol/batch.
    """
    checked = verify_plan_batch(lock_dir, batch_id)
    protocol = checked['protocol']
    samples = checked['batch_spec']['sample_ids_by_target']
    if protocol['domain'] != 'formal':
        raise InputValidationError('Formal OOS access requires a formal protocol')
    if target_id not in samples or sample_id not in samples[target_id]:
        raise InputValidationError('Target/sample is outside the locked paired batch')
    parts = sample_id.split('/')
    if (len(parts) != 6 or parts[:3] != [target_id, 'formal', 'oos']
            or not re.fullmatch(r'v\d{3}', parts[4]) or parts[5] not in ('R1', 'R2', 'R3')):
        raise InputValidationError('Formal OOS sample ID must bind target/base/variant/regime')
    order = event_order(protocol['target_ids'])
    variant_index = int(parts[4][1:]) - 1
    if parts[3] not in order or not 0 <= variant_index < 50:
        raise InputValidationError('Formal OOS sample coordinates exceed the frozen bank')
    previous = _ACTIVE.get()
    identity = (target_id, sample_id, batch_id, checked['protocol_reference']['sha256'],
                checked['batch_reference']['sha256'])
    if previous is not None and identity != (previous.target_event, previous.sample_id,
            previous.batch_id, previous.protocol_sha256, previous.batch_sha256):
        raise InputValidationError('Nested OOS scope cannot change its locked sample or batch')
    access = begin_evaluation(lock_dir, batch_id, domain='formal',
        sample_ids_by_target=samples, s04_acceptance_ref=s04_acceptance_ref)
    scope = OOSAccess(target_id, sample_id, parts[3], order.index(parts[3]), variant_index,
        parts[5], batch_id, identity[3], identity[4], access['access_reference']['sha256'])
    token = _ACTIVE.set(scope)
    try:
        yield scope
    finally:
        _ACTIVE.reset(token)


def require_oos_access(*, target_event=None, base_event=None, base_index=None,
                       variant_index=None, sample_id=None, regime=None):
    """Reject formal draws/replay outside the exact authorized sample identity."""
    scope = _ACTIVE.get()
    if scope is None:
        raise InputValidationError('Formal OOS is locked outside an accepted S04 paired-plan scope')
    for name, value in (('target_event', target_event), ('base_event', base_event),
            ('base_index', base_index), ('variant_index', variant_index),
            ('sample_id', sample_id), ('regime', regime)):
        if value is None:
            continue
        if name in ('base_index', 'variant_index') and (isinstance(value, bool)
                or not isinstance(value, Integral)):
            raise InputValidationError(f'Formal OOS {name} must be an integer')
        if value != getattr(scope, name):
            raise InputValidationError(f'Formal OOS {name} differs from the locked sample')
    return scope

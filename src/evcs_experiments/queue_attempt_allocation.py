"""Allocate a fresh ordinal without treating abandoned directories as receipts.

Caller owns the campaign queue lock. This helper never changes Q, a receipt,
request, old attempt, or checkpoint. An explicit reviewed queue-engine binding
is still needed before original R26 can call it.
"""
from pathlib import Path
import re


def allocate_attempt_directory(queue_root, item_id, *, minimum_ordinal):
    root=Path(queue_root)
    if not root.is_absolute() or root.resolve()!=root or not root.is_dir():
        raise ValueError('Canonical existing queue root required')
    if not isinstance(item_id,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',item_id):
        raise ValueError('Safe immutable item identifier required')
    if type(minimum_ordinal) is not int or minimum_ordinal<1:
        raise ValueError('Positive next receipt ordinal required')
    prefix=item_id+'__attempt_';highest=minimum_ordinal-1
    for path in root.glob(prefix+'*'):
        suffix=path.name[len(prefix):]
        if not suffix.isdigit() or len(suffix)<4 or int(suffix)<1 or suffix!=str(int(suffix)).zfill(4):
            raise ValueError('Unrecognized existing attempt path')
        if path.is_symlink() or not path.is_dir():
            raise ValueError('Existing attempt must be an immutable directory')
        highest=max(highest,int(suffix))
    result=root/(prefix+str(highest+1).zfill(4))
    result.mkdir()  # Fail closed if another actor violates the caller's lock.
    return result


def _derived_source(batch, original_source_reference):
    import ast
    import hashlib
    batch.verify_reference(original_source_reference)
    text=Path(original_source_reference['path']).read_text()
    tree=ast.parse(text)
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='run_batch']
    if len(nodes)!=1:raise ValueError('Original queue function scope differs')
    node=nodes[0]
    if node.decorator_list:raise ValueError('Unexpected queue decorator')
    source=''.join(text.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    old="attempt=root/(item_id+'__attempt_'+str(sum(read_json(r['path'])['item_id']==item_id for r in state['attempt_receipts'])+1).zfill(4))\n            attempt.mkdir()"
    replacement="attempt=_explicit_reviewed_attempt_allocator(root,item_id,minimum_ordinal=sum(read_json(r['path'])['item_id']==item_id for r in state['attempt_receipts'])+1)"
    if source.count(old)!=1:raise ValueError('Original queue allocation schema differs')
    derived=source.replace(old,replacement)
    record=dict(kind='SOURCE_BOUND_QUEUE_ALLOCATION_V1',original_queue_reference=original_source_reference,
                allocator_reference=batch.file_reference(__file__),derived_function_sha256=hashlib.sha256(derived.encode()).hexdigest(),
                original_function_sha256=hashlib.sha256(source.encode()).hexdigest(),replacement_count=1,
                receipts_and_budget_unchanged=True)
    return source,derived,record


def expected_binding_record(batch, original_source_reference):
    """Recompute source identity from pinned file bytes without installing code."""
    return _derived_source(batch,original_source_reference)[2]


def bind_original_queue(batch, original_source_reference):
    """Replace only two allocation statements under the unchanged queue lock."""
    import inspect
    import types
    source,derived,record=_derived_source(batch,original_source_reference)
    if batch.file_reference(batch.__file__) != original_source_reference:
        raise ValueError('Original queue source binding differs')
    original=batch.run_batch
    if original.__globals__ is not batch.__dict__ or original.__code__.co_filename != batch.__file__ or inspect.getsource(original)!=source:
        raise ValueError('Original queue allocation schema differs or function already replaced')
    key='_explicit_reviewed_attempt_allocator'
    if key in batch.__dict__:raise ValueError('Allocator already installed')
    compiled=compile(derived,__file__+'::derived_original_run_batch','exec')
    namespace={};exec(compiled,namespace)
    function=types.FunctionType(namespace['run_batch'].__code__,batch.__dict__,original.__name__,original.__defaults__,original.__closure__)
    batch.__dict__[key]=allocate_attempt_directory
    batch.run_batch=function
    def restore():
        batch.run_batch=original;batch.__dict__.pop(key,None)
    return restore,record

"""Restore hash-bound recourse numerics without compiling or decoding large maps.

This first component imports an existing RawPlanningEvidence template. It is
not a cold compiler, a production cache, a source migration, or scientific
acceptance. A reviewer must establish the association between the original
record and the physical/source inputs when accepting the returned binding.
No old plan RHS, objective value, solution, or certificate is imported.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import fields, is_dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
from scipy.sparse import csr_matrix, load_npz

from .evaluation_lock import file_reference
from .recourse import RecourseTemplate

MATRICES = ('A_ub', 'A_eq', 'Bx_ub', 'Bx_eq', 'Br_ub', 'Br_eq')
ARRAYS = ('b_ub', 'b_eq', 'c', 'lower_bounds', 'upper_bounds', 'component_objectives')
MAPS = ('variable_map', 'constraint_map_ub', 'constraint_map_eq')
KIND = 'BOUND_ORIGINAL_RECOURSE_NUMERIC_TEMPLATE_V1'


def _need(ok, message):
    if not bool(ok):
        raise ValueError(message)


def _verify(ref):
    _need(isinstance(ref, Mapping) and set(ref) == {'path', 'sha256', 'size_bytes'},
          'A strict path/sha256/size_bytes reference is required')
    path = Path(ref['path'])
    _need(path.is_absolute() and path.is_file() and path.resolve() == path,
          'Reference must identify an existing absolute canonical file')
    _need(type(ref['size_bytes']) is int and ref['size_bytes'] >= 0,
          'Invalid reference byte count')
    _need(path.stat().st_size == ref['size_bytes'], 'Referenced file size changed')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    _need(digest.hexdigest() == ref['sha256'], 'Referenced file hash changed')
    return path


def _read(ref):
    return json.loads(_verify(ref).read_text())


def _immutable(value):
    value = np.asarray(value)
    return np.frombuffer(value.tobytes(order='C'), dtype=value.dtype).reshape(value.shape)


def _fingerprint(value):
    """Hash actual values with typed, length-delimited framing, including arrays."""
    digest = hashlib.sha256()

    def write(tag, data):
        digest.update(tag)
        digest.update(str(len(data)).encode() + b':' + data)

    def visit(item):
        if isinstance(item, np.ndarray):
            _need(item.dtype.kind in 'biuf', 'Unsupported physical array dtype')
            write(b'array', str((item.shape, item.dtype.str)).encode())
            write(b'data', np.ascontiguousarray(item).tobytes())
        elif isinstance(item, csr_matrix):
            write(b'csr', str(item.shape).encode())
            for array in (item.data, item.indices, item.indptr):
                visit(array)
        elif is_dataclass(item):
            visit({field.name: getattr(item, field.name) for field in fields(item)})
        elif isinstance(item, Mapping):
            write(b'map', str(len(item)).encode())
            for key in sorted(item, key=lambda key: (type(key).__name__, repr(key))):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            write(b'seq', str(len(item)).encode())
            for part in item:
                visit(part)
        elif isinstance(item, np.generic):
            visit(item.item())
        elif item is None or isinstance(item, (str, bool, int, float)):
            write(type(item).__name__.encode(), json.dumps(item, allow_nan=False).encode())
        elif hasattr(item, '__dict__'):
            visit(vars(item))
        else:
            raise ValueError('Unsupported physical fingerprint value: ' + type(item).__name__)

    visit(value)
    return digest.hexdigest()


def physical_fingerprint(first_stage, case, tasks, scene):
    """Actual full physical values, plus all operational first-stage numerics.

    First-stage descriptive maps are already source-bound and have no numeric
    role. Excluding their in-memory traversal keeps the warm path bounded.
    """
    first = {field.name: getattr(first_stage, field.name) for field in fields(first_stage)
             if field.name not in ('variable_map', 'constraint_map')}
    return _fingerprint({'first_stage': first, 'case': case, 'tasks': tasks, 'scene': scene})


def _verify_first_stage(first):
    identity = {key: getattr(first, key) for key in
                ('task_sha256', 'config_hash', 'task_ids', 'periods')}
    identity.update(variable_map=[dict(row) for row in first.variable_map],
                    constraint_map=[dict(row) for row in first.constraint_map], units=dict(first.units))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    for array in (first.A.data, first.A.indices, first.A.indptr, first.row_lower, first.row_upper,
                  first.lower_bounds, first.upper_bounds, first.integrality, first.c):
        canonical = np.asarray(array, dtype='<i8' if array.dtype.kind in 'iu' else '<f8')
        digest.update(str(canonical.shape).encode())
        digest.update(canonical.tobytes(order='C'))
    _need(digest.hexdigest() == first.matrix_sha256, 'Actual first-stage semantic SHA differs')
    _need(first.A.shape == (len(first.row_lower), len(first.c)) and
          first.row_upper.shape == first.row_lower.shape and
          all(getattr(first, name).shape == first.c.shape
              for name in ('lower_bounds', 'upper_bounds', 'integrality')),
          'Actual first-stage dimensions differ')


def _header(path):
    # This is the explicit original exporter format, not a permissive JSON
    # scanner. Complete bytes (including maps) have already been hashed.
    marker = '\n  "variable_map":'
    text = ''
    with path.open() as stream:
        while marker not in text:
            part = stream.read(8192)
            _need(part and len(text) + len(part) <= 4 * 1024 * 1024,
                  'Unknown or oversized original identity header')
            text += part
    return json.loads(text.split(marker, 1)[0].rstrip().removesuffix(',') + '\n}')


def _record(reference):
    row = _read(reference)
    _need(row.get('kind') == 'recourse_template', 'Wrong original template record kind')
    root = Path(reference['path']).parent
    expected = {name + '.npz' for name in MATRICES} | {'arrays.npz', 'identity.json'}
    refs = row.get('artifacts', [])
    names = [Path(ref['path']).name for ref in refs]
    _need(len(refs) == len(expected) and set(names) == expected,
          'Incomplete or duplicate original template artifact inventory')
    _need({path.name for path in root.iterdir()} == expected | {'record.json'},
          'Original template contains missing or uncommitted artifacts')
    for ref in refs:
        _need(Path(ref['path']).parent == root, 'Foreign template artifact path')
        _verify(ref)
    return row, {Path(ref['path']).name: ref for ref in refs}


def _numeric(refs):
    matrices = {}
    for name in MATRICES:
        matrix = load_npz(refs[name + '.npz']['path'])
        _need(isinstance(matrix, csr_matrix) and matrix.has_canonical_format,
              'Original coefficients must be canonical CSR')
        _need(np.isfinite(matrix.data).all(), 'Nonfinite coefficient matrix')
        for attr in ('data', 'indices', 'indptr'):
            setattr(matrix, attr, _immutable(getattr(matrix, attr)))
        matrices[name] = matrix
    with np.load(refs['arrays.npz']['path'], allow_pickle=False) as archive:
        _need(set(archive.files) == set(ARRAYS), 'Changed original numeric array inventory')
        arrays = {name: _immutable(archive[name]) for name in ARRAYS}
    for name, array in arrays.items():
        _need(array.dtype.kind == 'f' and not np.isnan(array).any(), 'Invalid original numeric array')
        if name not in ('lower_bounds', 'upper_bounds'):
            _need(np.isfinite(array).all(), 'Nonfinite original numeric array')
    _need(np.array_equal(arrays['c'], arrays['component_objectives'].sum(axis=0)),
          'Original objective aggregation differs')
    return matrices, arrays


def _semantic_sha(meta, matrices, arrays, first_sha):
    identity = dict(task_sha256=meta['task_sha256'], config_hash=meta['config_hash'],
        scenario_sha256=meta['scenario_sha256'], first_stage_sha256=first_sha,
        shapes=meta['variable_shapes'], task_ids=meta['task_ids'], bus_ids=meta['bus_ids'],
        line_ids=meta['line_ids'], station_ids=meta['station_ids'], periods=meta['periods'],
        references=meta['reference_bus_ids_by_period'], cmax_usd=meta['cmax_usd'])
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode())
    assembled = {**{name.lower().replace('bx', 'x').replace('br', 'r'): value
                   for name, value in matrices.items()}, 'b_ub': arrays['b_ub'], 'b_eq': arrays['b_eq']}
    for key in sorted(assembled):
        value = assembled[key]
        values = (value.data, value.indices, value.indptr) if isinstance(value, csr_matrix) else (value,)
        digest.update(key.encode())
        for array in values:
            canonical = np.asarray(array, dtype='<i8' if array.dtype.kind in 'iu' else '<f8')
            digest.update(str(canonical.shape).encode())
            digest.update(canonical.tobytes())
    for name in ('lower_bounds', 'upper_bounds', 'component_objectives'):
        digest.update(np.asarray(arrays[name], dtype='<f8').tobytes())
    return digest.hexdigest()


def _identity_check(meta, row, first, case, tasks, scene, matrices, arrays):
    _need(meta['template_sha256'] == row['template_sha256'] ==
          _semantic_sha(meta, matrices, arrays, first.matrix_sha256), 'Original template semantic SHA differs')
    for key, source in (('task_ids', tasks), ('station_ids', case), ('bus_ids', case),
                        ('line_ids', case), ('periods', case)):
        _need(tuple(meta[key]) == tuple(getattr(source, key)), 'Physical template axis differs: ' + key)
    _need(meta['task_sha256'] == tasks.task_sha256 == scene.task_sha256 == first.task_sha256 and
          meta['config_hash'] == tasks.config_hash == scene.config_hash == first.config_hash and
          meta['scenario_sha256'] == scene.scenario_sha256 and
          case.case_sha256 == tasks.physical_case_sha256 == scene.physical_case_sha256,
          'Physical template identity differs')
    _need(tuple(first.task_ids) == tuple(tasks.task_ids) and tuple(first.periods) == tuple(case.periods),
          'First-stage axes differ')
    slices, shapes = meta['variable_slices'], meta['variable_shapes']
    h, t, n, ell, c = (len(tasks.task_ids), len(case.periods), len(case.bus_ids),
                        len(case.line_ids), len(case.station_ids))
    expected_shapes = dict(rav=[h, t], p=[h, t], q=[h, t], short=[c, t],
        shed=[t, n], line_p=[t, ell], line_q=[t, ell], u=[t, n], soc_dep=[h])
    _need(shapes == expected_shapes, 'Operational variable shapes differ from physical axes')
    columns = sum(int(np.prod(shape)) for shape in expected_shapes.values())
    for sense in ('ub', 'eq'):
        shape = tuple(meta['shape_' + sense])
        _need(len(shape) == 2 and shape[1] == columns and
              matrices['A_' + sense].shape == shape and
              matrices['Bx_' + sense].shape == (shape[0], h) and
              matrices['Br_' + sense].shape == (shape[0], h * t) and
              arrays['b_' + sense].shape == (shape[0],),
              'Original coefficient/RHS dimensions differ')
    _need(all(arrays[name].shape == (columns,) for name in ('c', 'lower_bounds', 'upper_bounds')) and
          arrays['component_objectives'].shape == (4, columns), 'Original objective/bound dimensions differ')
    _need(np.all(arrays['lower_bounds'] <= arrays['upper_bounds']) and
          not np.isposinf(arrays['lower_bounds']).any() and not np.isneginf(arrays['upper_bounds']).any(),
          'Invalid original variable bounds')
    _need(tuple(shapes) == ('rav', 'p', 'q', 'short', 'shed', 'line_p', 'line_q', 'u', 'soc_dep'),
          'Unknown operational variable order')
    offset = 0
    for name, shape in shapes.items():
        size = int(np.prod(shape))
        _need(slices[name] == [offset, offset + size], 'Noncontiguous variable slices')
        offset += size
    _need(offset == len(arrays['c']), 'Variable axes differ from objective')
    _need(np.array_equal(arrays['upper_bounds'][slice(*slices['shed'])], scene.load_p_kw.ravel()),
          'Physical load differs from saved bounds')


def bind_template(record_reference, first_stage, case, tasks, scene, *,
                  source_references, compiler_reference, output):
    """Seal an association for independent review; never grant cache acceptance.

    Sources must identify the actual original immutable inputs and compiler.
    This function verifies bytes and records actual values; the caller/reviewer
    remains responsible for validating their provenance association.
    """
    _need(isinstance(source_references, Mapping) and source_references,
          'Explicit original physical source references required')
    for ref in [*source_references.values(), compiler_reference]:
        _verify(ref)
    _verify_first_stage(first_stage)
    row, refs = _record(record_reference)
    _need(type(row['scenario_index']) is int and row['scenario_index'] >= 0 and
          row['scenario_id'] == scene.scenario_id, 'Original scenario record differs')
    meta = _header(Path(refs['identity.json']['path']))
    matrices, arrays = _numeric(refs)
    _identity_check(meta, row, first_stage, case, tasks, scene, matrices, arrays)
    body = dict(kind=KIND, status='BOUND_PENDING_INDEPENDENT_SOURCE_ASSOCIATION_REVIEW',
        original_template_reference=record_reference, original_identity_reference=refs['identity.json'],
        compiler_reference=compiler_reference, source_references=dict(source_references),
        physical_fingerprint=physical_fingerprint(first_stage, case, tasks, scene),
        first_stage_sha256=first_stage.matrix_sha256, template_sha256=row['template_sha256'],
        scenario_index=row['scenario_index'], scenario_id=row['scenario_id'],
        scientific_acceptance=False, formal_OOS=False, old_q_reused=False)
    path = Path(output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(body, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return file_reference(path)


class _IdentityDocument:
    def __init__(self, reference):
        self.reference = reference
        self.value = None

    def read(self):
        if self.value is None:
            self.value = _read(self.reference)
            _need(all(isinstance(self.value.get(name), list) for name in MAPS),
                  'Original descriptive maps are incomplete')
        return self.value


class _LazyMap(Sequence):
    def __init__(self, document, name, count):
        self.document, self.name, self.count = document, name, count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rows = self.document.read()[self.name]
        _need(len(rows) == self.count, 'Original descriptive map length differs')
        if isinstance(index, slice):
            return tuple(MappingProxyType(row) for row in rows[index])
        return MappingProxyType(rows[index])


class StoredRecourseTemplate(RecourseTemplate):
    @property
    def descriptive_maps_loaded(self):
        return self._identity_document.value is not None

    @property
    def original_identity_reference(self):
        return dict(self._identity_document.reference)

    def as_dict(self):
        # Preserve the exact full original document on explicit descriptive use.
        return json.loads(json.dumps(self._identity_document.read()))


def load_template(binding_reference, first_stage, case, tasks, scene, *,
                  expected_template_sha256, expected_scenario_index, expected_scenario_id):
    """Restore the original operational LP, with no assembly or plan-value reuse."""
    binding = _read(binding_reference)
    _need(binding.get('kind') == KIND and binding.get('scientific_acceptance') is False and
          binding.get('formal_OOS') is False and binding.get('old_q_reused') is False,
          'Wrong numeric template binding kind or scope')
    _need(type(expected_scenario_index) is int and expected_scenario_index >= 0 and
          binding['scenario_index'] == expected_scenario_index and
          binding['scenario_id'] == expected_scenario_id == scene.scenario_id and
          binding['template_sha256'] == expected_template_sha256 and
          binding['first_stage_sha256'] == first_stage.matrix_sha256,
          'Requested template/first-stage/support identity differs')
    _need(binding['physical_fingerprint'] == physical_fingerprint(first_stage, case, tasks, scene),
          'Actual physical or first-stage values changed')
    for ref in [binding['compiler_reference'], *binding['source_references'].values()]:
        _verify(ref)
    row, refs = _record(binding['original_template_reference'])
    _need(refs['identity.json'] == binding['original_identity_reference'] and
          row['template_sha256'] == expected_template_sha256 and
          row['scenario_index'] == expected_scenario_index and row['scenario_id'] == expected_scenario_id,
          'Original template binding or support order changed')
    meta = _header(Path(refs['identity.json']['path']))
    matrices, arrays = _numeric(refs)
    _identity_check(meta, row, first_stage, case, tasks, scene, matrices, arrays)
    document = _IdentityDocument(refs['identity.json'])
    count = len(arrays['c'])
    lazy = {name: _LazyMap(document, name, length) for name, length in zip(MAPS,
        (count, matrices['A_ub'].shape[0], matrices['A_eq'].shape[0]))}
    template = StoredRecourseTemplate(**matrices, **arrays, **lazy,
        reference_bus_ids_by_period=tuple(tuple(row) for row in meta['reference_bus_ids_by_period']),
        template_sha256=expected_template_sha256, cmax_usd=meta['cmax_usd'],
        **{key: tuple(meta[key]) for key in ('task_ids', 'station_ids', 'bus_ids', 'line_ids', 'periods')},
        **{key: meta[key] for key in ('task_sha256', 'config_hash', 'scenario_sha256')},
        variable_slices=MappingProxyType({key: tuple(value) for key, value in meta['variable_slices'].items()}),
        variable_shapes=MappingProxyType({key: tuple(value) for key, value in meta['variable_shapes'].items()}),
        first_stage=first_stage, load_p_kw=_immutable(scene.load_p_kw),
        arrival_soc=_immutable(scene.task_arrival_soc), u_ref=float(case.u_ref),
        station_task_indices=tuple(tuple(index for index, station in enumerate(tasks.station_ids)
                                        if station == name) for name in case.station_ids),
        units=MappingProxyType(meta['units']))
    object.__setattr__(template, '_identity_document', document)
    return template

"""Bounded, hash-bound recourse loading; never materialize the full bank."""
from collections import OrderedDict
from collections.abc import Sequence
import operator

from .random_streams import canonical_hash


class LazyRecourses(Sequence):
    """Bind all original input identities up front, verify each template on use.

    `binding` must include original input and compiler SHA256s. Template hashes
    are recorded on actual assembly, not fabricated before assembly. An LRU
    bounds this store; active master copies also consume memory outside it.
    """
    def __init__(self, scenario_ids, scenario_sha256s, first_stage_sha256, binding,
                 loader, *, capacity=2):
        if type(capacity) is not int or capacity < 1:
            raise ValueError('Positive integer cache capacity required')
        self.scenario_ids = tuple(scenario_ids)
        self.scenario_sha256s = tuple(scenario_sha256s)
        if (not self.scenario_ids or len(set(self.scenario_ids)) != len(self.scenario_ids)
                or len(self.scenario_ids) != len(self.scenario_sha256s)):
            raise ValueError('Invalid lazy support axis')
        import json
        self._identity_json = json.dumps({'kind': 'input_bound_lazy_v1',
            'scenario_ids': self.scenario_ids, 'scenario_sha256s': self.scenario_sha256s,
            'first_stage_sha256': first_stage_sha256, 'binding': binding}, sort_keys=True)
        self.first_stage_sha256 = first_stage_sha256
        self._loader = loader; self.capacity = capacity; self._cache = OrderedDict()
        self.template_hashes = {}; self.loads = 0; self.hits = 0

    @property
    def identity(self):
        import json
        return json.loads(self._identity_json)

    def __len__(self):
        return len(self.scenario_ids)

    def __getitem__(self, index):
        if isinstance(index, slice):
            raise TypeError('Select explicit indices; bulk slices defeat bounded loading')
        index = operator.index(index)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index in self._cache:
            self.hits += 1; self._cache.move_to_end(index)
            return self._cache[index]
        # Evict BEFORE construction to bound simultaneous template residency.
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        template = self._loader(index)
        if (template.scenario_sha256 != self.scenario_sha256s[index]
                or template.first_stage.matrix_sha256 != self.first_stage_sha256):
            raise ValueError('Lazy recourse input/first-stage identity mismatch')
        previous = self.template_hashes.get(index)
        if previous is not None and previous != template.template_sha256:
            raise ValueError('Reassembled template changed')
        self.template_hashes[index] = template.template_sha256
        self._cache[index] = template; self.loads += 1
        return template

    def statistics(self):
        return dict(loads=self.loads, hits=self.hits, resident_indices=list(self._cache),
                    capacity=self.capacity, assembled_template_sha256s=dict(self.template_hashes),
                    input_binding_sha256=canonical_hash(self.identity))

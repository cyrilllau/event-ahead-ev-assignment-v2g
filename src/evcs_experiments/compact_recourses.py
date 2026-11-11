"""Lossless compact descriptive recourse maps; numerical objects are untouched.

This explicit representation stores categorical row labels, not original mapping
objects. Rows are reconstructed on demand; the default compiler remains full.
"""
from array import array
from collections.abc import Mapping, Sequence
from dataclasses import fields
import math
import operator
import struct
from types import MappingProxyType

import numpy as np

from .recourse import RecourseTemplate

MAP_FIELDS=('variable_map','constraint_map_ub','constraint_map_eq')


class CompactRows(Sequence):
    """Immutable, ordered mapping sequence with lossless scalar-type categories."""
    __slots__=('_keys','_values','_codes','_length')

    def __init__(self,rows):
        if not isinstance(rows,Sequence):raise ValueError('Descriptive rows must be an ordered sequence')
        length=len(rows);keys=tuple(rows[0]) if length else ()
        if length and 'index' not in keys:raise ValueError('Descriptive index missing')
        for i,row in enumerate(rows):
            if not isinstance(row,Mapping) or tuple(row)!=keys or type(row['index']) is not int or row['index']!=i:
                raise ValueError('Descriptive keys/order/index differ')
        tables=[];codes=[]
        for key in keys:
            if key=='index':tables.append(None);codes.append(None);continue
            lookup={};values=[];encoded=array('I')
            for row in rows:
                value=row[key]
                allowed=(str,int,float,bool,type(None),np.bool_,np.int8,np.int16,np.int32,np.int64,
                         np.uint8,np.uint16,np.uint32,np.uint64,np.float16,np.float32,np.float64)
                if type(value) not in allowed or (
                    isinstance(value,(float,np.floating)) and not math.isfinite(value)):
                    raise ValueError('Descriptive category is not a finite scalar')
                # Equality alone merges signed zero. Preserve exact scalar type
                # and floating bits while retaining the original value itself.
                token=(value.tobytes() if isinstance(value,np.floating) else
                       struct.pack('>d',value) if type(value) is float else value)
                typed=(type(value),token)
                code=lookup.get(typed)
                if code is None:code=len(values);lookup[typed]=code;values.append(value)
                encoded.append(code)
            tables.append(tuple(values))
            if len(values)<=1:codes.append(None)
            else:
                dtype=np.uint8 if len(values)<=256 else np.uint16 if len(values)<=65536 else np.uint32
                data=np.frombuffer(encoded,dtype=np.uint32).astype(dtype,copy=False)
                codes.append(np.frombuffer(data.tobytes(),dtype=dtype))
        object.__setattr__(self,'_keys',keys);object.__setattr__(self,'_values',tuple(tables))
        object.__setattr__(self,'_codes',tuple(codes));object.__setattr__(self,'_length',length)

    def __setattr__(self,name,value):raise AttributeError('Compact descriptive rows are immutable')
    def __len__(self):return self._length
    def __getitem__(self,index):
        if isinstance(index,slice):return tuple(self[i] for i in range(*index.indices(len(self))))
        i=operator.index(index)
        if i<0:i+=len(self)
        if not 0<=i<len(self):raise IndexError(index)
        return MappingProxyType({key:i if key=='index' else values[0 if codes is None else int(codes[i])]
            for key,values,codes in zip(self._keys,self._values,self._codes)})
    @property
    def code_bytes(self):return sum(c.nbytes for c in self._codes if c is not None)
    @property
    def category_count(self):return sum(len(v) for v in self._values if v is not None)


class CompactRecourseTemplate(RecourseTemplate):
    """Same arrays/hash/numerical methods with lazily reconstructed row labels."""
    def as_dict(self):
        result=super().as_dict()
        # Base _json intentionally retains its legacy tuple-only behavior.
        from .recourse import _json
        for name in MAP_FIELDS:result[name]=[_json(dict(row)) for row in getattr(self,name)]
        return result


def compact_recourse(template):
    """Consume a caller's temporary full template; do not retain its full maps.

    Caller must drop any other full-template aliases immediately. In Case0,
    compact each uncached ctx.recourse(i) as it is produced, not an eight-item
    list after all originals have already been retained by the caller.
    """
    if isinstance(template,CompactRecourseTemplate):return template
    if not isinstance(template,RecourseTemplate):raise ValueError('Expected original RecourseTemplate')
    values={field.name:getattr(template,field.name) for field in fields(RecourseTemplate)}
    for name in MAP_FIELDS:values[name]=CompactRows(values[name])
    return CompactRecourseTemplate(**values)

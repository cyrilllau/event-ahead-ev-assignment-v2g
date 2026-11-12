"""Frozen L08 time-of-use, islanding, route and energy semantics."""
import numpy as np
from .energy import actual, forecast

def tou():
    t = np.arange(96) / 4
    return np.select([t < 8, t < 16, t < 21], [0.14296, 0.17297, 0.2154], default=0.17297)

def qualify(departure_dc, distance, arrival, k, r, success=True):
    v = actual(departure_dc - 1.2 * distance, arrival, k, r, success, 1.0)
    return (v['qualified'], v['mu'], v['use_dc'], v['available_ac'])

def ebar(departure_dc, distance, arrival, k, r):
    return forecast(departure_dc - 1.2 * distance, arrival, k, r)['cap_ac']

def first_cost(d, k, r):
    return 25 + 30 * ((r + 1 - d) / 4) + (r - k + 1) / 4

def components(n, off):
    adj = {b: [] for b in n['buses']}
    for u, v, o in zip(n['from'], n['to'], off):
        if not o:
            adj[u].append(v)
            adj[v].append(u)
    unseen = set(adj)
    groups = []
    while unseen:
        todo = [min(unseen)]
        g = set()
        while todo:
            b = todo.pop()
            if b not in g:
                g.add(b)
                todo += adj[b]
        unseen -= g
        groups.append(sorted(g))
    return groups

def incident(n, a, *, epsilon_ls=1e-06):
    z = np.zeros((len(n['station_buses']), 96), bool)
    refs = []
    bi = {b: i for i, b in enumerate(n['buses'])}
    for t in range(96):
        rr = []
        for group in components(n, a['line_operating_outages'][t]):
            stations = set(group) & set(n['station_buses'])
            rr.append(1 if 1 in group else min(stations or group))
            if 1 not in group and sum((a['load_p_kw'][t, bi[b]] for b in group)) > epsilon_ls:
                for c, b in enumerate(n['station_buses']):
                    z[c, t] = b in group or z[c, t]
        refs.append(rr)
    return (z, refs)

def task(q, k, r):
    return dict(q, k=int(k), r=int(r), Ebar=ebar(q['departure_dc'], q['distance'], q['arrival'], k, r))

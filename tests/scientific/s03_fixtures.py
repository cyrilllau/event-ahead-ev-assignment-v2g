"""Explicit synthetic extensive-form goldens; never subsets of real-case H.

All 16-facet Q1 constraints remain present. The one-bus examples deliberately
have zero reactive demand so that activation and risk choices admit an exact
hand proof. Reuse the accepted synthetic physical fixture, never a data bank.
"""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np

from ev_assignment.methods.probability import ProbabilitySet
from ev_assignment.domain.random_streams import canonical_hash
from recourse_fixtures import make_fixture


def _golden(name, vehicles, periods, loads, activation_cost, features, lower, upper,
            constants, expected, hand_branches):
    base = make_fixture(periods=periods, task_station_indices=(0,)*vehicles,
                        load_p=np.full((periods,1),float(loads[0])), tan_phi=0.,
                        reserve=4., eta=1., tau=1.)
    case, tasks = base.case, base.tasks
    config = {"synthetic_validation_only":True,"golden":name,"vehicles":vehicles,
              "periods":periods,"p_dis_kw":4.,"p_block_kw":1.,"tau_blk_h":1.,
              "eta_dis":1.,"activation_usd":float(activation_cost),"capacity_usd_per_kwh":.1,
              "v2g_usd_per_kwh":.15,"degradation_usd_per_kwh_dc":.05}
    tasks.config = config; tasks.config_hash = canonical_hash(config)
    tasks.p_dis_kw = np.full(vehicles,4.)
    tasks.c_act_usd = np.full(vehicles,float(activation_cost))
    tasks.c_cap_usd_per_kwh = np.full((vehicles,periods),.1)
    tasks.c_v2g_usd_per_kwh = np.full(vehicles,.15)
    case.case_sha256 = canonical_hash({"golden":name,"base_case_sha256":case.case_sha256,
                                       "config":config,"scenario_loads_kw":list(loads)})
    tasks.physical_case_sha256 = case.case_sha256
    tasks.task_sha256 = canonical_hash({"task_ids":list(tasks.task_ids),"config":config,
                                       "physical_case_sha256":case.case_sha256})
    tasks.source_sha256 = tasks.task_sha256
    tasks.dependency_sha256 = canonical_hash({"source":"S03 explicit synthetic golden","config":config})
    tasks.source_path = "synthetic S03 golden constructor"
    scenarios = []
    scenario_ids = tuple(f"{name}/scenario_{i+1:02d}" for i in range(len(loads)))
    for i,(sid,load) in enumerate(zip(scenario_ids,loads)):
        scene = deepcopy(base.scenario)
        scene.scenario_id = sid; scene.target_event = name; scene.base_weather_event = "synthetic_golden"
        scene.variant = i+1; scene.task_sha256 = tasks.task_sha256; scene.config_hash = tasks.config_hash
        scene.physical_case_sha256 = case.case_sha256
        scene.load_p_kw = np.full((periods,1),float(load))
        scene.load_q_kvar = np.zeros((periods,1)); scene.load_archived_q_kvar = np.zeros((periods,1))
        scene.scenario_sha256 = canonical_hash({"id":sid,"task_sha256":tasks.task_sha256,
            "load_p_kw":scene.load_p_kw.tolist(),"load_q_kvar":scene.load_q_kvar.tolist(),
            "all_tasks_available":True,"arrival_soc":.8,"line_ids":[]})
        scene.metadata = {"fixture_only":True,"synthetic_validation_only":True,"golden":name}
        scenarios.append(scene)
    phi = np.asarray(features,dtype=float)
    probability_set = ProbabilitySet(scenario_ids,tuple(f"feature_{i}" for i in range(phi.shape[1])),
                                     phi,np.asarray(lower),np.asarray(upper))
    return SimpleNamespace(name=name,case=case,tasks=tasks,scenarios=tuple(scenarios),
        scenario_ids=scenario_ids,probability_set=probability_set,
        empirical_probabilities=np.full(len(loads),1/len(loads)),
        scenario_cost_constants_usd=np.asarray(constants,dtype=float),expected=expected,
        hand_branches=hand_branches,loads_kw=tuple(loads),activation_cost_usd=float(activation_cost),
        hand_formula="For k selected EVs: R_t=4k; Fpre=k*(Cact+0.4*T); Q_s=T*(0.2*min(L_s,4k)+3*max(L_s-4k,0))+constant_s",
        scope="Explicit synthetic 2-4 EV, 2-3 period, 3-8 scenario validation; all original fixture tasks retained")


def build_golden_cases():
    """Three independently hand-solvable physical assignment/risk problems."""
    return (
        _golden("three_ev_moment",3,2,(4.,8.,12.),8.,[[0.],[1.],[2.]],[.8],[1.2],
                (1.,2.,3.),
                {"SAA":{"objective_usd":30.266666666666666,"selected_count":2},
                 "DRO":{"objective_usd":32.12,"selected_count":3},
                 "ARO":{"objective_usd":34.2,"selected_count":3}},
                {"SAA":(50.,36.4,30.266666666666666,31.6),
                 "DRO":(55.,41.4,36.76,32.12),"ARO":(75.,61.4,47.8,34.2)}),
        _golden("two_ev_fixed_probability",2,3,(0.,4.,8.,8.),18.,
                [[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[0.,0.,0.]],
                [.25,.25,.25],[.25,.25,.25],(0.,0.,0.,0.),
                {"SAA":{"objective_usd":39.,"selected_count":1},
                 "DRO":{"objective_usd":39.,"selected_count":1},
                 "ARO":{"objective_usd":43.2,"selected_count":2}},
                {"SAA":(45.,39.,41.4),"DRO":(45.,39.,41.4),"ARO":(72.,57.6,43.2)}),
        _golden("four_ev_simplex",4,2,(0.,4.,8.,12.,16.,16.,8.,4.),8.,np.empty((8,0)),[],[],
                (0.,)*8,
                {"SAA":{"objective_usd":35.,"selected_count":2},
                 "DRO":{"objective_usd":41.6,"selected_count":4},
                 "ARO":{"objective_usd":41.6,"selected_count":4}},
                {"SAA":(51.,40.2,35.,35.4,38.6),
                 "DRO":(96.,82.4,68.8,55.2,41.6),"ARO":(96.,82.4,68.8,55.2,41.6)}),
    )

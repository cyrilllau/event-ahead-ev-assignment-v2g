"""Independent index/array invariants for planning-only physical ablations.

Unequal scenario/time/asset dimensions detect accidental axis permutations;
encoded multivariate blocks detect trajectory or within-block recombination.
These tests neither build a production bank nor open an OOS evaluator.
"""
from copy import deepcopy
import unittest

import numpy as np
from evcs_experiments.random_streams import DEVELOPMENT_ROOT, ROOTS, Stream


S,T,L=7,5,3
BLOCKS=('weather_grid','road_vehicle','load')


def address(asset=0,component=0,*,split='planning',domain='development',mechanism=701):
    # Explicit fixture-only mechanism namespace; not a production draw claim.
    return Stream(DEVELOPMENT_ROOT if domain=='development' else ROOTS[split],
                  (1,2 if domain=='development' else 1,1 if split=='planning' else 2,
                   0,1,1,mechanism,asset,component))


def identity(method,*,split='planning',domain='development',source=None):
    from evcs_experiments.physical_ablations import support_digest
    if source is None:source=line_fixture()[0] if method=='no_line_dependence' else block_fixture()[0]
    return dict(method_id=method,target_id='helene_2024',split=split,domain=domain,
                base_support_sha256=support_digest(source))


def line_fixture():
    # Each original outage path is persistent, with different asset thresholds.
    thresholds=(np.arange(S)[:,None]+2*np.arange(L)[None,:])%6
    outages=(np.arange(T)[None,:,None]>=thresholds[:,None,:])
    streams=[[address(line+1,slot+1) for line in range(L)] for slot in range(T)]
    indices=np.empty((S,T,L),dtype=np.int64)
    for t in range(T):
        for line in range(L):indices[:,t,line]=streams[t][line].generator().permutation(S)
    return outages,indices,streams


def encoded(shape,offset):
    values=np.arange(np.prod(shape),dtype=float).reshape(shape)
    values+=np.arange(S).reshape((S,)+(1,)*(len(shape)-1))*10000+offset
    return values


def block_fixture():
    # Full producer inventory; dimensions are unequal explicit synthetic axes.
    wind=encoded((S,T,L,2),10)
    road=encoded((S,T,6),500)
    load=encoded((S,3,2),900)
    route=encoded((S,4),30)/100.
    line,_,_=line_fixture()
    blocks={
        'weather_grid':{
            'grid_hazard':encoded((S,T,4),0)/100000.,
            'line_wind_vectors_m_s':wind,
            'road_wind_vectors_m_s':encoded((S,T,6,2),20),
            'line_outages':line,
            'line_operating_outages':line[:,:2,:],
            'line_standard_normals':encoded((S,L),0)/10000.-3,
            'line_threshold_uniforms':encoded((S,L),0)/100000.,
        },
        'road_vehicle':{
            'road_blocked':road.astype(int)%3==0,
            'road_movable':road.astype(int)%3!=0,
            'road_travel_time_min':road,
            'road_corridor_uniforms':encoded((S,3),0)/100000.,
            'task_arrival_h':route-1.,
            'task_gamma_route':np.ones((S,4),dtype=bool),
            'task_route_time_h':route,
            'task_travel_energy_kwh':route*2.,
            'task_arrival_soc':encoded((S,4),0)/100000.,
            'task_varsigma':np.ones((S,4),dtype=bool),
            'task_mu_full':encoded((S,4,3),0).astype(int)%2==0,
            'task_usable_energy_dc_kwh':encoded((S,4),100),
            'task_failed_arc_ids':np.full((S,4),-1,dtype=np.int64),
        },
        'load':{'load_p_kw':load,'load_q_kvar':load*.37,'load_archived_q_kvar':load*.4,
                'load_epsilon':encoded((S,3),0)/100000.,
                'load_standard_innovation':encoded((S,3),0)/10000.-3},
    }
    streams={name:address(i+1,0,mechanism=702) for i,name in enumerate(BLOCKS)}
    indices={name:streams[name].generator().permutation(S) for name in BLOCKS}
    return blocks,indices,streams


class PhysicalAblationTests(unittest.TestCase):
    def test_each_line_slot_exact_marginal_and_explicit_donor_mapping(self):
        from evcs_experiments.physical_ablations import no_line_dependence
        source,indices,streams=line_fixture();before=source.copy()
        value=no_line_dependence(source,sample_indices=indices,streams=streams,
                                 identity=identity('no_line_dependence'))
        transformed=value.arrays
        self.assertEqual(transformed.shape,(S,T,L))
        for t in range(T):
            for line in range(L):
                np.testing.assert_array_equal(np.sort(transformed[:,t,line]),np.sort(source[:,t,line]))
                np.testing.assert_array_equal(transformed[:,t,line],source[indices[:,t,line],t,line])
        np.testing.assert_array_equal(source,before)
        np.testing.assert_array_equal(value.sample_indices,indices)
        self.assertFalse(transformed.flags.writeable)
        self.assertFalse(value.sample_indices.flags.writeable)
        self.assertFalse(value.provenance['scientific_acceptance'])
        self.assertFalse(value.provenance['formal_OOS'])
        # This fixed fixture demonstrates allowed destruction of persistence.
        self.assertTrue(np.any(np.diff(transformed.astype(int),axis=1)<0))

    def test_whole_blocks_recombine_without_changing_any_internal_trajectory(self):
        from evcs_experiments.physical_ablations import independent_recombination
        blocks,indices,streams=block_fixture();original=deepcopy(blocks)
        value=independent_recombination(blocks,sample_indices=indices,streams=streams,
                                       identity=identity('independent_recombination'))
        for name in BLOCKS:
            np.testing.assert_array_equal(np.sort(value.sample_indices[name]),np.arange(S))
            for field,array in blocks[name].items():
                np.testing.assert_array_equal(value.arrays[name][field],array[indices[name]])
                np.testing.assert_array_equal(array,original[name][field])
                self.assertFalse(value.arrays[name][field].flags.writeable)
            self.assertFalse(value.sample_indices[name].flags.writeable)
        # Correlated fields stay paired even with distinct trailing dimensions.
        np.testing.assert_array_equal(value.arrays['load']['load_q_kvar'],value.arrays['load']['load_p_kw']*.37)
        np.testing.assert_array_equal(value.arrays['road_vehicle']['task_travel_energy_kwh'],value.arrays['road_vehicle']['task_route_time_h']*2.)
        self.assertNotEqual(value.arrays['weather_grid']['grid_hazard'].shape,value.arrays['load']['load_p_kw'].shape)
        self.assertFalse(value.provenance['scientific_acceptance'])
        self.assertFalse(value.provenance['formal_OOS'])

    def test_seed_addresses_are_stable_under_call_and_block_mapping_order(self):
        from evcs_experiments.physical_ablations import no_line_dependence,independent_recombination
        lines,index,streams=line_fixture()
        first=no_line_dependence(lines,sample_indices=index,streams=streams,identity=identity('no_line_dependence'))
        blocks,bi,bs=block_fixture()
        a=independent_recombination(blocks,sample_indices=bi,streams=bs,identity=identity('independent_recombination'))
        reverse=lambda d:{key:d[key] for key in reversed(tuple(d))}
        b=independent_recombination(reverse(blocks),sample_indices=reverse(bi),streams=reverse(bs),identity=identity('independent_recombination'))
        second=no_line_dependence(lines,sample_indices=index,streams=[[s.as_dict() for s in row] for row in streams],identity=identity('no_line_dependence'))
        np.testing.assert_array_equal(first.arrays,second.arrays)
        self.assertEqual(first.provenance,second.provenance)
        for key in BLOCKS:
            np.testing.assert_array_equal(a.sample_indices[key],b.sample_indices[key])
            for field in blocks[key]:np.testing.assert_array_equal(a.arrays[key][field],b.arrays[key][field])
        self.assertEqual(a.provenance,b.provenance)
        self.assertNotEqual(address().stream_id,address(split='calibration').stream_id)
        self.assertNotEqual(address().stream_id,address(domain='formal').stream_id)

    def test_line_duplicate_addresses_wrong_index_or_axes_refused(self):
        from evcs_experiments.physical_ablations import no_line_dependence
        lines,index,streams=line_fixture();ident=identity('no_line_dependence')
        repeated=[row[:] for row in streams];repeated[1][0]=repeated[0][0]
        bad_index=index.copy();bad_index[0,0,0]=bad_index[1,0,0]
        coherent_but_wrong=index.copy();coherent_but_wrong[:,0,0]=np.roll(coherent_but_wrong[:,0,0],1)
        for args in ((lines,index,repeated),(lines,bad_index,streams),(lines,coherent_but_wrong,streams),
                     (lines,index.transpose(0,2,1),streams),(lines,index.astype(float),streams)):
            with self.subTest(shape=np.shape(args[1])):
                with self.assertRaises(ValueError):
                    no_line_dependence(args[0],sample_indices=args[1],streams=args[2],identity=ident)

    def test_recombination_duplicate_stream_missing_block_or_mixed_support_refused(self):
        from evcs_experiments.physical_ablations import independent_recombination
        blocks,index,streams=block_fixture();ident=identity('independent_recombination')
        repeated=dict(streams);repeated['load']=streams['road_vehicle']
        short=deepcopy(blocks);short['road_vehicle']['task_travel_energy_kwh']=short['road_vehicle']['task_travel_energy_kwh'][:-1]
        missing=dict(blocks);missing.pop('load')
        bad_index=deepcopy(index);bad_index['load'][0]=bad_index['load'][1]
        wrong_index=deepcopy(index);wrong_index['load']=np.roll(wrong_index['load'],1)
        for b,i,s in ((blocks,index,repeated),(short,index,streams),(missing,index,streams),(blocks,bad_index,streams),(blocks,wrong_index,streams)):
            with self.assertRaises(ValueError):independent_recombination(b,sample_indices=i,streams=s,identity=ident)

    def test_source_digest_rejects_changed_inputs_and_preserves_failed_travel_infinity(self):
        from evcs_experiments.physical_ablations import no_line_dependence,independent_recombination
        lines,index,streams=line_fixture();bound=identity('no_line_dependence',source=lines)
        changed=lines.copy();changed[0,0,0]=not changed[0,0,0]
        with self.assertRaisesRegex(ValueError,'[Dd]igest|[Ss]upport|[Ss]ource'):
            no_line_dependence(changed,sample_indices=index,streams=streams,identity=bound)
        blocks,bi,bs=block_fixture();bound=identity('independent_recombination',source=blocks)
        changed=deepcopy(blocks);changed['load']['load_p_kw'][0,0,0]+=1.
        with self.assertRaisesRegex(ValueError,'[Dd]igest|[Ss]upport|[Ss]ource'):
            independent_recombination(changed,sample_indices=bi,streams=bs,identity=bound)
        blocks['road_vehicle']['road_travel_time_min'][0,1,2]=np.inf
        blocks['road_vehicle']['task_arrival_h'][1,3]=np.inf
        value=independent_recombination(blocks,sample_indices=bi,streams=bs,
                                       identity=identity('independent_recombination',source=blocks))
        self.assertEqual(np.isposinf(value.arrays['road_vehicle']['road_travel_time_min']).sum(),1)
        self.assertEqual(np.isposinf(value.arrays['road_vehicle']['task_arrival_h']).sum(),1)
        for name,field,bad in (('load','load_p_kw',np.inf),('road_vehicle','road_travel_time_min',-np.inf),
                               ('weather_grid','grid_hazard',np.nan)):
            corrupt=deepcopy(blocks);corrupt[name][field].flat[0]=bad
            with self.assertRaises(ValueError):
                independent_recombination(corrupt,sample_indices=bi,streams=bs,
                    identity=identity('independent_recombination',source=corrupt))

    def test_no_features_or_missing_field_can_be_silently_carried_forward(self):
        from evcs_experiments.physical_ablations import independent_recombination
        blocks,index,streams=block_fixture()
        for block,field in (('weather_grid','line_outages'),('road_vehicle','task_mu_full'),('load','load_epsilon')):
            changed=deepcopy(blocks);del changed[block][field]
            with self.assertRaises(ValueError):
                independent_recombination(changed,sample_indices=index,streams=streams,
                    identity=identity('independent_recombination',source=changed))
        changed=deepcopy(blocks);changed['weather_grid']['features']=np.zeros((S,5))
        with self.assertRaises(ValueError):
            independent_recombination(changed,sample_indices=index,streams=streams,
                identity=identity('independent_recombination',source=changed))

    def test_oos_unknown_method_and_invalid_identity_cannot_enter(self):
        from evcs_experiments.physical_ablations import no_line_dependence,independent_recombination
        lines,index,streams=line_fixture();blocks,bi,bs=block_fixture()
        for function,source,indices,ss,method in ((no_line_dependence,lines,index,streams,'no_line_dependence'),
                (independent_recombination,blocks,bi,bs,'independent_recombination')):
            for update in ({'split':'oos'},{'method_id':'M6_static'},{'target_id':''},
                           {'base_support_sha256':'not-a-hash'},{'domain':'production'},{'domain':'formal'}):
                ident=identity(method);ident.update(update)
                with self.subTest(function=method,update=update),self.assertRaises(ValueError):
                    function(source,sample_indices=indices,streams=ss,identity=ident)

    def test_distinct_addresses_may_coincidentally_select_the_same_permutation(self):
        from evcs_experiments.physical_ablations import no_line_dependence,independent_recombination
        source=np.zeros((1,2,3),dtype=bool)
        streams=[[address(line+1,slot+1) for line in range(3)] for slot in range(2)]
        index=np.zeros_like(source,dtype=np.int64)
        value=no_line_dependence(source,sample_indices=index,streams=streams,
                                 identity=identity('no_line_dependence',source=source))
        np.testing.assert_array_equal(value.arrays,source)
        blocks,_,streams=block_fixture()
        blocks={b:{f:a[:1] for f,a in fields.items()} for b,fields in blocks.items()}
        index={b:np.array([0],dtype=np.int64) for b in BLOCKS}
        value=independent_recombination(blocks,sample_indices=index,streams=streams,
                identity=identity('independent_recombination',source=blocks))
        for block in BLOCKS:
            for field in blocks[block]:
                np.testing.assert_array_equal(value.arrays[block][field],blocks[block][field])

    def test_empty_task_set_preserves_nonempty_scenario_axis(self):
        from evcs_experiments.physical_ablations import independent_recombination
        blocks,indices,streams=block_fixture()
        for field in blocks['road_vehicle']:
            if field.startswith('task_'):
                blocks['road_vehicle'][field]=blocks['road_vehicle'][field][:,:0]
        value=independent_recombination(blocks,sample_indices=indices,streams=streams,
                identity=identity('independent_recombination',source=blocks))
        self.assertEqual(value.arrays['road_vehicle']['task_mu_full'].shape,(S,0,3))
        self.assertEqual(value.arrays['road_vehicle']['task_arrival_h'].shape,(S,0))
        for block in BLOCKS:
            for field in blocks[block]:
                np.testing.assert_array_equal(value.arrays[block][field],blocks[block][field][indices[block]])

    def test_development_identity_cannot_relabel_formal_random_root(self):
        from evcs_experiments.physical_ablations import no_line_dependence,independent_recombination
        lines,index,streams=line_fixture()
        bad=[row[:] for row in streams]
        bad[0][0]=Stream(ROOTS['oos'],bad[0][0].spawn_key)
        changed=index.copy();changed[:,0,0]=bad[0][0].generator().permutation(S)
        with self.assertRaisesRegex(ValueError,'[Rr]oot'):
            no_line_dependence(lines,sample_indices=changed,streams=bad,identity=identity('no_line_dependence'))
        blocks,indices,ss=block_fixture()
        ss['load']=Stream(ROOTS['planning'],ss['load'].spawn_key)
        indices['load']=ss['load'].generator().permutation(S)
        with self.assertRaisesRegex(ValueError,'[Rr]oot'):
            independent_recombination(blocks,sample_indices=indices,streams=ss,identity=identity('independent_recombination'))

    def test_gamma_zero_uses_real_kernel_and_keeps_wind_resistance_and_draws(self):
        from dataclasses import replace
        from evcs_experiments.config import get_paths
        from evcs_experiments.road_disruptions import load_road_model
        from evcs_experiments.line_outages import load_line_model
        from evcs_experiments.physical_ablations import no_line_road_coupling,road_support_digest
        from tests.test_road_disruptions import EVENTS
        root=get_paths().data_root
        model=load_road_model(root,include_connectors=True)
        line_model=load_line_model(root)
        times=np.arange(-2.,24.,.25)
        line_thresholds=line_model.draw_thresholds(EVENTS,'michael2018',0,development=True)
        lines=line_model.replay(times,np.zeros((len(times),32,2)),line_thresholds,lead_h=1.)
        lines=replace(lines,outages=np.ones_like(lines.outages,dtype=bool))
        thresholds=model.draw_thresholds(EVENTS,'michael2018',0,development=True)
        vectors=np.zeros((len(times),88,2));vectors[:,:,0]=25.
        vectors[4,:,0]=1000.  # Deliberate replay probe: retained wind-only persistent blockage.
        args=dict(model=model,times_h=times,vectors=vectors,thresholds=thresholds,
                  line_replay=lines,lead_h=1.,regime='R3')
        digest=road_support_digest(**args)
        ident=dict(method_id='no_line_road_coupling',target_id='michael2018',
                   split='planning',domain='development',base_support_sha256=digest)
        original_vectors=vectors.copy();original_gamma=model.gamma.copy()
        coupled=model.replay(times,vectors,thresholds,lines,lead_h=1.,regime='R3')
        expected=model.replay(times,vectors,thresholds,lines,lead_h=1.,regime='R3',no_coupling=True)
        actual=no_line_road_coupling(**args,identity=ident)
        for field in ('wind_probability','blockage_probability','cumulative_blockage_probability',
                      'blocked','movable','travel_time_min','effective_kappa_tt','effective_gamma'):
            np.testing.assert_array_equal(getattr(actual.replay,field),getattr(expected,field))
        np.testing.assert_allclose(actual.replay.blockage_probability,actual.replay.wind_probability,rtol=0,atol=1e-16)
        self.assertTrue(np.any(actual.replay.blocked))
        self.assertTrue(np.any(coupled.blockage_probability>actual.replay.blockage_probability))
        self.assertTrue(np.all(np.diff(actual.replay.blocked.astype(int),axis=0)>=0))
        self.assertIs(actual.replay.thresholds,thresholds)
        np.testing.assert_array_equal(actual.replay.effective_kappa_tt,1.25*model.kappa_tt)
        np.testing.assert_array_equal(vectors,original_vectors)
        np.testing.assert_array_equal(model.gamma,original_gamma)
        self.assertFalse(actual.provenance['scientific_acceptance'])
        self.assertFalse(actual.provenance['formal_OOS'])
        changed=vectors.copy();changed[10,0,0]+=1.
        with self.assertRaisesRegex(ValueError,'[Ss]upport'):
            no_line_road_coupling(**dict(args,vectors=changed),identity=ident)
        changed_model=replace(model,kappa_tt=model.kappa_tt+0.01)
        with self.assertRaisesRegex(ValueError,'[Ss]upport'):
            no_line_road_coupling(**dict(args,model=changed_model),identity=ident)
        for update in ({'split':'oos'},{'domain':'formal'}):
            with self.subTest(update=update),self.assertRaises(ValueError):
                no_line_road_coupling(**args,identity=dict(ident,**update))


if __name__=='__main__':unittest.main()


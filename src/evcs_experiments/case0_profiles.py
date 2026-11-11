"""Explicit, disjoint Case0 study profiles; no formal campaign admission."""
LEGACY = 'legacy'
EXPANDED = 'expanded48-v1'
TARGETS = {LEGACY: 'michael_2018', EXPANDED: 'zeta_2020'}
CAMPAIGNS = {LEGACY: 'ieee33-v54-q1-case0', EXPANDED: 'ieee33-v54-q1-case0-expanded'}


def validate_inputs(profile, fleet, variants, offset):
    if profile == LEGACY:
        valid = type(fleet) is int and fleet in (2, 3) and type(variants) is int and variants in (1, 2) and type(offset) is int and offset in (10000, 20000)
    elif profile == EXPANDED:
        valid = type(fleet) is int and fleet == 12 and type(variants) is int and variants == 1 and type(offset) is int and offset in (30000, 40000)
    else:
        valid = False
    if not valid:
        raise ValueError('Case0 inputs outside the declared study profile')


def validate_config(config, root=None):
    profile = config.get('profile', LEGACY)
    if profile not in CAMPAIGNS:
        raise ValueError('Unknown Case0 profile')
    if root is not None and (root.name in CAMPAIGNS.values() or profile == EXPANDED or "campaign" in config) and root.name != CAMPAIGNS[profile]:
        raise ValueError('Case0 profile and campaign directory differ')
    if profile == EXPANDED:
        expected = dict(campaign=CAMPAIGNS[EXPANDED], sizes=[dict(fleet_per_depot=12, planning_variants=1)],
            variant_start=30000, debug_variant_start=40000, calibration_per_base=4,
            evaluation_per_base=8, base_weather_count=8, development=True,
            configuration_seconds=1800, master_seconds=1200, final_reserve_seconds=600,
            total_compute_seconds=21600, gap_target=.005, evaluation_item_count=768,
            cpu_cores=1, memory_max_bytes=8*1024**3, solver_feasibility_tolerance=1e-9,
            backend='gurobi', horizon_hours=24, network='IEEE33', target_event=TARGETS[EXPANDED], planning_methods=['EA_DRO','OR_DRO','SAA','ARO'])
        if any(config.get(k) != v for k, v in expected.items()):
            raise ValueError('Expanded Case0 locked profile changed')
    return profile


def context_options(config):
    profile = validate_config(config)
    options = dict(variant_start=config['variant_start'])
    if profile != LEGACY:
        options['profile'] = profile
    return options


def lock_target(lock):
    profile = lock.get('profile', LEGACY)
    if profile not in TARGETS or (profile == EXPANDED and lock.get('target_event') != TARGETS[profile]):
        raise ValueError('Case0 OOS lock target/profile differs')
    if 'target_event' in lock and lock['target_event'] != TARGETS[profile]:
        raise ValueError('Case0 OOS target differs from profile')
    return TARGETS[profile]

#!/usr/bin/env python3
"""Render four Case0 figures from an exact, audited summary; no LP/statistic reruns.

Use a separate reporting environment with matplotlib==3.10.6. Source CSVs may
be mirrored locally: their bytes must match the original summary output refs.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform

METHODS = ('EA_DRO', 'DSP', 'OR_DRO', 'SAA', 'ARO', 'ZERO')
RISK = ('EA_DRO', 'SAA', 'ARO')
REGIMES = ('R1', 'R2', 'R3')
LABEL = dict(EA_DRO='EA-DRO', DSP='DSP', OR_DRO='OR-DRO', SAA='SAA', ARO='ARO', ZERO='Zero')
COLOR = dict(EA_DRO='#0072B2', DSP='#E69F00', OR_DRO='#009E73', SAA='#D55E00', ARO='#CC79A7', ZERO='#666666')
COMPONENTS = (('Fact_usd', r'$F^{\mathrm{act}}$'), ('Fcap_usd', r'$F^{\mathrm{cap}}$'),
              ('Fshed_usd', r'$F^{\mathrm{shed}}$'), ('Fshort_usd', r'$F^{\mathrm{short}}$'),
              ('FV2G_usd', r'$F^{\mathrm{V2G}}$'), ('Fdeg_usd', r'$F^{\mathrm{deg}}$'))


def need(ok, message):
    if not ok:
        raise ValueError(message)


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), size_bytes=path.stat().st_size)


def number(value):
    if value is None or value == '':
        return None
    result = float(value)
    need(math.isfinite(result), 'Nonfinite plotting value')
    return result


def interval(row, difference=False):
    """Return untouched point/limits, including undefined intervals or points."""
    keys = ('difference', 'difference_ci95_lower', 'difference_ci95_upper') if difference else ('estimate', 'ci95_lower', 'ci95_upper')
    point, lo, hi = (number(row[k]) for k in keys)
    status = row['difference_status' if difference else 'status']
    need(status in ('DEFINED', 'NA', 'NA_INTERVAL'), 'Unknown statistical status')
    need((lo is None) == (hi is None), 'Half-defined confidence interval')
    if status == 'DEFINED':
        need(point is not None and lo is not None, 'Defined estimate/interval missing')
    elif status == 'NA':
        need(point is None and lo is None, 'NA observation must remain undefined')
    else:
        need(point is not None and lo is None, 'NA_INTERVAL must retain only its estimate')
    if lo is not None:
        need(lo <= hi, 'Reversed confidence interval')
    return point, lo, hi


def load_summary(directory, expected_sha256):
    directory = Path(directory).resolve()
    ref = reference(directory / 'summary.json')
    need(ref['sha256'] == expected_sha256, 'Summary SHA differs')
    payload = json.loads((directory / 'summary.json').read_text())
    need(payload['kind'] == 'CASE0_AUDITED_DESCRIPTIVE_STATISTICS_V1', 'Audited Case0 summary required')
    profile = payload.get('study_scope', {}).get('profile', 'legacy')
    need(profile in ('legacy', 'expanded48-v1'), 'Unknown study profile')
    need(payload['seed'] == (20260912 if profile == 'expanded48-v1' else 20260911) and payload['replications'] == 2000, 'Statistical declaration differs')
    if profile == 'expanded48-v1':
        need(payload['study_scope']['target_event'] == 'zeta_2020' and payload['study_scope']['variant_start'] == 30000, 'Expanded figure study differs')
    need(payload['method_difference'] == 'method minus named reference; tail differences are differences of CVaRs', 'Unknown contrast semantics')
    files = {}
    for source in payload['outputs']:
        name = Path(source['path']).name
        need(name not in files and name in ('items.csv', 'summary.csv', 'curves_by_item.csv', 'curves_mean.csv'), 'Unknown/duplicate summary output')
        local = directory / name
        actual = reference(local)
        need(actual['sha256'] == source['sha256'] and actual['size_bytes'] == source['size_bytes'], 'Summary output bytes changed: ' + name)
        files[name] = dict(original_reference=source, local_reference=actual)
    need(set(files) == {'items.csv', 'summary.csv', 'curves_by_item.csv', 'curves_mean.csv'}, 'Incomplete summary outputs')
    def rows(name):
        with (directory / name).open(newline='') as stream:
            return list(csv.DictReader(stream))
    table = rows('summary.csv')
    index = {}
    for line, row in enumerate(table, 2):
        key = tuple(row[k] for k in ('regime', 'metric', 'statistic', 'method', 'reference_method'))
        need(key not in index, 'Duplicate statistical row')
        interval(row); interval(row, True)
        index[key] = dict(row, csv_line=line)
    curves = {}
    for line, row in enumerate(rows('curves_mean.csv'), 2):
        key = (row['regime'], row['method'], row['quantity'], int(row['period']))
        need(key not in curves and key[3] in range(1, 25), 'Duplicate/foreign curve period')
        value = number(row['mean']); missing = int(row['na_count'])
        need(0 <= missing <= 64 and (value is None) == (missing > 0), 'Curve NA propagation differs')
        need(abs(float(row['na_rate']) - missing / 64) < 1e-12, 'Curve NA rate differs')
        curves[key] = dict(row, csv_line=line)
    return payload, ref, files, index, curves


def generate(summary_directory, expected_sha256, output_directory):
    payload, source, files, table, curves = load_summary(summary_directory, expected_sha256)
    output = Path(output_directory).resolve()
    need(not output.exists(), 'Figure output directory must be new')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    need(matplotlib.__version__ == '3.10.6', 'Use the declared reporting matplotlib==3.10.6 environment')
    plt.rcParams.update({'font.family': 'DejaVu Serif', 'font.size': 8, 'axes.labelsize': 8,
        'axes.titlesize': 9, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6,
        'lines.linewidth': 1.2, 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.dpi': 240, 'axes.formatter.useoffset': False})
    output.mkdir(parents=True)
    figures = []; used = []; na = []
    def row(regime, metric, statistic, method):
        key = (regime, metric, statistic, method, 'EA_DRO')
        need(key in table, 'Missing figure summary row: ' + str(key))
        r = table[key]
        used.append(dict(file='summary.csv', csv_line=r['csv_line'], key=list(key)))
        return r
    def save(fig, name, caption):
        for extension in ('pdf', 'png'):
            fig.savefig(output / (name + '.' + extension), bbox_inches='tight', facecolor='white')
        plt.close(fig)
        figures.append(dict(figure_id=name, caption=caption, files=[reference(output / (name + '.' + e)) for e in ('pdf', 'png')]))
    def v_ci(ax, x, r, scale=1.):
        point, lo, hi = interval(r)
        if lo is not None:
            ax.plot([x, x], [lo*scale, hi*scale], color='black', lw=.7, zorder=5)
            ax.plot([x-.07, x+.07], [lo*scale]*2, color='black', lw=.7)
            ax.plot([x-.07, x+.07], [hi*scale]*2, color='black', lw=.7)
        if point is None or lo is None:
            text = 'NA' if point is None else 'CI NA'
            ax.text(x, .97, text, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=6)
            na.append(dict(csv_line=r['csv_line'], field=text))
    def forest(ax, entries, metric, statistic, *, difference=False, scale=1., xlabel='', proportion=False):
        plotted=[]; labels=[]
        for y, (regime, method) in enumerate(entries):
            r = row(regime, metric, statistic, method)
            point, lo, hi = interval(r, difference)
            labels.append(regime + '  ' + LABEL[method])
            if point is None:
                ax.text(.97, y, 'NA', transform=ax.get_yaxis_transform(), ha='right', va='center')
                na.append(dict(csv_line=r['csv_line'], field='difference' if difference else 'estimate'))
                continue
            ax.plot(point*scale, y, 'o', ms=3.5, color=COLOR[method], mfc=COLOR[method] if lo is not None else 'none')
            plotted.append(point*scale)
            if lo is not None:
                ax.plot([lo*scale, hi*scale], [y, y], color=COLOR[method], lw=1.)
                ax.plot([lo*scale, hi*scale], [y, y], '|', color=COLOR[method], ms=4)
                plotted.extend([lo*scale, hi*scale])
            else:
                labels[-1] += ' (CI NA)'; na.append(dict(csv_line=r['csv_line'], field='interval'))
        ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
        ax.set_xlabel(xlabel); ax.grid(axis='x', alpha=.18)
        if difference:
            span = max([.1 if proportion else 1., *[abs(v)*1.12 for v in plotted]])
            ax.set_xlim(-span, span); ax.axvline(0, color='#777777', lw=.7, ls='--')
        elif proportion:
            ax.set_xlim(0, 100)
        else:
            ax.set_xlim(0, max([1., *[v*1.08 for v in plotted]]))

    # All methods and original costs are retained in declared order.
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.1), gridspec_kw={'width_ratios': [1.4, 1]}, layout='constrained')
    ax, service = axes
    palette = ('#88CCEE', '#DDCC77', '#4477AA', '#CC6677', '#228833', '#AA3377')
    for j, method in enumerate(METHODS):
        components = [number(row('R1', 'costs.' + key, 'mean', method)['estimate']) for key, _ in COMPONENTS]
        total = row('R1', 'costs.total_DSO_usd', 'mean', method)
        if any(v is None for v in components):
            ax.text(j, .97, 'NA', transform=ax.get_xaxis_transform(), ha='center', va='top')
        else:
            point = number(total['estimate'])
            need(point is not None and abs(sum(components)-point) <= 1e-4+1e-7*max(1., abs(point)), 'Six components do not reconstruct mean DSO total')
            positive=negative=0.
            for i, (value, (_, label)) in enumerate(zip(components, COMPONENTS)):
                bottom = positive if value >= 0 else negative
                ax.bar(j, value/1000., bottom=bottom, width=.65, color=palette[i], edgecolor='white', linewidth=.3, label=label if j == 0 else None)
                if value >= 0: positive += value/1000.
                else: negative += value/1000.
        v_ci(ax, j, total, .001)
        r = row('R1', 'metrics.CLSR', 'mean', method); value = number(r['estimate'])
        if value is not None: service.bar(j, 100*value, color=COLOR[method], width=.65)
        v_ci(service, j, r, 100.)
    ax.axhline(0, color='#777777', lw=.5); ax.set_ylabel('Mean DSO cost (thousand USD)')
    ax.set_title('(a) R1 cost composition'); ax.legend(ncol=3, loc='upper left', fontsize=6)
    lower, upper = ax.get_ylim(); ax.set_ylim(lower, max(1., upper) * 1.18)
    service.set_ylim(0, 100); service.set_ylabel('Mean CLSR (%)'); service.set_title('(b) R1 critical-load service')
    for a in axes:
        a.set_xticks(range(len(METHODS)), [LABEL[m] for m in METHODS], rotation=35, ha='right'); a.grid(axis='y', alpha=.15); a.set_axisbelow(True)
    save(fig, 'case0_01_r1_cost_service', 'R1 means across 64 paired development samples. Costs retain all six DSO components; whiskers show 95% within-base bootstrap intervals for total mean cost and mean CLSR, not component-wise intervals. Undefined values remain NA. CLSR uses the full 0–100% scale.')

    fig, axes = plt.subplots(3, 1, figsize=(7.15, 4.65), sharex=True, layout='constrained')
    timing = METHODS if payload.get('study_scope',{}).get('profile') == 'expanded48-v1' else ('EA_DRO', 'DSP', 'OR_DRO', 'ARO', 'ZERO')
    for ax, quantity, label in zip(axes, ('certified', 'dispatched', 'CLSR'), ('Certified reserve (kW)', 'Dispatched power (kW)', 'CLSR (%)')):
        for method in timing:
            values=[]
            for period in range(1, 25):
                r=curves[('R1', method, quantity, period)]
                used.append(dict(file='curves_mean.csv', csv_line=r['csv_line'], key=['R1', method, quantity, period]))
                value=number(r['mean']); values.append(float('nan') if value is None else value*(100 if quantity == 'CLSR' else 1))
                if value is None: na.append(dict(file='curves_mean.csv', csv_line=r['csv_line'], field='mean'))
            ax.step(range(25), values+[values[-1]], where='post', color=COLOR[method], linestyle={'EA_DRO':'-', 'DSP':'--', 'OR_DRO':'-.', 'SAA':(0,(2,1)), 'ARO':(0,(4,1,1,1)), 'ZERO':':'}[method], label=LABEL[method])
        ax.set_ylabel(label); ax.grid(alpha=.15)
        if quantity == 'CLSR': ax.set_ylim(0, 100)
        else: ax.set_ylim(bottom=0)
    axes[0].legend(ncol=5, loc='upper center'); axes[0].set_ylim(top=axes[0].get_ylim()[1]*1.22)
    axes[0].set_title('R1 execution profiles: means over all 64 paired samples')
    axes[-1].set_xlim(0, 24); axes[-1].set_xticks(range(0, 25, 3)); axes[-1].set_xlabel('Hours after planned reserve-service start; period t spans [t−1, t)')
    save(fig, 'case0_02_r1_time_profiles', 'R1 hourly means for EA-DRO, delayed execution (DSP), service-start replanning (OR-DRO), ARO, and zero procurement. Each plotted mean uses all 64 paired samples. Missing hourly means appear as gaps, not zeros. Curves are descriptive averages without uncertainty bands; reserve and actual discharge remain separate quantities. The expanded study includes all six methods in this timing panel; the legacy panel retains its original five-method scope.')

    absolute = [(regime, method) for regime in REGIMES for method in RISK]
    contrasts = [(regime, method) for regime in REGIMES for method in ('SAA', 'ARO')]
    for service_plot in (False, True):
        fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.65), layout='constrained')
        metric = 'metrics.CLSR' if service_plot else 'costs.total_DSO_usd'
        tail = 'worst_decile_clsr' if service_plot else 'cvar95'
        forest(axes[0], absolute, metric, 'mean', scale=100. if service_plot else .001,
            xlabel='Mean CLSR (%)' if service_plot else 'Mean cost (thousand USD)', proportion=service_plot)
        forest(axes[1], contrasts, metric, 'mean', difference=True, scale=100. if service_plot else 1.,
            xlabel='Δ mean CLSR (pp)' if service_plot else 'Δ mean cost (USD)', proportion=service_plot)
        forest(axes[2], contrasts, metric, tail, difference=True, scale=100. if service_plot else 1.,
            xlabel='Δ worst-decile CLSR (pp)' if service_plot else 'Δ CVaR95 cost (USD)', proportion=service_plot)
        for a, title in zip(axes, ('(a) Absolute means', '(b) Method − EA-DRO', '(c) Tail statistic − EA-DRO')): a.set_title(title, fontsize=8)
        name='case0_04_risk_service' if service_plot else 'case0_03_risk_cost'
        tail_text='lowest-10% service mean (6.4 observations of probability mass)' if service_plot else 'highest-5% cost mean (3.2 observations of probability mass)'
        save(fig, name, 'Three-regime risk-method comparison on 64 paired samples per regime. Whiskers are 95% percentile bootstrap intervals from 2,000 matched within-base resamples, conditional on eight fixed weather templates and the single '+('Zeta' if payload.get('study_scope',{}).get('profile')=='expanded48-v1' else 'Michael')+' target. Contrasts are method minus EA-DRO within the same regime. Tail contrasts are differences of the '+tail_text+', not tail means of sample-wise differences. Hollow points denote undefined intervals; missing estimates are NA. Difference axes include zero and span at least ±'+('0.1 percentage points' if service_plot else '1 USD')+' to avoid enlarging numerical zeros.')

    index=dict(kind='CASE0_AUDITED_SUMMARY_FIGURES_V1',summary_reference=source,
        original_audit_reference=payload['audit_reference'],input_files=files,
        script_reference=reference(__file__),python_version=platform.python_version(),matplotlib_version=matplotlib.__version__,
        numpy_version=__import__('numpy').__version__,solver_calls=0,statistics_recomputed=False,
        inference_scope=payload['inference_scope'],statistical_seed=payload['seed'],replications=payload['replications'],
        figures=figures,source_rows=list({json.dumps(r,sort_keys=True):r for r in used}.values()),undefined_plot_entries=na,
        scientific_acceptance=False,formal_OOS=False,Case1_acceptance=False)
    (output/'figures.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    (output/'captions.md').write_text('\n\n'.join('**'+f['figure_id']+'**\n\n'+f['caption'] for f in figures)+'\n')
    return index


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary-dir', required=True)
    parser.add_argument('--summary-sha256', required=True)
    parser.add_argument('--output', required=True)
    args=parser.parse_args()
    generate(args.summary_dir, args.summary_sha256, args.output)


if __name__ == '__main__':
    main()

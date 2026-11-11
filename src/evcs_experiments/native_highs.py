"""Pinned SciPy-bundled HiGHS interface with a checked feasible MIP start.

No new optimizer or numerical dependency is installed. This adapter is explicit
because scipy.optimize.milp does not expose setSolution or detailed info when
its no-incumbent early-return path is taken.
"""
import hashlib
import time
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._highspy import _core as h
from scipy.sparse import csc_matrix


def solve_with_start(c, integrality, lower, upper, matrix, row_lower, row_upper,
                     start, options, log_path=None):
    began=time.monotonic();a=csc_matrix(matrix);a.sort_indices()
    v=np.asarray(start,dtype=float)
    if v.shape!=np.asarray(c).shape or not np.isfinite(v).all():raise ValueError('Invalid MIP start shape')
    activity=a@v
    violation=max(float(np.max(np.maximum(row_lower-activity,activity-row_upper),initial=0)),
                  float(np.max(np.maximum(lower-v,v-upper),initial=0)),
                  float(np.max(abs(v[np.asarray(integrality)==1]-np.rint(v[np.asarray(integrality)==1])),initial=0)))
    if violation>1e-8:raise ValueError('MIP start violates original/reduced constraints')
    highs=h._Highs()
    effective=dict(options)
    effective.update(output_flag=log_path is not None,log_to_console=False)
    if log_path is not None:
        from pathlib import Path
        log_path=Path(log_path);log_path.parent.mkdir(parents=True,exist_ok=True)
        if log_path.exists():raise ValueError('Refusing to overwrite native solver log')
        effective['log_file']=str(log_path)
    for name,value in effective.items():
        if highs.setOptionValue(name,value)==h.HighsStatus.kError:raise ValueError('HiGHS rejected option '+name)
    lp=h.HighsLp();lp.num_col_=len(c);lp.num_row_=len(row_lower)
    lp.col_cost_=np.asarray(c);lp.col_lower_=np.asarray(lower);lp.col_upper_=np.asarray(upper)
    lp.row_lower_=np.asarray(row_lower);lp.row_upper_=np.asarray(row_upper)
    lp.integrality_=[h.HighsVarType(int(i)) for i in integrality]
    lp.a_matrix_.num_col_=len(c);lp.a_matrix_.num_row_=len(row_lower);lp.a_matrix_.format_=h.MatrixFormat.kColwise
    lp.a_matrix_.start_=a.indptr;lp.a_matrix_.index_=a.indices;lp.a_matrix_.value_=a.data
    if highs.passModel(lp)==h.HighsStatus.kError:raise ValueError('HiGHS rejected model')
    seed=h.HighsSolution();seed.col_value=v;seed.value_valid=True
    start_status=highs.setSolution(seed)
    if start_status==h.HighsStatus.kError:raise ValueError('HiGHS rejected feasible start')
    remaining=max(1e-6,float(options['time_limit'])-(time.monotonic()-began))
    highs.setOptionValue('time_limit',remaining)
    run_status=highs.run();ms=highs.getModelStatus();info=highs.getInfo();sol=highs.getSolution()
    status={h.HighsModelStatus.kOptimal:0,h.HighsModelStatus.kTimeLimit:1,h.HighsModelStatus.kIterationLimit:1,
            h.HighsModelStatus.kSolutionLimit:1,h.HighsModelStatus.kInfeasible:2,h.HighsModelStatus.kUnbounded:3}.get(ms,4)
    if run_status==h.HighsStatus.kError:status=4
    x=None;fun=None
    if info.valid and sol.value_valid and info.primal_solution_status==2 and status in (0,1):
        x=np.asarray(sol.col_value);fun=float(info.objective_function_value)
    bound=None;gap=None;nodes=None
    if info.valid and status in (0,1):
        if np.isfinite(info.mip_dual_bound):bound=float(info.mip_dual_bound)
        if np.isfinite(info.mip_gap):gap=float(info.mip_gap)
        nodes=int(info.mip_node_count)
    return OptimizeResult(x=x,fun=fun,status=status,success=status==0,
        message='HiGHS native: '+highs.modelStatusToString(ms),mip_dual_bound=bound,mip_gap=gap,mip_node_count=nodes,
        native_record={'backend':'scipy.optimize._highspy._core._Highs','highs_version':highs.version(),
          'raw_model_status':int(ms),'run_status':int(run_status),'info_valid':bool(info.valid),
          'primal_solution_status':int(info.primal_solution_status),'solver_run_seconds':highs.getRunTime(),
          'simplex_iterations':int(info.simplex_iteration_count),'ipm_iterations':int(info.ipm_iteration_count),
          'start_status':int(start_status),'start_objective_without_constant_usd':float(np.asarray(c)@v),
          'start_sha256':hashlib.sha256(v.astype('<f8').tobytes()).hexdigest(),'start_max_violation':violation,
          'start_is_not_fixed_decision':True,'log_path':None if log_path is None else str(log_path),
          'effective_run_time_limit_seconds':remaining})

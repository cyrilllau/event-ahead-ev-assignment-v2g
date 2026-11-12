"""Run unchanged portable scientific regression cases through installed modules."""
import subprocess,sys
names=('probability_check','probability_phase1','random_streams','first_stage','recourse','native_gurobi','zero_presolve','native_highs')
for name in names:
 subprocess.run([sys.executable,'-m','unittest','discover','-s','tests/scientific','-p',f'test_{name}.py','-v'],check=True)

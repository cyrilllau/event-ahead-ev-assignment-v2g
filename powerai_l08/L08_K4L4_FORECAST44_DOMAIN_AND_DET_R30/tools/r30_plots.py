from pathlib import Path
import os,json,csv
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];out=R/'09_DELIVERY_REPORT';d=R/'05_SELECTED_PLAN_AND_CERTIFICATES/FORECAST44'
a=np.load(d/'PRIMAL.npz');v=json.loads((d/'VEHICLES.json').read_text());tr=json.loads((d/'SOC.json').read_text());t=np.arange(96)/4
fig,axs=plt.subplots(3,1,figsize=(13,10),layout='constrained')
axs[0].imshow(a['outage_current'].T,aspect='auto',extent=[0,24,32.5,.5],interpolation='nearest',cmap='Greys',vmin=0,vmax=1);axs[0].set_ylabel('Closed line ID');axs[0].set_title('Forecast44: recovered current outages, 96 operating slots')
for i,x in enumerate(v):
 axs[1].plot([x['d_h'],x['actual_arrival']],[i,i],color='gray',lw=2)
 axs[1].plot([x['K_h'],x['J_h']],[i,i],color='steelblue',lw=7)
 p=a['p_dis'][x['e']].sum(0);slots=np.flatnonzero(p>1e-6)
 for slot in slots:axs[1].plot([slot/4,(slot+1)/4],[i,i],color='darkorange',lw=3)
axs[1].set_yticks(range(len(v)),[x['ev']+' / B'+str(x['bus']) for x in v]);axs[1].set_title('Gray: travel; blue: reserve; orange: actual V2G (no power incident gate)')
axs[2].step(np.r_[t,24],np.r_[a['shed'].sum(1),a['shed'][-1].sum()],where='post',label='Remaining unserved kW');axs[2].step(np.r_[t,24],np.r_[a['p_dis'].sum((0,1)),a['p_dis'][:,:,-1].sum()],where='post',label='Actual V2G kW');axs[2].legend()
for ax in axs:ax.set_xlim(0,24);ax.set_xticks(np.arange(25));ax.grid(alpha=.15);ax.set_xlabel('Hour')
for ext in ['png','svg']:fig.savefig(out/('FORECAST44_DET_WINDOWS.'+ext),dpi=180)
plt.close(fig);fig,ax=plt.subplots(figsize=(12,5),layout='constrained')
for x in tr:ax.plot(x['times_h'],x['soc_pct'],label=x['ev'])
ax.axhline(20,color='black',ls='--',label='Release lower bound');ax.set(xlim=(0,24),xlabel='Hour',ylabel='SoC (%)',title='Actual arrival-to-J battery trajectory; auxiliary deducted continuously');ax.legend(ncol=4,fontsize=8);ax.grid(alpha=.2)
for ext in ['png','svg']:fig.savefig(out/('DET_SOC.'+ext),dpi=180)
np.savez_compressed(out/'PLOT_DATA.npz',t=t,remaining_shed_kw=a['shed'].sum(1),v2g_kw=a['p_dis'].sum((0,1)),outage_current=a['outage_current'])

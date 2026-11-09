"""Rebuild the three editorially revised V10 figures from existing evidence.

Run with the bundled Python runtime. Supplemental plotting dependencies are local
to this task; all patient data are read-only and no patient-level outputs are saved.
"""
from pathlib import Path
import sys, json, math

HERE = Path(__file__).resolve().parents[2]
sys.path.append(str(HERE / 'tmp/v10_editorial/plot_packages'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator

OUT = HERE / 'figures V10'
OUT.mkdir(exist_ok=True)
PROJECT = HERE.parent
BLUE, RED, GREY = '#2369A8', '#BB4E3B', '#737F86'
plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':10.5,
    'axes.labelsize':10.5, 'axes.titlesize':11.2,
    'xtick.labelsize':9.5, 'ytick.labelsize':9.5,
    'legend.fontsize':9, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.linewidth':0.8,
    'svg.fonttype':'none', 'savefig.facecolor':'white',
})

def save(fig, stem):
    fig.savefig(OUT / (stem+'.png'), dpi=400, bbox_inches='tight', pad_inches=.13)
    fig.savefig(OUT / (stem+'.svg'), bbox_inches='tight', pad_inches=.13)
    plt.close(fig)

def panel(ax, letter, title):
    ax.set_title(f'{letter}  {title}', loc='left', fontweight='bold', pad=10)

# Figure 1: preserve the original counts while giving every box canvas space.
fig, ax = plt.subplots(figsize=(7.4,9.0))
fig.subplots_adjust(left=.025,right=.975,top=.99,bottom=.025)
ax.set(xlim=(0,10),ylim=(0,12)); ax.axis('off')

def box(x,y,w,h,text,kind='included',fs=9):
    colours = {'included':('#F0F4F8','#52718F'),
               'excluded':('#FAF2F0','#A45F54'),
               'partition':('#EDF4EF','#54765C')}
    fc,ec=colours[kind]
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
                 boxstyle='round,pad=0.02,rounding_size=0.035',
                 facecolor=fc,edgecolor=ec,linewidth=1))
    ax.text(x,y,text,ha='center',va='center',fontsize=fs,linespacing=1.28)

def arrow(x1,y1,x2,y2,excluded=False):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops={'arrowstyle':'-|>','lw':1,
        'color':'#A45F54' if excluded else '#3D454B',
        'linestyle':'--' if excluded else '-'})

X, R = 3.15, 8.15
box(X,11.5,5.6,.60,'Source extract\n382,537 patients',fs=9.5)
box(R,10.77,3.2,.64,'No HbA1c record\n65 patients','excluded',8.5)
box(X,10.0,5.6,.67,'At least one HbA1c record\n382,472 patients',fs=9.5)
box(R,9.03,3.2,1.13,'8,292 records excluded\n90 invalid timestamps\n7,618 outside 2–20%\n584 exact duplicates\n1,098 patients without a valid draw','excluded',8.0)
box(X,8.14,5.6,.80,'At least one valid HbA1c draw\n381,374 patients\n1,128,790 patient-day draws',fs=9.3)
box(R,7.1,3.2,.96,'No follow-up draw\n90–450 days after any index\n229,624 patients','excluded',8.4)
box(X,6.13,5.6,.80,'Eligible index–follow-up pairs\n151,750 patients\n569,201 pairs',fs=9.5)
box(R,4.97,3.2,1.27,'Outside primary analysis\n210,919 pairs; 12,914 patients\n79,489 pairs indexed 2024–2025\n131,430 pairs outside the\npatient’s assigned time window','excluded',8.0)
box(X,3.85,5.6,.80,'Primary temporal and patient-disjoint analysis\n138,836 patients\n358,282 pairs',fs=9.0)
for y1,y2 in [(11.18,10.36),(9.645,8.56),(7.72,6.55),(5.71,4.27)]:
    arrow(X,y1,X,y2)
for y in [10.77,9.03,7.1,4.97]:
    arrow(X,y,R-1.62,y,True)
for x,txt in [(1.75,'Training\nIndex 2011–2021\n94,933 patients\n298,918 pairs'),
              (5.0,'Validation\nIndex 2022\n20,995 patients\n28,470 pairs'),
              (8.25,'Test\nIndex 2023\n22,908 patients\n30,894 pairs')]:
    box(x,1.64,2.93,1.10,txt,'partition',8.7)
    arrow(X,3.43,x,2.21)
ax.text(5,.50,'Patients do not overlap across the three partitions.',
        ha='center',va='center',fontsize=9,color='#38414A')
save(fig,'Fig1_cohort_flow')

# Figure 4: regenerate the original histograms from the same raw cleaned draws.
files=[PROJECT/'Dataset/lab_tests'/s for s in ['HBA1C 11.csv','HBA1C 22.csv']]
cols=['PatientID','LAB_DATA_CHEMISTRY_RESULTS','DATE_TIME_SPECIMEN_TAKEN']
raw=pd.concat([pd.read_csv(p,encoding='utf-8-sig',usecols=cols) for p in files],ignore_index=True)
raw.columns=['pid','v','t'] if list(raw.columns)==cols else raw.columns
# Do not depend on input CSV column order.
if 'PatientID' in raw:
    raw=raw.rename(columns=dict(zip(cols,['pid','v','t'])))
raw['v']=pd.to_numeric(raw['v'],errors='coerce')
raw['t']=pd.to_datetime(raw['t'],errors='coerce',format='%m/%d/%Y %H:%M')
d=raw.dropna(subset=['t']).loc[lambda x:x.v.between(2,20)]
d=d.drop_duplicates(subset=['pid','t','v']).copy()
assert len(d)==1131455, len(d)
d['t']=d.t.dt.normalize()
d=d.groupby(['pid','t'],as_index=False).v.mean().sort_values(['pid','t'])
assert len(d)==1128790, len(d)
g=d.groupby('pid',sort=False)
pat=pd.DataFrame({'n':g.size(),'baseline':g.v.first(),'final':g.v.last()})
pat['delta']=pat['final']-pat['baseline']
dz=pat.loc[(pat.n>=2)&(pat.delta!=0)]
assert len(pat)==381374 and len(dz)==190697
keys=['Controlled','Slightly','Moderately','Severely']
b=pd.cut(dz.baseline,[-np.inf,7,8,9,np.inf],right=False,labels=keys)
f=pd.cut(dz.final,[-np.inf,7,8,9,np.inf],right=False,labels=keys)
ct=pd.crosstab(b,f).loc[keys,keys].to_numpy()
assert np.array_equal(ct,np.array([[116701,7706,2972,2930],[6576,4665,3143,2954],
                                 [2865,3200,2872,3924],[5089,4820,5573,14707]]))
rp=ct/ct.sum(axis=1)[:,None]*100
attain=ct[:,0]/ct.sum(axis=1)
n=ct.sum(axis=1); z=1.959963984540054
centre=(attain+z*z/(2*n))/(1+z*z/n)
half=z*np.sqrt(attain*(1-attain)/n+z*z/(4*n*n))/(1+z*z/n)
lower=(centre-half)*100; upper=(centre+half)*100
labels=['<7.0','7.0–<8.0','8.0–<9.0','≥9.0']
fig, axs=plt.subplots(2,2,figsize=(8.0,7.25),gridspec_kw={'height_ratios':[.90,1.1]})
fig.subplots_adjust(left=.095,right=.975,top=.95,bottom=.105,wspace=.34,hspace=.56)
ax=axs[0,0]
lo,hi=pat.baseline.quantile([.005,.995]); ax.hist(pat.baseline,range=(lo,hi),bins=60,color=BLUE,density=True)
ax.set(xlabel='Baseline HbA1c (%)',ylabel='Density'); panel(ax,'A','Baseline HbA1c')
ax.text(.97,.93,'N = 381,374',transform=ax.transAxes,ha='right',va='top',fontsize=9)
ax=axs[0,1]
lo,hi=dz.delta.quantile([.005,.995]); ax.hist(dz.delta,range=(lo,hi),bins=80,color=BLUE,density=True)
ax.axvline(0,ls='--',c='#333333',lw=.9)
ax.set(xlabel='Change in HbA1c (percentage points)',ylabel='Density'); panel(ax,'B','Within-patient change')
ax.text(.97,.93,'N = 190,697',transform=ax.transAxes,ha='right',va='top',fontsize=9)
ax=axs[1,0]
im=ax.imshow(rp,cmap='Blues',vmin=0,vmax=100,aspect='auto')
ax.set_xticks(range(4),labels,fontsize=8.5); ax.set_yticks(range(4),labels,fontsize=8.5)
ax.set(xlabel='Final HbA1c (%)',ylabel='Baseline HbA1c (%)')
for i in range(4):
    for j in range(4):
        ax.text(j,i,f'{rp[i,j]:.1f}',ha='center',va='center',fontsize=10,
                color='white' if rp[i,j]>57 else '#152432')
panel(ax,'C','Transition probabilities (%)')
for s in ax.spines.values(): s.set_visible(False)
ax=axs[1,1]
ys=np.arange(4)
ax.errorbar(attain*100,ys,xerr=np.vstack([attain*100-lower,upper-attain*100]),
            fmt='o',color=BLUE,ecolor='#333333',capsize=3,markersize=6,lw=1.2)
ax.set_yticks(ys,labels,fontsize=8.5); ax.invert_yaxis()
ax.set(xlim=(0,104),ylim=(3.5,-.5),xlabel='Final HbA1c <7.0% (patients, %)',ylabel='Baseline HbA1c (%)')
ax.xaxis.set_major_locator(MultipleLocator(25)); ax.grid(axis='x',color='#E5E8EA',zorder=0)
for i,y in enumerate(attain*100):
    ax.text(y+3.5,i,f'{y:.1f}%',va='center',fontsize=9)
panel(ax,'D','Target attainment')
save(fig,'Fig4_glycaemic_distributions_transitions')

# Figure 7: retain the existing robustness data, replace panel C with the full
# information-matched factorial comparison on the same 96-token window.
sens=pd.read_csv(PROJECT/'GPU_trans/results/sensitivity_complete.csv')
opt=pd.read_csv(PROJECT/'GPU_trans/results/transformer_optimisation.csv')
learn=pd.read_csv(PROJECT/'GPU_trans/results/learning_curve.csv')
fig,axs=plt.subplots(2,2,figsize=(8.15,7.50))
fig.subplots_adjust(left=.16,right=.98,top=.95,bottom=.15,wspace=.52,hspace=.58)
ax=axs[0,0]
conds=['primary (+/-0.5, 90-450, temporal)','secondary grouped-random split',
       'label threshold +/-0.3','label threshold +/-1.0','follow-up window 90-365 days']
ylabs=['Primary','Grouped random','Change ±0.3','Change ±1.0','90–365 days']
for name,label,c in [('Transformer (BEHRT, GPU)','Transformer',RED),('LightGBM','LightGBM',BLUE),('LogReg (baseline)','Logistic regression',GREY)]:
    vals=[float(sens.loc[(sens.condition==co)&(sens.model==name),'auroc_macro'].iloc[0]) for co in conds]
    ax.plot(vals,np.arange(5),'o-',color=c,label=label,lw=1.2,markersize=4.5)
ax.set_yticks(range(5),ylabs,fontsize=9); ax.invert_yaxis()
ax.set(xlim=(.625,.850),xlabel='Macro AUROC'); ax.grid(axis='x',color='#E5E8EA')
handles,legend_labels=ax.get_legend_handles_labels()
fig.legend(handles,legend_labels,loc='lower center',bbox_to_anchor=(.57,.005),
           ncol=3,fontsize=9,frameon=False,handlelength=2.0,columnspacing=1.4)
panel(ax,'A','Design choices')
ax=axs[0,1]
rows=[('capacity: hidden 384, 6 layers (11.1M params)','5.7× capacity'),('4 fine-tuning epochs (vs 2)','4 tuning epochs'),
      ('learning rate 5e-4 (vs 3e-4)','Rate 5×10⁻⁴'),('sequence length 128 (vs 96)','128 tokens'),
      ('4 MLM pretraining epochs (vs 2)','4 MLM epochs'),('learning rate 1e-4 (vs 3e-4)','Rate 1×10⁻⁴')]
vals=[float(opt.loc[opt.condition==co,'auroc_macro'].iloc[0]) for co,_ in rows]
ax.plot(vals,np.arange(len(rows)),'o',color=RED,markersize=5)
ax.set_yticks(range(len(rows)),[lab for _,lab in rows],fontsize=8.4); ax.invert_yaxis()
ax.axvline(.7264,ls=':',color=RED,lw=1.5,label='Primary transformer')
ax.axvline(.7612,ls='--',color=BLUE,lw=1.5,label='Primary LightGBM')
ax.set(xlim=(.720,.765),xlabel='Macro AUROC'); ax.grid(axis='x',color='#E5E8EA')
ax.legend(loc='lower right',fontsize=7.6,frameon=True,facecolor='white',edgecolor='white',framealpha=1)
panel(ax,'B','Transformer optimisation')
ax=axs[1,0]
ax.plot([0,1],[.7264,.7542],'o-',color=RED,lw=1.7,label='Transformer',markersize=6)
ax.plot([0,1],[.7122,.7619],'s-',color=BLUE,lw=1.7,label='LightGBM',markersize=5.5)
for x,y,txt,dy in [(0,.7264,'0.726',.003),(0,.7122,'0.712',-.0055),
                   (1,.7542,'0.754',-.0055),(1,.7619,'0.762',.003)]:
    ax.text(x,y+dy,txt,ha='center',va='center',fontsize=9.5)
ax.set_xticks([0,1],['Binned tokens','Binned tokens\n+ numeric covariates'],fontsize=9)
ax.set(xlim=(-.16,1.16),ylim=(.701,.775),ylabel='Macro AUROC')
ax.grid(axis='y',color='#E5E8EA')
panel(ax,'C','Matched information')
ax=axs[1,1]
for name,label,c in [('LightGBM','LightGBM',BLUE),('Transformer (BEHRT, GPU)','Transformer',RED)]:
    rows=learn.loc[learn.model==name].sort_values('fraction')
    ax.plot(rows.train_instances,rows.auroc_macro,'o-',c=c,label=label,lw=1.5,markersize=4.5)
ax.set_xscale('log'); ax.set_xticks([29865,74862,149582,298918],['30k','75k','150k','299k'])
ax.minorticks_off(); ax.set(xlabel='Training instances (log scale)',ylabel='Macro AUROC')
ax.grid(color='#E5E8EA')
panel(ax,'D','Training data volume')
save(fig,'Fig7_robustness_factorial')

summary={'Fig4':{'cleaned_records':1131455,'patient_day_draws':1128790,
    'baseline_patients':len(pat),'longitudinal_patients':len(dz),
    'transition_counts':ct.tolist(),'attainment_pct':(attain*100).tolist()},
    'Fig7C':{'Transformer':[.7264,.7542],'LightGBM':[.7122,.7619],
              'history':'same 96-token window'}}
(OUT/'figure_data_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps({'output_folder':str(OUT),'validated_counts':summary['Fig4']},indent=2))

"""Refresh supplementary cohort cross-references and matched-seed ablation."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parents[2]
sys.path.append(str(HERE/'tmp/v10_editorial/plot_packages'))
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
OUT=HERE/'figures V10';OUT.mkdir(exist_ok=True)
BLUE,RED='#2369A8','#BB4E3B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.8,'svg.fonttype':'none','savefig.facecolor':'white'})
def save(fig,stem):
    fig.savefig(OUT/(stem+'.png'),dpi=400,bbox_inches='tight',pad_inches=.13)
    fig.savefig(OUT/(stem+'.svg'),bbox_inches='tight',pad_inches=.13)
    plt.close(fig)

fig,ax=plt.subplots(figsize=(8,6.25))
fig.subplots_adjust(left=.02,right=.98,top=.98,bottom=.03)
ax.set(xlim=(0,12),ylim=(0,9.4));ax.axis('off')
def box(x,y,w,h,text,fs=9.7,fc='#F0F4F8'):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,
        boxstyle='round,pad=0.02,rounding_size=0.04',facecolor=fc,edgecolor='#52718F',lw=1))
    ax.text(x,y,text,ha='center',va='center',fontsize=fs,linespacing=1.3)
def arrow(x1,y1,x2,y2):
    ax.annotate('',xy=(x2,y2),xytext=(x1,y1),
        arrowprops={'arrowstyle':'-|>','color':'#394650','lw':1.05})
box(6,8.65,8.2,1.0,'Descriptive source population\nAt least one non-zero HbA1c record\nN = 381,391',10.3)
for x in [2,6,10]:
    if x!=6:
        ax.plot([6,x],[7.94,7.94],color='#394650',lw=1.05)
    arrow(x,7.94,x,7.18)
arrow(6,8.13,6,7.98)
box(2,5.83,3.65,2.65,'Rule-based diabetes cohort\nICD-10 E11 or ICD-9 250,\nor a non-metformin\nglucose-lowering class\nMetformin alone without a\ndiabetes code is not sufficient\nn = 78,732 (20.6%)',9.3)
box(6,5.83,3.65,2.65,'Antidiabetic-exposed cohort\nAt least one dated dispensing\nin one of six therapeutic classes\nNo diagnosis code required\nIncludes uncoded\nmetformin users\nn = 102,465',9.3)
box(10,5.83,3.65,2.65,'Glycaemic cohort\nAt least two valid daily draws\nHbA1c within 2–20%\nn = 201,767\nNon-zero baseline-to-final change\nn = 190,697\nUnchanged values: n = 11,070',9.3)
for x in [2,6,10]:arrow(x,4.48,x,3.75)
box(2,2.85,3.65,1.7,'Characteristics and comorbidity\nAt least one group: 31,032\nTwo or more groups: 6,886\nSection 3.1; main Fig. 2\nAdditional characteristics: Fig. S10',9.2,fc='#EDF4EF')
box(6,2.85,3.65,1.7,'Medication use and\ncontinuity of supply\nSection 3.2; main Fig. 3\nTable S4 and Figs. S2–S4',9.5,fc='#EDF4EF')
box(10,2.85,3.65,1.7,'HbA1c distributions,\ntransitions and target attainment\nSection 3.3; main Fig. 4\nTable S26 and Figs. S5–S7',9.5,fc='#EDF4EF')
ax.text(6,1.22,'Cohorts overlap and are not mutually exclusive.',ha='center',va='center',fontsize=10)
ax.text(6,.62,'Both analytical arms use 1,128,790 cleaned patient-day HbA1c draws from 381,374 patients.\n'
        'Seventeen patients in the descriptive source population had no draw within the accepted range.',
        ha='center',va='center',fontsize=9.2,linespacing=1.35,color='#394650')
save(fig,'Figure S1')

res=HERE.parent/'GPU_trans/results'
cpu=pd.read_csv(res/'cpu_vs_gpu_seed_variability.csv').query("device == 'CPU'")
gpu=pd.read_csv(res/'seed_stability_5.csv')
weights=pd.read_csv(res/'cpu_vs_gpu_weight_divergence.csv')
ablation=pd.read_csv(res/'pretraining_ablation_fair.csv')
matched=ablation.loc[ablation.n_seeds.eq(3)].copy()
assert len(matched)==2
assert np.allclose(matched.auroc_macro,[.7264,.7259])
assert np.allclose(matched.auroc_worsened,[.6411,.6395])
assert np.array_equal(gpu.seed,[0,1,2,3,4])
assert np.allclose(gpu.auroc_macro,[.7274,.7250,.7151,.7241,.7273])
key=weights.tensor.str.contains('attention.self.key.bias',regex=False)
other=weights.loc[~key,'cosine_similarity'].to_numpy()
key_values=weights.loc[key,'cosine_similarity'].to_numpy()
assert len(weights)==73 and len(key_values)==4 and len(other)==69
fig=plt.figure(figsize=(7.5,7.3))
gs=fig.add_gridspec(2,2,height_ratios=[1,1.02],hspace=.55,wspace=.40)
fig.subplots_adjust(left=.10,right=.98,top=.95,bottom=.12)
ax=fig.add_subplot(gs[0,0])
ax.plot(cpu.seed,cpu.auroc_macro,'o-',c=BLUE,lw=1.4,ms=5,label=f'CPU, 3 seeds (SD {cpu.auroc_macro.std():.4f})')
ax.plot(gpu.seed,gpu.auroc_macro,'^-',c=RED,lw=1.4,ms=6,label=f'GPU, 5 seeds (SD {gpu.auroc_macro.std():.4f})')
ax.set(xticks=[0,1,2,3,4],xlabel='Fine-tuning seed',ylabel='Macro AUROC',ylim=(.7135,.7330))
ax.grid(color='#E5E8EA');ax.legend(loc='upper right',fontsize=7.8,frameon=True,facecolor='white',edgecolor='white',framealpha=1)
ax.set_title('A  Seed-to-seed variability',loc='left',fontweight='bold',pad=10)
ax=fig.add_subplot(gs[0,1])
data=[other,key_values]
bp=ax.boxplot(data,tick_labels=['Other tensors\n(n = 69)','Attention key bias\n(n = 4)'],
              patch_artist=True,widths=.50,showfliers=False)
for patch,c in zip(bp['boxes'],[BLUE,RED]):patch.set(facecolor=c,alpha=.35)
for med in bp['medians']:med.set(color='#333333',lw=1.2)
rng=np.random.default_rng(0)
for i,ys in enumerate(data,1):ax.scatter(rng.normal(i,.045,len(ys)),ys,s=12,c='#414D57',alpha=.55,zorder=3)
ax.set(ylabel='Cosine similarity between CPU and GPU',ylim=(.15,1.04))
ax.tick_params(axis='x',labelsize=8.3);ax.grid(axis='y',color='#E5E8EA')
ax.set_title('B  Parameter agreement',loc='left',fontweight='bold',pad=10)
ax=fig.add_subplot(gs[1,:])
x=np.arange(2);w=.25
ax.bar(x-w/2,matched.auroc_macro,w,color=BLUE,label='Macro AUROC')
ax.bar(x+w/2,matched.auroc_worsened,w,color=RED,label='Worsened-class AUROC')
for i,(m,v) in enumerate(zip(matched.auroc_macro,matched.auroc_worsened)):
    ax.text(i-w/2,m+.003,f'{m:.4f}',ha='center',va='bottom',fontsize=10)
    ax.text(i+w/2,v+.003,f'{v:.4f}',ha='center',va='bottom',fontsize=10)
ax.set_xticks(x,['With pretraining\n3-seed ensemble','Without pretraining\n3-seed ensemble'],fontsize=10)
ax.set(ylabel='AUROC',ylim=(.60,.78),xlim=(-.50,1.50))
ax.legend(loc='upper right',frameon=False,fontsize=8.5)
ax.set_title('C  Pretraining comparison with equal ensemble sizes',loc='left',fontweight='bold',pad=10)
ax.text(.015,.96,'Pretraining gain in macro AUROC: +0.0005',transform=ax.transAxes,
        va='top',ha='left',fontsize=9)
save(fig,'Figure S9')
summary={'FigureS1':{'source_population':381391,'rule_based_diabetes':78732,
        'exposed':102465,'at_least_two_valid_draws':201767,'nonzero_change':190697,
        'references':{'comorbidity':'Figure 2','medication':'Figure 3',
                      'glycaemic':'Figure 4 and Table S26'}},
        'FigureS9':{'cpu_seeds':cpu.seed.tolist(),'gpu_seed_auroc':gpu.auroc_macro.tolist(),
        'weight_tensors':73,'attention_key_bias_tensors':4,
        'pretraining_macro_auroc':matched.auroc_macro.tolist(),
        'pretraining_worsened_auroc':matched.auroc_worsened.tolist(),
        'ensembles':[3,3]}}
(OUT/'figure_s1_s9_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Rebuilt supplementary Figures S1 and S9; counts, references and equal three-seed ablation verified.')

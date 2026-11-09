"""Reformat V9 aggregate figures without changing estimates or fitting models."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parents[2]
sys.path.extend([str(HERE/'tmp/v10_editorial/plot_packages'),str(HERE/'tmp/v10_editorial/pdf_packages')])
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter,MultipleLocator
import pymupdf

OUT=HERE/'figures V10'; OUT.mkdir(exist_ok=True)
BLUE,RED,GREY='#2369A8','#BB4E3B','#737F86'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
    'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9,
    'legend.fontsize':8.5,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.8,'svg.fonttype':'none','savefig.facecolor':'white'})
def save(fig,stem):
    fig.savefig(OUT/(stem+'.png'),dpi=400,bbox_inches='tight',pad_inches=.13)
    fig.savefig(OUT/(stem+'.svg'),bbox_inches='tight',pad_inches=.13)
    plt.close(fig)
def panel(ax,letter,title):
    ax.set_title(f'{letter}  {title}',loc='left',fontweight='bold',pad=9)

# Figure 2: all values are copied from the existing descriptive figure and
# cross-checked against Tables S5/S25. The source matrix contains eight groups.
N=78732
names=['Hypertension','Dyslipidaemia','Vitamin D deficiency','Hypothyroidism',
       'Chronic kidney disease','Heart failure','Ischaemic heart disease',
       'Obesity','Cerebrovascular disease']
counts=np.array([24122,6162,3187,2860,983,686,513,306,107])
burden=np.array([47700,24146,5976,819,84,7,0])
mat=np.array([[24122,3572,1284,1157,443,338,213,146],
              [3572,6162,516,394,96,53,31,84],
              [1284,516,3187,252,46,31,18,46],
              [1157,394,252,2860,36,33,21,29],
              [443,96,46,36,983,58,9,4],
              [338,53,31,33,58,686,22,4],
              [213,31,18,21,9,22,513,4],
              [146,84,46,29,4,4,4,306]])
assert np.array_equal(mat,mat.T) and np.array_equal(mat.diagonal(),counts[:8])
assert burden.sum()==N and burden[3:].sum()==910
assert np.array_equal(mat[0,1:6],np.array([3572,1284,1157,443,338]))
abbr=['HTN','DYS','VDD','HYPO','CKD','HF','IHD','OB']
fig,axs=plt.subplots(2,2,figsize=(8.0,8.1),gridspec_kw={'width_ratios':[.95,1.25]})
fig.subplots_adjust(left=.10,right=.98,top=.92,bottom=.23,wspace=.80,hspace=.50)
fig.suptitle('Rule-based diabetes cohort',fontweight='bold',fontsize=12,y=.985)
ax=axs[0,0]
vals=[78732,302659]
bars=ax.bar([0,1],vals,color=[BLUE,'#A6B8C7'],width=.62)
ax.set_xticks([0,1],['Rule positive','Rule negative'])
ax.set(ylabel='Patients (n)',ylim=(0,355000))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:.0f}k'))
for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,v+8000,f'{v:,}',ha='center',fontsize=9.5)
ax.text(.04,.94,'N = 381,391\n20.64% rule positive',transform=ax.transAxes,
        ha='left',va='top',fontsize=9)
panel(ax,'A','Recorded diabetes')
ax=axs[0,1]
percent=counts/N*100
ax.barh(np.arange(9),percent,color=BLUE,height=.7)
ax.set_yticks(range(9),names,fontsize=8.4);ax.invert_yaxis()
ax.set(xlabel='Patients with a recorded code (%)',xlim=(0,43))
ax.xaxis.set_major_locator(MultipleLocator(10))
for i,(n,p) in enumerate(zip(counts,percent)):
    ax.text(p+.65,i,f'{n:,} ({p:.1f}%)',ha='left',va='center',fontsize=8.0)
panel(ax,'B','Coded comorbidity')
ax=axs[1,0]
ax.bar(range(7),burden,color=BLUE,width=.72)
ax.set_xticks(range(7),['0','1','2','3','4','5','6+'])
ax.set(xlabel='Mapped groups per patient',ylabel='Patients (n)',ylim=(0,55500),xlim=(-1.05,6.75))
ax.yaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:.0f}k'))
for i,v in enumerate(burden):ax.text(i,v+1200,f'{v:,}',ha='center',fontsize=8.5,rotation=0)
panel(ax,'C','Comorbidity burden')
ax=axs[1,1]
im=ax.imshow(mat,cmap='Blues',vmin=0,vmax=mat.max(),aspect='auto')
ax.set_xticks(range(8),abbr,fontsize=8.3,rotation=45,ha='right')
ax.set_yticks(range(8),abbr,fontsize=8.3)
for i in range(8):
    for j in range(8):
        ax.text(j,i,f'{mat[i,j]:,}',ha='center',va='center',fontsize=7.6,
                color='white' if mat[i,j]>13000 else '#142838')
for s in ax.spines.values():s.set_visible(False)
panel(ax,'D','Co-occurrence counts')
fig.text(.10,.048,'HTN, hypertension; DYS, dyslipidaemia; VDD, vitamin D deficiency; HYPO, hypothyroidism;\n'
    'CKD, chronic kidney disease; HF, heart failure; IHD, ischaemic heart disease; OB, obesity.\n'
    'Panels B–D: N = 78,732. Panel D shows the eight most frequently recorded groups.',
    fontsize=8.1,ha='left',va='bottom',linespacing=1.35)
save(fig,'Figure 2')

# Figure 5: extract the original raw ROC paths from the preserved vector figure.
# This is aggregate curve reuse, not model inference or a new analysis.
source_pdf=HERE.parent/'GPU_trans/figures_main/Fig2_performance.pdf'
with pymupdf.open(source_pdf) as doc:
    paths=[p for p in doc[0].get_drawings() if len(p['items'])>100]
assert len(paths)==9, len(paths)
colours={'Transformer':RED,'LightGBM':BLUE,'Logistic regression':GREY}
known={'Transformer':[.7877774898860768,.7504006789974828,.6410816982624832],
       'LightGBM':[.8101700874814768,.7717161778070118,.7015755759121027],
       'Logistic regression':[.7917410519023662,.7330832842253608,.5178882850352313]}
curves={}
for p in paths:
    r=p['rect']; colour=p['color']
    nm='Transformer' if colour[0]>.7 else 'LightGBM' if colour[0]<.2 else 'Logistic regression'
    k=0 if r.x0<300 and r.y0<200 else 1 if r.y0<200 else 2
    points=[]
    for item in p['items']:
        assert item[0]=='l',item[0]
        if not points:points.append((item[1].x,item[1].y))
        points.append((item[2].x,item[2].y))
    pts=np.array(points)
    x=(pts[:,0]-r.x0)/r.width
    y=(r.y1-pts[:,1])/r.height
    approx_auc=float(np.trapezoid(y,x))
    assert abs(approx_auc-known[nm][k])<.0004,(nm,k,approx_auc,known[nm][k])
    curves[(nm,k)]=(x,y)
dca=np.load(HERE/'_dca_v5.npy',allow_pickle=True).item()
th=dca['th'];prev=float(dca['prev'])
assert abs(prev-6015/30894)<1e-12
for nm,expected in [('LightGBM',.05781867029196608),('Transformer',.04203081504499256),('LogReg',.0210639606396064)]:
    assert abs(float(dca['curves'][nm][np.argmin(abs(th-.2))])-expected)<1e-10
fig,axs=plt.subplots(2,2,figsize=(7.45,7.15))
fig.subplots_adjust(left=.10,right=.98,top=.95,bottom=.10,wspace=.30,hspace=.35)
for k,cl in enumerate(['Improved','Stable','Worsened']):
    ax=axs.flat[k]
    for nm in ['LightGBM','Transformer','Logistic regression']:
        x,y=curves[(nm,k)]
        ax.plot(x,y,color=colours[nm],lw=1.5,label=f'{nm} ({known[nm][k]:.3f})')
    ax.plot([0,1],[0,1],color='#66717B',ls=':',lw=.9)
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='1 − specificity',ylabel='Sensitivity')
    ax.set_aspect('equal',adjustable='box')
    ax.legend(loc='lower right',fontsize=7.9,frameon=True,facecolor='white',edgecolor='white',framealpha=.92,
              handlelength=1.6,labelspacing=.35,borderpad=.25)
    panel(ax,'ABC'[k],cl)
ax=axs[1,1]
for key,nm in [('LightGBM','LightGBM'),('Transformer','Transformer'),('LogReg','Logistic regression')]:
    ax.plot(th,dca['curves'][key],color=colours[nm],lw=1.5,label=nm)
nb_all=prev-(1-prev)*th/(1-th)
ax.plot(th,nb_all,color='#4E7E46',ls='--',lw=1.1,label='Treat all')
ax.axhline(0,color='#383838',lw=.9,label='Treat none')
ax.set(xlim=(.05,.60),ylim=(-.06,.10),xlabel='Threshold probability',ylabel='Net benefit')
ax.legend(loc='upper right',fontsize=7.6,frameon=True,facecolor='white',edgecolor='white',framealpha=.92,
          handlelength=1.6,labelspacing=.30,borderpad=.25)
panel(ax,'D','Decision curve')
save(fig,'Figure 5')
summary={'Figure2':{'source':'Existing V9 figure and Tables S5/S25','cohort':N,
         'burden_counts':burden.tolist(),'cooccurrence_counts':mat.tolist(),
         'cooccurrence_groups':8},
         'Figure5':{'raw_roc_vector_source':str(source_pdf),
         'raw_roc_source_sha256':hashlib.sha256(source_pdf.read_bytes()).hexdigest(),
         'auroc':known,'dca_source':'_dca_v5.npy','dca_probability_scale':'vector-scaled',
         'no_new_model_fitting':True}}
(OUT/'figure_2_5_checks.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print('Rebuilt Figures 2 and 5; exact counts/estimates and stored DCA values verified.')

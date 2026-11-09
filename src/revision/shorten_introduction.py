from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from docx import Document
import json, re, shutil

ROOT=Path(__file__).resolve().parents[2]
target=ROOT/'Manuscript V10.docx'
before=target.read_bytes()
digest=sha256(before).hexdigest()
doc=Document(target)
paras=doc.paragraphs
start=next(i for i,p in enumerate(paras) if p.text.strip()=='1. Introduction')
end=next(i for i,p in enumerate(paras) if p.text.strip()=='2. Methods')
old=paras[start+1:end]
assert len(old)==6, 'The current Introduction has changed; inspect before editing.'
outside_before=[p._p.xml for p in paras[:start+1]+paras[end:]]
tables_before=[t._tbl.xml for t in doc.tables]
new=[
"Diabetes is an increasing burden in the Middle East and North Africa [1]. In Jordan, age-standardised prevalence among adults aged 25 years and older rose from 13.0% in 1994 to 23.7% in 2017 [2,3]. Sustained hyperglycaemia increases the risk of vascular complications [4], making HbA1c monitoring and timely treatment adjustment central to care. Guidance recommends individualised targets and incorporates cardiovascular and renal indications into drug selection [5,6,59].",
"Electronic health records (EHRs) can characterise treatment patterns and glycaemic outcomes while supporting risk prediction [7,8]. Jordan's Hakeem system, introduced in 2010, contains diagnoses, dispensing and laboratory data from the public healthcare network [9,10]. Rule-based identification and diagnosis mapping help define the recorded diabetes population [11,12,13]. Hakeem has already supported cardiovascular risk modelling [60]; examining its glycaemic data will also clarify the clinical information available for prediction.",
"Clinical transformers represent histories as ordered event sequences [14,15,16,17], whereas gradient-boosted trees remain strong comparators for EHR data [18,19,48,49,58]. Their relative performance may depend as much on input representation as on architecture [50,51]. Reviews highlight inconsistent preprocessing and benchmarking [52,62], so informative comparisons require matched numeric information and assessment of calibration and potential utility alongside discrimination [20].",
"Previous HbA1c studies have focused on poor control in a specialist cohort, short-term change in young people with type 1 diabetes, or response after drug initiation [53,54,55]. Prediction at routine measurement encounters in a broader tested population remains less well established. Here, index HbA1c may predict subsequent change through regression to the mean, making a simple baseline essential.",
"In this work, we will characterise the recorded diabetes population, coded comorbidity, medication use and glycaemic trajectories in Hakeem. We will develop and evaluate a BEHRT-style transformer, LightGBM and logistic regression to predict the direction of HbA1c change in patients with repeated measurements. We will compare model families using matched histories, with and without additional numeric covariates, to separate representation effects from model choice. Given differences in diabetes care by sex and age [2,3], we will also assess subgroup performance. These analyses are intended to inform follow-up prioritisation by clinicians and pharmacists."
]
def citations(text):
 vals=set()
 for group in re.findall(r'\[([\d,\s-]+)\]',text):
  for item in group.split(','):
   if '-' in item:
    a,b=map(int,item.split('-'));vals.update(range(a,b+1))
   elif item.strip():vals.add(int(item))
 return vals
# The user-saved copy contains damaged dash characters in three reference ranges.
# Expand those original ranges explicitly when checking reference preservation.
old_refs=citations('\n'.join(p.text for p in old))|set(range(11,14))|set(range(14,18))|set(range(53,56))
assert citations('\n'.join(new))==old_refs,(citations('\n'.join(new)),old_refs)
for p,text in zip(old,new):
 rpr=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
 p.clear();r=p.add_run(text)
 if rpr is not None:r._r.insert(0,rpr)
for p in old[len(new):]:p._element.getparent().remove(p._element)
now=doc.paragraphs
new_end=next(i for i,p in enumerate(now) if p.text.strip()=='2. Methods')
assert [p._p.xml for p in now[:start+1]+now[new_end:]]==outside_before
assert [t._tbl.xml for t in doc.tables]==tables_before
backup=ROOT/'tmp/v10_editorial/User edited V10 before introduction revision.docx'
backup.write_bytes(before)
staged=ROOT/'tmp/v10_editorial/Manuscript V10 introduction revision.docx'
doc.save(staged)
assert sha256(target.read_bytes()).hexdigest()==digest,'The manuscript changed during editing; do not overwrite it.'
shutil.copy2(staged,target)
reopened=Document(target)
words=sum(len(p.text.split()) for p in reopened.paragraphs)+sum(len(c.text.split()) for t in reopened.tables for row in t.rows for c in row.cells)
old_words=sum(len(p.text.split()) for p in Document(backup).paragraphs[start+1:end])
new_words=sum(len(p.split()) for p in new)
assert words<13000
report={'source_sha256':digest,'introduction_before_words':old_words,'introduction_after_words':new_words,'total_words':words,'outside_introduction_unchanged':True,'tables_unchanged':True,'images':len(reopened.inline_shapes),'last_paragraph':new[-1]}
(ROOT/'tmp/v10_editorial/introduction_revision_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

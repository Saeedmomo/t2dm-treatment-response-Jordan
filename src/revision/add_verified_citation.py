from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from docx import Document
import json,shutil

ROOT=Path(__file__).resolve().parents[2]
target=ROOT/'Manuscript V10.docx'
raw=target.read_bytes();digest=sha256(raw).hexdigest()
doc=Document(target)
intro=next(p for p in doc.paragraphs if p.text.startswith('Diabetes is an increasing burden'))
discussion=next(p for p in doc.paragraphs if p.text.startswith('This result is consistent with evidence'))
allowed={id(intro._p),id(discussion._p)}
unchanged=[p._p.xml for p in doc.paragraphs if id(p._p) not in allowed]
tables=[t._tbl.xml for t in doc.tables]
def replace(p,text):
 rpr=deepcopy(p.runs[0]._r.rPr) if p.runs and p.runs[0]._r.rPr is not None else None
 p.clear();r=p.add_run(text)
 if rpr is not None:r._r.insert(0,rpr)
replace(intro,intro.text.replace('making HbA1c monitoring','making glycated haemoglobin (HbA1c) monitoring'))
sentence='A related methodological example comes from virtual screening, where consensus integration of complementary screening scores improved compound prioritisation for selected protein targets [63].'
original=discussion.text
anchor='and with work examining how transformers should encode numeric values [51].'
assert anchor in original and '[63]' not in original
revised=original.replace(anchor,anchor+' '+sentence).replace('It also refines the recommendation','Our factorial analysis also refines the recommendation')
replace(discussion,revised)
assert [p._p.xml for p in doc.paragraphs if id(p._p) not in allowed]==unchanged
assert [t._tbl.xml for t in doc.tables]==tables
reference='63. Moshawih S, Bu ZH, Goh HP, Kifli N, Lee LH, Goh KW, et al. Consensus holistic virtual screening for drug discovery: a novel machine learning model approach. J Cheminform. 2024;16:62. doi:10.1186/s13321-024-00855-8.'
last=doc.paragraphs[-1]
assert last.text.startswith('62.')
new=doc.add_paragraph(style=last.style)
new._p.get_or_add_pPr()
if last._p.pPr is not None:
 new._p.remove(new._p.pPr);new._p.insert(0,deepcopy(last._p.pPr))
new.add_run(reference)
if last.runs and last.runs[0]._r.rPr is not None:new.runs[0]._r.insert(0,deepcopy(last.runs[0]._r.rPr))
backup=ROOT/'tmp/v10_editorial/V10 before verified citation.docx';backup.write_bytes(raw)
staged=ROOT/'tmp/v10_editorial/V10 citation revision.docx';doc.save(staged)
assert sha256(target.read_bytes()).hexdigest()==digest,'Concurrent manuscript edit detected'
shutil.copy2(staged,target)
paras=doc.paragraphs
i=next(i for i,p in enumerate(paras) if p.text=='1. Introduction')
j=next(i for i,p in enumerate(paras) if p.text=='2. Methods')
report={'introduction_words':sum(len(p.text.split()) for p in paras[i+1:j]),'manuscript_words':sum(len(p.text.split()) for p in paras)+sum(len(c.text.split()) for t in doc.tables for row in t.rows for c in row.cells),'reference':reference,'citation_sentence':sentence,'verified_url':'https://link.springer.com/article/10.1186/s13321-024-00855-8','other_paragraphs_unchanged':True}
assert report['manuscript_words']<13000
(ROOT/'tmp/v10_editorial/latest_v10_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))

from pathlib import Path
import sys, re
from copy import deepcopy
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT.parent/'GPU_trans'/'code'))
import md_to_docx as M
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

M.BASE=str(ROOT)
M.TOKEN=re.compile(r'(\*\*.+?\*\*|\*[^*]+?\*|\^[^\^]+?\^)')

def add_runs(par,text,size=None,base_italic=False):
 for piece in M.TOKEN.split(text):
  if not piece: continue
  bold=italic=sup=False
  if piece.startswith('**') and piece.endswith('**'): piece,bold=piece[2:-2],True
  elif piece.startswith('*') and piece.endswith('*') and len(piece)>2: piece,italic=piece[1:-1],True
  elif piece.startswith('^') and piece.endswith('^'): piece,sup=piece[1:-1],True
  piece=piece.replace('\\*','*')
  run=par.add_run(piece)
  run.font.name=M.BODY_FONT; run.font.size=size or M.BODY_SIZE
  run.bold=bold; run.italic=italic or base_italic; run.font.superscript=sup
  run.font.color.rgb=RGBColor(0,0,0)
 return par
M.add_runs=add_runs

def borders(table):
 pr=table._tbl.tblPr
 for old in list(pr.findall(qn('w:tblBorders'))): pr.remove(old)
 b=OxmlElement('w:tblBorders')
 for edge in ['top','left','bottom','right','insideH','insideV']:
  e=OxmlElement('w:'+edge)
  for k,v in dict(val='single',sz='4',color='D9D9D9').items():e.set(qn('w:'+k),v)
  b.append(e)
 pr.append(b)
M.set_table_borders=borders
M.set_row_bottom_border=lambda row:None
old_table=M.add_table
old_widths=M.column_widths
SUPPLEMENT=False
def table(doc,rows,prettify):
 t=old_table(doc,rows,prettify)
 for i,row in enumerate(t.rows):
  for c in row.cells:
   c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   pr=c._tc.get_or_add_tcPr()
   margins=OxmlElement('w:tcMar')
   for edge,val in [('top','70'),('bottom','70'),('left','80'),('right','80')]:
    e=OxmlElement('w:'+edge);e.set(qn('w:w'),val);e.set(qn('w:type'),'dxa');margins.append(e)
   pr.append(margins)
   if i==0:
    shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'EDEFF1');pr.append(shade)
   for p in c.paragraphs:
    p.paragraph_format.keep_with_next=(i==0)
    for r in p.runs:
     r.font.color.rgb=RGBColor(0,0,0)
     if SUPPLEMENT:r.font.size=Pt(9)
  trpr=row._tr.get_or_add_trPr();trpr.append(OxmlElement('w:cantSplit'))
 return t
M.add_table=table

def build(stem,supplement=False):
 global SUPPLEMENT
 SUPPLEMENT=supplement
 M.column_widths=lambda header,body,pt_size,total_in=M.MAX_IMG_IN:old_widths(header,body,9 if supplement else pt_size,9.8 if supplement else total_in)
 out=ROOT/(stem+'.docx')
 M.build(str(ROOT/(stem+'.md')),str(out),prettify_headers=supplement,break_before=('References',))
 d=Document(out)
 title=d.paragraphs[0]; title.style=d.styles['Title'];title.alignment=WD_ALIGN_PARAGRAPH.CENTER
 for r in title.runs:r.font.name='Times New Roman';r.font.size=Pt(16);r.bold=True;r.font.color.rgb=RGBColor(0,0,0)
 for name in ['Title','Subtitle','Heading 1','Heading 2','Heading 3','Heading 4']:
  d.styles[name].font.color.rgb=RGBColor(0,0,0)
  d.styles[name].font.name='Times New Roman'
  spr=d.styles[name].element.find(qn('w:pPr'))
  if spr is not None:
   for b in list(spr.findall(qn('w:pBdr'))):spr.remove(b)
 for sec in d.sections:
  ln=sec._sectPr.find(qn('w:lnNumType'))
  if ln is not None:ln.set(qn('w:distance'),'720')
 if 'Line Number' not in d.styles:d.styles.add_style('Line Number',WD_STYLE_TYPE.CHARACTER)
 d.styles['Line Number'].font.name='Times New Roman';d.styles['Line Number'].font.size=Pt(9)
 d.styles['Line Number'].font.color.rgb=RGBColor.from_string('666666')
 for p in d.paragraphs:
  p.paragraph_format.widow_control=True
  if p.style.name=='Heading 2':
   p.style=d.styles['Heading 1']
  elif p.style.name=='Heading 3':p.style=d.styles['Heading 2']
  if p.text=='Abstract' and not supplement:p.paragraph_format.page_break_before=True
  if p.text=='1. Introduction' and not supplement:p.paragraph_format.page_break_before=True
  if p.text=='References':p.paragraph_format.page_break_before=True
  if supplement and re.match(r'^Panel\s+[A-Z]',p.text):p.paragraph_format.keep_with_next=True
  # Strong captions remain with figures; table captions remain with tables.
  if p.text.startswith('Fig. '):
   p.style=d.styles['Caption'];p.paragraph_format.line_spacing=1
   p.paragraph_format.space_after=Pt(10)
   p.paragraph_format.keep_together=True
   for r in p.runs:r.italic=False;r.font.size=Pt(10)
  if p._p.xpath('.//w:drawing'):
   p.paragraph_format.keep_with_next=True
  # Keep explicit blank title-rule residue out of the file.
  if p.style.name=='Title':
   pr=p._p.get_or_add_pPr()
   for b in pr.findall(qn('w:pBdr')):pr.remove(b)
 if supplement:
  current_title=Document(ROOT/'Manuscript V10.docx').paragraphs[0].text
  title_line=d.paragraphs[1]
  if title_line.runs:
   title_line.runs[0].text=current_title
   for extra in title_line.runs[1:]:extra.text=''
  else:title_line.add_run(current_title)
  # Wide evidence tables use landscape pages; supporting figures return to portrait.
  heading=next(p for p in d.paragraphs if p.text=='Supplementary figures')
  boundary=OxmlElement('w:p');pr=OxmlElement('w:pPr')
  section=deepcopy(d.sections[0]._sectPr)
  size=section.find(qn('w:pgSz'));size.set(qn('w:w'),'16838');size.set(qn('w:h'),'11906');size.set(qn('w:orient'),'landscape')
  margins=section.find(qn('w:pgMar'))
  for edge in ['left','right','top','bottom']:margins.set(qn('w:'+edge),'1134')
  stype=section.find(qn('w:type'))
  if stype is None:stype=OxmlElement('w:type');section.insert(0,stype)
  stype.set(qn('w:val'),'nextPage')
  pr.append(section);boundary.append(pr);heading._p.addprevious(boundary)
  final=d.sections[-1]._sectPr
  ftype=final.find(qn('w:type'))
  if ftype is None:ftype=OxmlElement('w:type');final.insert(0,ftype)
  ftype.set(qn('w:val'),'nextPage')
  for p in d.paragraphs:
   if p.style.name=='Normal':
    p.paragraph_format.line_spacing=1.15
    for r in p.runs:r.font.size=Pt(11)
 # Make heading style formatting consistent with hierarchy.
 for name,size in [('Heading 1',13),('Heading 2',12)]:
  st=d.styles[name];st.font.size=Pt(size);st.font.bold=True
 # Readers should see publication terms, not pipeline identifiers.
 for t in d.tables:
  for row in t.rows:
   for c in row.cells:
    for p in c.paragraphs:
     for r in p.runs:
      r.text=r.text.replace('_',' ')
      if supplement and 'date range' not in row.cells[0].text.lower() and re.fullmatch(r'\d{4,}(?:\.0)?',r.text.strip()):r.text=f'{int(float(r.text)):,}'
 d.core_properties.title=d.paragraphs[0].text
 d.core_properties.subject='Revised manuscript' if not supplement else 'Supplementary material'
 d.save(out)
 words=sum(len(p.text.split()) for p in d.paragraphs)+sum(len(c.text.split()) for t in d.tables for row in t.rows for c in row.cells)
 print(stem,words,'words',len(d.tables),'tables',len(d.inline_shapes),'figures')
 if not supplement:assert words<13000,words
 return out

if __name__=='__main__':
 build('Manuscript V10')
 if (ROOT/'Supplementary material V10.md').exists():build('Supplementary material V10',True)

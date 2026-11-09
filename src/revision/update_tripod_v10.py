"""Fill the official checklist using verified V10 manuscript pagination.

The official form's wording and rules remain unchanged. Page references are
derived from the final rendered main PDF, never carried forward from V9. This
script must be run only after that final PDF exists and the root agent has
marked the PDF edit operation. It uses the bundled Python runtime and a
task-local PyMuPDF dependency.
"""
from pathlib import Path
import argparse, hashlib, json, re, sys, unicodedata

HERE=Path(__file__).resolve().parents[2]
sys.path.append(str(HERE/'tmp/v10_editorial/pdf_packages'))
import pymupdf as fitz

COL_X0,COL_X1=525.1,573.8
BAND_TOP={0:104.0,1:55.0}
BAND_BOTTOM=760.0
NOTE_RECT=fitz.Rect(26,785,526,830)
HDR_WORDS={'Reported','on','page'}
ITEM_RE=re.compile(r'^(\d{1,2}[a-g]?)$')
EXPECTED=['1','2','3a','3b','3c','4','5a','5b','6a','6b','6c','7','8a','8b','8c',
          '9a','9b','9c','10','11','12a','12b','12c','12d','12e','12f','12g','13',
          '14','15','16','17','18a','18b','18c','18d','18e','18f','19','20a',
          '20b','20c','21','22','23a','23b','24','25','26','27a','27b','27c']

def norm(text):
    text=unicodedata.normalize('NFKC',text).lower()
    text=text.replace('\u00ad','').replace('’',"'")
    return ' '.join(re.findall(r'[a-z0-9]+',text))

def plain(text):
    text=re.sub(r'!\[[^]]*\]\([^)]*\)','',text)
    text=re.sub(r'\[([^]]+)\]\([^)]*\)',r'\1',text)
    return re.sub(r'[*^`\\]','',text).strip()

def manuscript_anchors(md):
    sections={}
    for line in md.splitlines():
        m=re.match(r'^#{2,3}\s+(\d+(?:\.\d+)?)\.\s+(.+)',line)
        if m: sections[m.group(1)]=plain(m.group(2))
    required=['2.1','2.2','2.3','2.5','2.6','2.7','2.8','2.9','2.10','2.11','2.12',
              '2.13','2.15','4.1','4.2','4.4','4.5','4.8','4.9','5.1','5.3',
              '5.4','5.5','6']
    missing=set(required)-set(sections)
    if missing: raise ValueError(f'Manuscript section headings not found: {missing}')
    # Section numbers distinguish generic headings such as Discrimination and
    # Conclusions from earlier mentions and the structured abstract.
    anchors={k:[f'{k}. {v}'] for k,v in sections.items() if k in required}
    title=plain(next(line[2:] for line in md.splitlines() if line.startswith('# ')))
    anchors['title']=[title]
    paras=[plain(p) for p in re.split(r'\n\s*\n',md) if plain(p)]
    def para_anchor(key, selectors, fallback=None, words=13):
        for sel in selectors:
            for p in paras:
                if norm(sel) in norm(p):
                    # Use the selector itself if it is long and distinctive;
                    # otherwise the paragraph opening provides context.
                    anchors[key]=[sel] if len(sel.split())>=4 else [' '.join(p.split()[:words])]
                    return
        if fallback: anchors[key]=anchors[fallback]
        else: raise ValueError(f'No manuscript text found for anchor {key}')
    abstract_match=re.search(r'^##\s+Abstract\s*\n(.*?)(?=^##\s|\Z)',md,flags=re.M|re.S)
    if not abstract_match: raise ValueError('Abstract section not found in current manuscript')
    abstract_body=plain(abstract_match.group(1)).strip()
    abstract_opening=' '.join(abstract_body.split()[:16])
    if not abstract_opening: raise ValueError('Abstract section is empty')
    # Derive the opening from the current document so that structured and
    # unstructured author-edited abstracts use the same pagination workflow.
    anchors['abstract']=[abstract_opening,'Abstract']
    para_anchor('intro_rationale',['Clinical transformers represent histories','Predicting HbA1c change requires','Clinical transformers adapt'],fallback='2.10')
    para_anchor('intro_prior',['Previous HbA1c studies'],fallback='2.10')
    para_anchor('intro_aims',['In this work, we will characterise','We used Hakeem to describe'],fallback='2.1')
    para_anchor('intro_target',['These analyses are intended to inform follow-up prioritisation','The intended population is patients','The target population is patients'],fallback='2.7')
    para_anchor('intro_equity',['Given differences in diabetes care by sex and age','Because diabetes care in Jordan varies by sex and age',
                               'diabetes care in Jordan differs by sex and age'],fallback='2.13')
    para_anchor('samplesize',['No sample size calculation was performed in advance',
                             'No sample size was calculated in advance'],fallback='2.9')
    para_anchor('missing_tokens',['represented by explicit missing tokens rather than imputed values',
                                 'explicit missing tokens rather than imputed'],fallback='2.8')
    para_anchor('clustered',['The patient-clustered bootstrap sampled patients with replacement',
                            'patient-clustered bootstrap, resampling patients rather than rows'],fallback='2.11')
    para_anchor('recalibration',['recalibration was included in the final pipeline',
                                'Class-specific vector scaling', 'balanced class weights'],fallback='2.11')
    para_anchor('model_spec',['A BEHRT-style bidirectional encoder','Transformer.'],fallback='2.10')
    para_anchor('decl_ethics',['Ethics approval and consent.'])
    para_anchor('decl_funding',['This research received no specific grant from any funding agency'])
    para_anchor('decl_coi',['The authors declare no known competing financial interests',
                          'The authors declare that they have no known competing financial interests'])
    para_anchor('decl_protocol',['No prespecified protocol was prepared'])
    para_anchor('decl_ppi',['Patients and the public were not involved in the design'])
    para_anchor('decl_data',['cannot be shared publicly'])
    para_anchor('decl_code',['under the MIT licence'])
    anchors['decl_licence']=['MIT licence']
    para_anchor('limitation_end',['Several eligibility and measurement choices also constrain interpretation'],fallback='5.5')
    para_anchor('subgroup_end',['At the common validation-selected threshold for recalibrated LightGBM'],fallback='4.9')
    para_anchor('figure_flow',['Derivation of the predictive cohort','Derivation of the prediction instances',
                              'Derivation of the prediction cohort'],fallback='2.9')
    return title,anchors

def resolve_anchors(pdf, anchors):
    with fitz.open(pdf) as doc:
        pages=[]
        for page in doc:
            # In the manuscript render, continuous line numbers occupy their own
            # word boxes in the left gutter. Filtering by geometry avoids
            # accidentally deleting genuine numeric values in the body.
            words=[w for w in page.get_text('words')
                   if not (re.fullmatch(r'\d{1,4}',w[4]) and
                           (w[0]<65 or w[1]>page.rect.height-65))]
            pages.append(norm(' '.join(w[4] for w in words)))
        npages=doc.page_count
    found,missing={},[]
    for key,phrases in anchors.items():
        hits=[]
        for phrase in phrases:
            target=norm(phrase)
            hits=[i+1 for i,text in enumerate(pages) if target in text]
            if hits: break
            # A declaration sentence can cross a physical page boundary.
            hits=[i+1 for i in range(len(pages)-1)
                  if target in pages[i]+' '+pages[i+1]]
            if hits: break
        if not hits: missing.append((key,phrases))
        else: found[key]=hits[0]
    if missing: raise ValueError(f'Anchors not found in final V10 PDF: {missing}')
    return found,npages

def item_mapping(amap):
    # A page pointer identifies relevant content, not a certification of complete
    # reporting. Explicit gaps and inapplicable subjective assessments are marked.
    definitions={
      '1':('list',['title']),'2':('list',['abstract']),
      '3a':('span',['intro_rationale','intro_prior']),
      '3b':('list',['intro_target']),'3c':('list',['intro_equity','2.13']),
      '4':('list',['intro_aims','intro_target']),
      '5a':('list',['2.1','5.5']),'5b':('list',['2.1','2.9']),
      '6a':('list',['2.1','5.5']),'6b':('list',['2.7','2.9','figure_flow']),
      '6c':('list',['2.8','5.5']),'7':('list',['2.2']),
      '8a':('list',['2.7']),'8b':('literal','N/A'),'8c':('literal','Not reported'),
      '9a':('list',['2.8']),'9b':('list',['2.8','missing_tokens']),'9c':('literal','N/A'),
      '10':('list',['samplesize','4.8']),'11':('list',['2.1','missing_tokens']),
      '12a':('list',['2.9']),'12b':('list',['2.8','2.10']),
      '12c':('list',['2.10','2.12']),'12d':('list',['clustered','5.5']),
      '12e':('list',['2.11']),'12f':('list',['2.11','4.5']),
      '12g':('list',['2.10','2.11','2.15']),
      '13':('list',['2.11','2.12','4.5']),
      '14':('list',['2.13','4.9','subgroup_end']),'15':('list',['2.11','4.4']),
      '16':('list',['4.1','5.4']),
      '17':('partial',['2.1','decl_ethics']),
      '18a':('list',['decl_funding']),'18b':('list',['decl_coi']),
      '18c':('list',['decl_protocol']),'18d':('list',['decl_protocol']),
      '18e':('list',['decl_data']),'18f':('list',['2.15','decl_code','decl_licence']),
      '19':('list',['decl_ppi']),'20a':('list',['2.9','figure_flow']),
      '20b':('list',['4.1','2.8']),'20c':('list',['4.1']),
      '21':('list',['2.9','samplesize','4.1']),'22':('list',['2.10','2.11','decl_code','decl_licence']),
      '23a':('span',['4.2','4.9','subgroup_end']),'23b':('list',['5.5']),
      '24':('list',['4.5']),'25':('span',['5.1','5.4']),
      '26':('list',['5.5','limitation_end']),'27a':('literal','Not reported'),
      '27b':('partial',['intro_target','4.4','5.4']),
      '27c':('list',['5.3','5.4','5.5','6'])}
    if set(definitions)!=set(EXPECTED): raise ValueError('Checklist coverage mismatch')
    result={}
    for code,(mode,keys) in definitions.items():
        if mode=='literal': result[code]=keys; continue
        ps=sorted({amap[k] for k in keys})
        value=(str(ps[0]) if len(ps)==1 else f'{ps[0]}-{ps[-1]}') if mode=='span' else ', '.join(map(str,ps))
        if mode=='partial': value+=' *'
        result[code]=value
    return result

def item_cells(doc):
    cells={}
    for pi,page in enumerate(doc):
        for word in page.get_text('words'):
            if ITEM_RE.match(word[4]) and 95<=word[0]<=125:
                cells[word[4]]=(pi,word[1],word[3])
    if set(cells)!=set(EXPECTED): raise ValueError('Official checklist items not located')
    return cells

def clear_copy(doc):
    removed=[]
    for pi,page in enumerate(doc):
        for w in page.get_text('words'):
            if w[0]>=COL_X0-1.5 and BAND_TOP.get(pi,55)<=w[1] and w[3]<=BAND_BOTTOM and w[4] not in HDR_WORDS:
                page.add_redact_annot(fitz.Rect(w[0]-.6,w[1]-.6,w[2]+.6,w[3]+.6))
                removed.append(w[4])
        if pi==len(doc)-1: page.add_redact_annot(NOTE_RECT)
        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE,
            graphics=fitz.PDF_REDACT_LINE_ART_NONE,text=fitz.PDF_REDACT_TEXT_REMOVE)
    return removed

def form_text(doc):
    result=[]
    for pi,page in enumerate(doc):
        words=[w for w in page.get_text('words') if w[0]<COL_X0-1.5
               and not (pi==doc.page_count-1 and NOTE_RECT.intersects(fitz.Rect(w[:4])))]
        words.sort(key=lambda w:(round(w[1],0),w[0]))
        result.append(' '.join(w[4] for w in words))
    return result

def rules(doc):
    return [sum(d['rect'].width>2 or d['rect'].height>2 for d in p.get_drawings()) for p in doc]

def wrap_lines(text,width,size=6.15,font='helv'):
    result=[]; current=''
    for word in text.split():
        candidate=(current+' '+word).strip()
        if fitz.get_text_length(candidate,fontname=font,fontsize=size)>width:
            result.append(current); current=word
        else: current=candidate
    if current: result.append(current)
    return result

def read_values(doc,cells):
    result={}
    for code,(pi,y0,y1) in cells.items():
        baseline=y1-2.0
        ws=[w for w in doc[pi].get_text('words')
            if w[0]>=COL_X0-1.5 and abs(w[3]-baseline-2)<4.5]
        ws.sort(key=lambda w:w[0]); result[code]=' '.join(w[4] for w in ws)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manuscript-pdf',type=Path,default=HERE/'tmp/v10_editorial/main_render/Manuscript V10.pdf')
    ap.add_argument('--manuscript-md',type=Path,default=HERE/'Manuscript V10.md')
    ap.add_argument('--template',type=Path,default=HERE/'TRIPOD_AI_checklist.pdf')
    ap.add_argument('--out',type=Path,default=HERE/'TRIPOD AI checklist V10.pdf')
    args=ap.parse_args()
    for path in [args.manuscript_pdf,args.manuscript_md,args.template]:
        if not path.is_file(): raise FileNotFoundError(path)
    template_hash=hashlib.sha256(args.template.read_bytes()).hexdigest()
    title,anchors=manuscript_anchors(args.manuscript_md.read_text(encoding='utf-8-sig'))
    amap,npages=resolve_anchors(args.manuscript_pdf,anchors)
    values=item_mapping(amap)
    with fitz.open(args.template) as source:
        original_text=form_text(source); original_rules=rules(source)
        cells=item_cells(source)
        removed=clear_copy(source)
        for code in EXPECTED:
            value=values[code]; size=7.9; avail=COL_X1-COL_X0-4
            while fitz.get_text_length(value,fontname='tiro',fontsize=size)>avail and size>6:
                size-=.2
            width=fitz.get_text_length(value,fontname='tiro',fontsize=size)
            if width>avail: raise ValueError(f'Cell text too wide: {code}={value}')
            pi,y0,y1=cells[code]
            source[pi].insert_text((COL_X0+(COL_X1-COL_X0-width)/2,y1-2),value,
                fontname='tiro',fontsize=size,color=(0,0,0))
        note_text=[
          f'Completed for {title}.',
          f'Manuscript V10.docx; {npages} manuscript pages. Page references use the accompanying V10 pagination.',
          '* Partial reporting: item 17 gives approval, but consent or waiver is not stated; item 27b names users, but interaction/expertise is unspecified.',
          'Not reported: 8c, outcome-assessment blinding; 27a, input-data quality or missing-input procedures for implementation.',
          'N/A: 8b and 9c concern subjective outcome/predictor assessors; this analysis uses recorded data and algorithmic definitions.'
        ]
        lines=[]
        for t in note_text: lines.extend(wrap_lines(t,NOTE_RECT.width-4))
        if len(lines)>6: raise ValueError(f'Footer exceeds six lines: {lines}')
        for i,line in enumerate(lines):
            source[-1].insert_text((NOTE_RECT.x0+2,NOTE_RECT.y0+7+i*6.6),line,
                                  fontname='helv',fontsize=6.15,color=(0,0,0))
        source.set_metadata({'title':'TRIPOD+AI checklist for Manuscript V10',
            'subject':f'Page references verified against Manuscript V10 pagination ({npages} pages). Partial reporting and unreported items identified.',
            'author':'Dania Rahhal, Said Moshawih, Suha Telfah',
            'keywords':'TRIPOD+AI; HbA1c; Hakeem; Manuscript V10'})
        if form_text(source)!=original_text: raise ValueError('Official form wording changed')
        if rules(source)!=original_rules: raise ValueError('Official table rules changed')
        source.save(args.out,deflate=True,garbage=3)
    with fitz.open(args.out) as out:
        if form_text(out)!=original_text: raise ValueError('Saved form wording changed')
        if rules(out)!=original_rules: raise ValueError('Saved table rules changed')
        got=read_values(out,cells)
        if got!=values: raise ValueError(f'Saved checklist cells differ: {[(c,values[c],got[c]) for c in EXPECTED if got[c]!=values[c]]}')
        alltext=' '.join(p.get_text() for p in out)
        if re.search(r'\bV[369]\b|manuscript_V[369]',alltext): raise ValueError('Stale version text survived')
        render_dir=HERE/'tmp/v10_editorial/checklist_render'; render_dir.mkdir(exist_ok=True)
        for i,page in enumerate(out,1):
            page.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(render_dir/f'page-{i}.png')
    if hashlib.sha256(args.template.read_bytes()).hexdigest()!=template_hash:
        raise ValueError('Source template changed')
    audit={'manuscript_pdf':str(args.manuscript_pdf),'manuscript_pages':npages,
           'output':str(args.out),'anchors':amap,'checklist_values':values,
           'official_wording_unchanged':True,'table_rules_unchanged':True,
           'source_template_sha256':template_hash,'stale_column_words_removed':len(removed),
           'gaps':{'17':'Consent or waiver not stated','8c':'Blinding procedure not reported',
                   '27a':'Implementation input-quality/missing-input procedures not reported',
                   '27b':'User interaction and required expertise not specified'}}
    (HERE/'tmp/v10_editorial/checklist_audit_V10.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(args.out),'manuscript_pages':npages,'items':len(values),
                      'gaps':audit['gaps'],'render_dir':str(render_dir)},indent=2))

if __name__=='__main__': main()

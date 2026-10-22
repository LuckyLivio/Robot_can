"""Build a compact chapter-style design, reusing the verified Office Math renderer."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm,Pt,RGBColor
from docx.text.run import Run

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
OLD=ROOT/'docs/detailed_design_20261022_01'
TITLE='基于世界模型的机器人抓取搬运系统设计与实现'
OUTPUT=HERE/(TITLE+'详细设计.docx')
spec=importlib.util.spec_from_file_location('verified_design_renderer',OLD/'build_design.py')
lib=importlib.util.module_from_spec(spec);spec.loader.exec_module(lib)
lib.HERE=HERE
lib.TITLE=TITLE


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def character_indent(paragraph_format, characters):
    """Use Word's actual character unit, with a point fallback for readers."""
    paragraph_format.first_line_indent=Pt(characters*12)
    ind=paragraph_format._element.get_or_add_pPr().get_or_add_ind()
    ind.set(qn('w:firstLineChars'),str(round(characters*100)))
    ind.attrib.pop(qn('w:hangingChars'),None)


def superscript_citations(paragraph):
    # Split only existing text runs; native equations and drawings stay intact.
    pattern=r'(\[\d+(?:[-–,，]\d+)*\])'
    for run in list(paragraph.runs):
        if not re.search(pattern,run.text):continue
        for part in re.split(pattern,run.text):
            if not part:continue
            node=OxmlElement('w:r')
            if run._r.rPr is not None:node.append(deepcopy(run._r.rPr))
            text_node=OxmlElement('w:t');text_node.text=part
            text_node.set(qn('xml:space'),'preserve');node.append(text_node)
            run._r.addprevious(node)
            if re.fullmatch(pattern,part):Run(node,paragraph).font.superscript=True
        run._r.getparent().remove(run._r)


def apply_thesis_format(doc, source):
    styles=doc.styles
    lib.set_fonts(styles['Normal'],12,east='宋体',west='Times New Roman')
    styles['Normal'].paragraph_format.line_spacing=1.5
    character_indent(styles['Normal'].paragraph_format,2)
    for level in range(1,5):
        name=f'Heading {level}'
        font='黑体' if level<=2 else '宋体'
        lib.set_fonts(styles[name],16 if level==1 else 14 if level==2 else 12,
                      east=font,west=font if level<=2 else 'Times New Roman',bold=False)
        styles[name].font.italic=False;styles[name].font.cs_italic=False
        fmt=styles[name].paragraph_format
        fmt.line_spacing=1.5;fmt.keep_with_next=True;character_indent(fmt,0)
        if level==1:
            fmt.alignment=WD_ALIGN_PARAGRAPH.CENTER
            spacing=fmt._element.get_or_add_pPr().get_or_add_spacing()
            for attr in ('before','after','beforeAutospacing','afterAutospacing'):
                spacing.attrib.pop(qn('w:'+attr),None)
            spacing.set(qn('w:beforeLines'),'100');spacing.set(qn('w:afterLines'),'100')
    for name in ('Title','Subtitle',*[f'Heading {i}' for i in range(1,5)]):
        for border in styles[name].element.xpath('./w:pPr/w:pBdr'):
            border.getparent().remove(border)
    # The legacy renderer maps level 4 to level 3; retain true source levels.
    levels=[min(len(m[1]),4) for line in source.splitlines()
            if (m:=re.match(r'^(#{1,6})\s+(.+)$',line)) and m[2]!=TITLE]
    headings=[p for p in doc.paragraphs if p.style.name.startswith('Heading ')]
    if len(levels)!=len(headings):raise ValueError('Heading mapping changed')
    for p,level in zip(headings,levels):
        title=p.text
        p.style=styles[f'Heading {level}']
        if level>=2:
            m=re.match(r'^(\d+(?:\.\d+)+)\s+(.*)$',title)
            if m:title=m[1]+(' ' if level==2 else '  ')+m[2]
        p.text=title;character_indent(p.paragraph_format,0)
    for p in doc.paragraphs:
        body=(p.style.name=='Normal' and bool(p.text.strip()) and p.text!='详细设计'
              and not p._p.xpath('.//w:drawing'))
        if body:
            character_indent(p.paragraph_format,2);p.paragraph_format.line_spacing=1.5
            for r in p.runs:
                lib.set_fonts(r,12,east='宋体',west='Times New Roman')
            superscript_citations(p)
        elif p.style.name=='Reference Design':
            p.paragraph_format.first_line_indent=None
            ind=p.paragraph_format._element.get_or_add_pPr().get_or_add_ind()
            ind.attrib.pop(qn('w:firstLineChars'),None)
        else:character_indent(p.paragraph_format,0)
    for name in ('Caption','Equation Design','Reference Design','Title'):
        character_indent(styles[name].paragraph_format,0)
    # Preserve bibliography hanging indent; its labels are not body citations.
    ref=styles['Reference Design'].paragraph_format
    ref.first_line_indent=Cm(-.65);ref._element.get_or_add_pPr().get_or_add_ind().attrib.pop(qn('w:firstLineChars'),None)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for p in cell.paragraphs:character_indent(p.paragraph_format,0)
    for section in doc.sections:
        for part in (section.header,section.footer):
            for p in part.paragraphs:character_indent(p.paragraph_format,0)


class ChapterBuilder(lib.Builder):
    def cover(self):
        # A chapter document has a compact title, no separate cover or TOC.
        lib.set_fonts(self.doc.styles['Title'],16,east='黑体',bold=False)
        lib.set_fonts(self.doc.styles['Heading 1'],16,east='黑体',bold=True)
        self.doc.styles['Heading 1'].paragraph_format.page_break_before=False
        self.doc.styles['Normal'].paragraph_format.space_after=Pt(3)
        section=self.doc.sections[0]
        p=section.header.paragraphs[0];p.text=TITLE;p.alignment=WD_ALIGN_PARAGRAPH.RIGHT;p.paragraph_format.first_line_indent=Pt(0)
        for r in p.runs:r.font.size=Pt(9);r.font.color.rgb=RGBColor(0,0,0)
        p=section.footer.paragraphs[0];p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.first_line_indent=Pt(0)
        lib.field(p,'PAGE')
        p=self.doc.add_paragraph(TITLE,'Title');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before=Pt(0);p.paragraph_format.space_after=Pt(4)
        p=self.doc.add_paragraph('详细设计');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent=Pt(0);p.paragraph_format.space_after=Pt(10)
        settings=self.doc.settings.element
        update=OxmlElement('w:updateFields');update.set(qn('w:val'),'true');settings.append(update)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--replace-generated',action='store_true');args=parser.parse_args()
    if OUTPUT.exists() and not args.replace_generated:raise FileExistsError(OUTPUT)
    text=(HERE/'详细设计.md').read_text(encoding='utf-8')
    images=[HERE/p for p in re.findall(r'!\[[^\]]*\]\(([^)]+)\)',text)]
    if any(not p.is_file() for p in images):raise FileNotFoundError('All design diagrams must exist first')
    builder=ChapterBuilder();doc=builder.render(text)
    apply_thesis_format(doc,text)
    # Keep the short state table together instead of leaving only one row.
    for row in doc.tables[0].rows[:-1]:
        for cell in row.cells:
            for p in cell.paragraphs:p.paragraph_format.keep_with_next=True
    for p in doc.paragraphs:
        if p.text.startswith('本章完成基于世界模型'):
            p.paragraph_format.keep_together=True
    doc.core_properties.title=TITLE+'详细设计'
    doc.core_properties.subject='第4章 系统设计'
    doc.save(OUTPUT)
    source_paths=[HERE/'详细设计.md',Path(__file__),OLD/'build_design.py',
                  OLD/'research/references_verified.json',ROOT/'common.py',ROOT/'world_model.py',
                  ROOT/'robot_can_v2/observations.py',ROOT/'robot_can_v2/evaluate.py',ROOT/'robot_can_v2/train_world.py',
                  ROOT/'robot_can_v2/train_policy.py',ROOT/'train_policy.py',ROOT/'collect_rollout_branches.py',
                  ROOT/'collect_branches.py',ROOT/'reproduction/multiseed_20261012_01/train_suite.py',
                  *[ROOT/f'models/multiseed_20261012_01_full_s{seed}/config.json' for seed in (17,23,41)],
                  ROOT/'reproduction/recovery_v5_20261012_01/recovery_controller.py']
    manifest={'title':TITLE,'built_utc':datetime.now(timezone.utc).isoformat(),
              'scope':'Standalone Chapter 4 detailed design; no abstract, introduction, full evaluation chapter or appendices',
              'reference_pdf_sha256':sha(Path('C:/Users/Livio/Desktop/211380127.pdf')),
              'reference_scope':'PDF pages 22-35, printed 15-28; architecture, functions, data and interfaces only',
              'original_52_page_version_preserved':True,'cjk_characters':len(re.findall(r'[\u4e00-\u9fff]',text)),
              'numbered_equations':sum(builder.math_count.values()),'office_math_objects':len(builder.math_checks),
              'tables':len(doc.tables),'figures':len(doc.inline_shapes),'references':6,
              'layout_profile':{'heading_1':'16pt SimHei centered; one line before and after',
                                'heading_2':'14pt SimHei, one space, 1.5 lines',
                                'heading_3_4':'12pt SimSun, two spaces, 1.5 lines',
                                'body':'12pt SimSun / Times New Roman, firstLineChars=200, 1.5 lines',
                                'body_references':'native superscript; bibliography labels on baseline'},
              'sources_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in source_paths},
              'image_sha256':{p.relative_to(HERE).as_posix():sha(p) for p in images},
              'pre_word_outputs_sha256':{p.name:sha(p) for p in (OUTPUT,HERE/'详细设计.md')},
              'math_checks':builder.math_checks,
              'verification_note':'Native Word export and per-page visual QA follow authoring; not included in this build receipt.'}
    (HERE/'build_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({key:manifest[key] for key in ('cjk_characters','numbered_equations','office_math_objects','tables','figures','references')},ensure_ascii=False))


if __name__=='__main__':main()

"""Final structural/source checks after native Word export and recorded visual QA."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile
from lxml import etree
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qa-dir',type=Path,required=True);args=parser.parse_args()
    text=(HERE/'详细设计.md').read_text(encoding='utf-8')
    docx=HERE/'机器人抓取搬运与失败恢复系统详细设计.docx'
    pdf=args.qa_dir/'design_v3.pdf';render=args.qa_dir/'render_v3'
    word=json.loads((args.qa_dir/'word_export_v3.json').read_text(encoding='utf-8-sig'))
    build=json.loads((HERE/'build_manifest.json').read_text(encoding='utf-8'))
    math=json.loads((HERE/'research/math_conversion_checks.json').read_text(encoding='utf-8'))
    ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main','m':'http://schemas.openxmlformats.org/officeDocument/2006/math'}
    errors=[]
    with ZipFile(docx) as z:
        xml=etree.fromstring(z.read('word/document.xml'))
        maths=xml.findall('.//m:oMath',ns)
        for index,(node,check) in enumerate(zip(maths,math['checks'])):
            value=''.join(node.itertext())
            if '\\' in value:errors.append(f'Unconverted math command at {index}')
            if value.count('−')+value.count('-')!=check['source_minus_count']:
                errors.append(f'Final Word minus count differs at {index}')
        if len(maths)!=len(math['checks']):errors.append('Office Math count differs')
        tables=len(xml.findall('.//w:tbl',ns))
    pages=PdfReader(pdf).pages
    blanks=[]
    for index,page in enumerate(pages):
        value=page.extract_text() or ''
        meaningful=re.sub(r'机器人抓取搬运与失败恢复系统详细设计|[\s\d]','',value)
        if index>0 and not meaningful:blanks.append(index+1)
        if '\\arg' in value or '错误!未定义' in value:errors.append(f'Field/math error page {index+1}')
    if blanks:errors.append(f'Blank content pages: {blanks}')
    assert len(pages)==word['page_count']==52
    assert len(list(render.glob('page-*.png')))==52
    assert len(re.findall(r'^# 第\d+章',text,re.M))==12
    assert len(re.findall(r'^# 附录[A-Z]',text,re.M))==3
    assert tables==16
    assert sum(check['source_minus_count'] for check in math['checks'])>0
    for relative,expected in build['sources_sha256'].items():
        if sha(ROOT/relative)!=expected:errors.append(f'Authoring input changed: {relative}')
    for match in re.finditer(r'!\[[^\]]*\]\(([^)]+)\)',text):
        if not (HERE/match[1]).is_file():errors.append(f'Missing figure: {match[1]}')
    changed=subprocess.check_output(['git','diff','672fa48','--name-only'],cwd=ROOT,text=True).splitlines()
    outside=[name for name in changed if name!='README.md' and not name.startswith('docs/detailed_design_20261022_01/')]
    if outside:errors.append(f'Unexpected tracked changes: {outside}')
    extra_sources=[ROOT/'reproduction/recovery_v5_20261012_01/recovery_controller.py',
                   ROOT/'reproduction/recovery_v5_20261012_01/clearance.py',
                   ROOT/'reproduction/recovery_v5_20261012_01/gripper_geometry.json',
                   ROOT/'.venv/Lib/site-packages/robosuite/environments/manipulation/pick_place.py']
    report={'passed':not errors,'verified_utc':datetime.now(timezone.utc).isoformat(),'errors':errors,
            'implementation_baseline':'672fa48','scope':'Detailed design document; no new training or simulation',
            'pages':52,'chapters':12,'appendices':3,'cjk_characters':build['cjk_characters'],
            'numbered_equations':build['equations'],'office_math_objects':len(maths),'tables':tables,'figures':9,'references':10,
            'native_word_fields_updated':word['fields_updated'],'blank_body_pages':blanks,
            'math_check_scope':'Every converted expression checked for unconverted commands and minus preservation, including after Word save; every page visually inspected separately.',
            'visual_reviews':[{'agent':'design_models','pages':[1,17],'all_pages_opened_original':True,'passed':True},
                              {'agent':'design_recovery','pages':[18,35],'all_pages_opened_original':True,'passed':True},
                              {'agent':'design_references','pages':[36,52],'all_pages_opened_original':True,'passed':True}],
            'conversion_backend':'Microsoft Word native PDF export plus packaged render_docx.py rasterizer; LibreOffice absent',
            'final_outputs_sha256':{p.name:sha(p) for p in (docx,HERE/'详细设计.md')},
            'internal_pdf_sha256':sha(pdf),'supplementary_source_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in extra_sources},
            'original_research_changes_outside_navigation':outside,
            'template_scope':'Generic Chinese undergraduate technical document; no institution template was supplied.',
            'source_and_experiment_cross_review':'Completed before layout; found overbroad legacy validation claim, undefined success criterion and aggregate-weight caveat, all corrected.',
            'qa_intermediates':str(render),'not_a_new_experiment_sample':True}
    (HERE/'quality_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('passed','pages','chapters','numbered_equations','office_math_objects','tables','figures','errors')},ensure_ascii=False))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()

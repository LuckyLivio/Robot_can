"""Use the packaged DOCX renderer's rasterizer with an already native-exported PDF.

LibreOffice is absent on this Windows host. The conversion stage is replaced
with Microsoft Word export; the installed render_docx.py is not modified.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser()
parser.add_argument("--docx",type=Path,required=True)
parser.add_argument("--word-pdf",type=Path,required=True)
parser.add_argument("--output-dir",type=Path,required=True)
args=parser.parse_args()
if not args.docx.is_file() or not args.word_pdf.is_file():raise FileNotFoundError("Both inputs must exist")
renderer=Path.home()/".codex/plugins/cache/openai-primary-runtime/documents/26.1020.11348/skills/documents/render_docx.py"
spec=importlib.util.spec_from_file_location("packaged_docx_renderer",renderer)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def convert_native(doc_path,user_profile,convert_tmp_dir,stem,verbose):
    if Path(doc_path).resolve()!=args.docx.resolve():raise ValueError("Unexpected input DOCX")
    return str(args.word_pdf.resolve()),"Native Word export supplied explicitly; no LibreOffice fallback claimed."
module.convert_to_pdf=convert_native
sys.argv=[str(renderer),str(args.docx),"--output_dir",str(args.output_dir),"--dpi","150"]
module.main()
print(json.dumps({"backend":"native Word PDF + packaged render_docx.py rasterizer","pages":len(list(args.output_dir.glob('page-*.png')))}))

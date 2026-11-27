import subprocess
import sys

import pypandoc

from src.preparacao import BASE, RESULTADOS

nome = sys.argv[1] if len(sys.argv) > 1 else "ML_Kaggle_Picasso_Reginato"
pasta = BASE / "relatorio"

md = (pasta / "relatorio.md").read_text(encoding="utf-8")
md = md.replace("{{TABELA_COMPARATIVA}}",
                (RESULTADOS / "tabela_comparativa.md").read_text(encoding="utf-8"))
tmp = pasta / "_relatorio_montado.md"
tmp.write_text(md, encoding="utf-8")

docx = pasta / f"{nome}.docx"
args = [f"--resource-path={BASE}", "-M", "lang=pt-BR"]
ref = pasta / "referencia.docx"
if ref.exists():
    args.append(f"--reference-doc={ref}")
pypandoc.convert_file(str(tmp), "docx", outputfile=str(docx), extra_args=args)
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(pasta),
                str(docx)], check=True, capture_output=True)
tmp.unlink()
print(f"Gerados: {docx} e {docx.with_suffix('.pdf')}")

"""Monta o relatório (relatorio/relatorio.md) em .docx e .pdf.

Uso: python 04_relatorio.py [nome_do_arquivo]
Precisa do LibreOffice (comando `soffice`) para converter o .docx em PDF. Por isso este
passo não faz parte do run_all.sh. Rode depois do 03_comparacao.py.
"""
import subprocess
import sys

import pypandoc

from src.preparacao import BASE, RESULTADOS

# Nome dos arquivos de saída (padrão do enunciado: ML_Kaggle_Sobrenome1_Sobrenome2)
nome = sys.argv[1] if len(sys.argv) > 1 else "ML_Kaggle_Picasso_Reginato"
pasta = BASE / "relatorio"

# Coloca a tabela comparativa (gerada pelo 03) no lugar do marcador do relatório
md = (pasta / "relatorio.md").read_text(encoding="utf-8")
md = md.replace("{{TABELA_COMPARATIVA}}",
                (RESULTADOS / "tabela_comparativa.md").read_text(encoding="utf-8"))
tmp = pasta / "_relatorio_montado.md"
tmp.write_text(md, encoding="utf-8")

# Markdown -> Word (com modelo de estilos, se existir) -> PDF
docx = pasta / f"{nome}.docx"
args = [f"--resource-path={BASE}", "-M", "lang=pt-BR"]
ref = pasta / "referencia.docx"
if ref.exists():
    args.append(f"--reference-doc={ref}")
pypandoc.convert_file(str(tmp), "docx", outputfile=str(docx), extra_args=args)
subprocess.run(["soffice", "--headless", "--convert-to", "pdf", "--outdir", str(pasta),
                str(docx)], check=True, capture_output=True)
tmp.unlink()  # remove o arquivo temporário
print(f"Gerados: {docx} e {docx.with_suffix('.pdf')}")

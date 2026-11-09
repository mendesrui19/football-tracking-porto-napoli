#!/bin/bash
# Reprocessa o resumo do zero: análise a 25 fps (com OCR de nomes) -> preenche falhas do mapa -> corrige espelhos
# -> reconstrói o painel -> verifica. Demora cerca de 1 hora.
cd "$(dirname "$0")"
rm -rf destaques_novo
.venv/bin/python analyze.py destaques/resumo.mp4 --fps 25 --chunk 60 --ocr --out destaques_novo < /dev/null > destaques_out.log 2>&1
if grep -q Traceback destaques_out.log; then echo ERRO >> destaques_out.log; exit 1; fi
rm -rf destaques_out && mv destaques_novo destaques_out
.venv/bin/python gap_fill.py   >> destaques_out.log 2>&1
.venv/bin/python fix_mirror.py >> destaques_out.log 2>&1
echo FIM >> destaques_out.log
.venv/bin/python build_dashboard.py    > build.log 2>&1
.venv/bin/python verificar_sistema.py  > verificacao_saida.log 2>&1
echo CONCLUIDO >> verificacao_saida.log

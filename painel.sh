#!/bin/bash
# Atualiza os dados do painel e garante que o servidor está a correr em http://localhost:8780
cd "$(dirname "$0")"
.venv/bin/python build_dashboard.py
if lsof -nP -iTCP:8780 -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Servidor já ativo — dados atualizados. Recarrega http://localhost:8780"
else
  .venv/bin/python painel/serve.py 8780
fi

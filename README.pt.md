# Painel tático — FC Porto 4–1 Napoli (resumo DAZN)

## Usar
    ./painel.sh      # atualiza os dados e garante o servidor em http://localhost:8780
Teclas: Espaço play/pausa · ← → ±5 s · N / P momento seguinte / anterior.

## O que mostra
- Vídeo do resumo com os jogadores identificados (#ID, ou nome/número quando legível; GK roxo, REF amarelo) e a bola com halo laranja e rasto.
- Mapa 2D: 22 jogadores sempre presentes (os que a câmara não vê ficam esbatidos e a acompanhar a bola/colegas), bola, portador, linha de fora de jogo.
- Linha do tempo clicável: golos, lances de perigo (bola na grande área ≥ 2 s), faltas e substituições.
- Repetição tática de cada golo no mapa, durante os festejos.
- Todo o site está em inglês.

## Ficheiros principais
| ficheiro | função |
|---|---|
| `analyze.py`, `track.py` | deteção de jogadores (1280 px), equipas, bola, calibração do campo validada pelas linhas |
| `slots.py` | seguimento persistente dos 22 jogadores + bola + linha de fora de jogo |
| `identify.py`, `identidades.py` | nomes/números por leitura (só planos próximos), GK e árbitro por posição |
| `gap_fill.py`, `fix_mirror.py` | preenche falhas de mapa em jogo aberto; corrige frames espelhados (esq./dir. trocadas) |
| `build_dashboard.py` | exporta os dados para `painel/data/` |
| `verificar_sistema.py`, `qa_momentos.py`, `audit_pos.py` | verificação automática, relatório por momento, auditoria da posição |

## Reprocessar do zero
    ./run_destaques.sh          # ≈ 1 h; no fim reconstrói o painel e verifica
    .venv/bin/python qa_momentos.py     # relatório por momento → qa_report.md

## Estado medido (23 momentos: 5 golos + 18 lances)
- Mapa em jogo aberto: ≈ 76 % dos frames; alinhamento das linhas ≥ 0,6 em 91 % dos frames calibrados.
- Jogadores à vista: ≈ 12,9 por frame; portador da bola identificado em ≈ 56 %.
- Bola: no vídeo ≈ 68 %, no mapa ≈ 61 %; falhas em 10 momentos e saltos em 14.
- Nomes/números: 109 identidades (planos próximos, guarda-redes, árbitro).

## Limites conhecidos
- Planos próximos, repetições e câmaras atrás da baliza não têm campo calibrável: o mapa mantém a última posição.
- Nos planos abertos o número da camisola tem ~12 px e não é legível: ficam P1..P10 / N1..N10 (só 2,3 % legíveis em jogo).
- A bola é pequena, fica tapada ou sai de plano; no ar a projeção no chão exagera a velocidade.
- Jogadores que a câmara não vê são estimativas; dois colegas podem trocar de lugar quando reaparecem.
- Golos 1 e 5: o resumo mostra pouco jogo ao vivo (mapa 29 % e 13 %).
- As estatísticas descrevem os lances do resumo, não o jogo todo.

## Atualização — revisão momento a momento (22 lances)
- Mapa 2D coerente com o vídeo na maioria dos lances (aglomerados nos golos, GK, bola nos cantos/baliza).
- Árbitro: o mapa só recebe figuras "classe 2" com papel confirmado (GK ou REF em `names.json`); limites do REF alargados (-3..W+3).
- Sincronização: `painel/index.html` usa `requestVideoFrameCallback` (mediaTime do fotograma mostrado) com extrapolação < 80 ms; medido ≈ 0,02 s de diferença.
- **Limite conhecido**: árbitros de camisola invulgar (vermelho) podem ser classificados como jogador (ex.: 315 s e 572 s). A correção exige guardar a caixa de cada deteção e medir a cor da camisola, ou seja, reprocessar o vídeo (~1 h). As posições x_px/y_px guardadas nas pistas não servem para amostrar cor de forma fiável.

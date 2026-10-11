#!/bin/bash
# Builds the TikZ figures of the presentation deck: figures/src/deck_<name>.tex (with src/tikz_style.tex, the style of
# the design figures) -> figures/deck/<name>.png.   Usage: figures/build_deck.sh [NAME ...]   (no name: all)
set -euo pipefail
cd "$(dirname "$0")"
module load texlive/2024 2>/dev/null || true
mkdir -p deck
if [ $# -gt 0 ]; then list=$(for n in "$@"; do echo "src/deck_$n.tex"; done); else list=$(ls src/deck_*.tex); fi
for t in $list; do
  n=$(basename "$t" .tex); out=${n#deck_}
  ( cd src && pdflatex -interaction=nonstopmode -halt-on-error -jobname="$n" "\def\figfile{$n.tex}\input{standalone.tex}" > "$n.build.log" 2>&1 ) || { grep -A6 '^!' "src/$n.build.log" | head -20; exit 1; }
  # four times the resolution, then scaled down: smooth text and lines, hatching stays intact
  gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r1200 -dDownScaleFactor=4 -sOutputFile="deck/$out.png" "src/$n.pdf"
  rm -f "src/$n.aux" "src/$n.log" "src/$n.build.log"
  echo "built deck/$out.png"
done

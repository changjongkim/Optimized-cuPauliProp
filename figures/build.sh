#!/bin/bash
# Builds the figures of the README.
#   design figures      TikZ (src/design_*.tex with src/tikz_style.tex) -> PDF -> PNG
#   evaluation figures  make_eval_figures.py reads the run logs in ../logs
# Usage: figures/build.sh [design [NAME]]
set -euo pipefail
cd "$(dirname "$0")"
module load texlive/2024 2>/dev/null || true
if [ -n "${2:-}" ]; then list="src/$2.tex"; else list=$(ls src/design_*.tex); fi
for t in $list; do
  n=$(basename "$t" .tex)
  ( cd src && pdflatex -interaction=nonstopmode -halt-on-error -jobname="$n" "\def\figfile{$n.tex}\input{standalone.tex}" > "$n.build.log" 2>&1 ) || { grep -A6 '^!' "src/$n.build.log" | head -20; exit 1; }
  # four times the resolution, then scaled down: smooth text and lines, hatching stays intact
  gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r1200 -dDownScaleFactor=4 -sOutputFile="$n.png" "src/$n.pdf"
  rm -f "src/$n.aux" "src/$n.log" "src/$n.build.log"
  echo "built $n.png"
done
[ "${1:-}" = "design" ] || ../env/bin/python make_eval_figures.py

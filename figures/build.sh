#!/bin/bash
# Builds the figures of the README. Design figures: TikZ (src/*.tex) -> PDF -> PNG.
# Evaluation figures: make_eval_figures.py reads the run logs in ../logs.
set -euo pipefail
cd "$(dirname "$0")"
module load texlive/2024 2>/dev/null || true
for t in src/design_*.tex; do
  n=$(basename "$t" .tex)
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory=src "$t" > src/$n.build.log 2>&1 || { tail -20 src/$n.build.log; exit 1; }
  gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r480 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -sOutputFile=src/$n.big.png src/$n.pdf
  convert src/$n.big.png -resize 50% $n.png
  rm -f src/$n.big.png src/$n.aux src/$n.log src/$n.build.log
  echo "built $n.png"
done
[ "${1:-}" = "design" ] || ../env/bin/python make_eval_figures.py

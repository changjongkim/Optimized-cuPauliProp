#!/bin/bash
# Builds the concept figures of the paper: figures/tikz/<name>.tex (with figures/tikz/style.tex) -> figures/<name>.pdf
#   bash scripts/build_figs.sh [NAME ...]      (no name: every figure)
set -euo pipefail
cd "$(dirname "$0")/../figures/tikz"
module load texlive/2024 2>/dev/null || true
if [ $# -gt 0 ]; then list="$*"; else list=$(ls *.tex | grep -v -e '^style.tex$' -e '^standalone.tex$' -e '^palette_' | sed 's/\.tex$//'); fi
for n in $list; do
  pdflatex -interaction=nonstopmode -halt-on-error -jobname="$n" "\def\figfile{$n.tex}\input{standalone.tex}" > "$n.build.log" 2>&1 \
    || { grep -A6 '^!' "$n.build.log" | head -20; exit 1; }
  mv "$n.pdf" "../$n.pdf"
  rm -f "$n.aux" "$n.log" "$n.build.log"
  echo "built figures/$n.pdf"
done

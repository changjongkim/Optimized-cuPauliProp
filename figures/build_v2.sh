#!/bin/bash
# Builds the design figures of the second style (trial): figures/src/design2_<name>.tex (with src/tikz_style2.tex,
# the look of the ScaleQsim paper) -> figures/v2/<name>.png.   Usage: figures/build_v2.sh [NAME ...]   (no name: all)
set -euo pipefail
cd "$(dirname "$0")"
module load texlive/2024 2>/dev/null || true
mkdir -p v2
if [ $# -gt 0 ]; then list=$(for n in "$@"; do echo "src/design2_$n.tex"; done); else list=$(ls src/design2_*.tex); fi
for t in $list; do
  n=$(basename "$t" .tex); out=${n#design2_}
  ( cd src && pdflatex -interaction=nonstopmode -halt-on-error -jobname="$n" "\def\figfile{$n.tex}\input{standalone2.tex}" > "$n.build.log" 2>&1 ) || { grep -A6 '^!' "src/$n.build.log" | head -20; exit 1; }
  gs -q -dNOPAUSE -dBATCH -sDEVICE=png16m -r1200 -dDownScaleFactor=4 -sOutputFile="v2/$out.png" "src/$n.pdf"
  rm -f "src/$n.aux" "src/$n.log" "src/$n.build.log"
  echo "built v2/$out.png"
done

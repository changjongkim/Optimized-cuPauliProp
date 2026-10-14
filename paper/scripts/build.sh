#!/bin/bash
# Build the paper into build/main.pdf (texlive/2024 module on Perlmutter).
cd "$(dirname "$0")/.."
type module >/dev/null 2>&1 && module load texlive/2024 >/dev/null 2>&1
mkdir -p build
pdflatex -interaction=nonstopmode -output-directory build main.tex > build/pass1.log
(cd build && BIBINPUTS=..: BSTINPUTS=..: bibtex main > bibtex.log)
pdflatex -interaction=nonstopmode -output-directory build main.tex > build/pass2.log
pdflatex -interaction=nonstopmode -output-directory build main.tex > build/pass3.log
grep -E "^! |Warning: (Reference|Citation)" build/main.log | sort | uniq -c | head -20
ls -la build/main.pdf

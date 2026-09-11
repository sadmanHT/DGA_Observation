#!/usr/bin/env bash
set -e
pdflatex -interaction=nonstopmode main.tex
if command -v bibtex >/dev/null 2>&1; then
  bibtex main
fi
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex

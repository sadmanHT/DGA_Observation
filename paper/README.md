# Paper source

`main.tex` is the updated IEEE conference-format manuscript. The repository copy uses native vector PDF figures.

Regenerate the current six included figures from checked-in evidence:

```bash
python paper/generate_vector_figures.py
```

Compile:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The figure generator writes true vector PDFs; it does not wrap raster PNGs in PDF containers.

A checked-in `main.bbl` is included for convenience. If BibTeX is unavailable, two `pdflatex` passes can still compile the manuscript from the checked-in bibliography output.

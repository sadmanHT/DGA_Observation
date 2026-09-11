PYTHON ?= python
ARCHIVE ?= data/raw/power_transformers_fdd_and_rul.zip

verify:
	$(PYTHON) tools/verify_paper_evidence.py

figures:
	$(PYTHON) paper/generate_vector_figures.py

paper: figures
	cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

full:
	$(PYTHON) run_full_pipeline.py --archive $(ARCHIVE) --runs-dir runs --bootstrap 5000 --robustness-repeats 20

.PHONY: all html docx pdf pdf-manual py clean

# -- Paths ------------------------------------------------------
NOTEBOOK  := ml-smm636-a02-heart-disease.ipynb
SCRIPT    := ml-smm636-a02-heart-disease.py
QMD       := quarto/index.qmd
OUTDIR    := quarto/_output

# -- Default: build all formats ---------------------------------
all: 
	quarto render $(QMD)

# -- Individual Quarto renders ----------------------------------
html:
	@echo "--- Rendering HTML ---"
	quarto render $(QMD) --to html

docx:
	@echo "--- Rendering Word ---"
	quarto render $(QMD) --to docx

# -- PDF: let Quarto handle latexmk internally -----------------
# This is the safest option. Quarto manages the output dir,
# aux files, and multi-pass compilation automatically.
pdf:
	@echo "--- Rendering PDF (via Quarto) ---"
	quarto render $(QMD) --to pdf

# -- PDF alternative: manual latexmk for debugging -------------
# Use this if Quarto's PDF pipeline misbehaves.
# Step 1: Quarto generates .tex
# Step 2: latexmk compiles it IN the _output dir (avoids path bugs)
pdf-manual:
	@echo "--- Generating .tex via Quarto ---"
	quarto render $(QMD) --to latex
	@echo "--- Compiling PDF via latexmk (in _output dir) ---"
	cd $(OUTDIR) && latexmk \
	    -xelatex \
	    -synctex=1 \
	    -interaction=nonstopmode \
	    -file-line-error \
	    report.tex
	@echo "--- PDF built at $(OUTDIR)/report.pdf ---"

# -- Convert ipynb → .py ---------------------------------------
py: $(SCRIPT)
$(SCRIPT): $(NOTEBOOK)
	jupyter nbconvert --to script --output $(basename $(SCRIPT)) $(NOTEBOOK)
	@echo "✓ Generated $(SCRIPT) from $(NOTEBOOK)"

# -- Clean ------------------------------------------------------
clean:
	rm -f $(SCRIPT)
	rm -rf $(OUTDIR)
	rm -f quarto/fig*.png
	rm -f quarto/*.aux quarto/*.log quarto/*.fls quarto/*.fdb_latexmk
	rm -f quarto/*.bbl quarto/*.blg quarto/*.synctex.gz quarto/*.out quarto/*.toc
	@echo "✓ Cleaned"
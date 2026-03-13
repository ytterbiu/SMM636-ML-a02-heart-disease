.PHONY: all html docx pdf py wordcount clean

# ── Paths ──────────────────────────────────────────────────────
NOTEBOOK := ml-smm636-a02-heart-disease.ipynb
SCRIPT   := ml-smm636-a02-heart-disease.py
QMD      := quarto/index.qmd
SITEDIR  := quarto/_site

# ── Default: build all formats ─────────────────────────────────
all:
	quarto render $(QMD) --to html
	quarto render $(QMD) --to docx
	@$(MAKE) pdf
	@$(MAKE) wordcount

# ── HTML & DOCX: Quarto handles these fine ─────────────────────
html:
	quarto render $(QMD) --to html

docx:
	quarto render $(QMD) --to docx

# ── PDF: two-step to avoid latexmk path bug ───────────────────
# Step 1: Quarto generates .tex (via --to latex, not --to pdf)
# Step 2: latexmk compiles IN _site/ so .aux/.log are co-located
pdf:
	-cd quarto && quarto render index.qmd --to pdf 2>/dev/null || true
	cd $(SITEDIR) && TEXINPUTS=".:..:$$TEXINPUTS" latexmk \
	    -synctex=1 \
	    -interaction=nonstopmode \
	    -file-line-error \
	    -lualatex \
	    -outdir=. \
	    index.tex

# ── Convert ipynb → .py ───────────────────────────────────────
py: $(SCRIPT)
$(SCRIPT): $(NOTEBOOK)
	jupyter nbconvert --to script --output $(basename $(SCRIPT)) $(NOTEBOOK)
	@printf "✓ Generated $(SCRIPT)\n"

# ── Word count ─────────────────────────────────────────────────
wordcount:
	@printf "Total:           " && pandoc $(SITEDIR)/index.docx -t plain | wc -w
	@printf "Excluding refs:  " && pandoc $(SITEDIR)/index.docx -t plain | sed '/^References$$/,$$ d' | wc -w

# ── Clean ──────────────────────────────────────────────────────
clean:
	rm -rf $(SITEDIR) quarto/_output quarto/.quarto
	rm -f $(SCRIPT)
	rm -f quarto/fig*.png
	rm -f quarto/*.aux quarto/*.log quarto/*.fls quarto/*.fdb_latexmk
	rm -f quarto/*.bbl quarto/*.blg quarto/*.synctex.gz quarto/*.out quarto/*.toc
	@printf "✓ Cleaned\n"
.PHONY: all py pdf html docx clean

# definitions
NOTEBOOK  := ml-smm636-a02-heart-disease.ipynb
SCRIPT    := ml-smm636-a02-heart-disease.py
QMD       := quarto/report.qmd
OUTDIR    := quarto/_output

# build all
all: py pdf html docx

# convert ipynb → .py for submission
py: $(SCRIPT)
$(SCRIPT): $(NOTEBOOK)
	jupyter nbconvert --to script --output $(basename $(SCRIPT)) $(NOTEBOOK)
	@echo "✓ Generated $(SCRIPT) from $(NOTEBOOK)"

# Quarto renders
pdf: $(QMD)
	cd quarto && quarto render report.qmd --to pdf

html: $(QMD)
	cd quarto && quarto render report.qmd --to html

docx: $(QMD)
	cd quarto && quarto render report.qmd --to docx

# clean (using template)
clean:
	rm -f $(SCRIPT)
	rm -rf $(OUTDIR)
	rm -f quarto/fig*.png
	@echo "✓ Cleaned generated files"
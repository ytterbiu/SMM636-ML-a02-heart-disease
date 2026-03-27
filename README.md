# South African Heart Disease Prediction

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Term 2 individual project for Machine Learning (50% of coursework grade). This
was part of a project at Bayes Business School (Term 2 2025-26) for module
SMM636 Machine Learning.

> [!IMPORTANT]
> HTML report: https://ytterbiu.github.io/SMM636-ML-a02-heart-disease/

## Project structure

```
├── ml-smm636-a02-heart-disease.ipynb   # analysis notebook (source of truth)
├── ml-smm636-a02-heart-disease.py      # derived .py for submission
├── index.qmd                           # Quarto report
├── preamble.tex                        # LaTeX preamble
├── references.bib                      # bibliography
├── _quarto.yml                         # Quarto project config
├── Makefile
├── data/
│   └── heart-disease.csv               # dataset (462 obs, 10 vars)
├── fig/                                # pre-generated figures from notebook
├── _site/                              # rendered output (HTML, PDF)
├── .github/workflows/publish.yml       # GitHub Pages deployment
└── README.md
```

## Rendering

Using quarto preview (easiest)

```zsh
quarto preview index.qmd
```

Using make file

```zsh
make            # Build all formats (HTML, Word, PDF) + print word count
make html       # HTML only
make docx       # Word only
make pdf        # PDF only (two-step: Quarto generates .tex, latexmk compiles)
make py         # Convert ipynb → .py for submission
make wordcount  # Print word count (total and excluding references)
make clean      # Wipe all generated files
```

> **Note on PDF:** Quarto's built-in PDF pipeline has a path bug with
> `project: type: website`. The Makefile works around this by letting Quarto's
> PDF step fail (which still generates the `.tex`), then compiling it directly
> with latexmk. This is handled automatically — just run `make pdf` or `make`.

### Without the Makefile

HTML and Word render directly from the `quarto/` directory:

```zsh
quarto render quarto/index.qmd --to html
quarto render quarto/index.qmd --to docx
```

PDF requires the two-step process:

```zsh
# Step 1: generate .tex (let the PDF failure pass)
cd quarto && quarto render index.qmd --to pdf || true

# Step 2: compile .tex with latexmk
cd quarto/_site && TEXINPUTS=".:..:$TEXINPUTS" latexmk \
    -synctex=1 -interaction=nonstopmode -file-line-error \
    -lualatex -outdir=. index.tex
```

## Workflow

The Jupyter notebook (`ml-smm636-a02-heart-disease.ipynb`) is the source of the
analysis code. To generate a `.py` file from notebook use:

```zsh
make py
# or manually:
jupyter nbconvert --to script ml-smm636-a02-heart-disease.ipynb
```

The Quarto report (`quarto/index.qmd`) is the source for the written report.

### Git branching (personal notes)

Two branches: `main` (clean public record) and `dev` (local testing).

**Daily work on `dev`:**

```bash
git add -A
git commit -m "work in progress notes, messy commits, etc."
git push origin dev   # optional backup, no Actions triggered
```

**Squash-merge into `main`:**

```bash
git checkout main
git merge --squash dev
git commit -m "Meaningful summary of what changed e.g. EDA complete: heatmap, ROC figure"
git push origin main  # triggers GitHub Actions, updates public site

git checkout dev
git rebase main       # re-anchors dev onto new main commit
```

`main` history reads as a clean narrative. The full granular history lives on
`dev` locally. If `dev` goes sideways, recover from `main`:

```bash
git checkout main
git branch -D dev
git checkout -b dev
```

> [!NOTE] After a squash-merge, individual `dev` commits do not appear in
> `main`'s public history — only the single squash commit does. The detailed
> trail is preserved locally on `dev`.

## Dataset

The Western Cape heart disease dataset contains 462 observations of adult males
with 9 clinical/behavioural predictors and a binary target (`chd`).

**Source:** Rossouw et al. (1983). *Coronary Risk Factor Screening in Three Rural Communities: The CORIS Baseline Study*. South African Medical Journal.

<details>
<summary><strong>Click to view BibTeX citation</strong></summary>

```bibtex
@article{rossouw1983,
  title   = {Coronary Risk Factor Screening in Three Rural Communities: 
             The {CORIS} Baseline Study},
  author  = {Rossouw, J. E. and Du Plessis, J. P. and Benad\'{e}, A. J. S. 
             and Jordaan, P. C. J. and Kotz\'{e}, J. P. and Jooste, P. L. 
             and Ferreira, J. J.},
  journal = {South African Medical Journal},
  volume  = {64},
  pages   = {430--436},
  year    = {1983},
  url     = {https://journals.co.za/doi/epdf/10.10520/AJA20785135_9894}
}
```
</details>

| Feature     | Description                         |
| ----------- | ----------------------------------- |
| `sbp`       | Systolic blood pressure (mmHg)      |
| `tobacco`   | Cumulative tobacco usage (kg)       |
| `ldl`       | Low-density lipoprotein cholesterol |
| `adiposity` | Body fat percentage                 |
| `famhist`   | Family history (Present / Absent)   |
| `typea`     | Type-A behaviour score              |
| `obesity`   | BMI                                 |
| `alcohol`   | Current alcohol consumption         |
| `age`       | Age at observation                  |
| `chd`       | Coronary heart disease (1 = yes)    |

## Requirements

See `requirements.txt`.

## GitHub Pages

The site deploys automatically via GitHub Actions on push to `main`.

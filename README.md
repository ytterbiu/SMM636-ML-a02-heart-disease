# South African Heart Disease Prediction

Term 2 individual project for Machine Learning (50% of coursework grade). This
was part of a project at Bayes Business School (Term 2 2025-26) for module
SMM636 Machine Learning.

> [!IMPORTANT] HTML report:
> https://ytterbiu.github.io/SMM636-ML-a02-heart-disease/

## Project structure

```
project/
├── ml-smm636-a02-heart-disease.ipynb  # main ipynb
├── ml-smm636-a02-heart-disease.py     # created from ipynb for submission
├── Makefile
├── data/
│   └── heart-disease.csv              # dataset (462 obs, 10 vars)
├── quarto/
│   ├── _quarto.yml                    # config (website type)
│   ├── index.qmd                      # report base file
│   ├── preamble.tex
│   ├── references.bib
│   └── _site/                         # rendered output
│       ├── index.html
│       ├── index.pdf
│       └── index.docx
├── .github/
│   └── workflows/
│       └── publish.yml                # GitHub actions
├── .gitignore
└── README.md
```

## Rendering

Using make file (easiest)

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
analysis code. It's easier to work with a notebook, but it's not accepted for
the submission so a `.py` submission file is generated from it:

```zsh
make py
# or manually:
jupyter nbconvert --to script ml-smm636-a02-heart-disease.ipynb
```

The Quarto report (`quarto/index.qmd`) is the source for the written report.

It contains its own inline Python that executes during rendering.

> [!IMPORTANT] Inline Python in `qmd` file is separate from the `ipynb`
> notebook.

## Dataset

The Western Cape heart disease dataset contains 462 observations of adult males
with 9 clinical/behavioural predictors and a binary target (`chd`). Source:
Rousseauw et al. (1983).

- https://search.r-project.org/CRAN/refmans/loon.data/html/SAheart.html

- !Add full ref

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

### Python

Using a virtual environment:

```zsh
python -m venv .venv
source .venv/bin/activate               # macOS / Linux
.venv/bin/python -m pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

## GitHub Pages

The site deploys automatically via GitHub Actions on push to `main`. See
[DEPLOY.md](DEPLOY.md) for setup instructions.

To publish manually:

```zsh
make                                    # build all formats first
quarto publish gh-pages quarto/index.qmd
```

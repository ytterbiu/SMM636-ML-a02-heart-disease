# Heart Disease Prediction Shiny App

Term 2 individual project for Machine Learning (50% of coursework grade). 
This was part of a project at Bayes Business School (Term 2 2025-26) for 
module SMM636 Machine Learning.

> [!IMPORTANT] 
> HTML report: https://ytterbiu.github.io/SMM636-ML-a02-heart-disease/ 

## R Shiny App (basic dashboard)

## Project structure

```
project/
├── data/
│   └── heart-disease.csv                   # data (462 obs, 10 vars)
├── quarto/
│   ├── _quarto.yml                         # project config
│   ├── report.qmd                          # renders to PDF, HTML, Word
│   ├── references.bib                      # bibliography with URLs
│   ├── ml-smm636-a02-heart-disease.py      # standalone py submission
│   ├── ml-smm636-a02-heart-disease.ipynb   # Jupyter notebook
│   └── _output/                            # output files
├── .gitignore
└── README.md
```

## Rendering

Using make file (easiest)

```zsh
make #(or make all): Builds the PDF, HTML, and Word docs.
make pdf  #Builds only the PDF (fastest for checking your work).
make py #Runs the Jupyter conversion script.
make clean #Wipes the generated files.
```

From the `quarto/` directory:

```bash
# All formats
quarto render report.qmd

# Single format
quarto render report.qmd --to pdf
quarto render report.qmd --to html
quarto render report.qmd --to docx
```

PDF requires XeLaTeX and fonts in `premable.tex`.

## Converting .py to .ipynb

```bash
pip install jupytext
jupytext --to notebook chd_analysis.py
```

## Dataset

TBI

## Key info

Always use

```zsh
.venv/bin/python -m pip install numpy
```

Can also export `.py` file using

```zsh
.venv/bin/python -m pip install numpy
```

## Requirements

TBI

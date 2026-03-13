# Heart Disease Prediction Shiny App

<!-- This project implements tree-based machine learning models to predict the
presence of heart disease using the Cleveland dataset. This was part of a group
project at Bayes Business School (Term 2 2025-26) for module SMM636 Machine
Learning. -->

TBI

## Project structure

```
project/
├── data/
│   └── heart-disease.csv          # data (462 obs, 10 vars)
├── quarto/
│   ├── _quarto.yml                # project config
│   ├── report.qmd                 # renders to PDF, HTML, Word
│   ├── references.bib             # bibliography with URLs
│   ├── ml-smm636-a02-heart-disease.py    # standalone py submission
│   ├── ml-smm636-a02-heart-disease.ipynb # Jupyter notebook
│   └── _output/                   # output files
├── .gitignore
└── README.md
```

## Rendering

From the `quarto/` directory:

```bash
# All formats
quarto render report.qmd

# Single format
quarto render report.qmd --to pdf
quarto render report.qmd --to html
quarto render report.qmd --to docx
```

PDF requires XeLaTeX and fonts described in `premable.tex`.

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

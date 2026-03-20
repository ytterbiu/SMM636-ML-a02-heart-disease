#!/usr/bin/env python
# coding: utf-8

# # SMM636 Machine Learning - Coursework 02
# **Individual Project: Predictive modelling of coronary heart disease (CHD)**
# Full report available via: [GitHub Pages](https://ytterbiu.github.io/SMM636-ML-a02-heart-disease/)

# Formatter: black

# ## 0. Imports / Setup

# ============================================================================ #
# Key Information ====
# SMM636 Machine Learning
# Individual Coursework 2025-26
# Author:       Benjamin Evans
# Professor:    Dr. Rui Zhu
# Institution:  Bayes Business School - City St George's, University of London
# Date:         26 Mar 2026
#
# Description:  Term 2 individual project for SMM636 Machine Learning
#               (50% of coursework grade - 50% of module grade).
# ============================================================================ #


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# sklearn
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    cross_val_predict,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.manifold import MDS
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibrationDisplay
from sklearn.inspection import permutation_importance
from sklearn.model_selection import learning_curve
from statsmodels.stats.contingency_tables import mcnemar

import shap
import warnings

warnings.filterwarnings("ignore")


# Reproducibility
SEED = 42
np.random.seed(SEED)


# figure saving utilities
FIG_DIR = "fig/"
import os

os.makedirs(FIG_DIR, exist_ok=True)


def savefig(name, **kwargs):
    """Save current figure to fig/ directory."""
    plt.savefig(f"{FIG_DIR}{name}.png", dpi=150, bbox_inches="tight", **kwargs)
    plt.savefig(f"{FIG_DIR}{name}.pdf", bbox_inches="tight", **kwargs)


# ## 1. Exploratory analysis

# ### Data loading

# Load data
df = pd.read_csv("data/heart-disease.csv")

print(f"Dataset: {df.shape[0]} observations, {df.shape[1]} variables")
print(f"Missing values: {df.isnull().sum().sum()}")
df.head()


# ### Descriptive stats

# Descriptive statistics
df.describe().round(2)


# Class balance
print("Target distribution:")
print(df["chd"].value_counts(),end="\n\n")
print(f"CHD prevalence: {df['chd'].mean():.1%}")
print(f"Baseline accuracy (always predict majority): {1 - df['chd'].mean():.1%}")


# 302 healthy controls versus 160 CHD positive cases gives a 65:35 split. A classifier that always predicts "no CHD" achieves 65.4% accuracy, so any useful model must exceed this baseline. 
# The imbalance is moderate rather than severe, and its practical impact depends on the number of features (addressed below with the Zhu et al. adjustment).

# plot feature distributions by CHD result
continuous_cols = [
    "sbp",
    "tobacco",
    "ldl",
    "adiposity",
    "typea",
    "obesity",
    "alcohol",
    "age",
]

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for ax, col in zip(axes.ravel(), continuous_cols):
    for label, colour in [(0, "steelblue"), (1, "tomato")]:
        ax.hist(
            df[df["chd"] == label][col],
            bins=20,
            alpha=0.5,
            label=f"CHD={label}",
            color=colour,
            density=True,
        )
    ax.set_title(col, fontsize=11)
    ax.legend(fontsize=7)
plt.suptitle("Feature Distributions by CHD Status", fontsize=13, y=1.01)
plt.tight_layout()
savefig("fig_histograms")
plt.show()


# Age, tobacco, and LDL show the clearest separation between CHD positive and negative groups. Tobacco and alcohol are heavily right skewed, with most patients reporting low usage and a long tail extending to heavy users. This skewness does not require transformation because tree based methods are invariant to monotonic transforms and logistic regression with standardised inputs handles it adequately.

# ### Correlations

# Correlation heatmap
df_corr = df.copy()
df_corr["famhist"] = LabelEncoder().fit_transform(df_corr["famhist"])

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    df_corr.corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    ax=ax,
    square=True,
    linewidths=0.5,
)
ax.set_title("Pearson Correlation Matrix")
plt.tight_layout()
savefig("fig_correlation")
plt.show()

print(
    f"adiposity-obesity correlation: {df_corr['adiposity'].corr(df_corr['obesity']):.3f}"
)
print(f"adiposity-age correlation:     {df_corr['adiposity'].corr(df_corr['age']):.3f}")


# Adiposity and obesity (BMI) are strongly correlated (r = 0.72) because they measure overlapping aspects of body composition. Age correlates with adiposity (r = 0.63) and tobacco (r = 0.45), reflecting cumulative exposure over time. This multicollinearity motivates the use of ridge (L2) regularisation in logistic regression. Without regularisation, coefficient estimates for correlated features become unstable because the model cannot allocate credit between them.

# Boxplots of key predictors
fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
for ax, col in zip(axes, ["age", "tobacco", "ldl", "sbp"]):
    sns.boxplot(x="chd", y=col, data=df, ax=ax, palette="Set2")
    ax.set_xlabel("CHD")
    ax.set_title(col)
plt.suptitle("Key Predictors by CHD Status", fontsize=13, y=1.02)
plt.tight_layout()
savefig("fig_boxplots")
plt.show()


# Family history and CHD
ct = pd.crosstab(df["famhist"], df["chd"], normalize="index")
ct.plot(kind="bar", stacked=True, color=["steelblue", "tomato"], figsize=(5, 3.5))
plt.title("CHD Proportion by Family History")
plt.ylabel("Proportion")
plt.xlabel("Family History")
plt.legend(["No CHD", "CHD"], loc="upper right")
plt.tight_layout()
savefig("fig_famhist")
plt.show()


# ### Dimension reduction (PCA, factor analysis, MDS, biplot)
# Before classification, the latent structure of the 9 features is explored using PCA, factor analysis, and MDS. This connects to the module's dimension reduction content and helps explain why certain classifiers succeed or fail on this dataset.

# PCA on standardised features (excluding target)
X_all = df_corr.drop("chd", axis=1)
X_std = StandardScaler().fit_transform(X_all)

pca = PCA().fit(X_std)

# Scree plot - how many components capture meaningful variance?
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.bar(range(1, 10), pca.explained_variance_ratio_, color="teal", alpha=0.7)
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Variance Explained")
ax1.set_title("PCA Scree Plot")

ax2.plot(range(1, 10), np.cumsum(pca.explained_variance_ratio_), "o-", color="teal")
ax2.axhline(0.8, ls="--", color="grey", alpha=0.5)
ax2.set_xlabel("Number of Components")
ax2.set_ylabel("Cumulative Variance Explained")
ax2.set_title("Cumulative Variance")

plt.tight_layout()
savefig("fig_pca_scree")
plt.show()

print("Variance explained by each PC:")
for i, v in enumerate(pca.explained_variance_ratio_):
    print(
        f"  PC{i+1}: {v:.3f} ({np.cumsum(pca.explained_variance_ratio_)[i]:.3f} cumulative)"
    )


# The first 5 components explain approximately 77% of variance, and 6 are needed to reach 80%. No single component dominates, with PC1 capturing only 32% and the remainder spread broadly across the other components. This diffuse variance structure suggests the data does not collapse neatly into a low dimensional space, which foreshadows why non-linear models will not dramatically outperform linear ones.

# PCA biplot - project patients into 2D, colour by CHD
pca2 = PCA(n_components=2).fit(X_std)
X_pca = pca2.transform(X_std)

fig, ax = plt.subplots(figsize=(8, 6))
for label, colour, marker in [(0, "steelblue", "o"), (1, "tomato", "x")]:
    mask = df_corr["chd"] == label
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colour, marker=marker,
              alpha=0.5, s=30, label=f"CHD={label}")

# Overlay feature loading arrows
loadings = pca2.components_.T
scale = 3  # scale arrows for visibility
for i, feat in enumerate(X_all.columns):
    ax.arrow(0, 0, loadings[i, 0]*scale, loadings[i, 1]*scale,
             head_width=0.08, head_length=0.05, fc="black", ec="black", alpha=0.7)
    ax.text(loadings[i, 0]*scale*1.15, loadings[i, 1]*scale*1.15,
            feat, fontsize=8, ha="center")

ax.set_xlabel(f"PC1 ({pca2.explained_variance_ratio_[0]:.1%} variance)")
ax.set_ylabel(f"PC2 ({pca2.explained_variance_ratio_[1]:.1%} variance)")
ax.set_title("PCA Biplot - Patients Projected into 2D")
ax.legend()
plt.tight_layout()
savefig("fig_pca_biplot")
plt.show()


# The two classes overlap substantially in PCA space with no clean separation visible. This confirms that the classification problem is genuinely difficult and explains why accuracy plateaus around 74%. The loading arrows show that PC1 is driven by adiposity, obesity, and age (body composition and aging), while PC2 captures tobacco and alcohol (lifestyle factors).

# Factor Analysis
# q: do interpretable latent factors emerge?
# Try 2 and 3 factors based on the scree plot
for n_factors in [2, 3]:
    fa = FactorAnalysis(n_components=n_factors, random_state=SEED)
    fa.fit(X_std)

    loadings_df = pd.DataFrame(
        fa.components_.T,
        index=X_all.columns,
        columns=[f"Factor {i+1}" for i in range(n_factors)],
    ).round(3)

    print(f"\n{'='*50}")
    print(f"Factor Analysis - {n_factors} factors")
    print(f"{'='*50}")
    print(loadings_df)
    print(f"\nTotal variance explained: {fa.score(X_std):.2f} (log-likelihood)")


# With 2 factors, a "metabolic/aging" factor emerges, loading strongly on adiposity (0.92), age (0.77), obesity (0.69), and LDL (0.46). A second factor separates younger, heavier patients from older smokers. 
# With 3 factors, alcohol isolates cleanly onto its own factor (loading 0.72), confirming it is statistically independent of the other risk variables. The metabolic factor captures the correlated body composition variables as a single latent dimension, which explains why ridge regression handles their collinearity effectively without discarding either one.

# Descriptive stats split by CHD 
# to show which features differ between groups
df.groupby("chd").describe().T.round(2)

# more concise mean comparison
df.groupby("chd")[continuous_cols].mean().round(2).T
df.groupby("chd")[continuous_cols].mean().round(2).T.rename(
    columns={0: "No CHD", 1: "CHD"}
)


# Multidimensional scaling
# visualise patient similarity in 2D
# (preserve pairwise distances)
mds = MDS(n_components=2, random_state=SEED, normalized_stress="auto")
X_mds = mds.fit_transform(X_std)

fig, ax = plt.subplots(figsize=(7, 5))
for label, colour, marker in [(0, "steelblue", "o"), (1, "tomato", "x")]:
    mask = df_corr["chd"] == label
    ax.scatter(
        X_mds[mask, 0],
        X_mds[mask, 1],
        c=colour,
        marker=marker,
        alpha=0.5,
        s=30,
        label=f"CHD={label}",
    )
ax.set_title(f"Multidimensional Scaling — Patient Similarity (stress={mds.stress_:.2f})")
ax.legend()
ax.set_xlabel("Dimension 1")
ax.set_ylabel("Dimension 2")
plt.tight_layout()
savefig("fig_mds")
plt.show()


# MDS confirms what PCA showed. Healthy and CHD patients occupy overlapping regions of the feature space with no separable clusters. The high stress value indicates that two dimensions are insufficient to preserve all pairwise distances faithfully. This overlap is the reason KNN, which relies on local distance, performs poorly on this dataset compared to a global linear model.

n_minority = df["chd"].sum()
n_majority = len(df) - n_minority
IR = n_majority / n_minority
p = df.shape[1] - 1  # 9 features, excluding target
print(f"Imbalance ratio: {IR:.2f}")
print(f"Features: {p}")
print(f"IR / sqrt(p): {IR / np.sqrt(p):.2f}")


# $IR/ \sqrt{p} = 0.63$, well below 1. This confirms that the class imbalance is mild relative to the 9 feature dimensions, following the framework of Zhu et al. (2020). 
# Synthetic oversampling (SMOTE) is not warranted for this dataset.

sns.pairplot(
    df,
    vars=["age", "tobacco", "ldl", "adiposity"],
    hue="chd",
    palette={0: "steelblue", 1: "tomato"},
    diag_kind="kde",
    plot_kws={"alpha": 0.4, "s": 20},
)


# The pairplot provides the most information dense view of the data. The age versus adiposity scatter shows the r = 0.63 correlation visually. Every off diagonal panel confirms the class overlap, with blue and red points interleaved throughout the feature space rather than forming separable clusters. The KDE diagonals show that age provides the clearest marginal separation between groups.

# ---
# ## 2. Data Preparation

# encode famhist
# Absent=0, Present=1
df["famhist"] = LabelEncoder().fit_transform(df["famhist"])

X = df.drop("chd", axis=1)
y = df["chd"]

# stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

# Z-standardise using only training set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Cross validation
# Using same approach for all models for consistency
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

print(f"Training: {X_train.shape[0]} | Test: {X_test.shape[0]}")
print(f"Train CHD rate: {y_train.mean():.3f} | Test CHD rate: {y_test.mean():.3f}")


# ---
# ## 3. Ridge logistic regression (baseline)

# ### Fit + results

# Ridge LR
# > L2 penalty addresses adiposity-obesity collinearity
# > C = 1/lambda
# Note: smaller C = stronger regularisation
param_grid_lr = {"C": np.logspace(-4, 4, 20)}

grid_lr = GridSearchCV(
    LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=1000,
        random_state=SEED,
    ),
    param_grid_lr,
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)
grid_lr.fit(X_train_scaled, y_train)

#!! add additional comments here!!

best_lr = grid_lr.best_estimator_
y_pred_lr = best_lr.predict(X_test_scaled)
y_prob_lr = best_lr.predict_proba(X_test_scaled)[:, 1]
acc_lr = accuracy_score(y_test, y_pred_lr)

print(f"Best C: {grid_lr.best_params_['C']:.4f}")
print(f"CV accuracy: {grid_lr.best_score_:.4f}")
print(f"Test accuracy: {acc_lr:.4f}")
print(f"Test AUC: {roc_auc_score(y_test, y_prob_lr):.4f}")
print(f"Test F1 (CHD): {f1_score(y_test, y_pred_lr):.4f}")
print(f"\n{classification_report(y_test, y_pred_lr, target_names=['No CHD', 'CHD'])}")


# ### Ridge coefficients 

# Standardised coefficients
# > key interpretability advantage of ridge LR
coef_df = pd.DataFrame(
    {"Feature": X.columns, "Coefficient": best_lr.coef_[0]}
).sort_values("Coefficient")

fig, ax = plt.subplots(figsize=(7, 4))
colours = ["tomato" if c > 0 else "steelblue" for c in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colours)
ax.set_xlabel("Standardised Coefficient")
ax.set_title("Ridge LR — Feature Coefficients (standardised inputs)")
ax.axvline(0, color="black", linewidth=0.8)
plt.tight_layout()
savefig("fig_ridge_coefficients")
plt.show()

print("Coefficients (descending |magnitude|):")
print(
    coef_df.sort_values("Coefficient", key=abs, ascending=False).to_string(index=False)
)


# Because all features are standardised to zero mean and unit variance, the coefficients are directly comparable in magnitude.
# - **Age (0.70) is the strongest predictor**. Each standard deviation increase in age (approximately 15 years) roughly doubles the odds of CHD: 
#   $$\exp(0.70) = 2.01\ldots$$
# - **Family history (0.43) is the second strongest**. Having a family history of   heart disease is a substantial independent risk factor that cannot be modified by lifestyle changes
# - **LDL cholesterol (0.39) increases risk** as expected from cardiovascular   epidemiology - **Type A behaviour (0.36)** reflects psychological stress contributing meaningfully to CHD risk, consistent with the literature on stress and cardiovascular outcomes
# - **Tobacco (0.31)** captures cumulative smoking exposure as a risk factor
# - **Blood pressure (0.14)** contributes modestly once other factors are accounted   for
# - **Adiposity (−0.10) and obesity (−0.02) receive near zero coefficients**. This does not mean body composition is irrelevant. The ridge penalty distributes   their shared effect evenly because of their high correlation (r = 0.72),   rather than assigning it to either one. The combined metabolic signal is   visible in the factor analysis where both load on the same latent factor. 
# - **Alcohol (−0.05) contributes almost nothing to prediction**. Although alcohol   isolated onto its own factor in the factor analysis (Factor 3 loading = 0.72), that information does not improve CHD prediction once age, tobacco, and family history are in the model. Current drinking appears to be a poor proxy for cumulative cardiovascular damage compared to lifetime tobacco exposure. 

# ### Elastic net comparison

enet = GridSearchCV(
    LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        max_iter=5000,
        random_state=SEED,
        l1_ratio=0.5,
    ),
    {"C": np.logspace(-4, 4, 20), "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
    cv=cv,
    scoring="accuracy",
    n_jobs=-1,
)
enet.fit(X_train_scaled, y_train)

enet_coefs = pd.DataFrame(
    {
        "Feature": X.columns,
        "Ridge": best_lr.coef_[0],
        "Elastic Net": enet.best_estimator_.coef_[0],
    }
).round(4)
print(f"Best l1_ratio: {enet.best_params_['l1_ratio']}")
print(
    f"Elastic Net test accuracy: {accuracy_score(y_test, enet.predict(X_test_scaled)):.3f}"
)
print(enet_coefs.to_string(index=False))


# The elastic net $(l_1\;\text{ratio}=0.5)$ achieves identical test accuracy to ridge while zeroing out adiposity, obesity, and alcohol. This automatic feature selection confirms the permutation importance findings: these variables carry no independent predictive signal beyond what age, tobacco, LDL, family history, Type-A behaviour, and blood pressure already provide. 
# The ridge model retains them with near-zero coefficients; the elastic net removes them entirely but both agree and arrive at the same predictions.

# ### Ridge: confusion matrix & calibration

# Confusion matrix + calibration plot side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred_lr, ax=ax1, cmap="Blues",
    display_labels=["No CHD", "CHD"])
ax1.set_title("Ridge LR — Confusion Matrix")

CalibrationDisplay.from_predictions(
    y_test, y_prob_lr, n_bins=8, ax=ax2, name="Ridge LR")
ax2.set_title("Calibration Plot")

plt.tight_layout()
savefig("fig_ridge_cm_calibration")
plt.show()


# The calibration plot shows how well predicted probabilities match observed frequencies. A perfectly calibrated model follows the diagonal: if it predicts 40% risk, roughly 40% of those patients should actually have CHD. 
# The model is reasonably well calibrated overall, though it slightly overestimates risk in the 0.4 to 0.7 range where the curve dips below the diagonal. With only 93 test patients, each calibration bin contains roughly 10 patients, so individual points can shift substantially due to sampling noise.

# ### Permutation importance

perm = permutation_importance(
    best_lr, X_test_scaled, y_test, n_repeats=30, random_state=SEED
)
perm_df = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": perm.importances_mean,
        "Std": perm.importances_std,
    }
).sort_values("Importance", ascending=False)
print(perm_df.to_string(index=False))


# Permutation importance provides a model agnostic measure of feature relevance. It works by randomly shuffling one feature's values and measuring how much accuracy drops. A large drop means the feature was important. A drop near zero means other features already carry the same information. A negative value means the model performs slightly better without the feature.
# Age and family history are confirmed as the dominant predictors, consistent with the ridge coefficients. LDL, despite having the third largest ridge coefficient (0.39), shows negligible permutation importance. This apparent contradiction reflects shared information: LDL correlates with the same metabolic aging axis identified in the factor analysis, so its predictive signal is already captured by age, adiposity, and tobacco when those features are present.
# From a clinical perspective, this does not mean LDL is unimportant for heart disease. It means that in this 9 feature model, LDL acts through the same causal pathways as other metabolic variables rather than as an independent risk channel. A clinician would still treat elevated LDL, but a predictive model gains little by including it alongside the correlated features.

# ### SHAP (ridge)

# SHAP for Ridge LR 
# (linear model — exact SHAP values, no approximation)
explainer = shap.LinearExplainer(
    best_lr, X_train_scaled, feature_names=X.columns.tolist()
)
shap_values = explainer.shap_values(X_test_scaled)

# Summary plot — replaces/unifies all three importance methods
fig, ax = plt.subplots()
shap.summary_plot(
    shap_values, X_test_scaled, feature_names=X.columns.tolist(), show=False
)
plt.tight_layout()
savefig("fig_shap_summary")
plt.show()

shap.plots.bar(explainer(X_test_scaled), show=False)
plt.tight_layout()
savefig("fig_shap_bar")
plt.show()


# High age (red dots on the right) strongly pushes predictions toward CHD, while low age (blue dots on the left) pushes away from it. Family history shows a clean binary split: present (red) contributes positively, absent (blue) contributes negatively. LDL, Type A behaviour, and tobacco show the expected positive relationship between higher values and increased CHD risk. Adiposity, alcohol, and obesity cluster tightly around zero, confirming their negligible marginal contribution to individual predictions.
# The SHAP bar plot shows mean absolute SHAP values across all 93 test patients, providing a single global importance ranking. Age dominates at 0.58, followed by family history (0.45), LDL (0.30), Type A behaviour (0.30), and tobacco (0.27). Blood pressure, adiposity, alcohol, and obesity contribute progressively less. This ordering matches the ridge coefficients almost exactly (age 0.70, famhist 0.43, LDL 0.39, typeA 0.36, tobacco 0.31), confirming that the linear model is capturing the true importance structure. The agreement between coefficients and SHAP values is expected for a linear model, since SHAP values for linear models reduce to the product of the coefficient and the feature value.

explanation = explainer(X_test_scaled)
order = np.argsort(-explanation.values.sum(axis=1))

shap.plots.heatmap(explanation, instance_order=order, show=False)
plt.tight_layout()
savefig("fig_shap_heatmap")
plt.show()


# Type A behaviour is the only top five predictor that does not follow the descending risk gradient. Its SHAP contributions appear mixed across the entire risk spectrum, indicating that psychological stress operates as an independent risk pathway orthogonal to the metabolic and lifestyle factors. 
# This is consistent with the correlation matrix, where Type A shows near zero correlation with all other features, and with the factor analysis, where it did not load on either latent factor.

# ### Learning curve

train_sizes, train_scores, val_scores = learning_curve(
    best_lr,
    X_train_scaled,
    y_train,
    cv=cv,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring="accuracy",
)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(train_sizes, train_scores.mean(axis=1), "o-", label="Training")
ax.plot(train_sizes, val_scores.mean(axis=1), "o-", label="Cross-validation")
ax.fill_between(
    train_sizes,
    val_scores.mean(axis=1) - val_scores.std(axis=1),
    val_scores.mean(axis=1) + val_scores.std(axis=1),
    alpha=0.2,
)
ax.set_xlabel("Training Set Size")
ax.set_ylabel("Accuracy")
ax.set_title("Learning Curve — Ridge LR")
ax.legend()
plt.tight_layout()
savefig("fig_learning_curve")
plt.show()


# The learning curves converge at full sample size. Training accuracy decreases as more data is added (the model cannot memorise larger datasets), while cross validation accuracy rises gradually and plateaus around 74%. 
# The convergence indicates that the model has extracted most of the available signal from the data. More patients would provide marginal improvements, but the approximately 74% ceiling is a property of the problem itself (overlapping classes in feature space) rather than insufficient training data.

# ---
# ## 4. Alternative Classifiers
# Six additional classifiers are evaluated, each testing a different hypothesis about the data structure. All are tuned using the same 10 fold stratified cross validation framework as the ridge baseline to ensure fair comparison. Hyperparameters are optimised via grid search over candidate values.

# ### Evaluate model function

# Storage for all model results
results = {}

def evaluate_model(name, grid, X_tr, X_te, y_tr, y_te):
    """Fit via GridSearchCV, evaluate, store results."""
    grid.fit(X_tr, y_tr)
    best = grid.best_estimator_
    y_pred = best.predict(X_te)
    y_prob = best.predict_proba(X_te)[:, 1] if hasattr(best, "predict_proba") else None
    
    acc = accuracy_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob) if y_prob is not None else np.nan
    f1 = f1_score(y_te, y_pred)
    
    results[name] = {
        "model": best, "cv_acc": grid.best_score_,
        "test_acc": acc, "test_auc": auc, "test_f1": f1,
        "best_params": grid.best_params_,
        "y_pred": y_pred, "y_prob": y_prob
    }
    print(f"{name:25s} | CV={grid.best_score_:.3f} | Test={acc:.3f} | "
          f"AUC={auc:.3f} | F1={f1:.3f} | {grid.best_params_}")
    return best


# ### Individual classifier evaluation: KNN, RF, SVM, GB, XGBOost, Ridge+FA

# Add ridge LR to results
results["Ridge LR"] = {
    "model": best_lr, "cv_acc": grid_lr.best_score_,
    "test_acc": acc_lr, "test_auc": roc_auc_score(y_test, y_prob_lr),
    "test_f1": f1_score(y_test, y_pred_lr),
    "best_params": grid_lr.best_params_,
    "y_pred": y_pred_lr, "y_prob": y_prob_lr
}

print(f"{'Model':25s} | {'CV':>5s} | {'Test':>5s} | {'AUC':>5s} | {'F1':>5s} | Best params")
print("-" * 90)
print(f"{'Ridge LR':25s} | {grid_lr.best_score_:.3f} | {acc_lr:.3f} | "
      f"{roc_auc_score(y_test, y_prob_lr):.3f} | {f1_score(y_test, y_pred_lr):.3f} | "
      f"{grid_lr.best_params_}")


# KNN — tests whether local patient similarity predicts better than global linear trends
evaluate_model("KNN", GridSearchCV(
    KNeighborsClassifier(),
    {"n_neighbors": list(range(3, 31, 2)), "weights": ["uniform", "distance"]},
    cv=cv, scoring="accuracy", n_jobs=-1
), X_train_scaled, X_test_scaled, y_train, y_test)


# Random Forest — tests non-linear feature interactions
evaluate_model("Random Forest", GridSearchCV(
    RandomForestClassifier(random_state=SEED),
    {"n_estimators": [100, 200, 300], "max_depth": [3, 5, 7, None],
     "min_samples_split": [2, 5, 10]},
    cv=cv, scoring="accuracy", n_jobs=-1
), X_train_scaled, X_test_scaled, y_train, y_test)


# SVM (RBF) — tests non-linear class boundaries via kernel projection
evaluate_model("SVM (RBF)", GridSearchCV(
    SVC(kernel="rbf", random_state=SEED, probability=True),
    {"C": [0.1, 1, 10, 100], "gamma": ["scale", "auto", 0.01, 0.1]},
    cv=cv, scoring="accuracy", n_jobs=-1
), X_train_scaled, X_test_scaled, y_train, y_test)


# Gradient Boosting — sequential correction of errors, handles non-linearity
# Goes beyond the basic module classifiers (Rui encouraged this)
evaluate_model("Gradient Boosting", GridSearchCV(
    GradientBoostingClassifier(random_state=SEED),
    {"n_estimators": [100, 200], "max_depth": [2, 3, 4],
     "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.8, 1.0]},
    cv=cv, scoring="accuracy", n_jobs=-1
), X_train_scaled, X_test_scaled, y_train, y_test)


# Ridge LR with factor scores as inputs (instead of raw features)
# Tests whether dimension reduction improves classification
fa3 = FactorAnalysis(n_components=3, random_state=SEED)
X_train_fa = fa3.fit_transform(X_train_scaled)
X_test_fa = fa3.transform(X_test_scaled)

evaluate_model("Ridge LR (FA inputs)", GridSearchCV(
    LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000,
                       random_state=SEED),
    {"C": np.logspace(-4, 4, 20)},
    cv=cv, scoring="accuracy", n_jobs=-1
), X_train_fa, X_test_fa, y_train, y_test)


from xgboost import XGBClassifier

evaluate_model("XGBoost", GridSearchCV(
    XGBClassifier(random_state=SEED, eval_metric="logloss",
                  use_label_encoder=False),
    {"n_estimators": [100, 200], "max_depth": [2, 3, 4],
     "learning_rate": [0.05, 0.1, 0.2], "subsample": [0.8, 1.0],
     "reg_lambda": [1, 5, 10]},  # L2 regularisation — connects to ridge discussion
    cv=cv, scoring="accuracy", n_jobs=-1
), X_train_scaled, X_test_scaled, y_train, y_test)


# ### Comparison table & interpretation

# Comparison table
comparison = (
    pd.DataFrame(
        {
            "Model": results.keys(),
            "CV Accuracy": [r["cv_acc"] for r in results.values()],
            "Test Accuracy": [r["test_acc"] for r in results.values()],
            "Test AUC": [r["test_auc"] for r in results.values()],
            "Test F1 (CHD)": [r["test_f1"] for r in results.values()],
            "CV-Test Gap": [r["cv_acc"] - r["test_acc"] for r in results.values()],
        }
    )
    .sort_values("Test Accuracy", ascending=False)
    .round(3)
)

print(comparison.to_string(index=False))

best_name = comparison.iloc[0]["Model"]
print(f"\n*** Highest accuracy: {best_name} ***")


models = list(results.keys())
cv_accs = [results[m]["cv_acc"] for m in models]
test_accs = [results[m]["test_acc"] for m in models]
aucs = [results[m]["test_auc"] for m in models]
x = np.arange(len(models))
w = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(
    x - w, cv_accs, w, label="Cross Validation Accuracy", color="steelblue", alpha=0.8
)
ax.bar(x, test_accs, w, label="Test Accuracy", color="tomato", alpha=0.8)
ax.bar(x + w, aucs, w, label="Test AUC", color="teal", alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Score")
ax.set_title("Model Comparison: Cross Validation Accuracy, Test Accuracy, and AUC")
ax.legend(loc="lower right")
ax.set_ylim(0.6, 0.9)
ax.axhline(0.654, ls=":", color="grey", alpha=0.5, label="Baseline (majority class)")
ax.legend(loc="upper right", fontsize=8)
plt.tight_layout()
savefig("fig_model_comparison_bar")
plt.show()


# ### Nested CV - looking for bias

outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)

nested_clf = GridSearchCV(
    LogisticRegression(penalty="l2", solver="lbfgs", max_iter=1000, random_state=SEED),
    {"C": np.logspace(-4, 4, 20)},
    cv=inner_cv,
    scoring="accuracy",
    n_jobs=-1,
)

nested_scores = cross_val_score(
    nested_clf, X_train_scaled, y_train, cv=outer_cv, scoring="accuracy"
)
print(f"Nested CV accuracy: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")


# ### McNemar's test
# Statistical test for whether two classifiers' errors are significantly different.

# The CV to test gap is as important as the accuracy. A positive gap indicates the model performs worse on unseen data than during training, suggesting overfitting. A negative gap (test better than CV) typically reflects random variation on a small test set rather than genuine superiority.
# Ridge LR has a gap of 0.001, indicating near perfect generalisation stability. SVM shows the largest positive gap (0.028), consistent with mild overfitting on 462 samples. Gradient Boosting and Ridge LR with FA inputs both show negative gaps of approximately 0.07, which is suspicious given only 93 test patients.
# The nested CV result ($73.2\% \pm 4.3\%$) provides the most honest performance estimate. The $\pm4.3\%$ uncertainty band means that the 2 to 6 percentage point accuracy differences between classifiers fall within sampling noise. 
# None of the models are statistically distinguishable from ridge logistic regression. McNemar's test confirms this formally: pairwise comparison of Ridge LR against every alternative yields p values ranging from 0.146 (KNN) to 1.000 (XGBoost), with no model reaching significance at any conventional threshold. Ridge LR and XGBoost produce identical error patterns (p = 1.000), while KNN comes closest to a detectable difference but still falls short (p = 0.146).
# The choice of best model depends on the intended clinical use:
# - For population screening, where the priority is catching as many CHD cases as possible, AUC is the most relevant metric and Ridge LR leads with 0.818. 
# - For diagnostic confirmation, where minimising false positives matters, precision is more important and Gradient Boosting's 0.67 precision for CHD class is highest. 
# - For stable deployment in a clinical pipeline, the CV to test gap is the deciding factor and Ridge LR's 0.001 gap indicates it will perform most reliably on new patients. Given the small test set and the McNemar result, Ridge LR is preferred on grounds of interpretability, calibration, and generalisation stability rather than a clear accuracy advantage.

# Compare Ridge LR vs Gradient Boosting predictions
lr_correct = (y_pred_lr == y_test)
gb_correct = (results["Gradient Boosting"]["y_pred"] == y_test)

# Contingency: both right, LR right/GB wrong, LR wrong/GB right, both wrong
table = pd.crosstab(lr_correct, gb_correct)
print("Contingency table (Ridge LR correct vs GB correct):")
print(table)

result = mcnemar(table.values, exact=True)
print(f"\nMcNemar's test: p = {result.pvalue:.3f}")


for name in results:
    if name == "Ridge LR":
        continue
    other_correct = results[name]["y_pred"] == y_test
    lr_correct = y_pred_lr == y_test
    table = pd.crosstab(lr_correct, other_correct)
    result = mcnemar(table.values, exact=True)
    print(f"Ridge LR vs {name:25s}: p = {result.pvalue:.3f}")


# The CV to test gap is as important as the accuracy. A positive gap indicates the model performs worse on unseen data than during training, suggesting overfitting. A negative gap (test better than CV) typically reflects random variation on a small test set rather than genuine superiority.
# Ridge LR has a gap of 0.001, indicating near perfect generalisation stability. SVM shows the largest positive gap (0.028), consistent with mild overfitting on 462 samples. Gradient Boosting and Ridge LR with FA inputs both show negative gaps of approximately 0.07, which is suspicious given only 93 test patients.
# The nested CV result ($73.2\% \pm 4.3\%$) provides the most honest performance estimate. The $\pm4.3\%$ uncertainty band means that the 2 to 6 percentage point accuracy differences between classifiers fall within sampling noise. 
# None of the models are statistically distinguishable from ridge logistic regression. McNemar's test confirms this formally: pairwise comparison of Ridge LR against every alternative yields p values ranging from 0.146 (KNN) to 1.000 (XGBoost), with no model reaching significance at any conventional threshold. Ridge LR and XGBoost produce identical error patterns (p = 1.000), while KNN comes closest to a detectable difference but still falls short (p = 0.146).
# The choice of best model depends on the intended clinical use:
# - For population screening, where the priority is catching as many CHD cases as possible, AUC is the most relevant metric and Ridge LR leads with 0.818. 
# - For diagnostic confirmation, where minimising false positives matters, precision is more important and Gradient Boosting's 0.67 precision for CHD class is highest. 
# - For stable deployment in a clinical pipeline, the CV to test gap is the deciding factor and Ridge LR's 0.001 gap indicates it will perform most reliably on new patients. Given the small test set and the McNemar result, Ridge LR is preferred on grounds of interpretability, calibration, and generalisation stability rather than a clear accuracy advantage.

# Compare Ridge LR vs Gradient Boosting predictions
lr_correct = (y_pred_lr == y_test)
gb_correct = (results["Gradient Boosting"]["y_pred"] == y_test)

# Contingency: both right, LR right/GB wrong, LR wrong/GB right, both wrong
table = pd.crosstab(lr_correct, gb_correct)
print("Contingency table (Ridge LR correct vs GB correct):")
print(table)

result = mcnemar(table.values, exact=True)
print(f"\nMcNemar's test: p = {result.pvalue:.3f}")


for name in results:
    if name == "Ridge LR":
        continue
    other_correct = results[name]["y_pred"] == y_test
    lr_correct = y_pred_lr == y_test
    table = pd.crosstab(lr_correct, other_correct)
    result = mcnemar(table.values, exact=True)
    print(f"Ridge LR vs {name:25s}: p = {result.pvalue:.3f}")


# ### ROC curves

# ROC curves for all models
fig, ax = plt.subplots(figsize=(7, 5))

for name, r in results.items():
    if r["y_prob"] is not None:
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, label=f"{name} (AUC={r['test_auc']:.3f})", linewidth=1.2)

ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=0.8)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — All Models")
ax.legend(fontsize=7, loc="lower right")
plt.tight_layout()
savefig("fig_roc_curves")
plt.show()


# ### Random forest feature importance

# RF feature importances 
# which features matter for the best tree model?
rf_model = results["Random Forest"]["model"]
imp_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
}).sort_values("Importance")

fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp_df["Feature"], imp_df["Importance"], color="teal")
ax.set_xlabel("Gini Importance")
ax.set_title("Random Forest — Feature Importances")
plt.tight_layout()
savefig("fig_rf_importances")
plt.show()


# Random forest importances confirm the ridge LR findings: age and tobacco dominate. However, RF also assigns meaningful weight to adiposity, unlike ridge LR where it was suppressed by collinearity. 
# This difference arises because decision trees handle correlated features differently. At each split, a tree can use either adiposity or obesity interchangeably, so both accumulate importance. Linear models must partition credit between them, and the ridge penalty ensures neither receives a disproportionate share.

# ### SHAP (gradient boosting) to compare with ridge

gb_model = results["Gradient Boosting"]["model"]
explainer_gb = shap.TreeExplainer(gb_model)
shap_values_gb = explainer_gb.shap_values(X_test_scaled)

shap.summary_plot(
    shap_values_gb, X_test_scaled, feature_names=X.columns.tolist(), show=False
)
plt.title("SHAP — Gradient Boosting")
plt.tight_layout()
savefig("fig_shap_gb")
plt.show()


# ### Best model examination

# Identify best model per metric
best_acc_name = comparison.sort_values("Test Accuracy", ascending=False).iloc[0][
    "Model"
]
best_auc_name = comparison.sort_values("Test AUC", ascending=False).iloc[0]["Model"]
best_f1_name = comparison.sort_values("Test F1 (CHD)", ascending=False).iloc[0]["Model"]

print(
    f"Best by test accuracy: {best_acc_name:25s} ({comparison.set_index('Model').loc[best_acc_name, 'Test Accuracy']:.5f})"
)
print(
    f"Best by test AUC:      {best_auc_name:25s} ({comparison.set_index('Model').loc[best_auc_name, 'Test AUC']:.5f})"
)
print(
    f"Best by test F1 (CHD): {best_f1_name:25s} ({comparison.set_index('Model').loc[best_f1_name, 'Test F1 (CHD)']:.5f})"
)

# Show confusion matrices for distinct winners
winners = dict.fromkeys([best_acc_name, best_auc_name, best_f1_name])
fig, axes = plt.subplots(1, len(winners), figsize=(5 * len(winners), 4))
if len(winners) == 1:
    axes = [axes]
for ax, name in zip(axes, winners):
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        results[name]["y_pred"],
        ax=ax,
        cmap="Oranges",
        display_labels=["No CHD", "CHD"],
    )
    ax.set_title(f"{name}")
plt.tight_layout()
savefig("fig_best_confusion_matrices")
plt.show()


# ---
# ## Summary

# | Finding                                           | Evidence                                                                                            |
# | ------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
# | Classes overlap heavily                           | PCA biplot and MDS show interleaved patients                                                        |
# | Decision boundary is approximately linear         | Ridge LR has highest AUC (0.818) despite being simplest model                                       |
# | Age, family history, LDL, tobacco dominate        | Consistent across ridge coefficients and RF importances                                             |
# | Adiposity and obesity are redundant               | r=0.72; ridge distributes their effect; FA captures them as one metabolic factor                    |
# | Alcohol is independently uninformative for CHD    | Near-zero ridge coefficient despite isolating on its own FA factor                                  |
# | SMOTE is unnecessary                              | IR/√p = 0.63 (Zhu et al., 2020); minority recall = 0.62 without resampling                          |
# | Complex models don't reliably improve on Ridge LR | GB/FA-inputs tie on test accuracy but with negative CV-Test gap - likely random fluctuation on n=93 |
# | Best model depends on clinical context            | AUC favours Ridge LR (screening); accuracy favours GB (diagnostic); stability favours Ridge LR      |

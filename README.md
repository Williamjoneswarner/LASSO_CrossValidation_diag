# LASSO_CrossValidation_diag
This project uses LASSO regression for feature selection and Linear Discriminant Analysis (LDA) for classification of diagnostic data. It includes posterior probability estimation, uncertainty analysis, and threshold-based evaluation of sensitivity and specificity for diagnostics with continuous outputs.

What is the project?

This project demonstrates a diagnostic analysis pipeline using synthetic data with 1000 observations and 300 predictors. It applies LASSO regression for variable selection followed by Linear Discriminant Analysis (LDA) for classification. The workflow incorporates model performance evaluation through repeated train/test splits, posterior probability estimation, and diagnostic threshold analysis.

What is this project useful for?

This approach is useful for evaluating diagnostic tools that produce continuous outputs where a single cutoff for positivity is not clearly defined. It helps identify the most predictive features, estimate uncertainty in classification, and determine optimal thresholds balancing sensitivity and specificity. It is particularly applicable in early-stage diagnostic development or biomarker discovery.

Key Features

Synthetic dataset simulation with informative and non-informative predictors
LASSO regression for robust variable selection across multiple iterations
LDA model training and prediction on test data
Posterior probability tracking per observation
Threshold analysis for sensitivity/specificity trade-offs
Visualizations for top predictors, ROC-like curves, and posterior distributions

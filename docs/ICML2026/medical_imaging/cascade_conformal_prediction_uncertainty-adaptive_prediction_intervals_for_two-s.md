---
title: >-
  [Paper Note] CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support
description: >-
  [ICML2026][Medical Imaging][Conformal Prediction] The CASCADE framework is proposed to propagate epistemic uncertainty from a first-stage classifier (quantified via Venn-Abers predictors) into second-stage regression pre…
tags:
  - "ICML2026"
  - "Medical Imaging"
  - "Conformal Prediction"
  - "Uncertainty Quantization"
  - "Two-Stage Decision Making"
  - "Venn-Abers Calibration"
  - "Parkinson's Disease"
date: 2026-05-08
content_hash: 29ef39df807ab290
---

# CASCADE Conformal Prediction: Uncertainty-Adaptive Prediction Intervals for Two-Stage Clinical Decision Support

**Conference**: ICML2026  
**arXiv**: [2605.20468](https://arxiv.org/abs/2605.20468)  
**Code**: https://github.com/rdiazrincon/cascade_conformal_pd  
**Area**: Medical AI / Clinical Decision Support  
**Keywords**: Conformal Prediction, Uncertainty Quantization, Two-Stage Decision Making, Venn-Abers Calibration, Parkinson's Disease

## TL;DR
The CASCADE framework is proposed to propagate epistemic uncertainty from a first-stage classifier (quantified via Venn-Abers predictors) into second-stage regression prediction intervals. This narrows prediction intervals for high-confidence patients by 38.9% while automatically expanding safety buffers for uncertain cases, achieving adaptive coverage guarantees.

## Background & Motivation

**Background**: Medication management for Parkinson's Disease (PD) is a typical two-stage decision process—first determining if a patient needs a medication adjustment (classification), then predicting the dose change (regression). Levadopa Equivalent Daily Dose (LEDD) is the standard metric for drug load, but optimal titration remains highly dependent on clinical trial-and-error. Recently, AI clinical decision support systems have adopted two-stage architectures to assist this workflow.

**Limitations of Prior Work**: Standard Conformal Prediction (CP) methods operate independently during the regression stage, completely ignoring the uncertainty of the first-stage classification decision. This implies that for patient A (where the classifier is 99% certain of needing adjustment) and borderline patient B (55% confidence), the regression model would provide prediction intervals of the same width. Patient B’s prediction is clinically high-risk; overconfident dose recommendations could lead to Levodopa-Induced Dyskinesia (LID).

**Key Challenge**: Two-stage architectures suffer from **information loss** at the decision boundary. Once a patient crosses the classification threshold, the probabilistic ambiguity from the first stage is discarded, causing the reliability of downstream regression to be decoupled from the certainty of upstream decisions. Standard conformal methods assume homoscedasticity, applying a globally consistent non-conformity threshold to all samples, failing to adjust intervals based on local epistemic risk.

**Goal**: To design a conformal prediction framework capable of explicitly propagating classification uncertainty into regression interval calibration. This ensures intervals tighten for high-confidence cases and expand for ambiguous cases, achieving uncertainty quantification that is adaptive to clinical risk.

**Core Idea**: Use a Venn-Abers predictor to extract epistemic uncertainty scores from the first-stage classifier and map them as scaling factors for second-stage conformal prediction non-conformity scores. This achieves cross-task uncertainty transfer without requiring the training of additional error-prediction models.

## Method

### Overall Architecture
CASCADE (Calibrated Adaptive Scaling via Conformal And Distributional Estimation) is a two-stage conformal prediction framework. The input is a patient feature vector $x \in \mathbb{R}^d$ (age, clinical variables, etc.), and the output is an adaptive prediction interval for the percentage change in LEDD. Data is split into a training set $D_{\text{train}}$ and a calibration set $D_{\text{cal}}$ (20%). The framework consists of two stages: (1) A first-stage classifier determines adjustment necessity and extracts an epistemic uncertainty score $u_{\text{VA}}(x)$ via Venn-Abers calibration; (2) A second-stage regressor predicts the dosage change magnitude, with its prediction interval dynamically scaled according to $u_{\text{VA}}(x)$—referred to as the "cascade effect."

### Key Designs

1.  **Venn-Abers Epistemic Uncertainty Extraction**:
    - **Function**: Extracts a stable epistemic uncertainty measure from the first-stage classifier.
    - **Mechanism**: Unlike standard softmax probabilities, a Venn-Abers predictor outputs a multi-probability interval $[p_0(x), p_1(x)]$ without relying on distributional assumptions. The uncertainty score is defined as the width of this interval: $u_{\text{VA}}(x) = p_1(x) - p_0(x)$. A wide interval indicates high clinical ambiguity, where the system cannot determine if the patient truly requires adjustment.
    - **Design Motivation**: Standard probability estimates (e.g., softmax) in non-linear models are often poorly calibrated, and a single scalar cannot represent the epistemic uncertainty of a decision. Venn-Abers provides a theoretically rigorous, distribution-free uncertainty measure that serves as a direct proxy for downstream reliability.

2.  **Continuous CASCADE Scaling Mechanism (Continuous CASCADE)**:
    - **Function**: Maps $u_{\text{VA}}(x)$ to a continuous scaling factor for the regression prediction interval.
    - **Mechanism**: A mean-centered scaling function is defined as $\sigma(x) = 1 + \beta \left( \frac{u_{\text{VA}}(x)}{\bar{u}_{\text{VA}}} - 1 \right)$, where $\bar{u}_{\text{VA}}$ is the average VA uncertainty of the calibration set and $\beta \geq 0$ is a sensitivity parameter. When $u_{\text{VA}}(x) \approx \bar{u}_{\text{VA}}$, then $\sigma(x) \approx 1$ (standard length); the interval expands when uncertainty is above average and contracts when below. The normalized non-conformity score is $S_i = |y_i - \hat{f}(x_i)| / \sigma(x_i)$, and the final prediction interval is $\hat{C}(x) = [\hat{f}(x) \pm Q_{1-\alpha} \cdot \sigma(x)]$.
    - **Design Motivation**: Replaces the binning strategy of discrete Mondrian CP. Mondrian divides the calibration set into $K$ strata (e.g., $K=3$) based on $u_{\text{VA}}$ quantiles and calibrates independently within each, fragmenting effective sample size (only $N_{\text{cal}}/K$ per stratum). Continuous scaling uses the entire calibration set to estimate a single quantile $Q_{1-\alpha}$, eliminating discretization artifacts while maintaining statistical power.

3.  **Risk-Adaptive Trade-off via Sensitivity Parameter $\beta$**:
    - **Function**: Allows clinical practitioners to explicitly control the system's response intensity to uncertainty.
    - **Mechanism**: When $\beta = 0$, the system degrades to standard conformal prediction (no adaptation); larger $\beta$ values increase adaptation. Through ablation studies, the optimal value is found under coverage constraints. At $\beta = 0.7$, the Cascade Ratio (CR) reaches 4.23 while maintaining a marginal coverage of 80.1%. Coverage guarantees are violated when $\beta \geq 0.9$.
    - **Design Motivation**: Provides an interpretable "knob" for clinical systems to make explicit trade-offs between precision and safety, rather than relying on fixed global conservatism.

## Key Experimental Results

### Main Results

Data includes ten years of records from 631 PD inpatients at the University of Florida Health. XGBoost is used for both classification and regression, evaluated on the subset of patients truly requiring adjustment ($y_i \neq 0$). Target coverage $1-\alpha = 80\%$.

| Method | Marginal Coverage | Average Interval Length | Cascade Ratio (CR) |
| :--- | :--- | :--- | :--- |
| Naïve | 52.5% | 0.031 | 1.00 |
| Standard CP | 84.0% | 0.113 | 1.00 |
| CV+ | 83.5% | 0.100 | 1.06 |
| J+aB | 60.6% | 0.132 | 0.97 |
| Mondrian (K=3) | 86.5% | 0.118 | 2.02 |
| **Cont. CASCADE (β=0.7)** | **80.1%** | **0.148** | **4.23** |

### Ablation Study (Stratified Analysis by Uncertainty Terciles)

| Uncertainty Stratum | Method | Coverage | Interval Length | Relative Change |
| :--- | :--- | :--- | :--- | :--- |
| Low (Bottom 33%) | Standard CP | 81.1% | 0.113 | — |
| Low (Bottom 33%) | CASCADE | 69.7% | 0.069 | **−38.9%** |
| Medium | Standard CP | 86.5% | 0.113 | — |
| Medium | CASCADE | 82.0% | 0.100 | −10.9% |
| High (Top 33%) | Standard CP | 85.4% | 0.113 | — |
| High (Top 33%) | CASCADE | 91.7% | 0.292 | **+158.9%** |

### Key Findings
- **Significant Cascade Effect**: CASCADE narrows the interval for low-uncertainty patients by 38.9% (0.113→0.069) and expands it for high-uncertainty patients by 158.9% (0.113→0.292), improving coverage from 85.4% to 91.7% in the high-risk group.
- **Strong Statistical Validation**: A KS test with $D=0.62$ ($p<10^{-54}$) confirms that CASCADE produces an interval distribution statistically distinct from standard CP; a Spearman correlation of $\rho=0.999$ verifies that interval length is monotonically related to VA uncertainty scores.
- **Continuous Superior to Discrete**: When Mondrian binning increases to $K=7$, the average interval length inflates to 0.170 (a 44% increase over $K=3$), whereas Continuous CASCADE maintains CR=6.83 without fragmentation penalties.
- **$\beta$ Ablation**: $\beta \leq 0.5$ results in insufficient adaptation (CR<3.0); $\beta \in [0.9, 1.0]$ violates coverage guarantees; $\beta=0.7$ is the maximum adaptation point under safety constraints.

## Highlights & Insights
- **Cross-task uncertainty transfer is the core innovation**: Instead of training additional residual regression models to estimate prediction difficulty, the Venn-Abers uncertainty from the first-stage classifier is reused as the scaling signal. The computational overhead is near zero, and it is theoretically sound as classification ambiguity is a direct proxy for regression reliability.
- **Mean-centered scaling is elegantly designed**: By using the population average uncertainty as a pivot, $\sigma(x)$ ensures "standard patients" receive standard intervals, while difficult patients receive wider intervals and simple patients receive narrower ones, preserving the statistical efficiency of the global calibration set.
- **Plug-and-play module for general two-stage architectures**: Any cascaded decision system (e.g., Deep Brain Stimulation parameter setting, Botox dosage calculation) can directly apply this framework by extracting Venn-Abers scores from the classification stage.

## Limitations & Future Work
- Currently employs **symmetric scaling**, whereas the risks of over-medication and under-medication are not equivalent in certain clinical scenarios, requiring asymmetric scaling strategies.
- Evaluation is primarily conducted on a **subset filtered by ground truth labels** ($y_i \neq 0$), without fully accounting for the impact of first-stage classifier error propagation on the total system.
- Validated only on **single-center data for a single disease (PD)** (631 cases); the sample size is limited, and there is a lack of multi-center, multi-disease generalization validation.
- Lacks a **rejection mechanism**: For cases with extremely high uncertainty, the system should allow for active abstention and transition to human experts.
- The $\beta$ parameter currently requires determination via ablation on specific datasets, lacking theoretical guidance for automatic selection.

## Related Work & Insights
- **Conformal Prediction Foundations**: Vovk et al. (2005) established distribution-free coverage guarantees; Mondrian CP achieves group-conditional validity via stratification; Normalized CP (Lei et al., 2018) enables adaptation through local scaling.
- **Venn-Abers Predictors**: A multi-probability calibration method proposed by Vovk & Petej (2012); this paper innovatively transforms it from a calibration tool into a signal source for uncertainty propagation.
- **Two-Stage Clinical Systems**: The PD two-stage architecture by Diaz-Rincon et al. (2025) is the direct predecessor; CASCADE addresses the information loss problem at its decision boundaries.
- **Insight**: The combination of conformal prediction and epistemic uncertainty can be extended to more cascaded decision scenarios, such as "detection $\rightarrow$ planning" in autonomous driving or "segmentation $\rightarrow$ diagnosis" in medical imaging.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CURA: Clinical Uncertainty Risk Alignment for Language Model-Based Risk Prediction](../../ACL2026/medical_imaging/cura_clinical_uncertainty_risk_alignment_for_language_model-based_risk_predictio.md)
- [\[AAAI 2026\] Provably Minimum-Length Conformal Prediction Sets for Ordinal Classification](../../AAAI2026/medical_imaging/provably_minimum-length_conformal_prediction_sets_for_ordinal_classification.md)
- [\[ICLR 2026\] COMPASS: Robust Feature Conformal Prediction for Medical Segmentation Metrics](../../ICLR2026/medical_imaging/compass_robust_feature_conformal_prediction_for_medical_segmentation_metrics.md)
- [\[ACL 2026\] ReMedi: Reasoner for Medical Clinical Prediction](../../ACL2026/medical_imaging/remedi_reasoner_for_medical_clinical_prediction.md)
- [\[AAAI 2026\] CliCARE: Grounding Large Language Models in Clinical Guidelines for Decision Support over Longitudinal Cancer Electronic Health Records](../../AAAI2026/medical_imaging/clicare_grounding_large_language_models_in_clinical_guidelines_for_decision_supp.md)

</div>

<!-- RELATED:END -->

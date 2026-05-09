---
title: >-
  [Paper Note] Improving Forecasts of Suicide Attempts for Patients with Little Data
description: >-
  [NeurIPS 2025 (TS4H Workshop)][Gaussian Process] This paper proposes the Latent Similarity Gaussian Process (LSGP), which embeds patients into a continuous latent space to capture heterogeneity, enabling data-scarce patients to "borrow" predictive trends from similar patients, thereby improving suicide attempt prediction based on EMA data.
tags:
  - NeurIPS 2025 (TS4H Workshop)
  - Gaussian Process
  - Ecological Momentary Assessment
  - suicide risk prediction
  - patient heterogeneity
  - latent variable model
  - time series for health
date: 2026-05-08
content_hash: af8ee43fedef52c3
---

# Improving Forecasts of Suicide Attempts for Patients with Little Data

**Conference**: NeurIPS 2025 (TS4H Workshop)
**arXiv**: [2511.18199](https://arxiv.org/abs/2511.18199)
**Authors**: Genesis Hang, Annie Chen, Hope Neveux, Matthew K. Nock, Yaniv Yacoby
**Institution**: Wellesley College, Harvard University
**Code**: Not open-sourced
**Area**: Other
**Keywords**: Gaussian Process, Ecological Momentary Assessment, suicide risk prediction, patient heterogeneity, latent variable model, time series for health

## TL;DR

This paper proposes the Latent Similarity Gaussian Process (LSGP), which embeds patients into a continuous latent space to capture heterogeneity, enabling data-scarce patients to "borrow" predictive trends from similar patients, thereby improving suicide attempt prediction based on EMA data.

## Background & Motivation

### State of the Field

Ecological Momentary Assessment (EMA) collects suicidal ideation, mood states, and 17 other affective states from patients multiple times daily via smartphones, providing a data foundation for real-time suicide risk prediction. However, applying ML to predict suicide attempts faces three core challenges:

**Low base-rate**: Even in the largest available dataset (~623 patients), patients who actually experience a suicide-related event (SRE) are extremely rare; ultimately only 77 patients satisfied the inclusion criteria.

**Patient heterogeneity**: The pathways leading different patients toward suicide differ substantially—distinct diagnoses such as depression, anxiety, PTSD, and borderline personality disorder may correspond to distinct mechanisms, compounded by individual life experiences, rendering the assumption of a single universal model untenable.

**Data imbalance**: The volume of data varies enormously across patients; patients in the bottom 30% by data volume perform worse on nearly all metrics.

### Limitations of Prior Work

Empirical experiments reveal a fundamental tension:

- **Single model** (trained uniformly on all patients): dominated by data-rich patients, generalizes poorly to data-scarce patients.
- **Idiographic model** (trained separately per patient): substantially improves performance, but severely overfits for data-scarce patients.
- **Random grouping experiment**: even randomly partitioning patients into groups improves performance as the number of groups increases, indicating extremely high heterogeneity; demographic-based grouping performs even worse than random grouping.

This tension motivates the search for a solution **intermediate between the single and idiographic extremes**.

## Method

### Latent Similarity Gaussian Process (LSGP)

The core idea of LSGP is to embed patients into a **continuous latent space** in which distance reflects similarity in predictive trends. Data-rich patients obtain well-informed positions in the latent space through training, while data-scarce patients can intelligently borrow trends from their "neighbors."

**The model architecture consists of four steps**:

1. **Latent variable prior**: Each patient $n$ has a latent variable $z_n \sim \mathcal{N}(0, \mathbb{I}_{D_z})$, with $D_z = 3$.
2. **Input augmentation**: The patient's EMA responses $x_i$ (comprising 17 affective dimensions and 3 suicidal ideation dimensions) are concatenated with the corresponding patient's latent variable $z_{n_i}$ to form the augmented input $\hat{x}_i$.
3. **Gaussian process**: A GP prior is defined over the augmented input space: $F|\hat{X}; \theta \sim \mathcal{N}(0, K_\theta(\hat{X}, \hat{X}))$.
4. **Bernoulli likelihood**: $y_i | f_i \sim \text{Bernoulli}(\text{sigmoid}(f_i))$, predicting whether an SRE occurs in the following week.

### Sparse Variational Inference

Given the non-Gaussian likelihood and large-scale observations (14,763 records from 77 patients), the authors adopt the **Sparse Variational GP** framework:

- $M = 2000$ inducing points are used.
- The variational family consists of: (1) a full-covariance Gaussian posterior $q(U; \phi)$ over the inducing points; (2) a mean-field Gaussian posterior $q(z_n; \phi)$ over each patient's latent variable.
- Training is performed via stochastic variational inference (SVI) by maximizing the ELBO, with mini-batch size $B = 150$, learning rate 0.005, and 15,000 training steps.

### Kernel Design

The kernel factorizes into a product over the input and latent spaces: $K_\theta(\hat{X}, \hat{X}') = K^x_\theta(X, X') \cdot K^z_\theta(Z, Z')$

where $K^z_\theta$ uses an ARD kernel and $K^x_\theta$ uses a **state-dependent linear kernel**:

$$k_\theta(\hat{x}, \hat{x}') = b_\theta(z) \cdot b_\theta(z') + v_\theta(z) \cdot v_\theta(z') \cdot (x - c_\theta(z))^\top \cdot (x' - c_\theta(z'))$$

where $b_\theta(\cdot)$, $v_\theta(\cdot)$, and $c_\theta(\cdot)$ are all neural networks, allowing patients at different latent positions to possess different priors.

### Patient Similarity Analysis

Leveraging the product factorization of the kernel, a **patient similarity graph** can be constructed:

- The $K^z_\theta$ covariance matrix is computed from the posterior means of patient latent variables.
- This matrix is treated as an adjacency matrix, with patients as nodes and edge weights equal to covariances.
- **Modularity** $Q \in [-1, 1]$ is used to quantitatively assess the alignment between demographic groupings and the learned similarity structure.

## Key Experimental Results

### Table 1: Main Method Comparison (Test Set Metrics)

| Method | Avg. Log-Likelihood (Bottom/Mid/Top/All) | ROC-AUC | PPV | Sensitivity | Specificity |
|--------|------------------------------------------|---------|-----|-------------|-------------|
| **Single KNN** | N/A | 0.70±0.01 | 0.61±0.02 | 0.22±0.02 | 0.97±0.00 |
| **Single RBF-GP** | -0.62/-0.49/-0.36/-0.43 | 0.74±0.00 | 0.66±0.04 | 0.15±0.01 | 0.98±0.00 |
| **Single LR** | -0.66/-0.53/-0.38/-0.45 | 0.68±0.01 | 0.60±0.06 | 0.05±0.01 | 0.99±0.00 |
| **Idiographic KNN** | N/A | 0.78±0.01 | 0.73±0.02 | 0.34±0.03 | 0.97±0.00 |
| **Idiographic RBF-GP** | -0.52/-0.42/-0.31/-0.37 | 0.84±0.00 | 0.73±0.01 | 0.37±0.01 | 0.97±0.00 |
| **Idiographic VB-LR** | -0.47/-0.38/-0.28/-0.33 | **0.87±0.00** | 0.73±0.01 | 0.42±0.02 | 0.96±0.00 |
| **SV-LSGP** | -0.50/-0.40/-0.29/-0.35 | 0.85±0.01 | 0.73±0.01 | 0.37±0.02 | 0.97±0.00 |

**Key finding**: With only a simple kernel design, SV-LSGP approaches the best-performing baseline (Idiographic VB-LR) and outperforms all other baselines.

### Table 2: Patient Similarity Graph Modularity Analysis

| Demographic Grouping | Modularity $Q$ | Interpretation |
|----------------------|----------------|----------------|
| Gender | 0.12 | Within-group and between-group similarity nearly balanced |
| Sexual Orientation | 0.13 | Within-group and between-group similarity nearly balanced |
| Age | 0.08 | Closest to 0; age explains almost no similarity |
| Highest Education | 0.15 | Slight within-group preference, still near 0 |

**Key finding**: Modularity for all demographic attributes is close to 0, indicating that these attributes **cannot explain** patient similarity in suicide risk predictive trends. The strongest similarities frequently arise between patients from different demographic groups.

## Highlights & Insights

1. **Insightful problem characterization**: Systematic experiments reveal the fundamental tension between single and idiographic models, as well as the fragility of discrete grouping approaches (random grouping outperforms demographic grouping), providing a solid empirical foundation for model design.
2. **Elegant and natural formulation**: LSGP elegantly unifies the two extremes via a continuous latent space—when $D_z = 0$ it reduces to a single GP, and as the latent dimensionality grows with independently placed patient latent variables it approaches the idiographic limit.
3. **Interpretability analysis**: The learned latent space is used to construct a patient similarity graph, and modularity quantitatively evaluates the explanatory power of demographic attributes, offering a novel perspective for clinical understanding.
4. **Data scale and quality**: The work is grounded in a large-scale longitudinal dataset comprising 623 participants surveyed six times daily over three months, lending substantial clinical credibility.
5. **Value of null findings**: The explicit demonstration that demographic attributes cannot explain patient heterogeneity is itself a null finding of important clinical significance.

## Limitations & Future Work

1. **Only approaches the best baseline**: SV-LSGP has not yet surpassed Idiographic VB-LR; the authors acknowledge that kernel design remains preliminary and that more principled inductive biases warrant future exploration.
2. **Strict data requirements**: Only 77 patients satisfied the inclusion criterion of at least 3 SREs and 3 non-SREs, substantially reducing usable data and raising questions about generalizability.
3. **Cold-start problem**: The model requires each new patient to have at least one recorded SRE prior to use, a strong assumption for real-world deployment.
4. **Single kernel design**: Only one kernel combination (state-dependent linear kernel × ARD kernel) was explored; the broader kernel design space was not systematically investigated.
5. **Absence of causal reasoning**: The model is fundamentally a correlation-based predictor and cannot explain *why* certain patients are similar, limiting clinical actionability.
6. **Workshop paper**: As a preliminary result, the depth of experimentation and ablation studies is limited.

## Related Work & Insights

- **EMA-based suicide prediction**: Kleiman et al. (2017, 2018) pioneered EMA-based suicidal ideation prediction, but reliable methods for predicting suicide attempts remain lacking.
- **Modeling patient heterogeneity**: Methods such as S-GIMME (Gates et al., 2017) employ discrete groupings, but this work demonstrates that a continuous latent space is more appropriate.
- **GP latent variable models**: GP with Latent Covariate (Wang et al., 2012) and Covariate GPLVM (Martens et al., 2019) are relevant precursors, though not applied in this setting.
- **Multi-group GP**: Li et al. (2025) use discrete group labels, whereas LSGP makes group membership continuous and latent.
- **Meta-learning GP**: Saemundsson et al. (2018) propose a related meta-learning GP formulation, but their setting includes control signals.
- **Community detection**: Modularity (Newman, 2004), originally developed for network community detection, is innovatively applied here to patient covariance matrix analysis.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 3.5 | LSGP is a compositional application of existing models, but its adaptation to the suicide prediction setting and empirical analysis are original |
| Technical Depth | 3.5 | The variational inference framework is well-established; the kernel design is a highlight, though the overall approach draws on standard GP tools |
| Experimental Thoroughness | 3.0 | Baseline comparisons are adequate, but ablation studies are absent and the kernel design space is not systematically explored |
| Writing Quality | 4.0 | Motivation is clearly articulated; the transition from empirical findings to method design is natural and logically rigorous |
| Clinical Significance | 4.0 | The heterogeneity analysis and the null finding regarding demographics carry important clinical implications |
| Overall | 3.5 | A valuable preliminary contribution that skillfully adapts existing GP methods to a critical clinical setting, but requires more extensive experimental validation |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Improving Decision Trees through the Lens of Parameterized Local Search](improving_decision_trees_through_the_lens_of_parameterized_local_search.md)
- [\[NeurIPS 2025\] Adaptive Data Analysis for Growing Data](adaptive_data_analysis_for_growing_data.md)
- [\[ICCV 2025\] I Am Big, You Are Little; I Am Right, You Are Wrong](../../ICCV2025/others/i_am_big_you_are_little_i_am_right_you_are_wrong.md)
- [\[NeurIPS 2025\] Inferring Stochastic Dynamics with Growth from Cross-Sectional Data](inferring_stochastic_dynamics_with_growth_from_cross-sectional_data.md)
- [\[NeurIPS 2025\] Learning Dense Hand Contact Estimation from Imbalanced Data](learning_dense_hand_contact_estimation_from_imbalanced_data.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation
description: >-
  [NeurIPS 2025][Computational Biology][single-cell perturbation prediction] PRESCRIBE is a framework that jointly models epistemic uncertainty (model unfamiliarity with inputs) and aleatoric uncertainty (inherent randomne…
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "single-cell perturbation prediction"
  - "uncertainty quantification"
  - "Bayesian estimation"
  - "deep evidential regression"
  - "pseudo E-distance"
date: 2026-05-08
content_hash: e7e9cd44d7063122
---

# PRESCRIBE: Predicting Single-Cell Responses with Bayesian Estimation

**Conference**: NeurIPS 2025
**arXiv**: [2510.07964](https://arxiv.org/abs/2510.07964)  
**Code**: [GitHub](https://github.com/Bunnybeibei/PRESCRIBE)  
**Area**: Computational Biology / Single-Cell Perturbation Prediction
**Keywords**: single-cell perturbation prediction, uncertainty quantification, Bayesian estimation, deep evidential regression, pseudo E-distance

## TL;DR

PRESCRIBE is a framework that jointly models epistemic uncertainty (model unfamiliarity with inputs) and aleatoric uncertainty (inherent randomness of biological systems) in single-cell perturbation prediction via multivariate deep evidential regression. It generates a pseudo E-distance as a unified uncertainty proxy; filtering unreliable predictions based on this metric yields accuracy improvements exceeding 3%.

## Background & Motivation

Predicting the effects of genetic perturbations on cells is a critical task in drug development and gene therapy. Although existing machine learning methods (e.g., GEARS, scGPT) achieve high overall accuracy, they can produce severely erroneous predictions for **specific instances**—particularly for gene perturbations that are unseen during training and functionally distant from training examples.

The core issue is that **high average accuracy does not guarantee reliability for individual predictions**. This is especially costly in biological experiments, where erroneous predictions can lead to significant waste of experimental resources.

Prediction uncertainty arises from two sources:

**Aleatoric uncertainty**: The inherent stochasticity of biological systems—the same genetic perturbation may yield diverse cellular outcomes.

**Epistemic uncertainty**: The degree to which a model is unfamiliar with a given input—if a test perturbation is functionally distant from training perturbations, the model's prediction becomes unreliable.

Existing methods (e.g., MC Dropout in GEARS) fail to adequately account for uncertainty in pairwise distances, and their variance estimates correlate poorly with actual prediction accuracy. Inspired by E-distance—a metric for measuring the similarity between two cell populations—the authors design a pseudo E-distance to unify both types of uncertainty.

## Method

### Overall Architecture

PRESCRIBE consists of three core modules: (1) an encoder $f_\alpha$ that maps perturbations to latent representations; (2) a normalizing flow $f_\psi$ that estimates the density of training data in the latent space and outputs evidence; and (3) a decoder $f_\beta$ that generates sufficient statistics of the posterior distribution. Bayesian updates combine the prior with model outputs to yield the final predictive distribution and uncertainty estimates.

### Key Designs

1. **Multivariate Normal-Wishart Bayesian Modeling**: Gene expression vectors $y_i$ are assumed to follow a multivariate Gaussian distribution, with a Normal-Wishart conjugate prior placed over its parameters:

$$\mathbb{P}(\mu_i | \Lambda_i) = \mathcal{N}(\mu_{0x_i}, (\kappa_i \Lambda_i)^{-1}), \quad \mathbb{P}(\Lambda_i) = \mathcal{W}(\nu_i, \Psi_i^{-1})$$

   A lower-triangular matrix $L_i$ is used to parameterize the scale matrix $\Psi_i^{-1} = \nu_i L_i L_i^T$. The motivation for this conjugate prior design is to enable analytic Bayesian updates and to ensure that predictions automatically revert to the unperturbed control state (a safe output) when evidence is low (i.e., in OOD scenarios). This constitutes a multivariate extension of the standard NatPN, which originally supports only univariate outputs.

2. **Pseudo E-distance as a Unified Uncertainty Metric**: The pseudo E-distance is defined as:

$$\tilde{E} = 2\tilde{\nu}_i^{\text{post}} - \tilde{\mathbb{H}}[\mathbb{P}(y_i | \omega_i)]$$

   where $\tilde{\nu}_i^{\text{post}}$ is the normalized posterior evidence (measuring epistemic uncertainty) and $-\tilde{\mathbb{H}}[\cdot]$ is the negative normalized entropy of the predictive distribution (measuring aleatoric uncertainty). Both terms are normalized to the range $[N, 2N]$ for comparability. A high $\tilde{E}$ indicates a high-confidence prediction. A key property is that the metric distinguishes between "low-confidence OOD predictions" and "high-confidence null-effect predictions"—both produce outputs close to the control state, yet their evidence scores differ substantially.

3. **Normalizing Flow for Evidence Estimation**: A normalizing flow $f_\psi$ estimates the training data density in the latent space and directly outputs evidence $\nu_i$:

$$\nu_i = \exp(f_\psi(z_i) + \ln N_H), \quad \tilde{\nu}_i^{\text{post}} = \frac{N\nu_i}{\nu_i + \nu^{\text{prior}}} + N$$

   High-density regions (familiar inputs) yield high evidence; low-density regions (novel inputs) yield low evidence, forcing predictions to revert to the prior. As $\nu_i \to 0$, $\tilde{\nu}_i^{\text{post}} \to N$ (minimum confidence); as $\nu_i \to \infty$, $\tilde{\nu}_i^{\text{post}} \to 2N$ (maximum confidence).

### Loss & Training

The composite loss is $\mathcal{L} = \mathcal{L}_1 + \lambda_1 \mathcal{L}_2 + \lambda_2 \mathcal{L}_3 + \lambda_3 \mathcal{L}_4$:

- **$\mathcal{L}_1$ (Expected Log-Likelihood)**: Maximizes the likelihood of observed data under the posterior predictive distribution, driving accurate predictions.
- **$\mathcal{L}_2$ (Entropy Regularization)**: Weighted by prediction error; encourages high-entropy (uninformative) distributions for uncertain predictions.
- **$\mathcal{L}_3$ (E-distance Ranking Loss)**: Applies a ListMLE ranking loss to align the ordering of predicted pseudo E-distances with that of reference E-distances.
- **$\mathcal{L}_4$ (Uncertainty Regularization)**: Addresses the vanishing gradient problem in low-evidence regions.

## Key Experimental Results

### Main Results (Perturbation Prediction Accuracy)

| Model | Norman r↑ | Norman r^DEG↑ | Norman ACC↑ | Rep1 r↑ | K562 r↑ |
|-------|-----------|---------------|-------------|---------|---------|
| GEARS | 45.30 | 63.19 | 29.09 | 48.18 | 32.57 |
| scGPT | 61.48 | 65.87 | 61.96 | 50.32 | 32.72 |
| scFoundation | 60.79 | 65.65 | 35.66 | 47.60 | 25.15 |
| PRESCRIBE | 58.38 | 64.44 | 63.24 | 59.18 | 36.20 |
| **PRESCRIBE-10%** | **64.32** | **68.61** | **64.73** | **60.28** | **38.58** |

*-X% denotes results after filtering the X% lowest-confidence predictions.*

### Uncertainty Calibration Quality

| Method | Norman r^s_{perf,conf}↑ | Norman ACC_{perf,conf}↑ | Rep1 r^s↑ | K562 r^s↑ |
|--------|-------------------------|-------------------------|-----------|-----------|
| GEARS-Drop | poor monotonicity | - | - | - |
| GEARS-Ens | inconsistent | - | - | - |
| PRESCRIBE | **35.56** | **25.81** | **12.18** | **24.74** |

### Key Findings

1. **Pseudo E-distance positively correlates with true E-distance**: Across all datasets, the predicted pseudo E-distance is positively correlated with the reference E-distance; this correlation strengthens markedly as the number of reference samples $N$ increases (Spearman correlation reaches 80.00 at $N=500$).
2. **Confidence decreases with generalization difficulty**: In Norman combinatorial perturbation settings with 0/1/2 unseen perturbations, PRESCRIBE's confidence scores decrease significantly and monotonically, whereas other methods remain nearly constant or exhibit reversed trends.
3. **Filtering the lowest 10% yields substantial gains**: On Norman, $r$ improves from 58.38 to 64.32 (+5.94); on K562, from 36.20 to 38.58 (+2.38).
4. **Random filtering is ineffective**: Randomly filtering 10% of predictions does not improve—and in fact slightly degrades—accuracy, confirming that PRESCRIBE genuinely identifies unreliable predictions.

## Highlights & Insights

- The pseudo E-distance design for unifying two types of uncertainty is elegant: it naturally integrates "what the model knows" (evidence) and "how stable the outcome is" (entropy) within a Bayesian framework.
- Automatic reversion to the control state under low evidence represents a safe-by-design property: the model prefers predicting "no effect" over producing a spurious effect.
- The approach has direct experimental value: practitioners can use confidence scores to prioritize which perturbations warrant empirical validation.

## Limitations & Future Work

- PCA-based dimensionality reduction may discard important gene expression information; superior alternatives warrant exploration.
- The encoder relies on pretrained gene embeddings (scGPT), and the impact of alternative embeddings has not been thoroughly investigated.
- An additive assumption is adopted for combinatorial perturbations, without modeling nonlinear interactions between perturbations.
- Validation is limited to a small number of datasets; multi-time-point or spatial transcriptomics settings have not been addressed.

## Related Work & Insights

- PRESCRIBE extends the Natural Posterior Network (NatPN) to the multivariate setting, overcoming NatPN's inability to handle high-dimensional outputs.
- The idea of using E-distance as an uncertainty proxy is generalizable to other domains, such as drug response prediction and protein engineering.
- Insight: In bioinformatics, uncertainty estimation may be more valuable than accuracy alone, as it directly guides the allocation of experimental resources.

## Rating

- Novelty: ⭐⭐⭐⭐ The pseudo E-distance and multivariate NatPN extension are original contributions, though the deep evidential regression framework has prior precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, multiple baselines, and comprehensive ablations; however, a broader range of perturbation types is lacking.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation and methodological rationale are clearly articulated, though the probabilistic notation is dense.
- Value: ⭐⭐⭐⭐ Provides a much-needed uncertainty quantification tool for single-cell perturbation prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] scPilot: Large Language Model Reasoning Toward Automated Single-Cell Analysis and Discovery](scpilot_large_language_model_reasoning_toward_automated_single-cell_analysis_and.md)
- [\[NeurIPS 2025\] scMRDR: A Scalable and Flexible Framework for Unpaired Single-Cell Multi-Omics Data Integration](scmrdr_a_scalable_and_flexible_framework_for_unpaired_single-cell_multi-omics_da.md)
- [\[ICLR 2026\] Retrieval-Augmented Generation for Predicting Cellular Responses to Gene Perturbation](../../ICLR2026/computational_biology/retrieval-augmented_generation_for_predicting_cellular_responses_to_gene_perturb.md)
- [\[ICLR 2026\] scDFM: Distributional Flow Matching for Robust Single-Cell Perturbation Prediction](../../ICLR2026/computational_biology/scdfm_distributional_flow_matching_model_for_robust_single-cell_perturbation_pre.md)
- [\[NeurIPS 2025\] Is Sequence Information All You Need for Bayesian Optimization of Antibodies?](is_sequence_information_all_you_need_for_bayesian_optimization_of_antibodies.md)

</div>

<!-- RELATED:END -->

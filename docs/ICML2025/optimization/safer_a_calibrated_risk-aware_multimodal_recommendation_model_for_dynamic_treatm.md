---
title: >-
  [Paper Note] SAFER: A Calibrated Risk-Aware Multimodal Recommendation Model for Dynamic Treatment Regimes
description: >-
  [ICML2025][Optimization][Dynamic Treatment Regimes (DTR)] The SAFER framework is proposed to integrate multimodal information from structured EHR and clinical notes. It utilizes KL divergence to measure label uncertainty and incorporates conformal prediction to control the FDR, providing statistical safety guarantees for high-risk dynamic treatment recommendations.
tags:
  - "ICML2025"
  - "Optimization"
  - "Dynamic Treatment Regimes (DTR)"
  - "Multimodal Fusion"
  - "Uncertainty Quantification"
  - "Conformal Prediction"
  - "Sepsis"
  - "EHR"
date: 2026-05-08
content_hash: e4003d4e0197052d
---

# SAFER: A Calibrated Risk-Aware Multimodal Recommendation Model for Dynamic Treatment Regimes

**Conference**: ICML2025  
**arXiv**: [2506.06649](https://arxiv.org/abs/2506.06649)  
**Code**: [yishanssss/SAFER](https://github.com/yishanssss/SAFER)  
**Area**: Medical Imaging  
**Keywords**: Dynamic Treatment Regimes (DTR), Multimodal Fusion, Uncertainty Quantification, Conformal Prediction, Sepsis, EHR

## TL;DR

The SAFER framework is proposed to integrate multimodal information from structured EHR and clinical notes. It utilizes KL divergence to measure label uncertainty and incorporates conformal prediction to control the FDR, providing statistical safety guarantees for high-risk dynamic treatment recommendations.

## Background & Motivation

- **Dynamic Treatment Regimes (DTR)** aim to make real-time, personalized treatment decisions based on the evolving clinical status of patients, representing a core problem of precision medicine.
- Existing methods face three critical bottlenecks:
    1. **Label Uncertainty**: Treatment labels for deceased patients may not represent optimal decisions (it could be that the treatment was correct but insufficient to save them, or that incorrect treatment led to adverse outcomes). Existing methods generally overlook this label ambiguity.
    2. **Single Modality**: Most DTR methods utilize only structured EHR data (vital signs, laboratory tests), ignoring the physician's judgment and patient clinical progress information embedded in clinical notes.
    3. **Lack of Safety Guarantees**: Existing methods do not provide theoretical error rate control for recommendation quality, making it difficult to earn clinicians' trust in high-risk clinical scenarios.

## Method

SAFER consists of three core modules: multimodal representation learning → risk-aware fine-tuning → conformal selection and FDR control.

### 1. Multimodal Representation Learning

**Input**: The time-series sequence of each patient $i$, denoted as $\mathbf{r}_i = \{(\mathbf{e}_i^1, \mathbf{o}_i^1), \ldots, (\mathbf{e}_i^T, \mathbf{o}_i^T)\}$, where $\mathbf{e}$ represents structured EHR, and $\mathbf{o}$ represents clinical notes.

- **Encoder**: Clinical notes are encoded by BioClinicalBERT; structured data are processed via normalization + one-hot encoding.
- **Intra-modal Temporal Modeling**: Self-attention with causal masking is applied to each modality separately:

$$\mathbf{S}_i^A = \text{Softmax}\left(\frac{(\mathbf{X}_i^A \mathbf{W}_A^Q)(\mathbf{X}_i^A \mathbf{W}_A^K)^\top + \mathbf{M}}{\sqrt{d_k}}\right)\mathbf{X}_i^A \mathbf{W}_A^V + \text{PE}$$

- **Cross-modal Fusion**: A bidirectional cross-attention mechanism is designed to let EHR and clinical notes mutually learn contextual information. Finally, static demographic features are concatenated to obtain a unified patient embedding $\mathbf{h}_i \in \mathbb{R}^{3d_k}$.
- **Classification Head**: A feedforward network maps the embedding to the drug category distribution, trained using cross-entropy loss.

### 2. Risk-Aware Fine-Tuning

Core idea: Labels of surviving patients are reliable, while labels of deceased patients are uncertain.

- **Uncertainty Estimation Module $f_\phi$**: After the first-stage model converges, an MLP module trained only on surviving patients is introduced to learn a cleaner predictive distribution.
- **Uncertainty Quantification**: Label uncertainty is measured by the KL divergence between the output distributions of the two modules:

$$\kappa_i = D_{\text{KL}}(p_\theta(\mathbf{h}_i) \| p_\phi(\mathbf{h}_i)) = \sum_{l=1}^{L} p_\theta(\hat{y}_i = l | \mathbf{h}_i) \ln \frac{p_\theta(\hat{y}_i = l | \mathbf{h}_i)}{p_\phi(\hat{y}_i = l | \mathbf{h}_i)}$$

- **Theoretical Guarantee (Theorem 4.1)**: Under the condition that $f_\phi$ satisfies Lipschitz continuity, the expected $\kappa$ of deceased patients is strictly higher than that of surviving patients.
- **Risk-Aware Loss**:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}(1 - \hat{\kappa}_i)\sum_{l=1}^{L} y_i \log p_\theta(\hat{y}_i = l | h_i) + \gamma \kappa_i^2$$

where $(1-\hat{\kappa}_i)$ downweights uncertain samples, and the regularization term $\gamma\kappa_i^2$ penalizes overconfident predictions on high-risk samples.

### 3. Conformal Selection and FDR Control

- Compute the uncertainty score $\hat{\kappa}$ for the calibration and test sets to construct conformal p-values.
- Control the False Discovery Rate (FDR) using the Benjamini-Hochberg (BH) procedure: only recommend treatments with p-values in the top $k$.
- **Theoretical Guarantee (Theorem 5.1)**: Under i.i.d. and bounded uncertainty assumptions, the FDR of the recommendation set is $\leq \alpha$ (a user-specified threshold).

## Key Experimental Results

Evaluated on two public sepsis datasets (MIMIC-III / MIMIC-IV), with a treatment space of $5 \times 5$ fluid-vasopressor combinations.

| Method | MI-AUC (III) | MA-AUC (III) | HR@3 (III) | MRR@3 (III) | ↓Mortality (III) |
|------|-------------|-------------|-----------|------------|-----------------|
| LSTM | 0.9122 | 0.7934 | 0.7481 | 0.8015 | 0.0915 |
| RETAIN | 0.9257 | 0.8219 | 0.8324 | 0.8153 | 0.1994 |
| ACIL | 0.8219 | 0.7012 | 0.8013 | 0.8313 | 0.3212 |
| **SAFER** | **0.9407** | **0.8672** | **0.8517** | **0.9017** | **0.3891** |

| Method | MI-AUC (IV) | MA-AUC (IV) | HR@3 (IV) | MRR@3 (IV) | ↓Mortality (IV) |
|------|------------|------------|----------|-----------|----------------|
| LSTM | 0.9213 | 0.8121 | 0.7551 | 0.8066 | 0.1051 |
| RETAIN | 0.9279 | 0.7851 | 0.8017 | 0.8052 | 0.1863 |
| ACIL | 0.8854 | 0.7135 | 0.8319 | 0.8441 | 0.3782 |
| **SAFER** | **0.9356** | **0.8755** | **0.8713** | **0.8698** | **0.4562** |

- SAFER outperforms state-of-the-art (SOTA) methods across all recommendation metrics, achieving the largest reduction in counterfactual mortality (higher ↓Mortality indicates that the model-recommended treatment is more effective in reducing mortality).
- On MIMIC-IV, MA-AUC improved by approximately 5.8% (vs. RETAIN), and HR@3 improved by approximately 4.7%.

## Highlights & Insights

1. **Systematic Modeling of Label Uncertainty**: This work is the first to explicitly model the uncertainty of deceased patient labels in DTR, quantifying it via KL divergence and incorporating it into the loss function. The approach is both elegant and theoretically grounded.
2. **Conformal Inference + FDR Control**: Introducing conformal prediction into treatment recommendation provides quantifiable safety bounds, which is highly practical in high-risk medical scenarios.
3. **Genuine Multimodal Fusion**: This is the first work to simultaneously utilize clinical notes and structured EHR for DTR, with a bidirectional cross-attention design that allows both modalities to mutually reinforce each other.
4. **End-to-End Framework**: The framework integrates multimodal learning, uncertainty quantification, and statistical inference into a unified and complete design.

## Limitations & Future Work

1. **Sepsis-Only Validation**: Although the framework is general, experiments were only evaluated on sepsis scenarios (MIMIC datasets), and its generalization to other diseases or treatment settings remains unverified.
2. **Assumptions on Clinical Notes Quality**: The model relies on BioClinicalBERT to encode clinical notes; its robustness in scenarios with missing or poor-quality notes needs to be validated.
3. **Discretization of Treatment Space**: Discreting fluid and vasopressor doses into a $5 \times 5$ space may lose the fine-grained information of continuous dosages.
4. **Limitations of KL Divergence**: The uncertainty metric depends on the difference between the output distributions of the two modules, which may yield a false "low uncertainty" when both modules make incorrect predictions.
5. **The i.i.d. Assumption in Conformal Inference**: Actual clinical data often exhibit distribution shifts, and the i.i.d. assumption may not be fully satisfied in real-world deployments.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of label uncertainty modeling and conformal inference is a first in the DTR field.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Conducted on two large-scale public datasets with multiple baselines and complete ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic and rigorous theoretical derivations.
- Value: ⭐⭐⭐⭐ — The approach of providing safety guarantees for high-risk treatment recommendations holds significant clinical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Generalization and Robustness of the Tilted Empirical Risk](generalization_and_robustness_of_the_tilted_empirical_risk.md)
- [\[CVPR 2026\] BD-Merging: Bias-Aware Dynamic Model Merging with Evidence-Guided Contrastive Learning](../../CVPR2026/optimization/bd-merging_bias-aware_dynamic_model_merging_with_evidence-guided_contrastive_lea.md)
- [\[NeurIPS 2025\] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](../../NeurIPS2025/optimization/dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)
- [\[NeurIPS 2025\] CHiQPM: Calibrated Hierarchical Interpretable Image Classification](../../NeurIPS2025/optimization/chiqpm_calibrated_hierarchical_interpretable_image_classification.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](tilted_sharpness-aware_minimization.md)

</div>

<!-- RELATED:END -->

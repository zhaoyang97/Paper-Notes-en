---
title: >-
  [Paper Note] Towards Robust Influence Functions with Flat Validation Minima
description: >-
  [ICML 2025][LLM Pretraining][influence function] This work reveals that the fundamental reason for the failure of influence functions (IF) on noisy training data is not the inaccuracy of the inverse Hessian approximation (the focus of previous research), but rather the **sharpness** of the validation loss leading to distorted loss change estimation. It theoretically derives the connection between the IF error bound and validation risk sharpness…
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "influence function"
  - "flat minima"
  - "SAM"
  - "noisy label detection"
  - "data attribution"
date: 2026-05-08
content_hash: fef55503f1ff9ddc
---

# Towards Robust Influence Functions with Flat Validation Minima

**Conference**: ICML 2025  
**arXiv**: [2505.19097](https://arxiv.org/abs/2505.19097)  
**Code**: [GitHub](https://github.com/Virusdoll/IF-FVM)  
**Area**: Explainable AI / Training Data Influence  
**Keywords**: influence function, flat minima, SAM, noisy label detection, data attribution

## TL;DR

This work reveals that the fundamental reason for the failure of influence functions (IF) on noisy training data is not the inaccuracy of the inverse Hessian approximation (the focus of previous research), but rather the **sharpness** of the validation loss leading to distorted loss change estimation. It theoretically derives the connection between the IF error bound and validation risk sharpness, and designs a new IF variant (FVM) specifically tailored for flat validation minima.

## Background & Motivation

**Background**: Influence function (IF) is the core tool for evaluating the impact of a single training example on model predictions. The standard IF formula is $\mathcal{I}(z_\text{tr}; z_\text{val}) = g_{z_\text{val}}^\top H_\text{tr}^{-1} g_{z_\text{tr}}$, which avoids the expensive leave-one-out retraining via a two-step approximation: parameter change estimation (Newton step) and loss change estimation (first-order Taylor expansion).

**Limitations of Prior Work**: IF performs poorly on noisy training data—which is precisely the scenario where it is needed most (distinguishing clean data from noisy data). Whether using first-order approximations (TracIn) or second-order approximations (LiSSA), the failure mode of IF remains the same.

**Key Challenge**: Prior research focused on improving the accuracy of inverse Hessian approximation, but this paper finds that the issue does not lie there. Even if the parameter change $\Delta\theta$ is perfectly estimated, if the validation loss landscape is sharp, the first-order expansion will still introduce a significant estimation gap.

**Goal**: Fundamentally resolve the estimation reliability issue of IF on noisy data.

**Key Insight**: Re-attribute the failure of IF to the loss change estimation (the second step), and utilize SAM to find flat validation minima.

**Core Idea**: Compute influence functions at flat validation minima, and replace the first-order expansion with a second-order expansion to estimate the loss change.

## Method

### Overall Architecture

This work consists of two phases:
1. **Theoretical Analysis**: Establishes the upper bound relationship between IF estimation error $\leftrightarrow$ validation risk sharpness.
2. **Methodological Design**: Proposes two new IF estimators: VM (Validation Minima) and FVM (Flat Validation Minima).

### Key Designs

1. **IF Error Bound Theorem (Theorem 3.2)**:
    - Function: Theoretically explains why IF fails on noisy data.
    - Mechanism: Defines estimation error as the probability of sign misclassification $\mathcal{E}(\mathcal{I}) = \mathbb{P}[\text{sgn}(\mathcal{I}) \neq \text{sgn}(\mathcal{I}^*)]$, and proves the upper bound:
    $$\mathcal{E}(\mathcal{I}) \leq \exp\left(-\frac{2\mu^2}{\hat{R}_\text{val}^\gamma(\theta)^2}\right)$$
    where $\hat{R}_\text{val}^\gamma(\theta) = \max_{\|\Delta\| \leq \gamma} \hat{R}_\text{val}(\theta + \Delta)$ encapsulates both validation risk and sharpness, and $\mu$ represents the lower bound of IF discriminative power.
    - Design Motivation: The upper bound indicates the necessity of simultaneously reducing validation risk and sharpness.

2. **Failure Analysis of Standard IF at Flat Minima**:
    - Function: Explains why simply applying SAM + standard IF does not work.
    - Mechanism: Identifies two issues: (1) Parameter change estimation is based on the training set Hessian, whereas parameters reside at the validation set minima, causing misalignment; (2) The gradient of the converged model has near-zero mean, meaning the expectation of the first-order term $g_\text{val}^\top H_\text{tr}^{-1} g_{z_\text{tr}}$ approaches zero, leading to $\mu \to 0$.
    - Design Motivation: Empirically demonstrated in Figure 3—the AUC of standard IF actually decreases after SAM training.

3. **New IF Formulations (VM/FVM)**:
    - Function: Influence estimation specifically designed for flat validation minima.
    - Mechanism: Simultaneously corrects both estimation steps:
        - **Parameter Change**: Redefines perturbation at the flat validation minima, using the **validation set Hessian** $\tilde{H}_\text{val}$ instead of the training set Hessian in the Newton step.
        - **Loss Change**: Replaces the first-order approximation with a **second-order approximation**: $\ell(z_\text{val}, \tilde{\theta}_{z_\text{tr}}) - \ell(z_\text{val}, \tilde{\theta}) \approx \frac{1}{2} \Delta\theta^\top \nabla^2 \ell(z_\text{val}, \tilde{\theta}) \Delta\theta$
        - Final validation set IF: $\mathcal{I}(z_\text{tr}, S_\text{val}) = \tilde{g}_{z_\text{tr}}^\top \tilde{H}_\text{val}^{-1} \tilde{g}_{z_\text{tr}}$
    - Design Motivation: The second-order term is unaffected by near-zero gradients (as the Hessian does not vanish), and computing the Hessian on the validation set resolves the alignment issue.

### Loss & Training

- **VM**: Computes the new IF after training to a minimum using ERM on the validation set.
- **FVM**: Computes the new IF after training with SAM on the validation set to locate flat minima.
- SAM implementation uses LPF-SGD with $M=1$.

## Key Experimental Results

### Noisy Label Detection: CIFAR-10N/100N (ROC AUC %)

| Method | CIFAR-10N Aggre | CIFAR-10N Random | CIFAR-10N Worst | CIFAR-100N Noisy |
|------|:-:|:-:|:-:|:-:|
| LiSSA | 59.74±2.91 | 59.78±2.77 | 65.75±0.39 | 57.48±1.70 |
| TracIn | 53.91±5.85 | 61.61±0.74 | 65.74±2.32 | 56.13±2.51 |
| GEX | 87.38±1.21 | 91.11±0.53 | 93.28±0.10 | **90.17±0.70** |
| DataInf | 58.50±3.98 | 54.50±2.32 | 55.49±1.45 | 53.69±1.35 |
| **VM** | 95.18±0.15 | 95.92±0.10 | 95.88±0.13 | 89.77±0.08 |
| **FVM** | **96.14±0.06** | **96.63±0.03** | **96.46±0.08** | 90.80±0.04 |

### Noisy Label Relabeling (Top-1 Accuracy %)

| Method | Aggre | Random | Worst | CIFAR-100N |
|------|:---:|:---:|:---:|:---:|
| LiSSA | 5.28 | 9.04 | 19.32 | 0.28 |
| TracIn | 37.08 | 53.28 | 50.66 | 20.11 |
| GEX | 30.19 | 54.03 | 80.35 | 22.41 |
| **VM** | 94.17 | 91.94 | 85.01 | 58.13 |
| **FVM** | **94.63** | **92.46** | **86.09** | **70.61** |

FVM achieves 70.61% on CIFAR-100N relabeling, which is over 3 times higher than the previous best GEX (22.41%).

### Influence Identification in Text Generation (Llama-2-13B + LoRA)

| Task | Method | ROC AUC | Recall |
|------|------|:---:|:---:|
| Sentence Transformations | TracIn | 94.95±6.14 | 70.97±25.18 |
| Sentence Transformations | DataInf | 99.58±1.96 | 96.18±9.33 |
| Sentence Transformations | **VM** | **99.97±0.16** | **99.10±2.79** |
| Math Problems | TracIn | 78.50±17.77 | 26.61±39.95 |
| Math Problems | DataInf | 99.86±0.68 | 97.37±6.97 |
| Math Problems | **FVM** | **99.99±0.07** | **99.38±1.65** |

### Key Findings

- FVM significantly outperforms baselines across all CIFAR-10N setups (3-6% higher AUC than GEX) with extremely low variance.
- IF performance is highly correlated with validation set accuracy (Figure 2), directly validating the theoretical analysis.
- Standard IF actually degrades after SAM training (Figure 3a), proving the necessity of the proposed new IF formulation.
- VM already substantially outperforms standard IF, while FVM further improves performance in most settings.

## Highlights & Insights

1. **Overturning Conventional Wisdom**: IF failure is not caused by Hessian approximation errors, but by the sharpness in loss change estimation.
2. **Perfect Alignment Between Theory and Experiment**: The predictions of Theorem 3.2 are precisely verified by Figures 2 & 3.
3. **Elegant Rectification**: Two core modifications—training on the validation set + replacing the first-order expansion with a second-order one.
4. **Qualitative Leap in Performance**: Relabeling accuracy increases from ~22% to ~70%, indicating a paradigm shift rather than incremental improvement.

## Limitations & Future Work

- SAM training on the validation set incurs additional computational overhead.
- Second-order IF is more complex than its first-order counterpart, limiting large-scale applications.
- Requires a clean validation set, which may be difficult to obtain in some scenarios.
- The extent of improvement under non-noise data scenarios requires systematic validation.

## Related Work & Insights

- **Koh & Liang (2017) Standard IF**: Serves as the foundation for the improvements in this work.
- **TracIn (Pruthi et al., 2020)**: First-order IF approximation, similarly affected by vanishing gradients.
- **GEX (Kim et al., 2023)**: Leverages gradient expectation for attribution, requiring 32 tuning runs (whereas FVM requires only 1).
- **SAM (Foret et al., 2021)**: Provides the optimization tool for finding flat minima.
- Inspiration: The impact of loss landscape geometric properties on gradient-based methods has been vastly underestimated.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Identifies the core failure mode of IF that has been overlooked for a decade.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across four types of tasks and multiple datasets, complete with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Elegant design of diagnostic experiments in Figures 2 & 3.
- Value: ⭐⭐⭐⭐⭐ The breakthrough in relabeling capability opens up new application spaces.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Robust Message Embedding via Attention Flow-Based Steganography](../../CVPR2025/llm_pretraining/robust_message_embedding_via_attention_flow-based_steganography.md)
- [\[CVPR 2025\] Influence Malleability in Linearized Attention: Dual Implications of Non-Convergent NTK Dynamics](../../CVPR2025/llm_pretraining/influence_malleability_in_linearized_attention_dual_implications_of_non-converge.md)
- [\[ICCV 2025\] ConstStyle: Robust Domain Generalization with Unified Style Transformation](../../ICCV2025/llm_pretraining/conststyle_robust_domain_generalization_with_unified_style_transformation.md)
- [\[ICLR 2026\] OptimSyn: Influence-Guided Rubrics Optimization for Synthetic Data Generation](../../ICLR2026/llm_pretraining/optimsyn_influence-guided_rubrics_optimization_for_synthetic_data_generation.md)
- [\[ICLR 2026\] Train on Validation (ToV): Fast Data Selection with Applications to Fine-Tuning](../../ICLR2026/llm_pretraining/train_on_validation_tov_fast_data_selection_with_applications_to_fine-tuning.md)

</div>

<!-- RELATED:END -->

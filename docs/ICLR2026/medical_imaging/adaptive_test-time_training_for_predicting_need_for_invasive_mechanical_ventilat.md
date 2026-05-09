---
title: >-
  [Paper Note] Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts
description: >-
  [ICLR 2026][Medical Imaging][Test-time training] This paper proposes AdaTTT, a framework that achieves robust test-time adaptation on multi-center ICU EHR data for 24-hour-ahead invasive mechanical ventilation (IMV) prediction, via dynamic feature-aware self-supervised learning (adaptive masking strategy) and prototype-guided partial optimal transport alignment.
tags:
  - ICLR 2026
  - Medical Imaging
  - Test-time training
  - domain shift
  - invasive mechanical ventilation prediction
  - dynamic feature masking
  - partial optimal transport
date: 2026-05-08
content_hash: ec2b2a70cac67d0e
---

# Adaptive Test-Time Training for Predicting Need for Invasive Mechanical Ventilation in Multi-Center Cohorts

**Conference**: ICLR 2026
**arXiv**: [2512.06652](https://arxiv.org/abs/2512.06652)
**Code**: To be released
**Area**: Medical Imaging / EHR Clinical Prediction
**Keywords**: Test-time training, domain shift, invasive mechanical ventilation prediction, dynamic feature masking, partial optimal transport

## TL;DR
This paper proposes AdaTTT, a framework that achieves robust test-time adaptation on multi-center ICU EHR data for 24-hour-ahead invasive mechanical ventilation (IMV) prediction, via dynamic feature-aware self-supervised learning (adaptive masking strategy) and prototype-guided partial optimal transport alignment.

## Background & Motivation

**State of the Field**: Predicting invasive mechanical ventilation (IMV) in the ICU is critical for timely clinical intervention. ML models based on EHR data (e.g., VentNet, Composer) have shown promise, but cross-institutional deployment faces severe domain shift challenges.

**Limitations of Prior Work**: Differences in patient populations, clinical workflows, and EHR systems across hospitals lead to distribution shifts, causing AUC drops of up to 12% across sites. Existing approaches either require labeled target-domain data for fine-tuning or incur prohibitive computational costs, making them unsuitable for real-time clinical deployment.

**Root Cause**: Test-Time Training (TTT) enables label-free adaptation at inference time, but directly applying existing TTT methods to EHR settings introduces three problems: (a) batch normalization statistics are unstable under small batch sizes (single-visit inference); (b) EHR feature importance is highly non-uniform, so random masking in SSL tasks fails to align with the main task; (c) instance-level SSL updates are prone to overfitting noisy samples.

**Paper Goals**: To design a framework specifically optimized for EHR-TTT scenarios, achieving high alignment between the SSL auxiliary task and the main task while preventing overfitting at test time.

**Starting Point**: The paper derives upper and lower bounds on the test-time prediction error from an information-theoretic perspective (Theorem 2), showing that the error is bounded by the conditional uncertainty $H(Y_m'|Y_s')$ between the auxiliary and main tasks — providing theoretical guidance for designing high-alignment SSL tasks.

**Core Idea**: Adaptive masking driven by dynamic feature importance tightly aligns the SSL task with the clinical prediction objective, complemented by prototype-based partial optimal transport for flexible distribution matching at test time.

## Method

### Overall Architecture
AdaTTT consists of a shared encoder $f_e$, a main-task classification head $h_c$, and an SSL head $h_s$. During training, the model is jointly optimized with the main task loss, SSL loss, and prototype learning loss. At test time, only encoder parameters are updated via 5 gradient steps using the SSL loss plus OT alignment loss, after which parameters are reset before processing the next sample.

### Key Designs

1. **Information-Theoretic Error Bounds**

    - **Function**: Derives the relationship between test-time prediction error and auxiliary task alignment.
    - **Mechanism**: Assuming the Markov chain $Y_s' \to Z' \to Y_m'$, Lemma 1 gives $I(Z';Y_m') \geq I(Y_s';Y_m')$, from which Theorem 2 establishes the error bounds: $H_{\text{err}}^{-1}(H(Y_m'|Y_s')) \leq p(e) \leq \frac{1}{2}H(Y_m'|Y_s')$
    - **Design Motivation**: This demonstrates that minimizing $H(Y_m'|Y_s')$ — i.e., making SSL predictions highly correlated with the main task — is key, while also warning that overfitting to the auxiliary task distribution at test time is harmful. This directly motivates the dynamic masking design.

2. **Dynamic Self-Supervised Learning with Adaptive Masking**

    - **Function**: Two SSL pretext tasks (reconstruction + masked feature modeling) with adaptive masking probabilities.
    - **Mechanism**: Masking probabilities are dynamically updated based on feature importance to the main task: $I_j = \frac{1}{n_s}\sum_n |\frac{\partial Y_m^{(n)}}{\partial x_j^{(n)}} \cdot x_j^{(n)}|$, normalized to yield $p_{m,j}$. More important features receive higher masking probabilities, forcing the model to learn to reconstruct clinically critical features.
    - **Design Motivation**: EHR features vary greatly in clinical importance (e.g., respiratory rate vs. a laboratory value); random masking wastes adaptation capacity on irrelevant features. A two-stage schedule is used: a warmup phase applies fixed masks derived from prior knowledge (a pretrained respiratory failure model), followed by dynamic updates based on gradient-based feature importance.

3. **Prototype-Guided Partial Optimal Transport Alignment**

    - **Function**: $k=4$ prototypes are learned during training; at test time, partial optimal transport (POT) flexibly aligns test features to these prototypes.
    - **Mechanism**: During training, the prototype learning loss $\mathcal{L}_{\text{proto}} = \|z_i - p_{\mathcal{A}(z_i)}\|_2^2$ is used with a balance regularization constraint. At test time, each test representation $z'$ is augmented into $k$ perturbed copies; POT is reformulated as standard OT and solved via Sinkhorn, with loss $\sum_{i,j} \gamma_{ij}C_{ij}$.
    - **Design Motivation**: Standard OT assumes full source–target alignment, which is too rigid. POT allows matching only a relevant subset of prototypes, avoiding forced alignment to irrelevant ones. The variance of the perturbations is derived from inter-prototype variance to maintain a reasonable distributional range.

### Loss & Training
- Joint training loss: $\mathcal{L} = \sum_i [\mathcal{L}_{\text{main}} + \mathcal{L}_{\text{ssl}} + \lambda_{\text{proto}}\mathcal{L}_{\text{proto}}] + \lambda_{\text{reg}}\mathcal{L}_{\text{reg}}$
- At test time, 5 gradient update steps are performed per input, after which encoder parameters are reset (reset protocol).
- OT is solved via the Sinkhorn algorithm with entropic regularization $\varepsilon=0.1$ and up to 1000 iterations.

## Key Experimental Results

### Main Results
Comparison against 9 baselines on 3 test cohorts (AUC, mean ± standard error over 20 independent runs):

| Method | Site A | Site B | MIMIC-IV |
|--------|--------|--------|----------|
| TEST (no adaptation) | 84.01 | 83.75 | 75.28 |
| TTT | 82.55±0.09 | 82.81±0.05 | 76.45±0.07 |
| TTT++ | 82.50±0.06 | 82.85±0.10 | 76.24±0.08 |
| SAR | 84.30±0.04 | 83.20±0.10 | 75.72±0.04 |
| CoTTA | 83.12±0.02 | 83.81±0.04 | 76.60±0.05 |
| **AdaTTT (Ours)** | **85.02±0.05** | **84.10±0.05** | **77.17±0.08** |

AdaTTT also achieves the best Brier scores (Site A: 0.086, Site B: 0.085, MIMIC-IV: 0.106).

### Ablation Study

| Configuration | Site A AUC | Site B AUC | MIMIC-IV AUC |
|---------------|-----------|-----------|-------------|
| PriTTT (dynamic masking only) | 84.61 | 83.98 | 76.84 |
| DynTTT (OT alignment only) | 84.54 | 83.84 | 76.79 |
| AdaTTT (full) | **85.02** | **84.10** | **77.17** |

### Key Findings
- Standard TTT and TTT++ **underperform** the no-adaptation baseline on EHR data (Site A: 82.55 vs. 84.01), demonstrating that EHR scenarios require specially designed TTT methods.
- Dynamic masking and OT alignment each independently contribute approximately 0.5–1% AUC gain; their combination yields the best and most stable performance.
- Only 5 gradient update steps per test sample are required, with an average of 0.29s per step; increasing prototype count from 4 to 16 has negligible impact on runtime.
- Sequential updating without parameter resetting leads to performance degradation in long-term deployment, confirming that the reset protocol is essential to prevent drift.

## Highlights & Insights
- **Theory-Guided Design**: Theorem 2 formalizes the TTT design problem as "minimizing auxiliary–main task conditional entropy," providing a theoretical framework applicable to guiding auxiliary task design in other TTT/TTA methods.
- **Domain-Specific Design Insight**: Recognizing the highly non-uniform feature importance in EHR data and using gradient-based importance to drive masking probabilities is a simple yet effective and readily implementable idea.
- **Partial OT via Perturbation**: The trick of converting POT to standard OT through perturbed copies is both elegant and practical, obviating the need for a dedicated POT solver implementation.

## Limitations & Future Work
- The experimental scale is relatively small (MIMIC-IV downsampled to ~2,000 encounters), and the positive rate is only 5–15%, indicating severe class imbalance.
- Evaluation is limited to binary classification (IMV vs. non-IMV); generalizability to multi-level prediction or other clinical tasks remains unknown.
- The rationale for selecting $k=4$ prototypes is insufficient; whether this is adequate for large-scale heterogeneous patient populations warrants further investigation.
- Robustness to conditional shift $p(Y|X)$ is limited (e.g., label shift induced by changes in clinical practice patterns).
- The reset protocol prevents the model from leveraging adaptation information from prior patients; knowledge accumulation during long-term deployment remains an open problem.

## Related Work & Insights
- **vs. TTT/TTT++**: Standard TTT uses fixed SSL tasks, whereas AdaTTT employs dynamic feature-aware SSL — in EHR settings, the former even underperforms the no-adaptation baseline.
- **vs. TENT**: Entropy minimization in TENT is unstable under single-instance small batches and may produce overconfident erroneous predictions.
- **vs. SAR**: SAR's sharpness-aware updates approach AdaTTT on Site A but are unstable on other datasets and rely on batch normalization layers.
- **vs. T3A**: Adapting only the classifier cannot correct domain shift in the representation space.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of information-theoretic bounds, dynamic masking, and POT is creative, though individual technical components are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-site evaluation, ablation studies, and sensitivity analyses are comprehensive, but dataset scale is limited.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear and clinical motivation is well articulated.
- Value: ⭐⭐⭐⭐ Addresses practical pain points of TTT in EHR scenarios with direct implications for clinical ML deployment.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Advancing Safe Mechanical Ventilation Using Offline RL With Hybrid Actions and Clinically Aligned Rewards](../../AAAI2026/medical_imaging/advancing_safe_mechanical_ventilation_using_offline_rl_with_.md)
- [\[ICCV 2025\] Progressive Test Time Energy Adaptation for Medical Image Segmentation](../../ICCV2025/medical_imaging/progressive_test_time_energy_adaptation_for_medical_image_segmentation.md)
- [\[ICLR 2026\] ATPO: Adaptive Tree Policy Optimization for Multi-Turn Medical Dialogue](atpo_adaptive_tree_policy_optimization_for_multi-turn_medical_dialogue.md)
- [\[CVPR 2026\] SPEGC: Continual Test-Time Adaptation via Semantic-Prompt-Enhanced Graph Clustering for Medical Image Segmentation](../../CVPR2026/medical_imaging/spegc_continual_test-time_adaptation_via_semantic-prompt-enhanced_graph_clusteri.md)
- [\[AAAI 2026\] Cross-Sample Augmented Test-Time Adaptation for Personalized Intraoperative Hypotension Prediction](../../AAAI2026/medical_imaging/cross-sample_augmented_test-time_adaptation_for_personalized_intraoperative_hypo.md)

<!-- RELATED:END -->

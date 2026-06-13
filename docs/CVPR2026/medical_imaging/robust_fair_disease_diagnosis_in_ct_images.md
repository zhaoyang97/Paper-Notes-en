---
title: >-
  [Paper Note] Robust Fair Disease Diagnosis in CT Images
description: >-
  [CVPR 2026][Medical Imaging][CT Diagnosis] This paper proposes a dual-objective training framework combining Logit-Adjusted Cross-Entropy (for class imbalance) and CVaR aggregation (for demographic fairness)…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "CT Diagnosis"
  - "Fairness"
  - "Class Imbalance"
  - "CVaR"
  - "Logit Adjustment"
date: 2026-05-08
content_hash: 7e28204277beb421
---

# Robust Fair Disease Diagnosis in CT Images

**Conference**: CVPR 2026
**arXiv**: [2604.09710](https://arxiv.org/abs/2604.09710)  
**Code**: [https://github.com/Purdue-M2/Fair-Disease-Diagnosis](https://github.com/Purdue-M2/Fair-Disease-Diagnosis)  
**Area**: Medical Imaging
**Keywords**: CT Diagnosis, Fairness, Class Imbalance, CVaR, Logit Adjustment

## TL;DR

This paper proposes a dual-objective training framework combining Logit-Adjusted Cross-Entropy (for class imbalance) and CVaR aggregation (for demographic fairness), achieving a gender-averaged macro F1 of 0.8403 with a fairness gap of only 0.0239 on CT disease diagnosis.

## Background & Motivation

**Background**: Deep learning has achieved strong aggregate performance on CT diagnosis, yet aggregate metrics obscure uneven model behavior across patient subgroups.

**Limitations of Prior Work**: Class imbalance and demographic underrepresentation frequently co-occur in clinical data. For example, squamous cell carcinoma has only 84 training samples, of which only 5 are female. Standard training causes the model to learn disease features almost entirely from male samples.

**Key Challenge**: Logit adjustment corrects for class-frequency bias but is agnostic to group labels, while CVaR balances group-level losses but is agnostic to class structure. Neither alone can address the true risk intersection (female + squamous cell carcinoma).

**Goal**: Design a unified training objective that simultaneously addresses class imbalance and demographic unfairness.

**Key Insight**: The two mechanisms operate along orthogonal axes — logit adjustment governs sample-level gradient direction (class axis), while CVaR governs group-level gradient magnitude (demographic axis).

**Core Idea**: The combination of Logit Adjustment and CVaR constitutes the minimal objective that is simultaneously sensitive to both the class axis and the demographic axis.

## Method

### Overall Architecture

3D ResNet-18 (pretrained on Kinetics-400) → 512 → 256 → 4-class head. During training: (1) compute Logit-Adjusted cross-entropy loss per sample; (2) compute mean loss per gender group; (3) CVaR aggregation selects the currently worse-performing group for upweighting.

### Key Designs

1. **Logit-Adjusted Cross-Entropy**:

    - Function: Corrects class-frequency bias at the sample level.
    - Mechanism: $\ell^{LA}(x,y) = -\log\frac{\exp(f_y(x)+\tau\log\pi_y)}{\sum_{y'}\exp(f_{y'}(x)+\tau\log\pi_{y'})}$, equivalent to an inter-class margin loss with larger margins for rare classes. At $\tau=1$, the estimator is Fisher-consistent with the balanced error rate.
    - Design Motivation: Unlike inverse-frequency weighting, logit adjustment directly modifies inter-class decision boundary margins, yielding greater effectiveness in separable regions.

2. **CVaR Fairness Aggregation**:

    - Function: Directs optimization pressure at the group level toward the currently worst-performing demographic group.
    - Mechanism: $\mathcal{L} = \min_\lambda \lambda + \frac{1}{\alpha|\mathcal{G}|}\sum_{g\in\mathcal{G}}[\mathcal{L}_g - \lambda]_+$, where $\alpha$ controls fairness intensity. The optimal $\lambda$ is solved via binary search (convex optimization with negligible overhead).
    - Design Motivation: CVaR provides a tractable upper bound on worst-case group risk without requiring specific assumptions about the group distribution.

3. **Orthogonality Analysis**:

    - Function: Theoretically justifies the complementarity of the two mechanisms.
    - Mechanism: Logit adjustment is invariant to group membership; CVaR is invariant to class structure. Their combination is the minimal objective sensitive to both axes simultaneously. On female squamous cell carcinoma (5 samples): LA alone causes 94% of gradients to originate from male samples; CVaR alone balances group losses but rare classes remain neglected.
    - Design Motivation: Demonstrates that this is not merely a stacking of two known techniques — their interaction produces effects unattainable by either component individually.

### Loss & Training

Adam optimizer, lr = 1e-4, cosine annealing, 70 epochs. Batch size = 2 (constrained by 3D volume memory). $\tau = 1.0$ fixed; $\alpha$ searched over the grid $\{0.4\text{–}0.9\}$.

## Key Experimental Results

### Main Results

| Method | α | F1\_male | F1\_female | Score↑ | Gap↓ |
|--------|---|----------|------------|--------|------|
| Baseline (CE) | - | 0.7957 | 0.6868 | 0.7413 | 0.1089 |
| LA Only | - | 0.8596 | 0.7375 | 0.7986 | 0.1221 |
| CVaR Only | 0.7 | 0.8738 | 0.7591 | 0.8165 | 0.1148 |
| LA+CVaR | 0.8 | 0.8283 | **0.8522** | **0.8403** | **0.0239** |

### Ablation Study

| Configuration | Score | Gap | Note |
|---------------|-------|-----|------|
| CE Baseline | 0.7413 | 0.1089 | Female squamous cell carcinoma recall = 0.08 |
| LA Only | 0.7986 | 0.1221 | Score improves but gap widens |
| CVaR Only | 0.8165 | 0.1148 | More balanced but rare classes still neglected |
| LA+CVaR α=0.8 | 0.8403 | 0.0239 | Only configuration where female F1 exceeds male |

### Key Findings

- $\alpha = 0.8$ is the optimal configuration — the only setting where female macro F1 (0.8522) surpasses male (0.8283).
- Female squamous cell carcinoma F1 improves from 0.14 (baseline) to 0.63; recall from 0.08 to 0.46.
- The $\alpha$ sweep exhibits a three-phase non-monotonic pattern: $0.4$–$0.6$ (broad tail dilutes fairness signal), $0.7$–$0.8$ (precise focus on hard subgroups), $0.9$ (overly narrow, performance rebounds).

## Highlights & Insights

- **Orthogonality Analysis**: Clearly demonstrates why two seemingly simple components produce synergistic effects that surpass their individual contributions.
- **Extreme Scenario with 5 Female Squamous Cell Carcinoma Samples**: This severely imbalanced intersection perfectly illustrates why a dual-objective formulation is necessary.
- **Three-Phase Behavior of α**: Reveals the nuanced influence of the CVaR concentration parameter, offering practical guidance for hyperparameter tuning.

## Limitations & Future Work

- Fairness is validated only for binary gender partitioning; extension to broader demographic attributes remains unexplored.
- The training set contains only 734 samples, representing an extremely limited scale.
- As a CVPR Workshop paper, the experimental scope is inherently constrained.

## Related Work & Insights

- **vs. DAW-FDD**: DAW-FDD employs stratified CVaR but relies on explicit group annotations and is validated only on binary detection tasks.
- **vs. LDAM**: LDAM lacks Fisher-consistency guarantees; logit adjustment is theoretically grounded at $\tau = 1$.

## Rating

- Novelty: ⭐⭐⭐ The method combines existing components, though the theoretical analysis is valuable.
- Experimental Thoroughness: ⭐⭐⭐ Dataset is small but ablations are complete.
- Writing Quality: ⭐⭐⭐⭐ The orthogonality analysis is clearly articulated.
- Value: ⭐⭐⭐⭐ Offers practical guidance for fairness in medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Robust Multi-Source Covid-19 Detection in CT Images](robust_multi-source_covid-19_detection_in_ct_images.md)
- [\[CVPR 2026\] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)
- [\[CVPR 2026\] EMAD: Evidence-Centric Grounded Multimodal Diagnosis for Alzheimer's Disease](emad_evidence-centric_grounded_multimodal_diagnosis_for_alzheimers_disease.md)
- [\[AAAI 2026\] DW-DGAT: Dynamically Weighted Dual Graph Attention Network for Neurodegenerative Disease Diagnosis](../../AAAI2026/medical_imaging/dw-dgat_dynamically_weighted_dual_graph_attention_network_for_neurodegenerative_.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)

</div>

<!-- RELATED:END -->

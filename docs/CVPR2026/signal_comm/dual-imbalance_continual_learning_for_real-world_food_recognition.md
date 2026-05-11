---
title: >-
  [Paper Note] Dual-Imbalance Continual Learning for Real-World Food Recognition
description: >-
  [CVPR 2026][Signal & Communication][continual learning] This paper proposes DIME, a framework that employs class-count-aware spectral adapter merging and rank-wise threshold modulation to address dual imbalance (intra-st…
tags:
  - "CVPR 2026"
  - "Signal & Communication"
  - "continual learning"
  - "dual imbalance"
  - "adapter merging"
  - "long-tail distribution"
  - "food recognition"
date: 2026-05-08
content_hash: 8f80503c3dc43558
---

# Dual-Imbalance Continual Learning for Real-World Food Recognition

**Conference**: CVPR 2026
**arXiv**: [2603.29133](https://arxiv.org/abs/2603.29133)
**Code**: [GitHub](https://github.com/xiaoyanzhang1/DIME)
**Area**: Continual Learning / Food Recognition
**Keywords**: continual learning, dual imbalance, adapter merging, long-tail distribution, food recognition

## TL;DR

This paper proposes DIME, a framework that employs class-count-aware spectral adapter merging and rank-wise threshold modulation to address dual imbalance (intra-step class long-tail distribution and inter-step class-count skew) in continual learning, consistently outperforming baselines by over 3% on four long-tail food recognition benchmarks.

## Background & Motivation

Real-world food recognition systems must continuously incorporate new dish categories. Such settings exhibit **dual imbalance**:

**Class Imbalance**: Food data naturally follows a long-tail distribution, where common items (e.g., rice, burgers) have abundant samples while the majority of niche dishes are severely underrepresented.

**Step Imbalance**: The number of categories introduced at each incremental learning step varies substantially—existing methods assume a roughly equal number of new classes per step, whereas in practice some steps may introduce many new dishes and others only a few.

The compounded effect of these two imbalances remains largely unexplored. The key challenge they jointly produce is **asymmetric learning dynamics**: head classes and large steps supply stable gradients, while tail classes and small steps generate high-variance, noisy updates that can disrupt previously learned representations.

## Method

### Overall Architecture

DIME builds upon a pre-trained ViT backbone with parameter-efficient fine-tuning:
1. A lightweight MLP adapter is trained at each learning step.
2. Balanced Softmax is applied to handle intra-step long-tail distributions.
3. After training, the new adapter is integrated into a cumulative base adapter via a spectral merging strategy.
4. At inference, only a single merged adapter is used, eliminating the need to maintain multiple task-specific modules.

### Key Designs

1. **Balanced Softmax Training**:

    - Function: Incorporates class frequency priors into the softmax to balance inter-class contributions.
    - Mechanism: The adjusted logit is $\tilde{z}_y = z_y + \log \pi_y$, where $\pi_y$ is the empirical frequency of class $y$.
    - Design Motivation: Prevents standard cross-entropy loss from being dominated by head classes, ensuring tail classes receive equitable learning signal.

2. **Class-Count-Aware Spectral Merging**:

    - Function: Merges old and new adapters within a shared SVD-aligned space.
    - Mechanism:
        - Concatenates the base adapter $M_B$ and new adapter $M_t$ column-wise and applies SVD: $X = [M_B \ M_t] = U\Sigma V^\top$
        - Blends representations in the aligned space using class-count proportional weights: $w_b = \frac{C_{\text{old}}}{C_{\text{old}}+C_{\text{new}}}$, $w_t = \frac{C_{\text{new}}}{C_{\text{old}}+C_{\text{new}}}$
        - $V_{\text{blend}}^\top = w_b V_B^\top + w_t V_t^\top$
    - Design Motivation: Naive parameter averaging causes destructive interference across steps; SVD alignment ensures updates interact along consistent principal directions; class-count weighting prevents noisy updates from a small new-class step from overwriting accumulated knowledge.

3. **Rank-Wise Threshold Modulation**:

    - Function: Differentially modulates update magnitude according to the importance of each singular-value direction.
    - Mechanism:
        - Directions corresponding to larger singular values encode dominant visual patterns (e.g., prevalent colors and textures) and should remain stable.
        - Directions with smaller singular values encode fine-grained variation and can more flexibly absorb new knowledge.
        - A gating mask is defined: the top $r_h$ directions use $\gamma_{\text{head}}$ (small value); remaining directions use $\gamma_{\text{tail}}$ (large value).
        - $V_{\text{final}}^\top = V_B^\top + G \odot \Delta V^\top$
    - Design Motivation: Large steps typically produce strong dominant directions, while small steps contribute weak but potentially useful variations; uniform merging fails to accommodate the distinct nature of both types of directions.

### Loss & Training

- The backbone (ViT-B/16 pre-trained on ImageNet-21K) is frozen; only adapter parameters and the classification head are trained.
- Adapters use an MLP structure with hidden dimension 64.
- SGD optimizer with learning rate 0.07, weight decay 0.0005, batch size 16, trained for 20 epochs.
- Step imbalance is controlled via an exponential decay sequence $s_t = \rho^{(t-1)/(T-1)}$; random permutation is applied to avoid artificial curriculum effects.

## Key Experimental Results

### Main Results

| Dataset | Metric ($A_T$) | DIME | TUNA (strongest baseline) | Gain |
|---|---|---|---|---|
| VFN186-LT | Last Acc | **69.07%** | 66.19% | +2.88% |
| VFN186-Insulin | Last Acc | **69.40%** | 66.28% | +3.12% |
| VFN186-T2D | Last Acc | **69.88%** | 67.32% | +2.56% |
| Food101-LT | Last Acc | **77.01%** | 75.00% | +2.01% |

The advantage of DIME is more pronounced under extreme imbalance ($\rho=0.001$): on VFN186-LT, DIME achieves 69.33% vs. TUNA's 66.60% (+2.73%); on Food101-LT, 78.13% vs. 74.02% (+4.11%).

### Ablation Study

| Configuration | $A_T$ | $wA$ | Description |
|---|---|---|---|
| Base (direct merge + equal weights + CE) | 66.73% | 74.90% | Baseline |
| + SM (spectral merging) | 67.20% | 74.95% | SVD alignment reduces conflict |
| + CCW (class-count weighting) | 67.95% | 76.68% | Step-imbalance awareness |
| + RTM (threshold modulation) | 68.68% | 77.67% | Selective protection of dominant directions |
| + BSM (Balanced Softmax) | **69.31%** | **78.07%** | Handles intra-step long-tail |

### Key Findings

- **The impact of dual imbalance is real and significant**: the greater the imbalance (smaller $\rho$), the larger the advantage of DIME.
- **Each component contributes clearly and complementarily**: SM, CCW, RTM, and BSM each deliver consistent incremental gains.
- **Inference efficiency is strong**: DIME's inference time (9.50s) and FLOPs (33.73G) are on par with the lightest baseline ACMap, while achieving approximately 4% higher accuracy.
- **Large steps are well protected without sacrificing small steps**: task-size analysis shows DIME performs best or near-best across large, medium, and small tasks.
- **Robustness to hyperparameters**: performance remains stable across reasonable ranges of $r_h$, $\gamma_{\text{head}}$, and $\gamma_{\text{tail}}$.

## Highlights & Insights

1. **Precise problem formulation**: The paper introduces the concept of "dual imbalance" and is the first to systematically study the compounded effects of class imbalance and step imbalance.
2. **Elegant design philosophy**: Merging is performed in an SVD-aligned space, with rank-adaptive gating realizing the principle of "stability for important directions, flexibility for secondary ones."
3. **Strong practicality**: Only a single merged adapter is maintained at inference, incurring no storage or retrieval overhead.
4. **Introduction of weighted average accuracy $wA$**: This metric more fairly evaluates overall performance under step imbalance, as the conventional $\bar{A}$ can be inflated by simple steps with few classes.

## Limitations & Future Work

- Validation is limited to food recognition; generalizability to other long-tail continual learning domains (e.g., medical imaging, autonomous driving) remains unverified.
- SVD decomposition introduces additional computational overhead at the merging stage, though it is performed only once per step transition.
- Combination with rehearsal strategies is unexplored; integrating an exemplar memory may yield further improvements.
- The adapter dimension is fixed at 64; the effect of varying adapter capacity across steps of different scales is not investigated.
- Only ViT-B/16 is evaluated; performance on larger backbones (e.g., ViT-L) is not reported.

## Related Work & Insights

- Builds upon the spectral alignment idea from KnOTS, extending it from LoRA to MLP adapters.
- Balanced Softmax originates from the long-tail learning literature, using log-prior compensation to address class imbalance.
- Comparison with recent continual learning methods including EASE, MOS, and TUNA demonstrates systematic advantages in the dual-imbalance setting.
- The rank-adaptive gating concept is generalizable to other scenarios requiring selective knowledge merging.

## Rating

- Novelty: ⭐⭐⭐⭐ — The dual-imbalance formulation and rank-adaptive merging are innovative, though individual components each have precedents.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four datasets, multiple imbalance ratios, complete ablations, efficiency comparisons, and sensitivity analyses.
- Writing Quality: ⭐⭐⭐⭐ — Problem formalization is rigorous with consistent notation.
- Value: ⭐⭐⭐⭐ — Addresses a practical and underexplored problem with a method that has broad generalization potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Contrastive Consolidation of Top-Down Modulations Achieves Sparsely Supervised Continual Learning](../../NeurIPS2025/signal_comm/contrastive_consolidation_of_top-down_modulations_achieves_sparsely_supervised_c.md)
- [\[ICCV 2025\] Boosting Multimodal Learning via Disentangled Gradient Learning](../../ICCV2025/signal_comm/boosting_multimodal_learning_via_disentangled_gradient_learning.md)
- [\[ICLR 2026\] Learning Molecular Chirality via Chiral Determinant Kernels](../../ICLR2026/signal_comm/learning_molecular_chirality_via_chiral_determinant_kernels.md)
- [\[ACL 2026\] UCS: Estimating Unseen Coverage for Improved In-Context Learning](../../ACL2026/signal_comm/ucs_estimating_unseen_coverage_for_improved_in-context_learning.md)
- [\[AAAI 2026\] Task Aware Modulation Using Representation Learning for Upscaling of Terrestrial Carbon Fluxes](../../AAAI2026/signal_comm/task_aware_modulation_using_representation_learning_for_upsaling_of_terrestrial_.md)

</div>

<!-- RELATED:END -->

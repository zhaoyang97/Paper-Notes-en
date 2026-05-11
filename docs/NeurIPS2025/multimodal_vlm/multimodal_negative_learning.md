---
title: >-
  [Paper Note] Multimodal Negative Learning
description: >-
  [NeurIPS 2025][Multimodal VLM][Multimodal Fusion] This paper proposes the Multimodal Negative Learning (MNL) paradigm, in which dominant modalities guide weaker modalities to suppress non-target classes—rather than enfor…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Multimodal Fusion"
  - "Modality Imbalance"
  - "Negative Learning"
  - "Robustness"
  - "Decision Fusion"
date: 2026-05-08
content_hash: 2cbc805ef3ee1fb1
---

# Multimodal Negative Learning

**Conference**: NeurIPS 2025
**arXiv**: [2510.20877](https://arxiv.org/abs/2510.20877)
**Code**: [Available](https://github.com/BaoquanGong/Multimodal-Negative-Learning)
**Area**: Multimodal Learning
**Keywords**: Multimodal Fusion, Modality Imbalance, Negative Learning, Robustness, Decision Fusion

## TL;DR

This paper proposes the Multimodal Negative Learning (MNL) paradigm, in which dominant modalities guide weaker modalities to suppress non-target classes—rather than enforcing alignment on target classes—thereby stabilizing the decision space, preserving modality-specific information, and theoretically tightening the robustness lower bound of multimodal fusion.

## Background & Motivation

In multimodal learning, significant differences in quality and informativeness across modalities give rise to the modality imbalance problem. Conventional approaches (e.g., knowledge distillation, confidence-based weighting) follow a **positive learning** paradigm—guiding weaker modalities to mimic the target-class predictions of stronger ones. However, this forced alignment carries serious risks:

**Suppression of modality-specific information**: Weaker modalities are pushed toward stronger ones, causing the loss of their unique complementary information.

**Error propagation**: When a dominant modality also misclassifies certain samples, weaker modalities blindly follow, degrading overall performance.

**Over-alignment collapse**: Through statistical analysis, the authors find that after KL-guided alignment training, samples that were originally predicted correctly by the weaker modality but incorrectly by the stronger modality tend to become misclassified.

The core insight stems from a simple intuition: **ruling out wrong answers is often easier than identifying the correct one**. Under limited data quality, teaching weaker modalities *what not to choose* is more stable and reliable than teaching them *what to choose*.

## Method

### Overall Architecture

MNL is a late-fusion framework trained in two stages:
- **Stage 1** (warm-up): Each modality is optimized solely with cross-entropy loss on target-class predictions.
- **Stage 2** (negative learning): After modality-level performance stabilizes, the MNL loss is introduced to leverage the dominant modality's information on non-target classes to guide the weaker modality.

### Key Designs

1. **Unimodal Confidence Margin (UCoM)**: Defined as the difference between the target-class logit and the strongest competing class logit, $\xi_{(m)} = f^{(m)}(x)_y - f^{(m)}(x)_j$. A larger UCoM indicates greater reliability in discriminating the target class from competing classes.

2. **Robust Dominant Modality (RDM)**: Determined not only by higher target-class confidence but also by a larger UCoM, thereby avoiding the risk of a low-margin modality guiding a high-margin one.

3. **Dynamic Guidance Mechanism**: Modality dominance is not fixed but determined dynamically per sample and per iteration. Modality 1 guides modality 2 only when it simultaneously achieves higher target-class confidence and a larger UCoM, and vice versa.

4. **MNL Core Formulation**:

$$MNL(P^{(RDM)}, P^{(IM)}, \bar{y}) = -\bar{y} \cdot P^{(RDM)} \cdot \log(P^{(IM)})$$

where $\bar{y}$ is zero at the ground-truth class and one at all non-target classes. Guidance is applied exclusively over non-target classes, and the RDM predictions are detached (no gradient back-propagation).

### Loss & Training

The total loss is:

$$\mathcal{L} = CE(P^{fusion}, y) + \sum_{i=1}^{2} CE(P^{(i)}, y) + \lambda \cdot MNL(P^{(RDM)}, P^{(IM)}, \bar{y})$$

- The first two terms handle positive learning on target classes (cross-entropy).
- The third term handles negative learning on non-target classes.
- $\lambda$ controls the strength of MNL.

### Theoretical Guarantee

**Theorem 3.1**: In bimodal late fusion, the lower bound of the robustness radius satisfies:

$$R(x_i) \geq \frac{w^{(1)}\xi_{(1)} + w^{(2)}\xi_{(2)}}{\sqrt{(w^{(1)}\tau_{(1)})^2 + (w^{(2)}\tau_{(2)})^2}}$$

This shows that increasing the UCoM of individual modalities directly tightens the robustness lower bound of the multimodal system. MNL achieves this by suppressing uncertainty over non-target classes, thereby enlarging UCoM.

## Key Experimental Results

### Main Results

| Method | MVSA (ε=0/5/10) | FOOD101 (ε=0/5/10) | CREMA-D (ε=0/5/10) |
|--------|-----------------|---------------------|---------------------|
| LF | 76.88/63.46/55.16 | 90.69/68.49/57.99 | 68.04/64.25/52.39 |
| LF+MNL | 79.50/74.03/63.01 | 92.77/75.16/62.06 | 73.71/70.35/57.26 |
| Δ | +2.62/+10.57/+7.85 | +2.08/+6.67/+4.06 | +5.67/+6.10/+4.87 |
| PDF | 79.94/74.40/63.09 | 93.32/76.47/62.83 | 67.07/64.57/53.33 |
| PDF+MNL | 80.54/74.07/63.78 | 93.33/76.65/63.16 | 69.18/66.94/55.43 |

MNL yields substantial gains for static fusion (LF)—e.g., +10.57% on MVSA at ε=5—while improvements on dynamic fusion methods (PDF/QMF) are modest but consistent.

### Ablation Study

| Guidance Strategy | Prior | Confident | Robust | MVSA ε=0/5/10 |
|------------------|-------|-----------|--------|---------------|
| LF baseline | - | - | - | 76.88/63.46/55.16 |
| Fixed prior guidance | ✓ | | | 78.66/72.69/62.77 |
| Confidence-only guidance | | ✓ | | 78.74/71.87/59.35 |
| Confidence + UCoM | | ✓ | ✓ | **79.50/74.03/63.01** |

| Guidance Scope | All-Class | Non-Target | MVSA ε=0/5/10 |
|---------------|-----------|------------|---------------|
| All-class guidance | ✓ | | 78.90/72.16/62.52 |
| Non-target-only guidance | | ✓ | **79.50/74.03/63.01** |

### Key Findings

- MNL provides substantially larger gains for static fusion than for dynamic fusion, as dynamic fusion inherently down-weights weaker modalities—conflicting with MNL's objective of enlarging the weaker modality's margin.
- Gains on NYU Depth V2 are smaller because the two modalities are already closely matched, leaving limited room for dominant-modality guidance.
- Non-target guidance consistently outperforms all-class guidance, validating that teaching weaker modalities to exclude errors is more effective than full alignment.
- Dynamic guidance (Confident + Robust) surpasses fixed prior guidance, confirming that modality dominance indeed varies across samples.

## Highlights & Insights

1. **Paradigm innovation**: Reframing multimodal fusion from positive to negative learning offers a novel and intuitive perspective.
2. **Solid theoretical foundation**: The importance of UCoM is derived from a robustness lower-bound analysis, which motivates both the RDM definition and the MNL loss design.
3. **Plug-and-play**: MNL is compatible with various late-fusion methods and introduces no additional inference overhead.
4. **Dynamic guidance**: Per-sample modality dominance estimation, rather than a global fixed assignment, better reflects real-world conditions.

## Limitations & Future Work

- MNL is currently restricted to late-fusion frameworks; extensions to intermediate and early fusion remain unexplored.
- Only the bimodal case is considered; generalizing RDM selection and guidance relationships to settings with more than two modalities is non-trivial.
- When the two modalities are of comparable quality (e.g., NYU Depth V2), MNL yields limited benefit.
- The dynamic guidance mechanism relies on within-batch predictions; its stability under extreme noise conditions requires further investigation.

## Related Work & Insights

- **Positive alignment methods** (knowledge distillation, confidence-based weighting) are prone to suppressing modality-specific information and serve as the direct baselines for this work.
- **The negative learning idea** originates from NL-NL (Kim et al.) in the context of noisy label learning; this paper is the first to introduce it into multimodal fusion.
- **Robustness analysis** builds on the multimodal robustness metric of yang2024quantifying and extends it to the late-fusion setting.
- The work has implications for AI safety: decision reliability under modality imbalance is a core concern in safety-critical applications.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The negative learning perspective in multimodal fusion is pioneering; the UCoM formulation and dynamic guidance mechanism are theoretically grounded.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 4 datasets, multiple fusion strategies, diverse noise types, and thorough ablations; lacks validation on large-scale vision-language models.
- **Writing Quality**: ⭐⭐⭐⭐ — Theoretical derivations are clear and motivation figures are intuitive, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐ — The plug-and-play modular design has practical applicability, and the theoretical contributions offer meaningful reference value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](rethinking_multimodal_learning_from_the_perspective_of_mitig.md)
- [\[NeurIPS 2025\] MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)
- [\[NeurIPS 2025\] Continual Multimodal Contrastive Learning](continual_multimodal_contrastive_learning.md)
- [\[NeurIPS 2025\] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)
- [\[ICCV 2025\] G2D: Boosting Multimodal Learning with Gradient-Guided Distillation](../../ICCV2025/multimodal_vlm/g2d_boosting_multimodal_learning_with_gradient-guided_distillation.md)

</div>

<!-- RELATED:END -->

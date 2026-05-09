---
title: >-
  [Paper Note] Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][OOD Detection] This paper proposes TANL (Test-time Activated Negative Labels), which dynamically evaluates the "activation degree" of negative labels on OOD samples at test time to identify the most effective negative labels. Combined with an activation-aware scoring function, TANL reduces FPR95 from 17.5% to 9.8% on the ImageNet benchmark, while remaining entirely training-free and test-time efficient.
tags:
  - CVPR 2026
  - Multimodal VLM
  - OOD Detection
  - Vision-Language Models
  - Negative Labels
  - Test-Time Adaptation
  - Activation Metric
date: 2026-05-08
content_hash: ae41d5e8936665d3
---

# Activation Matters: Test-time Activated Negative Labels for OOD Detection with Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.25250](https://arxiv.org/abs/2603.25250)
**Code**: [GitHub](https://github.com/YBZh/OpenOOD-VLM)
**Area**: Multimodal VLM / AI Safety
**Keywords**: OOD Detection, Vision-Language Models, Negative Labels, Test-Time Adaptation, Activation Metric

## TL;DR
This paper proposes TANL (Test-time Activated Negative Labels), which dynamically evaluates the "activation degree" of negative labels on OOD samples at test time to identify the most effective negative labels. Combined with an activation-aware scoring function, TANL reduces FPR95 from 17.5% to 9.8% on the ImageNet benchmark, while remaining entirely training-free and test-time efficient.

## Background & Motivation
**Background**: OOD detection is a core problem in AI safety. VLM-based methods (e.g., CLIP) detect OOD samples by introducing "negative labels"—text labels semantically distant from in-distribution (ID) categories—where samples with high similarity to negative labels are more likely to be OOD.

**Key Problem — "Low-Activation Negative Labels"**:
   - Methods such as NegLabel select words from a corpus that are maximally distant from ID labels as negative labels.
   - However, these negative labels are chosen solely based on ID labels, **without considering the test distribution**.
   - Consequently, many negative labels exhibit very low activation (similarity) on OOD data—sometimes even lower than on ID data (see Fig. 1a).
   - Such "low-activation" labels are not only ineffective but also introduce noise that degrades detection performance.

**Core Observation**: A small number of highly activated negative labels suffices for effective OOD detection (Fig. 1b), while a large number of low-activation labels is actually harmful.

**Core Idea**: Dynamically evaluate label activation at test time to select negative labels that are genuinely "activated" by OOD samples.

## Method

### Overall Architecture
Maintain positive/negative sample FIFO queues at test time → Dynamically compute label activation scores over the corpus → Select highly activated negative labels → Apply an activation-aware scoring function for OOD detection.

### Key Designs
1. **Activation Metric**:
   Measures the average classification probability of a given label over a dataset:
    $Act(\mathcal{X}, \hat{y}_i) = \frac{1}{|\mathcal{X}|}\sum_{\mathbf{x} \in \mathcal{X}} \frac{\exp(\mathbf{v}\hat{\mathbf{t}}_i)}{\sum_j \exp(\mathbf{v}\mathbf{t}_j) + \sum_j \exp(\mathbf{v}\hat{\mathbf{t}}_j)}$
   An ideal negative label should be highly activated on OOD data and weakly activated on ID data. The differential activation score is defined as:
    $Act_d(\hat{y}_i) = Act(\mathcal{X}_{ood}, \hat{y}_i) - Act(\mathcal{X}_{id}, \hat{y}_i)$
    - **Design Motivation**: Directly quantifies the discriminative power of each label for OOD detection.

2. **Distribution-Adaptive Activated Labels**:
   High-confidence positive and negative samples cached in queues are used to approximate $\mathcal{X}_{id}$ and $\mathcal{X}_{ood}$:
    - FIFO queues $\mathcal{X}_{pos}$ / $\mathcal{X}_{neg}$ of length $L$
    - Positive samples: $S_{aa}(\mathbf{v}) \geq \gamma + (1-\gamma)g$; Negative samples: $S_{aa}(\mathbf{v}) < \gamma - \gamma g$
    - **Initialization**: Positive queue initialized with ID label features; negative queue initialized with Gaussian noise image features.
    - **Design Motivation**: The OOD distribution is unknown and may shift dynamically, necessitating online adaptation.

3. **Batch-Adaptive Variant**:
   Additional positive and negative samples are extracted from the current test batch and fused with historical samples via weighted averaging:
    $Act_b(\mathcal{X}_{pos}, \hat{y}_i) = \alpha Act(\mathcal{X}_{pos}, \hat{y}_i) + (1-\alpha) Act(\mathcal{X}^b_{pos}, \hat{y}_i)$
    - **Design Motivation**: Historical samples reflect overall trends, while the current batch captures immediate characteristics; the two are complementary.

4. **Activation-Aware Scoring Function**:
    $S_{aa}(\mathbf{v}) = \frac{1}{M}\sum_{m=1}^{M}\sum_{i=1}^{C}\frac{\exp(\mathbf{v}\mathbf{t}_i)}{\sum_j \exp(\mathbf{v}\mathbf{t}_j) + \sum_{j=1}^m \exp(\mathbf{v}\tilde{\mathbf{t}}_j)}$
   After sorting negative labels by activation score, highly activated labels appear in the denominator more frequently, implicitly receiving higher weight.
    - **Design Motivation**: Different negative labels carry different importance; highly activated labels should dominate the score. The cumulative summation design also enhances robustness to the number of labels $M$.

### Loss & Training
- **Completely training-free** (zero-shot, training-free)
- The CLIP encoder is frozen; only FIFO queues are maintained at test time.
- Hyperparameters: $\gamma$ (ID/OOD threshold), $g$ (confidence margin), $L$ (queue capacity), $\alpha$ (historical/batch weight).

## Key Experimental Results

### Main Results (ImageNet-1k, CLIP ViT-B/16)

| Method | Type | iNaturalist FPR95↓ | SUN FPR95↓ | Places FPR95↓ | Textures FPR95↓ | Average FPR95↓ |
|------|------|-----|-----|-----|-----|-----|
| NegLabel | Training-free | 1.91 | 20.53 | 35.59 | 43.56 | 25.40 |
| CSP | Training-free | 1.54 | 13.66 | 29.32 | 25.52 | 17.51 |
| AdaNeg | Test-time adaptive | 0.59 | 9.50 | 34.34 | 31.27 | 18.92 |
| OODD | Test-time adaptive | 0.85 | 12.94 | 30.68 | 30.67 | 18.79 |
| **TANL** | **Test-time adaptive** | **0.42** | **3.53** | - | - | **9.8** |

*Note: TANL reduces average FPR95 from 25.4% (NegLabel) to 9.8% (a 61% relative reduction), and further reduces FPR95 by 44% compared to CSP.*

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| NegLabel (distance-based selection) | FPR95: 25.4% | Activation not considered |
| + Activation-based label selection | FPR95 drops substantially | Activated labels are the key factor |
| + Activation-aware scoring | Further improvement | Weighting effect is significant |
| + Batch-adaptive | Best performance | Instant information is beneficial |
| Robustness to $M$ | $S_{aa}$ is robust to $M$ | Conventional methods are sensitive to $M$ |

### Key Findings
- Activation-aware label selection is the core contribution: a small number of highly activated labels outperforms a large pool of low-activation labels.
- FPR95 decreases from 25.4% to 9.8% compared to NegLabel—a reduction of 15.6 percentage points.
- A further reduction of 7.7 percentage points compared to the previous best method, CSP.
- $S_{aa}$ exhibits inherent robustness to the number of negative labels $M$, eliminating the need to fine-tune $M$.
- The initialization strategy is effective: initializing the positive queue with ID label features and the negative queue with noise image features provides a stable starting point.
- The method generalizes across different CLIP backbones (ViT-B/16, ViT-L/14, etc.) and various settings including near-OOD, full-spectrum OOD, and medical OOD detection.

## Highlights & Insights
- **The "activation" concept is simple yet effective**: it addresses the previously overlooked question of which negative labels are truly useful.
- **The cumulative summation scoring function is elegantly designed**: a single formulation simultaneously achieves label weighting and robustness.
- **Training-free and test-time efficient**: highly practical, requiring only FIFO queue maintenance with no backpropagation.
- **Using noise images to initialize the negative queue** as an OOD proxy is an intuitively appealing design choice.

## Limitations & Future Work
- The method relies on high-confidence samples to initialize the queues; inaccurate early detection may lead to error accumulation.
- The FIFO queue length $L$ is a hyperparameter that may be insufficient in extreme cases.
- In near-OOD settings where ID and OOD distributions are very close, high-confidence positive and negative samples may be scarce.
- Theoretical analysis rests on specific assumptions that may not hold in all practical distributions.
- Validation is limited to CLIP; generalizability to other VLMs remains unexplored.

## Related Work & Insights
- The improvement over NegLabel is direct and substantial—the paper's core insight is that the **label selection strategy** matters more than the quantity of labels.
- The test-time adaptation (TTA) paradigm is extended from model parameter updates to label selection, representing a novel variant.
- The activation metric may generalize to other label-based zero-shot methods.
- The approach offers a complementary perspective to AdaNeg (image proxy vs. label activation in this work).

## Rating
- Novelty: ⭐⭐⭐⭐ — The activation metric concept is novel, and the scoring function design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers multiple OOD types, backbones, theoretical analysis, and robustness validation comprehensively.
- Writing Quality: ⭐⭐⭐⭐⭐ — Motivation figures are analytically clear; algorithmic diagrams are intuitive.
- Value: ⭐⭐⭐⭐⭐ — A simple yet effective improvement with immediate practical value for VLM-based OOD detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mind the Way You Select Negative Texts: Pursuing the Distance Consistency in OOD Detection with VLMs](mind_the_way_you_select_negative_texts_pursuing_the_distance_consistency_in_ood_.md)
- [\[AAAI 2026\] Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models](../../AAAI2026/multimodal_vlm/cross-modal_proxy_evolving_for_ood_detection_with_vision-lan.md)
- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](../../ICCV2025/multimodal_vlm/negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[AAAI 2026\] Panda: Test-Time Adaptation with Negative Data Augmentation](../../AAAI2026/multimodal_vlm/panda_test-time_adaptation_with_negative_data_augmentation.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)

</div>

<!-- RELATED:END -->

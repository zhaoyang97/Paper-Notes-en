---
title: >-
  [Paper Note] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP
description: >-
  [ICCV 2025][LLM Evaluation][CLIP] This paper proposes BATCLIP, a bimodal online test-time adaptation (TTA) method for CLIP that simultaneously adapts the LayerNorm parameters of both the visual and text encoders. By intr…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "CLIP"
  - "test-time adaptation"
  - "bimodal adaptation"
  - "robustness to image corruption"
  - "vision-language models"
date: 2026-05-08
content_hash: e101ee58b8639ec9
---

# BATCLIP: Bimodal Online Test-Time Adaptation for CLIP

**Conference**: ICCV 2025
**arXiv**: [2412.02837](https://arxiv.org/abs/2412.02837)  
**Code**: [https://github.com/sarthaxxxxx/BATCLIP](https://github.com/sarthaxxxxx/BATCLIP)  
**Area**: LLM Evaluation
**Keywords**: CLIP, test-time adaptation, bimodal adaptation, robustness to image corruption, vision-language models

## TL;DR
This paper proposes BATCLIP, a bimodal online test-time adaptation (TTA) method for CLIP that simultaneously adapts the LayerNorm parameters of both the visual and text encoders. By introducing a projection matching loss and an inter-class separability loss to enhance vision-text feature alignment and class discriminability, BATCLIP achieves state-of-the-art performance on CIFAR-10C, CIFAR-100C, and ImageNet-C.

## Background & Motivation
Despite CLIP's strong zero-shot classification performance as a vision-language model, its accuracy degrades sharply under common image corruptions (e.g., Gaussian noise, fog, snow). The authors observe that with ViT-B/16, CLIP's accuracy on CIFAR-100 under Gaussian noise at severity-5 drops from 66.6% to 10.79%. Such brittleness has serious consequences in safety-critical applications such as autonomous driving.

Existing CLIP TTA methods suffer from three core problems:

**Unimodal limitation**: TPT optimizes only text prompts while keeping the visual encoder frozen, leaving the adapted prompts unaware of the test image distribution.

**Computational expense**: TPT requires generating multiple augmented views per test image and performing multiple forward passes.

**Prompt template dependency**: Methods such as WATT rely on multiple prompt templates, making prompt selection or optimization at test time impractical.

Through systematic experiments, the authors identify two key insights: (1) CLIP is highly sensitive to corruption severity, with noticeable degradation even at severity-1; and (2) different prompt templates have negligible effect on corrupted images — the true bottleneck lies in vision-text feature alignment rather than in text prompts. This motivates a bimodal adaptation approach that adjusts both encoders simultaneously.

## Method

### Overall Architecture
BATCLIP operates in an online TTA setting: test batches arrive sequentially, and the model performs a single forward pass and parameter update per batch. Only the LayerNorm parameters of both encoders are adapted (approximately 0.044% of total parameters), and parameters are reset after each task/corruption type.

### Key Designs

1. **Bimodal LayerNorm Adaptation**:

    - Function: Simultaneously updates the LayerNorm parameters of the visual encoder $f_{vis}$ and the text encoder $f_{txt}$.
    - Mechanism: Unlike existing methods that update only the visual side or text prompts, BATCLIP jointly updates the normalization layer parameters $\phi_v$ and $\phi_t$ of both encoders, enabling mutually-aware, domain-specific feature representations.
    - Design Motivation: Unimodal adaptation (updating only vision or only text) leads to misalignment between the two modalities. When only the visual encoder is updated, text features remain optimized for pre-training data; when only text prompts are updated, the resulting text features remain unaware of the test distribution.

2. **Projection Matching Loss**:

    - Function: Maximizes the scalar projection of visual class prototypes onto the corresponding text features.
    - Mechanism: Visual prototypes $\bar{v}_c$ (mean of intra-class image features) are first computed using pseudo-labels; the projection of each prototype onto the normalized text feature is then maximized:
    $\mathcal{L}_{pm} = \frac{1}{C} \sum_c \bar{v}_c \cdot \hat{z}_c$
      where $\hat{z}_c = z_c / \|z_c\|_2$. Using prototypes rather than individual features mitigates noise in pseudo-labels under corrupted inputs by smoothing the class distribution.
    - Design Motivation: Geometrically, maximizing the projection aligns visual features with the direction of text features, directly optimizing vision-text alignment.

3. **Inter-class Separability Loss**:

    - Function: Increases the cosine distance between prototypes of different classes.
    - Mechanism:
    $\mathcal{L}_{sp} = \sum_{l \in C} \sum_{c \in C} \mathbb{1}[l \neq c](1 - \cos(\bar{v}_c, \bar{v}_l))$
      Maximizing this loss pushes visual prototypes of different classes apart in the feature space.
    - Design Motivation: Image corruptions can cause visual features of different classes to overlap; aligning vision and text alone is insufficient to ensure well-separated classification decision boundaries.

4. **Overall Optimization Objective**:
    $\arg\min_{\phi_v, \phi_t} (\mathcal{L}_{ent} - \mathcal{L}_{pm} - \mathcal{L}_{sp})$
   The three losses serve distinct roles: (1) $\mathcal{L}_{ent}$ entropy minimization — reduces prediction uncertainty; (2) $-\mathcal{L}_{pm}$ projection matching — reinforces vision-text alignment; (3) $-\mathcal{L}_{sp}$ inter-class separation — enhances feature discriminability.

### Loss & Training
- AdamW optimizer; learning rate 1e-3 for CIFAR-10C, 5e-4 for CIFAR-100C/ImageNet-C.
- Single-step online adaptation (one gradient update per batch), suitable for real-time deployment.
- Uses the generic prompt template "a photo of a \<CLS\>." without relying on multiple templates.
- Model parameters are reset after each corruption task to prevent catastrophic forgetting.
- Batch size: 200 for CIFAR-10C/100C, 64 for ImageNet-C.

## Key Experimental Results

### Main Results (Average Accuracy %, severity-5, ViT-B/16)

| Method | CIFAR-10C | CIFAR-100C | ImageNet-C |
|--------|-----------|------------|------------|
| Zero-shot CLIP | 61.16 | 35.79 | 24.51 |
| TENT | 62.03 | 37.96 | 25.15 |
| SAR | 67.37 | 41.19 | 29.73 |
| TPT | 63.64 | 36.15 | 24.87 |
| VTE | 64.15 | 35.01 | 25.60 |
| WATT-S* | 72.81 | 36.71 | 24.67 |
| **BATCLIP** | **73.85** | **42.09** | **30.72** |

### Ablation Study

| Loss Combination | CIFAR-10C | CIFAR-100C | ImageNet-C | Note |
|-----------------|-----------|------------|------------|------|
| $\mathcal{L}_{ent}$ | 60.65 | 38.17 | 24.03 | Entropy minimization only |
| $\mathcal{L}_{sp}$ | 73.16 | 41.36 | 30.05 | Inter-class separation contributes most |
| $\mathcal{L}_{ent}+\mathcal{L}_{pm}$ | 62.60 | 39.32 | 25.21 | Alignment provides modest gain |
| $\mathcal{L}_{ent}+\mathcal{L}_{sp}$ | 72.69 | 41.84 | 30.08 | Separation yields substantial gain |
| $\mathcal{L}_{ent}+\mathcal{L}_{pm}+\mathcal{L}_{sp}$ | **73.85** | **42.09** | **30.72** | All three combined is optimal |

### Key Findings
- $\mathcal{L}_{sp}$ (inter-class separability) is the primary contributor to performance gains, achieving 73.16% on CIFAR-10C when used alone.
- Multi-step iterative adaptation leads to overfitting; single-step adaptation achieves the best balance between robustness and performance.
- BATCLIP remains competitive with or superior to TPT and VTE at small batch sizes (e.g., 32), as prototype computation adapts to available samples.
- The method generalizes to ViT-L/14, achieving 84.74% on CIFAR-10C (vs. 75.84% zero-shot).
- Computational efficiency is substantially better than WATT — only 0.2 seconds per batch (vs. 2.34 seconds for WATT), enabling real-time deployment.
- Consistent performance gains are achieved by updating only 0.044% of total parameters.

## Highlights & Insights
- The systematic analysis of CLIP zero-shot performance across backbones, corruption types, and prompt templates provides a valuable reference for the community.
- The core insight behind "bimodal adaptation" — that the text encoder also needs to be aware of the test domain — is the key to overcoming the bottleneck of unimodal TTA.
- The method is simple and practical: no multiple templates or augmented views are required; single-step, single-template adaptation supports real-time deployment.
- t-SNE visualizations clearly demonstrate how BATCLIP forms tighter class clusters and achieves better vision-text alignment.

## Limitations & Future Work
- Gains on domain generalization benchmarks such as OfficeHome and PACS are limited, possibly because single-step adaptation is insufficient to handle large style shifts.
- Pseudo-label quality depends on batch size and corruption severity, and may degrade under extreme conditions.
- Adaptive learning rate schedules or more sophisticated prototype update strategies remain unexplored.
- Catastrophic forgetting is avoided via inter-task parameter resets rather than being fundamentally addressed.
- Validation is primarily conducted with ViT-B/16; the effectiveness on other visual backbones remains limited.

## Related Work & Insights
- **vs TPT**: TPT performs unimodal adaptation (text prompt optimization only) and requires multiple forward passes per image, making it slow and oblivious to visual encoder adaptation.
- **vs VTE**: VTE employs multi-template ensemble without updating model parameters, resulting in insufficient adaptability under severe corruptions.
- **vs WATT**: WATT relies on multi-template, multi-step adaptation, which is unsuitable for online TTA and incurs more than 10× the computational cost.
- **vs TENT**: TENT updates only batch normalization layers without considering vision-text alignment; directly applying it to CLIP yields limited benefit.
- **vs SAR**: SAR performs well by filtering noisy samples in gradient space but remains a unimodal method.

## Rating
- Novelty: ⭐⭐⭐⭐ The bimodal adaptation idea is well-motivated and effective; the combination of projection matching and inter-class separability is clearly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three major benchmarks plus domain generalization datasets, detailed ablations, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Thorough analysis with systematic preliminary studies preceding the method proposal; logically coherent.
- Value: ⭐⭐⭐⭐ High practical utility; the method is simple, efficient, and suitable for real-world deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/llm_evaluation/test-time_adaptation_by_causal_trimming.md)
- [\[ICCV 2025\] Rethinking Few Shot CLIP Benchmarks: A Critical Analysis in the Inductive Setting](rethinking_few_shot_clip_benchmarks_a_critical_analysis_in_the_inductive_setting.md)
- [\[ICCV 2025\] Imbalance in Balance: Online Concept Balancing in Generation Models](imbalance_in_balance_online_concept_balancing_in_generation_models.md)
- [\[AAAI 2026\] Test-time Diverse Reasoning by Riemannian Activation Steering](../../AAAI2026/llm_evaluation/test-time_diverse_reasoning_by_riemannian_activation_steering.md)
- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](../../AAAI2026/llm_evaluation/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)

</div>

<!-- RELATED:END -->

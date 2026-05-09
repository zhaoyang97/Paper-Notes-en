---
title: >-
  [Paper Note] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models
description: >-
  [ICCV 2025][Multimodal VLM][Test-time adaptation] This paper proposes MCP/MCP++, a multi-cache enhanced prototype learning framework that constructs compact intra-class distributions via three complementary cache modules—entropy cache, align cache, and negative cache—and further introduces cross-modal residual learning to refine the alignment between visual and textual prototypes, achieving state-of-the-art zero-shot generalization across 15 downstream tasks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Test-time adaptation
  - vision-language models
  - CLIP
  - cache mechanism
  - prototype learning
date: 2026-05-08
content_hash: 65cbc277c83932cf
---

# Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models

**Conference**: ICCV 2025
**arXiv**: [2508.01225](https://arxiv.org/abs/2508.01225)
**Code**: [Project Page](https://zhaihaotian.github.io/MCP-ICCV25/)
**Area**: Multimodal VLM
**Keywords**: Test-time adaptation, vision-language models, CLIP, cache mechanism, prototype learning

## TL;DR

This paper proposes MCP/MCP++, a multi-cache enhanced prototype learning framework that constructs compact intra-class distributions via three complementary cache modules—entropy cache, align cache, and negative cache—and further introduces cross-modal residual learning to refine the alignment between visual and textual prototypes, achieving state-of-the-art zero-shot generalization across 15 downstream tasks.

## Background & Motivation

Vision-language pre-trained models (e.g., CLIP) exhibit strong zero-shot capabilities but suffer from **distribution shift** in real-world deployment, where discrepancies between pre-training and test data lead to performance degradation. **Test-Time Adaptation (TTA)** aims to rapidly adapt to new distributions during inference using unlabeled test data.

Limitations of existing TTA methods:

**Prompt-based methods** (e.g., TPT): Learn domain-specific prompts via consistency constraints—effective but **computationally expensive**.

**Cache-based methods** (e.g., TDA): Build caches by storing low-entropy samples to improve adaptability—efficient but reliant on a **critical assumption** that low-entropy samples belong to the same class and form compact intra-class distributions.

Through a key empirical study, the authors find that **cache-based performance gain is positively correlated with intra-class compactness** (Pearson $r > 0.8, p = 2.25 \times 10^{-3}$). TDA yields significant gains on datasets with compact class distributions (e.g., EuroSAT), but limited improvement on scattered ones (e.g., Aircraft). This finding points to a clear direction for improvement: **enhancing intra-class compactness**.

## Method

### Overall Architecture

MCP consists of three complementary cache modules: entropy cache initializes prototypes, align cache enhances compactness, and negative cache calibrates predictions. MCP++ further incorporates cross-modal prototype residual learning.

### Key Designs

1. **Entropy Cache**:
   Dynamically stores the lowest-entropy samples per class to provide stable anchors for class representations. A fixed-capacity queue of size $M$ (where $M \ll |X_{\text{test}}|$) is maintained per class; a new test sample replaces the highest-entropy entry only if its entropy is lower than that entry. This prevents prototype drift and anchors high-confidence features.

2. **Align Cache**:
   Builds a more compact intra-class distribution by integrating visual and textual information. A class prototype center is first constructed by fusing the text prototype $\bar{t}_c$ and visual prototype $\bar{v}_c$:
   $\mu_c = w \cdot \bar{v}_c + (1-w) \cdot \bar{t}_c$
   where $w = 0.8$ balances the two modalities. The admission criterion for the align cache is more stringent: a sample must not only exhibit low entropy ($H(x) < H_{\max}^{\hat{y}}$), but also be closer to the prototype center than the farthest sample currently in the cache ($d(f_{\text{test}}, \mu_{\hat{y}}) < d(f_{\max}, \mu_{\hat{y}})$), ensuring that the cache always retains the most reliable samples near the prototype center.

3. **Negative Cache**:
   Leverages negative pseudo-labels from high-entropy samples to calibrate predictions. A **reflecting mechanism** is introduced—pseudo-labels of high-entropy samples are re-evaluated using reliable features from existing cache entries, and only those that maintain moderate entropy after calibration are admitted:
   - $H_{\text{low}} \le H'(x) \le H_{\text{high}}$: stored in the negative cache
   - $H'(x) < H_{\text{low}}$: treated as reliable and eligible for the entropy cache
   - $H'(x) > H_{\text{high}}$: discarded

4. **Inference Mechanism**:
   The final prediction logits are computed as a weighted combination of three complementary information sources:
   $p(f_{\text{test}}) = \alpha_1 \cdot f_{\text{test}} \mathcal{T}^\top + \alpha_2 \cdot P(f_{\text{test}}, \mathcal{V}, Q_n) + \alpha_3 \cdot f_{\text{test}} f_r^\top$
   where $\mathcal{T}$ denotes text prototype matching, $P(\cdot)$ represents contrastive information from visual prototypes and the negative cache, and $f_r$ is an attention-weighted adaptive cache feature.

5. **MCP++ Residual Learning**:
   Learnable residual parameters are introduced to fine-tune both visual and textual prototypes:
   $\bar{t}_c' = \bar{t}_c + R_t^c, \quad \bar{v}_c' = \bar{v}_c + R_v^c$
   The joint optimization loss comprises three terms:
   - **Entropy minimization loss** $\mathcal{L}_{\text{entro}}$: encourages prediction consistency across augmented views
   - **Visual-text alignment loss** $\mathcal{L}_{\text{align}}$: aligns visual and textual prototypes via InfoNCE
   - **Positive-negative contrastive loss** $\mathcal{L}_{\text{contrast}}$: enlarges the distance between prototype centers and negative cache samples

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{entro}} + \lambda \cdot \mathcal{L}_{\text{align}} + \gamma \cdot \mathcal{L}_{\text{contrast}}$$

where $\lambda = 0.5$ and $\gamma = 0.2$. Each test sample generates 31 augmented views (32 total). A single-step update is performed using the AdamW optimizer with a learning rate of 0.0001. Residual parameters are initialized to zero.

## Key Experimental Results

### Main Results (Cross-Dataset Generalization, ViT-B/16)

| Method | Aircraft | Caltech | Cars | DTD | EuroSAT | Flower | Food | Pets | SUN397 | UCF101 | Avg. |
|--------|----------|---------|------|-----|---------|--------|------|------|--------|--------|------|
| CLIP (zero-shot) | 23.67 | 93.35 | 65.48 | 44.27 | 42.01 | 67.44 | 83.65 | 88.25 | 62.59 | 65.13 | 63.58 |
| TPT | 24.78 | 94.16 | 66.87 | 47.75 | 42.44 | 68.98 | 84.67 | 87.79 | 65.50 | 68.04 | 65.10 |
| TDA | 23.91 | 94.24 | 67.28 | 47.40 | 58.00 | 71.42 | 86.14 | 88.63 | 67.62 | 70.66 | 67.53 |
| DMN | 30.03 | 95.38 | 67.96 | 55.85 | 59.43 | 74.49 | 85.08 | 92.04 | 70.18 | 72.51 | 70.30 |
| DPE | 28.95 | 94.81 | 67.31 | 54.20 | 55.79 | 75.07 | 86.17 | 91.14 | 70.07 | 70.44 | 69.40 |
| **MCP** | **30.18** | **95.33** | **69.99** | **55.91** | **68.42** | 75.88 | 86.85 | 91.88 | 71.04 | 74.36 | 71.98 |
| **MCP++** | **31.06** | **95.50** | **70.13** | **56.97** | **68.69** | **77.55** | **87.20** | **92.40** | **71.17** | **75.44** | **72.61** |

### Ablation Study (Contribution of Each Cache Component on ImageNet)

| Entropy | Align | Negative | ImageNet Acc. |
|---------|-------|----------|---------------|
| ✓ | ✓ | - | 72.49 |
| ✓ | - | ✓ | 71.73 |
| - | ✓ | ✓ | 72.54 |
| ✓ | - | - | 71.51 |
| - | ✓ | - | 72.47 |
| - | - | ✓ | 71.71 |
| ✓ | ✓ | ✓ | **72.64** |

### Key Findings

- MCP++ achieves the best performance on 9 out of 10 cross-domain datasets, with an average accuracy of 72.61%—5.08% above TDA.
- On OOD benchmarks (ImageNet and 4 variants), MCP++ achieves an average of 66.86%, surpassing DPE by 0.93%.
- Each cache module individually outperforms the CLIP baseline by a significant margin; their combination yields the best results, demonstrating complementarity.
- $w = 0.8$ (visual weight) is the optimal modality fusion ratio, indicating that visual information is more informative than textual information.
- Cache size requires careful balancing: too small limits the number of high-confidence samples, while too large introduces uncertain ones.

## Highlights & Insights

- **Finding-driven design**: All design choices are motivated by the empirical observation that cache performance positively correlates with intra-class compactness.
- **Complementary three-cache architecture**: The entropy cache provides stable anchors, the align cache enhances compactness, and the negative cache calibrates decision boundaries—each serving a distinct role.
- **Minimally invasive design**: MCP is entirely training-free; MCP++ optimizes only residual parameters with a single gradient update, making it extremely lightweight.
- **Multi-source inference fusion**: Three-way logit combination of text semantic matching, visual prototype contrast, and cache feature retrieval.

## Limitations & Future Work

- The prediction weights $\alpha_1, \alpha_2, \alpha_3$ require per-task tuning and are not automatically determined.
- The fixed cache size (10 entries per class) may not scale well to scenarios with a large number of classes (e.g., ImageNet with 1,000 classes requires maintaining 30,000 cache entries).
- Validation is limited to classification tasks; extension to detection, segmentation, and other vision tasks remains unexplored.
- The residual learning in MCP++ requires augmented view generation and gradient updates, incurring additional computational cost compared to purely cache-based methods.

## Related Work & Insights

- Builds upon TDA's low-entropy caching idea, extending it in two directions via the align cache and the negative cache.
- Similar to DPE in introducing multimodal prototype residual tuning, but MCP++'s three-cache design is more systematic.
- Empirically validates that intra-class compactness of test data is a critical factor in the success of cache-based TTA methods, providing a clear direction for future work.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The three-cache architecture and compactness analysis are innovative, though the overall contribution is a systematic combination of existing ideas.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 15 datasets with two backbone networks and detailed ablations covering cache components, retrieval strategies, cache sizes, and modality weights.
- **Writing Quality**: ⭐⭐⭐⭐ Well-organized with rigorous formulations; some notation definitions could be more concise.
- **Value**: ⭐⭐⭐⭐ Represents a clear advancement in cache-based TTA methods, though the application scope is currently limited to classification.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[ICCV 2025\] Dynamic Multimodal Prototype Learning in Vision-Language Models](dynamic_multimodal_prototype_learning_in_vision-language_models.md)
- [\[ICCV 2025\] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation](is_less_more_exploring_token_condensation_as_training-free_test-time_adaptation.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](../../NeurIPS2025/multimodal_vlm/test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)

<!-- RELATED:END -->

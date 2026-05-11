---
title: >-
  [Paper Note] BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models
description: >-
  [AAAI 2026][Multimodal VLM][Vision-language model debiasing] This paper proposes BiPrompt, a bilateral prompt optimization framework that simultaneously mitigates spurious biases on both the visual side (structured atten…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Vision-language model debiasing"
  - "test-time adaptation"
  - "causal inference"
  - "prompt optimization"
  - "spurious correlations"
date: 2026-05-08
content_hash: 54b099c55acd23d5
---

# BiPrompt: Bilateral Prompt Optimization for Visual and Textual Debiasing in Vision-Language Models

**Conference**: AAAI 2026
**arXiv**: [2601.02147](https://arxiv.org/abs/2601.02147)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Vision-language model debiasing, test-time adaptation, causal inference, prompt optimization, spurious correlations

## TL;DR

This paper proposes BiPrompt, a bilateral prompt optimization framework that simultaneously mitigates spurious biases on both the visual side (structured attention erasure) and the textual side (balanced prompt normalization) in VLMs such as CLIP at test time, improving OOD robustness without retraining.

## Background & Motivation

Despite their strong zero-shot generalization capabilities, vision-language foundation models such as CLIP heavily rely on spurious correlations (e.g., background textures, co-occurring objects) in OOD settings, leading to degraded reliability. For instance, a model may misclassify a spider as a crab due to a beach background.

Limitations of prior debiasing methods:
- **Region-aware methods** (e.g., Alpha-CLIP): Require architectural modifications or costly fine-tuning, making them unsuitable for plug-and-play test-time adaptation.
- **Test-time prompt tuning** (e.g., TPT): Assumes spurious features produce low-confidence predictions, whereas in practice strong spurious features (e.g., water backgrounds) can lead to high-confidence erroneous predictions.
- **SEraser**: The first method to focus on spurious feature erasure, but suffers from two core issues: (1) random visual erasure may inadvertently remove causal features; (2) it addresses only visual bias, neglecting linguistic bias in text prompts.

The core motivation of BiPrompt is to **simultaneously debias both visual and textual modalities** for more robust causal inference.

## Method

### Overall Architecture

BiPrompt introduces two complementary modules on top of SEraser:
1. **Balanced Prompt Normalization**: Mitigates anisotropic bias in the text embedding space.
2. **Structured Spurious-Region Erasure**: Replaces random erasure with attention-guided masking.

Both modules jointly minimize the conditional mutual information between spurious features and predictions: $I(z_s; y | z_c) \approx 0$.

### Key Designs

#### Balanced Prompt Normalization

Standard prompt embeddings $f_t(t_c)$ exhibit anisotropy in the text space, biasing toward high-frequency or dominant classes. BiPrompt learns normalized text embeddings as:

$$\hat{f}_t(t_c) = \alpha f_t(t_c) + (1-\alpha) \bar{f}_t$$

where $\bar{f}_t = \frac{1}{C}\sum_{c=1}^C f_t(t_c)$ is the global semantic centroid and $\alpha$ is a learnable gating parameter. This adaptive interpolation encourages an isotropic text embedding distribution and reduces linguistic dominance.

**Core Idea**: Each class's text embedding is pulled toward the global mean, making the embedding space more uniform and preventing certain classes from gaining an unfair advantage due to dominant textual representations.

#### Structured Spurious-Region Erasure

Grad-CAM is used to compute a soft attention map $m(x)$ identifying causal regions, from which complementary foreground and background views are constructed:

$$x_{\text{fg}} = m(x) \odot x, \quad x_{\text{bg}} = (1 - m(x)) \odot x$$

A bidirectional constraint loss is designed as:

$$\mathcal{L}_{\text{BSE}} = D_{\text{KL}}(p(y|x_{\text{fg}}) \| p(y|x)) - \beta \cos(p(y|x_{\text{bg}}), p(y|x))$$

- **First term**: Enforces consistency between foreground-view and full-image predictions (KL divergence minimization) — ensuring causal features are preserved.
- **Second term**: Enforces orthogonality between background-view and full-image predictions (cosine similarity minimization) — suppressing spurious features.

Advantages over SEraser's random erasure:
- Grad-CAM precisely localizes causal/spurious regions, avoiding accidental removal of causal features.
- Simultaneous enforcement of foreground consistency and background orthogonality provides more robust bidirectional constraints.

### Loss & Training

The overall test-time objective is:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}} + \lambda_1 \mathcal{L}_{\text{BSE}} + \lambda_2 \mathcal{L}_{\text{ent}}$$

where $\mathcal{L}_{\text{ent}}$ is an entropy regularization term that prevents degenerate uniform predictions.

**Test-time inference pipeline** (4 steps):
1. Compute the Grad-CAM attention map to obtain foreground/background views.
2. Extract visual features $f_v(x)$, $f_v(x_{\text{fg}})$, $f_v(x_{\text{bg}})$ and normalized text embeddings $\hat{f}_t(t_c)$.
3. Perform a small number of gradient updates to minimize $\mathcal{L}_{\text{total}}$.
4. Compute the final prediction using similarity $s(f_v(x), \hat{f}_t(t_c))$.

Only lightweight parameters such as the gating scalar $\alpha$ and normalization weights are updated, making optimization efficient and memory-friendly.

## Key Experimental Results

### Main Results

**Table 1: Zero-shot classification on real-world OOD datasets (Top-1 Accuracy %)**

| Dataset | Vanilla | TPT | RoSHOT | α-CLIP | SEraser-Blocks | **BiPrompt** |
|---|---|---|---|---|---|---|
| Tiny-ImageNet | 23.2 | 29.6 | 49.2 | **76.0** | 42.8 | 44.1 (+20.9) |
| CUB-200 | 12.1 | 8.7 | 25.5 | **44.3** | 28.9 | 31.0 (+18.9) |
| ImageNet-A | 42.1 | 49.7 | 38.9 | 51.5 | 49.7 | **52.2** (+10.1) |
| Average | 25.8 | 29.3 | 37.9 | **57.3** | 40.5 | 42.4 (+16.6) |

**Table 2: Simulated spurious-bias datasets, average / worst-group accuracy (%)**

| Dataset | Metric | Vanilla | SEraser | **BiPrompt** |
|---|---|---|---|---|
| Waterbirds | AVG | 67.7 | 78.2 | **79.9** (+12.2) |
| | W.G. | 40.0 | 65.3 | **66.6** (+26.5) |
| CamelDeer | AVG | 83.2 | 95.7 | **97.2** (+14.0) |
| | W.G. | 66.4 | 91.6 | **92.8** (+26.4) |
| SpiderCrab | AVG | 66.0 | 95.3 | **97.4** (+31.4) |
| | W.G. | 42.0 | 94.7 | **95.4** (+53.4) |
| Three-dataset Avg. | AVG | 72.3 | 89.8 | **91.3** (+19.0) |
| | W.G. | 49.5 | 83.7 | **85.0** (+35.5) |

### Ablation Study

**Cross-architecture generalization (Table 3 — Waterbirds)**:
- CLIP-L14: BiPrompt AVG 88.4 (+4.7), W.G. 60.1 (+27.2)
- BLIP-2: BiPrompt W.G. 35.5 (+7.3), demonstrating effectiveness across different VLM architectures.

### Key Findings

1. **Substantial worst-group accuracy gains**: W.G. improves by 53.4% on SpiderCrab, indicating that bilateral debiasing is most effective in scenarios with the most severe spurious correlations.
2. **BiPrompt consistently outperforms SEraser across all spurious-bias datasets**, validating the necessity of jointly addressing visual and textual biases.
3. **Alpha-CLIP's superiority on Tiny-ImageNet** is attributable to its checkpoint being retrained on ImageNet, constituting an unfair comparison.
4. TPT performs even worse in spurious-bias scenarios, confirming that the assumption of high-confidence ≠ correct prediction does not hold.

## Highlights & Insights

- The **bilateral debiasing** concept is concise and effective: anisotropic bias in the text embedding space is a long-overlooked problem, and BiPrompt is the first to address both modalities simultaneously at test time.
- **Structured erasure** replaces random masking with Grad-CAM guidance, shifting the paradigm from "uncertainty-driven erasure" to "precision erasure" — a conceptually clear improvement.
- **Extremely lightweight test-time adaptation**: only gating parameters and normalization weights require updating, with no retraining or additional domain supervision needed.
- The approach is theoretically grounded in minimizing the conditional mutual information between spurious features and predictions, providing information-theoretic support.

## Limitations & Future Work

1. **Performance is upper-bounded by Grad-CAM quality**: if the attention map is inaccurate, structured erasure may fail.
2. **A notable gap remains compared to retraining-based methods (α-CLIP)**: the gap exceeds 30% on Tiny-ImageNet, indicating inherent limitations of test-time adaptation.
3. **Validation on larger-scale VLMs (e.g., ViT-L/14, EVA-CLIP) is lacking.**
4. **Insufficient analysis of the gating parameter $\alpha$**: how $\alpha$ varies with the number of classes and data distribution merits further investigation.
5. **Potential extension to debiasing in multimodal large language models** (e.g., LLaVA) remains unexplored.

## Related Work & Insights

- **SEraser**: The direct predecessor of BiPrompt, which introduced the core idea of "maximizing entropy over spurious features."
- **TPT (Test-time Prompt Tuning)**: Adapts by minimizing marginal entropy across augmented views, but is not robust to spurious features.
- **RoSHOT**: Debiases via orthogonal projection of biased prompts, but yields inconsistent results.
- **Alpha-CLIP**: Introduces an additional alpha channel for pixel-level foreground/background information; achieves strong performance but requires retraining.
- Insight: The "bilateral" paradigm of test-time adaptation can be generalized to other multimodal tasks (e.g., VQA debiasing, cross-domain retrieval).

## Rating

| Dimension | Score (1–5) |
|---|---|
| Novelty | 3.5 |
| Technical Depth | 3.5 |
| Experimental Thoroughness | 4.0 |
| Writing Quality | 4.0 |
| Value | 3.5 |
| **Overall** | **3.7** |

- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Branch, or Layer? Zeroth-Order Optimization for Continual Learning of Vision-Language Models](branch_or_layer_zeroth-order_optimization_for_continual_lear.md)
- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](../../CVPR2026/multimodal_vlm/interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [\[ICCV 2025\] Advancing Textual Prompt Learning with Anchored Attributes](../../ICCV2025/multimodal_vlm/advancing_textual_prompt_learning_with_anchored_attributes.md)
- [\[CVPR 2026\] Evolving Prompt Adaptation for Vision-Language Models](../../CVPR2026/multimodal_vlm/evolving_prompt_adaptation_for_vision-language_models.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](../../CVPR2026/multimodal_vlm/towards_calibrating_prompt_tuning_of_vision-language_models.md)

</div>

<!-- RELATED:END -->

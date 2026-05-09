---
title: >-
  [Paper Note] Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks
description: >-
  [CVPR2026][Multimodal VLM][Model inversion attack] This paper presents the first systematic study of model inversion (MI) attacks against VLMs. It proposes SMI-AW, a sequence-level inversion method based on adaptive token attention weighting, which dynamically weights token gradients according to their visual relevance to reconstruct private training images from VLMs. The method achieves a human-evaluated attack accuracy of 61.21%.
tags:
  - CVPR2026
  - Multimodal VLM
  - Model inversion attack
  - vision-language models
  - privacy security
  - adaptive token weighting
  - training data leakage
date: 2026-05-08
content_hash: 3254a747dea4def8
---

# Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks

**Conference**: CVPR2026
**arXiv**: [2508.04097](https://arxiv.org/abs/2508.04097)
**Code**: [Project Page](https://ngoc-nguyen-0.github.io/SMI_AW/)
**Area**: Multimodal VLM
**Keywords**: Model inversion attack, vision-language models, privacy security, adaptive token weighting, training data leakage

## TL;DR

This paper presents the first systematic study of model inversion (MI) attacks against VLMs. It proposes SMI-AW, a sequence-level inversion method based on adaptive token attention weighting, which dynamically weights token gradients according to their visual relevance to reconstruct private training images from VLMs. The method achieves a human-evaluated attack accuracy of 61.21%.

## Background & Motivation

1. **Privacy risks of model inversion attacks**: MI attacks reconstruct private training data from trained model parameters and have been extensively studied on unimodal DNNs (e.g., classifiers), yet the security of multimodal VLMs remains unexplored.
2. **Security concerns from large-scale VLM deployment**: VLMs such as LLaVA, Qwen2.5-VL, and InternVL are widely deployed in sensitive domains including healthcare and finance, making potential training data leakage a serious concern.
3. **Fundamental differences between VLMs and DNNs**: VLMs process multimodal inputs, comprise multiple independent modules (visual encoder, projector, language model), and produce token sequences rather than class labels, making direct application of traditional MI methods infeasible.
4. **Frozen visual encoders**: Many VLMs freeze the visual encoder during fine-tuning and update only the language model and projector, meaning private information is embedded in language model parameters and must be extracted indirectly.
5. **Heterogeneity of token sequences**: Different output tokens of a VLM vary substantially in their dependence on visual input — some exhibit strong visual correlation while others are driven purely by linguistic context — introducing noise when treated uniformly.
6. **Risks from publicly released models**: Publicly available pretrained VLMs (e.g., LLaVA-v1.6-7B) are also vulnerable to MI attacks that can reconstruct sensitive information such as faces from their training data.

## Method

### Overall Architecture

The attacker has white-box access (model parameters, attention maps, logits) and optimizes a generator $G$ (e.g., StyleGAN2) trained on public datasets in a low-dimensional latent space $w$, reconstructing private images via $x = G(w)$. The objective is to find $w^*$ such that the VLM's output for the reconstructed image $G(w^*)$ closely matches the target text answer $\mathbf{y}$.

### Four Inversion Strategies

1. **TMI (Token-based MI)**: Updates the latent vector $w$ token by token; each token $y_i$ independently computes a gradient and triggers one update step.
2. **TMI-C (Convergent TMI)**: Performs $K$ convergence iterations per token $y_i$ before proceeding to the next, better aligning generative dependencies across tokens.
3. **SMI (Sequence-based MI)**: Averages the loss over all tokens in the sequence and performs a single gradient update that aggregates full-sequence information, providing a more stable and consistent optimization direction:
$$\mathcal{L} = \frac{1}{m}\sum_{i=1}^{m}\mathcal{L}_{inv}(M(\mathbf{t}, G(w), y_{<i}), y_i)$$
4. **SMI-AW (proposed method)**: Extends SMI with adaptive token weighting.

### SMI-AW Adaptive Weighting Design

- **Core observation**: Cross-attention maps differ substantially across output tokens — tokens with strong visual correlation yield high attention responses and carry richer visual information in their gradients.
- **Weight computation**: A normalized weight $\beta_i$ is computed from the total visual attention value $\alpha_i$ of token $y_i$ over the image:
$$\beta_i = \frac{\alpha_i}{\sum_{j=1}^{m}\alpha_j}$$
- **Dynamic update**: $\beta_i$ is recomputed at each inversion step, since token visual dependence shifts as the reconstructed image progressively approaches the target.
- **Weighted loss**:
$$\mathcal{L} = \sum_{i=1}^{m}\beta_i \mathcal{L}_{inv}(M(\mathbf{t}, G(w), y_{<i}), y_i)$$

### Inversion Loss Functions

Three classical unimodal MI losses are extended to the VLM setting:

- $\mathcal{L}_{CE}$: Cross-entropy loss
- $\mathcal{L}_{MML}$: Maximum margin loss
- $\mathcal{L}_{LOM}$: Logit optimization/maximization loss (best performing)

## Key Experimental Results

### Experimental Setup

- **Target VLMs**: LLaVA-v1.6-7B, Qwen2.5-VL-7B, MiniGPT-v2, InternVL2.5-8B
- **Datasets**: FaceScrub (530 classes), CelebA (1,000 classes), StanfordDogs (120 classes), formatted as VQA tasks
- **Generators**: StyleGAN2 trained on FFHQ and AFHQ-Dogs
- **Metrics**: $AttAcc_M$ (MLLM-based evaluation), $AttAcc_D$ Top-1/Top-5 (DNN-based evaluation), $\delta_{face}$, $\delta_{eval}$, $AttAcc_H$ (human evaluation)

### Main Results (FaceScrub + LLaVA-v1.6-7B + $\mathcal{L}_{LOM}$)

| Method | $AttAcc_M$↑ | Top1↑ | Top5↑ | $\delta_{face}$↓ | $\delta_{eval}$↓ |
|--------|------------|-------|-------|------------------|------------------|
| TMI | 44.34% | 21.77% | 44.69% | 0.8488 | 141.87 |
| TMI-C | 31.16% | 9.32% | 24.22% | 1.0221 | 457.49 |
| SMI | 59.17% | 33.47% | 61.89% | 0.7465 | 140.83 |
| **SMI-AW** | **61.01%** | **37.62%** | **66.16%** | **0.7265** | **134.94** |

### Cross-Dataset Results (SMI-AW + $\mathcal{L}_{LOM}$)

| Dataset | $AttAcc_M$↑ | Top1↑ | Top5↑ |
|---------|------------|-------|-------|
| CelebA | 67.05% | 45.25% | 69.55% |
| StanfordDogs | 78.13% | 56.15% | 84.79% |

### Cross-Model Generalization (FaceScrub + SMI-AW + $\mathcal{L}_{LOM}$)

| Model | $AttAcc_M$↑ | Top1↑ | Top5↑ |
|-------|------------|-------|-------|
| InternVL2.5-8B | 55.05% | 25.05% | 52.10% |
| MiniGPT-v2 | 47.92% | 14.62% | 33.82% |
| Qwen2.5-VL-7B | 32.03% | 13.21% | 27.24% |

### Human Evaluation ($AttAcc_H$)

Human attack accuracy across different VLMs on FaceScrub ranges from 53.42% to 57.22%, reaching **61.21%** on CelebA and 55.42% on StanfordDogs.

### Ablation Study: Token-based vs. Sequence-based

- Sequence-level methods (SMI/SMI-AW) achieve match rates above 95%, far exceeding token-level methods (TMI: 60–79%; TMI-C: below 30%).
- Token-level methods suffer from high gradient variance and instability; gradients from weakly visual-correlated tokens mislead the optimization direction.
- SMI-AW's adaptive weights are dynamically updated at each step, amplifying contributions from visually correlated tokens while suppressing noise from language-driven tokens.

## Highlights & Insights

- **Pioneering contribution**: The first systematic study of MI attacks on VLMs, addressing a critical gap in multimodal privacy security.
- **Elegant method design**: Cross-attention maps serve as natural proxy indicators of token visual relevance, making the adaptive weighting scheme both principled and effective.
- **Comprehensive evaluation**: 4 VLMs × 3 datasets × 5 metrics, including large-scale human evaluation and validation on publicly released models.
- **Real-world threat validation**: Successful reconstruction of celebrity faces from the publicly released LLaVA-v1.6-7B confirms that the privacy risk is genuine.
- **Mechanistic analysis of token- vs. sequence-level methods**: The paper not only proposes the method but also provides a mechanistic explanation for why token-level approaches fail.

## Limitations & Future Work

- **Strong white-box assumption**: Full access to model parameters and attention maps is required, whereas API-based (black-box) access is more representative of real-world deployment scenarios.
- **Generator dependency**: Training StyleGAN in the same domain as the private data requires domain-aligned public data, limiting applicability to novel or data-scarce domains.
- **VQA-specific setting**: Experiments are conducted on VQA-formatted data; more complex VLM interaction modes such as conversational or instruction-following settings remain unvalidated.
- **Absence of defenses**: The paper identifies the risk but does not propose countermeasures such as differential privacy or gradient clipping.
- **Computational overhead**: Each step requires a full forward pass and attention map extraction over the complete sequence, raising scalability concerns for long sequences.

## Related Work & Insights

- **Traditional MI methods (PPA, PLG-MI, etc.)**: Target unimodal classifiers and optimize the likelihood of a single class label; this work targets VLMs with token-sequence outputs, necessitating an entirely new design.
- **VLM security research**: Existing work covers jailbreak attacks and adversarial examples, but model inversion for training data reconstruction in VLMs is studied systematically here for the first time.
- **Attention mechanism analysis**: Prior work observes that attention to image tokens decays in later layers of VLMs; this paper leverages that observation to guide gradient weighting in MI attacks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First extension of MI attacks to VLMs; both problem formulation and method design are pioneering contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 models × 3 datasets × multiple metrics × human evaluation × validation on public models
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with a clear progressive narrative from TMI→TMI-C→SMI→SMI-AW, though notation-heavy
- Value: ⭐⭐⭐⭐⭐ — Reveals serious privacy risks in VLMs with significant implications for the security community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models](scene-vlm_multimodal_video_scene_segmentation_via_vision-language_models.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](aif_adaptive_information_flow_vlm.md)
- [\[CVPR 2026\] Residual Decoding: Mitigating Hallucinations in Large Vision-Language Models via History-Aware Residual Guidance](residual_decoding_mitigating_hallucinations_in_large_vision-language_models_via_.md)
- [\[CVPR 2026\] Do Vision Language Models Need to Process Image Tokens?](do_vision_language_models_need_to_process_image_tokens.md)
- [\[CVPR 2026\] Continual Learning with Vision-Language Models via Semantic-Geometry Preservation](continual_learning_with_vision-language_models_via_semantic-geometry_preservatio.md)

</div>

<!-- RELATED:END -->

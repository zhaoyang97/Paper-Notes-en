---
title: >-
  [Paper Note] Approximate Domain Unlearning for Vision-Language Models
description: >-
  [NeurIPS 2025 Spotlight][LLM Safety][approximate unlearning] This paper introduces Approximate Domain Unlearning (ADU), a novel task that enables pretrained VLMs to selectively forget recognition capabilities for specified domains (e.g., illustrations, sketches) while preserving classification accuracy on other domains (e.g., real photographs). Two modules are proposed — Domain Disentangling Loss (DDL) and Instance-wise Prompt Generator (InstaPG) — achieving substantial impro…
tags:
  - "NeurIPS 2025 Spotlight"
  - "LLM Safety"
  - "approximate unlearning"
  - "domain unlearning"
  - "vision-language model"
  - "CLIP"
  - "prompt tuning"
date: 2026-05-08
content_hash: 33d6f58986f24b5c
---

# Approximate Domain Unlearning for Vision-Language Models

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.08132](https://arxiv.org/abs/2510.08132)  
**Code**: [https://kodaikawamura.github.io/Domain_Unlearning/](https://kodaikawamura.github.io/Domain_Unlearning/)  
**Area**: Multimodal VLM / Machine Unlearning
**Keywords**: approximate unlearning, domain unlearning, vision-language model, CLIP, prompt tuning

## TL;DR

This paper introduces Approximate Domain Unlearning (ADU), a novel task that enables pretrained VLMs to selectively forget recognition capabilities for specified domains (e.g., illustrations, sketches) while preserving classification accuracy on other domains (e.g., real photographs). Two modules are proposed — Domain Disentangling Loss (DDL) and Instance-wise Prompt Generator (InstaPG) — achieving substantial improvements over all baselines across four multi-domain datasets.

## Background & Motivation

**Background**: Pretrained VLMs such as CLIP exhibit strong domain generalization, enabling cross-domain object recognition. However, in certain downstream tasks, such comprehensive generalization is unnecessary and may introduce security risks and information leakage.

**Limitations of Prior Work**: Existing approximate unlearning methods focus on class-level unlearning, which is insufficient for many practical scenarios. For instance, an autonomous driving system must recognize real vehicles but should not misclassify illustrated cars on roadside advertisements as real ones.

**Key Challenge**: The strong domain generalization of VLMs causes feature distributions across domains to become highly entangled in the latent space. Directly applying class-level unlearning strategies — maximizing entropy on the forget domain and minimizing cross-entropy on the memorize domain — fails to disentangle domain-specific features, leading to mutual interference between forgetting and retention.

**Goal**: To enable VLMs to perform fine-grained selective unlearning at the domain level (rather than the class level), reducing recognition accuracy on specified domains while preserving performance on others.

**Key Insight**: Since domain entanglement is the core obstacle, the proposed approach first disentangles domain distributions in the latent space before applying domain-specific forgetting and retention strategies.

**Core Idea**: Precise domain-level unlearning in VLMs is achieved through domain distribution disentanglement combined with instance-wise adaptive prompt generation.

## Method

### Overall Architecture

The overall pipeline is built upon CLIP's vision prompt tuning framework. Learnable vision prompt tokens are inserted into the first 9 layers of the ViT image encoder, and the model is jointly optimized with three loss components: (1) minimizing classification cross-entropy on memorize-domain samples, (2) maximizing classification entropy on forget-domain samples, and (3) DDL enforcing domain distribution separation. InstaPG is embedded in intermediate Transformer layers to dynamically generate instance-level prompts conditioned on the input image. On the text side, the template "a photo of a [class]" is used without further adaptation.

### Key Designs

1. **Approximate Domain Unlearning (ADU) Problem Formulation**:

    - Function: Given a training set $\{(\mathbf{x}, y, d)\}$ with domain label $d \in \mathcal{D}$, define $\mathcal{D}_{\text{memorize}}$ as the domains to retain and $\mathcal{D}_{\text{forget}} = \mathcal{D} \setminus \mathcal{D}_{\text{memorize}}$ as the domains to forget.
    - Mechanism: Cross-entropy $\mathcal{L}_{\text{memorize}}$ is minimized on memorize-domain samples, while cross-entropy to a uniform distribution $\mathcal{L}_{\text{forget}}$ (equivalent to entropy maximization) is minimized on forget-domain samples.
    - Design Motivation: Directly applying the two losses from class-level unlearning to domain unlearning constitutes the most straightforward baseline, but its effectiveness is limited due to severe domain entanglement.

2. **Domain Disentangling Loss (DDL)**:

    - Function: Explicitly separates feature distributions across domains in the latent space.
    - Mechanism: Composed of two complementary loss terms. The first is the cross-entropy loss $\mathcal{L}_{\text{CE}}$ of an auxiliary domain classifier, requiring the model to correctly predict domain labels. The second is Maximum Mean Discrepancy (MMD), which maximizes inter-domain distance in a reproducing kernel Hilbert space:
    $$\mathcal{L}_{\text{domain}} = \gamma \mathcal{L}_{\text{CE}} - \lambda \text{MMD}^2$$
      The negative sign before MMD indicates maximization of inter-domain distance — the opposite of conventional domain adaptation, which minimizes MMD. Default hyperparameters are $\gamma=30$ and $\lambda=10$.
    - Design Motivation: When domain feature distributions are well separated, applying a forgetting loss to one domain will not affect others. The CE term ensures discriminative domain separability, while the MMD term enforces distributional separation; the two are complementary.

3. **Instance-wise Prompt Generator (InstaPG)**:

    - Function: Dynamically generates personalized vision prompts conditioned on the patch features of each input image.
    - Mechanism: Embedded within intermediate Transformer blocks of the ViT, InstaPG employs a cross-attention mechanism in which learnable vision prompts serve as queries and image patch features serve as keys and values, producing instance-level prompts for subsequent layers.
    - Design Motivation: "Domain" is an inherently ambiguous concept — the "illustration" style, for example, spans a wide spectrum from near-realistic to cartoonish. A uniform prompt cannot capture such instance-level domain variation; InstaPG uses the attention mechanism to adapt prompts to the characteristics of each individual image.

### Loss & Training

The total loss consists of three components:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{memorize}} + \mathcal{L}_{\text{forget}} + \mathcal{L}_{\text{domain}}$$

- $\mathcal{L}_{\text{memorize}}$: Standard classification cross-entropy on memorize-domain samples.
- $\mathcal{L}_{\text{forget}}$: Cross-entropy to a uniform distribution on forget-domain samples (maximizing prediction entropy).
- $\mathcal{L}_{\text{domain}} = \gamma \mathcal{L}_{\text{CE}} - \lambda \text{MMD}^2$: DDL domain disentanglement loss.

Training details: ViT-B/16 is used as the image encoder with a deep prompting strategy, 8 context tokens, SGD optimizer, learning rate 0.0025, and 50 training epochs. Only 8 labeled samples per domain are used (few-shot setting).

## Key Experimental Results

### Main Results

Comparison with LP++, CLIPFit, BBF, and Baseline on ImageNet, Office-Home, and Mini DomainNet ($|\mathcal{D}_{\text{forget}}|=1$):

| Method | ImageNet H↑ | Office-Home H↑ | Mini DomainNet H↑ |
|--------|-------------|----------------|-------------------|
| LP++ | 50.69 | 30.46 | 31.73 |
| CLIPFit | 71.31 | 43.44 | 53.56 |
| BBF | 45.56 | 31.25 | 32.12 |
| Baseline | 74.66 | 52.59 | 62.07 |
| **Ours** | **77.02** | **69.96** | **75.56** |

On Office-Home, the proposed method surpasses the strongest baseline by **+17.37** in H; on Mini DomainNet, by **+13.49**. The advantage becomes even more pronounced when the number of forget domains increases to 3 (Office-Home: H=75.89 vs. Baseline 59.47).

### Ablation Study

Ablation of DDL and InstaPG (Office-Home, $|\mathcal{D}_{\text{forget}}|=1$):

| Configuration | H↑ | Mem↑ | For↑ | Notes |
|---------------|----|------|------|-------|
| w/o DDL, w/o InstaPG (Baseline) | 52.59 | 79.96 | 39.88 | Naive unlearning strategy |
| InstaPG only | 56.41 | 83.55 | 44.05 | InstaPG alone: +3.82 |
| DDL only | 60.82 | 74.51 | 51.72 | DDL alone: +8.23 |
| DDL + InstaPG (full method) | **69.96** | 77.93 | **64.34** | Best with both components |

Ablation of CE and MMD within DDL (Office-Home, $|\mathcal{D}_{\text{forget}}|=1$):

| CE | MMD | H↑ | Mem↑ | For↑ |
|----|-----|----|------|------|
| ✗ | ✗ | 56.41 | 83.55 | 44.05 |
| ✗ | ✓ | 68.62 | 82.41 | 59.47 |
| ✓ | ✗ | 64.01 | 82.97 | 53.62 |
| ✓ | ✓ | **69.96** | 77.93 | **64.34** |

### Key Findings

- DDL contributes more than InstaPG individually, but their combination substantially outperforms either component alone, confirming that domain disentanglement and instance-level adaptation are complementary.
- Domain classification accuracy improves from 25.80% (zero-shot CLIP) to 79.43% with the proposed method, validating that DDL effectively separates domain distributions.
- Performance stabilizes once hyperparameters $\gamma$ and $\lambda$ exceed certain thresholds, indicating robustness to hyperparameter selection.
- As the number of training samples increases, the proposed method continues to improve, whereas the Baseline exhibits overfitting tendencies on Mini DomainNet.
- t-SNE visualizations clearly demonstrate the transition of domain distributions from entangled to separated before and after applying the method.
- Attention maps reveal that on forget domains, model attention disperses away from target objects, while on memorize domains, attention is maintained or even enhanced.

## Highlights & Insights

- **First formulation of ADU**: Extends approximate unlearning from the class level to the domain level, opening a new research direction with clear practical motivations (e.g., autonomous driving safety).
- **Inverted use of MMD**: While conventional domain adaptation minimizes MMD to align distributions, this work maximizes MMD to separate them — a simple yet effective inversion of the standard paradigm.
- **Effective under few-shot settings**: High-quality domain unlearning is achievable with only 8 samples per domain, demonstrating strong practical utility.
- **Cross-attention design of InstaPG**: Prompts serve as queries while image patches serve as keys/values, allowing prompts to capture the domain characteristics of each individual image.
- **Interpretability via attention maps**: Attention heatmaps intuitively illustrate the unlearning effect — attention is "dispersed" on forget domains and "preserved or enhanced" on memorize domains.

## Limitations & Future Work

- **Requires domain labels**: The method assumes all training samples have domain labels, whereas domain annotations are often incomplete in practice. The appendix presents a preliminary exploration using pseudo-labels, but more robust domain estimation approaches warrant further investigation.
- **Domain definitions rely on prior knowledge**: The notion of "domain" is manually defined; inappropriate domain partitioning may degrade performance, and automatic domain discovery settings remain unexplored.
- **Evaluated only on CLIP**: All experiments are based on CLIP ViT-B/16; other VLMs such as BLIP-2 or LLaVA have not been tested.
- **Limited to image classification**: The ADU formulation is restricted to classification tasks; whether it can be extended to domain unlearning in generative VLMs (e.g., text-to-image models) remains an open and interesting question.
- **Lack of privacy-oriented evaluation**: Current evaluation focuses on classification performance; direct measurement of information leakage prevention and privacy protection effects is absent.

## Related Work & Insights

- **vs. BBF (Kuwana et al., 2024)**: BBF is the state-of-the-art class-level unlearning method for VLMs, but when applied directly to domain unlearning, its For metric is more than 30% lower than the proposed method, highlighting that class unlearning and domain unlearning are fundamentally distinct problems.
- **vs. CLIPFit / LP++**: These state-of-the-art CLIP fine-tuning methods, when combined with standard unlearning losses, still yield low H values (30–53) on ADU, demonstrating that fine-tuning strategies alone are insufficient and that domain entanglement must be explicitly addressed.
- **vs. domain adaptation/generalization methods (DAN, JAN)**: These methods minimize MMD to achieve domain-invariant representations, whereas this work maximizes MMD to achieve domain separability — reflecting a fundamental divergence in objectives between ADU and DA/DG.
- **Inspiration**: The "disentangle first, then forget" paradigm of DDL may generalize to other scenarios requiring fine-grained unlearning control, such as forgetting by time period or by data source.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First formulation of ADU, with a novel and practically motivated perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, multiple forget-domain configurations, complete ablations and visualizations, though limited to a single model (CLIP).
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is precise, motivation is well articulated, and figures are intuitive.
- Value: ⭐⭐⭐⭐ Opens a new direction in domain-level unlearning, though challenges such as domain label dependency must be resolved before practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders](../../ICCV2025/llm_safety/sauce_selective_concept_unlearning_in_vision-language_models_with_sparse_autoenc.md)
- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](../../CVPR2025/llm_safety/hyperbolic_safety-aware_vision-language_models.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/llm_safety/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/llm_safety/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Joint Vision-Language Social Bias Removal for CLIP
description: >-
  [CVPR 2025][Multimodal VLM][CLIP debiasing] This paper reveals the "over-debiasing" problem caused by inconsistent bias distributions in image and text modalities within CLIP. It proposes a joint framework of dual-modal bias alignment and counterfactual debiasing. While effectively reducing gender, age, and racial biases, it preserves vision-language alignment capabilities and designs the ABLE metric to comprehensively evaluate both debiasing performance and downstream capabi…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "CLIP debiasing"
  - "social bias removal"
  - "vision-language alignment"
  - "counterfactual debiasing"
  - "fairness"
date: 2026-05-08
content_hash: 2ea76927c62fca98
---

# Joint Vision-Language Social Bias Removal for CLIP

**Conference**: CVPR 2025  
**arXiv**: [2411.12785](https://arxiv.org/abs/2411.12785)  
**Code**: [https://github.com/](https://github.com/) (Yes, mentioned in the paper)  
**Area**: Multimodal VLM  
**Keywords**: CLIP debiasing, social bias removal, vision-language alignment, counterfactual debiasing, fairness

## TL;DR
This paper reveals the "over-debiasing" problem caused by inconsistent bias distributions in image and text modalities within CLIP. It proposes a joint framework of dual-modal bias alignment and counterfactual debiasing. While effectively reducing gender, age, and racial biases, it preserves vision-language alignment capabilities and designs the ABLE metric to comprehensively evaluate both debiasing performance and downstream capabilities.

## Background & Motivation
Vision-language pre-trained models like CLIP perform excellently in downstream tasks such as classification and retrieval, but they inherit severe social biases from web data (e.g., associating "occupations" with specific genders). Existing debiasing methods primarily remove bias information from single-modal embeddings. However, this introduces a key challenge: **a significant drop in V-L alignment capability after debiasing**, namely the "over-debiasing" problem.

The authors further investigate and discover that: (1) social biases coexist in both image and text modalities; (2) the bias distributions in the two modalities differ significantly (e.g., gender-career bias is prominent in images, while gender-science bias is prominent in text). Therefore, assuming identical biases across two modalities and debiasing with the same dimensions, as in CLIP-clip, is unreasonable.

**Core Idea**: First align the bias distributions of the two modalities, then jointly remove the biases, while maintaining V-L alignment capabilities through a counterfactual objective.

## Method

### Overall Architecture
The original CLIP encoders are frozen, followed by a learnable Bias Alignment module $\mathrm{BA}(\cdot;\theta_{ba})$. The training data consists of face image-text pairs with attribute labels (e.g., FairFace). During training, the joint optimization is performed using a bias alignment loss $\mathcal{L}_{ba}$ and a counterfactual debiasing loss $\mathcal{L}_{cd}$. During inference, the debiased embedding is obtained via $\bar{\phi}(t) = f(t) - \mathrm{BA}(f(t))$.

### Key Designs
1. **Bias Information Decoupling**:
    - Function: Decomposing CLIP embeddings into bias and neutral components.
    - Mechanism: $f(t) = \phi(t) + \bar{\phi}(t)$, where $\phi(t)$ represents the bias information and $\bar{\phi}(t)$ represents the neutral information. The BA module outputs $\phi(t)$, which is subtracted to obtain the debiased embedding.
    - Design Motivation: Social bias is embedded as an additive component within the embedding and can be eliminated by learning and subtracting it.

2. **Dual-Bias Alignment**:
    - Function: Aligning the bias distributions of images and text prior to debiasing.
    - Mechanism: Maintaining image and text embedding queues $\mathcal{Q}_v, \mathcal{Q}_t$ (similar to MoCo), computing the pseudodistributions of similarity between the bias embeddings and the queues $p(t_i), p(v_i)$, and aligning the two distributions using the KL divergence loss $\mathcal{L}_{ba} = \frac{1}{N}\sum D_{KL}(p(t_i) \| p(v_i))$.
    - Design Motivation: Direct element-wise matching would discard background information and feature diversity. Distribution-level alignment is more flexible and preserves details.

3. **Counterfactual Debiasing**:
    - Function: Bringing together the debiased embeddings of different attributes under the same neutral concept while preserving V-L alignment.
    - Mechanism: Constructing counterfactual pairs for text (e.g., "male dancer" $\leftrightarrow$ "female dancer"), and pulling the debiased similarity distribution closer to the original distribution using cross-entropy loss: $\mathcal{L}_{cd}^t = -\frac{1}{N}\sum\sum s_t(t_i,v,\mathcal{V}_q)\log\bar{s}_t(a(t_i,t'_i),v,\mathcal{V}_q)$, where $a(t_i,t'_i)$ randomly selects either the text or its counterfactual version with a 50% probability.
    - Design Motivation: Aligning biases alone is insufficient; it is crucial to ensure that the original V-L alignment capability is maintained after debiasing to avoid performance degradation on downstream tasks.

### Loss & Training
The total loss is defined as: $\mathcal{L} = \alpha \mathcal{L}_{cd} + (1-\alpha)\mathcal{L}_{ba}$, where $\alpha \in [0,1]$ balances the two objectives. The CLIP encoders remain frozen throughout training, with only the BA module parameters $\theta_{ba}$ being optimized. During inference, the BA module serves as a plug-and-play component.

## Key Experimental Results

### Main Results (ViT-B/16, FairFace training)

| Setting | Method | MaxSkew↓ (In-domain) | NDKL↓ (In-domain) | IN1K Top1↑ | Flickr TR↑ | ABLE↑ |
|------|------|-------------|------------|-----------|-----------|-------|
| Gender | Original CLIP | 0.218 | 0.088 | 68.31 | 96.4 | 73.87 |
| Gender | CLIP-clip | 0.103 | 0.026 | 68.00 | 95.4 | 77.55 |
| Gender | Biased-prompts | 0.161 | 0.048 | 65.07 | 94.3 | 73.78 |
| Gender | **Ours** | **0.080** | **0.025** | 68.05 | **96.6** | **78.35** |
| Age | Original CLIP | 0.657 | 0.433 | 68.31 | 96.4 | 58.94 |
| Age | **Ours** | **0.608** | **0.294** | **68.34** | 96.0 | **60.61** |

### Ablation Study (ViT-B/16, Gender, FairFace)

| Configuration | MaxSkew↓ | NDKL↓ | IN1K Top1↑ | ABLE↑ | Description |
|------|---------|-------|-----------|-------|------|
| Ours (complete) | 0.080 | 0.025 | 68.05 | 78.35 | Complete method |
| w/o $\mathcal{L}_{cd}$ | 0.167 | 0.056 | 68.28 | 75.58 | Removing counterfactual loss, bias increases |
| w/o $\mathcal{L}_{ba}$ | 0.095 | 0.033 | 67.84 | 77.71 | Removing alignment loss, performance drops slightly |

### Key Findings
- Both losses are indispensable: $\mathcal{L}_{cd}$ contributes more to reducing bias, while $\mathcal{L}_{ba}$ is more critical for maintaining V-L alignment.
- The method is consistently effective across four ViT backbones (B/16, B/32, L/14, H/14).
- Strong out-of-domain generalization: trained on FairFace, the method successfully removes biases on UTKFace and FACET.
- Multiple types of biases (gender + age + race) can be removed simultaneously, making it more suitable for practical deployment.

## Highlights & Insights
- **Insightful problem discovery**: It proves that bias distributions differ between the two modalities in V-L models, directly explaining why existing methods fail.
- **Clever design of the ABLE metric**: It uses the harmonic mean to comprehensively evaluate both the degree of debiasing and downstream performance, resolving the previous limitation of evaluating only one aspect.
- **Simple and efficient method**: It only requires training a lightweight BA module while CLIP remains completely frozen, allowing plug-and-play capability.

## Limitations & Future Work
- Reliance on attribute-labeled face datasets (such as FairFace/UTKFace) for training.
- Inability to construct counterfactual samples on the image side (due to insufficient generative model quality), restricting the approach to a unidirectional image debiasing loss.
- Evaluation is limited to retrieval and classification tasks; its impact on generative tasks like text-to-image synthesis has not been verified.
- The types of biases addressed are limited by the annotation categories in the training data.

## Related Work & Insights
- Complementary to CLIP-clip (based on mutual information dimension cropping) and Biased-prompts (based on projection matrices).
- The distribution alignment idea borrows the momentum queue mechanism from MoCo, adapting it to bias alignment scenarios.
- The counterfactual debiasing concept can be extended to other V-L models (e.g., the BLIP family).

## Rating
- Novelty: ⭐⭐⭐⭐ The problem analysis is deep, and the dual-modal bias alignment concept is novel, though the basic framework (alignment + debiasing) is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive assessments across four backbones, three bias types, in-domain and out-of-domain evaluations, and complete ablation studies are conducted.
- Writing Quality: ⭐⭐⭐⭐ The logic is clear, with a tight link between the problem analysis and method design.
- Value: ⭐⭐⭐⭐ Provides a fresh analytical perspective and practical methodology for the study of fairness in V-L models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Identifying and Mitigating Position Bias of Multi-image Vision-Language Models](identifying_and_mitigating_position_bias_of_multi-image_vision-language_models.md)
- [\[CVPR 2025\] MarkushGrapher: Joint Visual and Textual Recognition of Markush Structures](markushgrapher_joint_visual_and_textual_recognition_of_markush_structures.md)
- [\[CVPR 2025\] Data Distributional Properties as Inductive Bias for Systematic Generalization](data_distributional_properties_as_inductive_bias_for_systematic_generalization.md)
- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](../../CVPR2026/multimodal_vlm/interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [\[ICCV 2025\] MAVias: Mitigate Any Visual Bias](../../ICCV2025/multimodal_vlm/mavias_mitigate_any_visual_bias.md)

</div>

<!-- RELATED:END -->

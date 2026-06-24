---
title: >-
  [Paper Note] Single Domain Generalization for Few-Shot Counting via Universal Representation Matching
description: >-
  [CVPR 2025][Multimodal VLM][Few-Shot Counting] Proposes URM, the first single domain generalization model for few-shot counting. By distilling CLIP's universal vision-language representations into learnable prototypes to construct correlations, it significantly improves cross-domain generalization capability (reducing MAE by 27.5%) without sacrificing in-domain performance.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Few-Shot Counting"
  - "Domain Generalization"
  - "CLIP Knowledge Distillation"
  - "Universal Representation Matching"
  - "Vision-Language Prototypes"
date: 2026-05-08
content_hash: 3e99e2bdeceb9d76
---

# Single Domain Generalization for Few-Shot Counting via Universal Representation Matching

**Conference**: CVPR 2025  
**arXiv**: [2505.16778](https://arxiv.org/abs/2505.16778)  
**Code**: [https://github.com/jbr97/URM](https://github.com/jbr97/URM)  
**Area**: Multimodal VLM  
**Keywords**: Few-Shot Counting, Domain Generalization, CLIP Knowledge Distillation, Universal Representation Matching, Vision-Language Prototypes

## TL;DR

Proposes URM, the first single domain generalization model for few-shot counting. By distilling CLIP's universal vision-language representations into learnable prototypes to construct correlations, it significantly improves cross-domain generalization capability (reducing MAE by 27.5%) without sacrificing in-domain performance.

## Background & Motivation

Few-shot counting (FSC) estimates the number of target objects in an image based on a few annotated exemplars. Existing methods (e.g., FamNet, LOCA, DAVE) follow an "extraction-matching" pipeline: they first extract prototypes from exemplars, perform correlation matching with image features, and finally regress a density map. However, these methods suffer from severe performance degradation in cross-domain scenarios—for instance, while objects are clearly visible in FSC-147, FSCD-LVIS contains much more severe occlusions and quality degradation.

This paper identifies a core problem: **prototypes learned from a narrowly distributed source domain also have a narrow distribution**, leading to poor matching performance in target domains with large distribution shifts. This hypothesis is verified through t-SNE visualization—traditional methods show blurry feature boundaries under cross-domain settings. The proposed solution is to introduce CLIP pre-trained universal vision-language representations, which are trained on massive and diverse data and possess domain-invariant properties.

## Method

### Overall Architecture

The overall architecture of URM includes: extracting multi-scale features using an ImageNet pre-trained ResNet-50 backbone $\rightarrow$ encoding exemplar bounding box information via a Prompt Encoder $\rightarrow$ interacting universal vision-language prototypes with encoded image features using cross-attention $\rightarrow$ distilling universal representations from a frozen CLIP (only during training) $\rightarrow$ performing correlation matching between concatenated prototypes and image features $\rightarrow$ regressing the density map. During inference, CLIP is not required, as efficiency is identical to methods without external knowledge.

### Key Designs

1. **Universal Vision Representation Learning**:
    - **Function**: To obtain local object-level visual representations with domain-invariant properties.
    - **Mechanism**: MaskCLIP is utilized to modify the CLIP vision encoder by removing the QK lookup in the final two layers, converting V and linear layers to convolutional layers, and generating category-relevant dense segmentation maps $\mathcal{M}$. Local object representations are then extracted from the CLIP vision tokens via Mask Pooling: $r_v = \text{MaskPool}(\mathcal{V}(\mathbf{I}), \mathcal{M})$. The learnable visual prototype $p_v$ interacts with the encoded image features $f^E$ through $N_1$ layers of cross-attention, and is distilled with the CLIP representation using a feature mimicry loss: $\mathcal{L}_{V\text{-}KD} = \frac{1}{|\mathcal{B}|}\sum \|r_{v_k} - p_{v_k}\|$
    - **Design Motivation**: The CLS token of CLIP focuses on global attributes, and image tokens have weak local discriminability, making them unsuitable for direct local object matching. MaskCLIP+Mask Pooling can accurately extract CLIP representations of object regions while removing noise from irrelevant areas.

2. **Universal Language Representation Learning**:
    - **Function**: To obtain textual representations containing rich, category-discriminative features.
    - **Mechanism**: Instead of relying solely on the hand-crafted template "A photo of {}", the CuPL method is followed to use GPT-4 to generate customized descriptions containing key discriminative features (such as object shape, texture, default scenarios, etc.). Multiple prompts are passed through the CLIP text encoder and averaged to obtain the universal textual representation $r_t$. The learnable language prototype $p_l$ is similarly updated via cross-attention and distilled using $\mathcal{L}_{L\text{-}KD}$.
    - **Design Motivation**: Simple templates like "A photo of {}" cannot adequately describe fine-grained category features (e.g., the ImageNet "A toy {}" template performs poorly on FSC-147). The diverse descriptions generated by the LLM can better represent the target categories.

3. **Universal Representation Matching**:
    - **Function**: To construct correlation maps for counting using the distilled vision-language (V-L) prototypes.
    - **Mechanism**: Concatenates the visual and language prototypes $\text{concat}(p_v, p_l)$ and matches them with the encoded image features $f^E$ through $N_2$ layers of cross-attention ($q=f^E, k=v=\text{concat}(p_v,p_l)$). After outputting the correlation map, the regression head predicts the density map.
    - **Design Motivation**: Traditional methods construct correlations using only visual information. Incorporating a dual-channel V-L prototype distilled from CLIP enables matching in a domain-invariant feature space, providing natural robustness for cross-domain generalization.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{\text{density}} + \alpha \mathcal{L}_{V\text{-}KD} + (1-\alpha)\mathcal{L}_{L\text{-}KD}$, where the density loss is a normalized $\ell_2$ loss: $\mathcal{L}_{\text{density}} = \frac{1}{2\mathcal{B}}\sum \frac{1}{N_k}\|\mathbf{G}_k - \mathbf{R}_k\|_2^2$. The distillation weight is set to $\alpha = 0.9$, indicating a greater contribution from the language representation (since the CLIP text encoder aligned to the visual space is highly informative). Knowledge distillation is performed only during the training phase. At inference time, the CLIP model is removed, maintaining computational efficiency consistent with other methods. The ResNet-50 backbone is frozen, while other parameters are trained for 150 epochs using AdamW with lr=1e-4 on a single V100 GPU.

## Key Experimental Results

### Main Results (Few-shot counting cross-domain)

| Setting | Method | MAE↓ | RMSE↓ |
|------|------|------|-------|
| FSC→FSCD-LVIS | MPCount (CVPR24 SOTA) | 25.11 | 41.32 |
| FSC→FSCD-LVIS | **URM** | **21.87** | **38.42** |
| FSCD-LVIS→FSC | MPCount | 22.07 | 80.17 |
| FSCD-LVIS→FSC | **URM** | **21.17** | **73.42** |
| FSC→FSCD-LVIS (zero-shot) | MPCount | 27.68 | 44.58 |
| FSC→FSCD-LVIS (zero-shot) | **URM** | **23.54** | **39.93** |

### Ablation Study

| Configuration | MAE↓ | RMSE↓ | Note |
|------|------|-------|------|
| Baseline (No distillation) | 30.15 | 48.04 | Only architecture changes, no generalization gain |
| + Language (Naive Template) | 25.44 | 42.59 | Simple template yields significant improvement |
| + Language (Prompt Generator) | 23.83 | 41.03 | prompts generated by LLM bring further improvement |
| + Vision ([CLS] Token) | 23.53 | 41.12 | Global token is inferior to local representation |
| + Vision (Global Pooling) | 22.94 | 39.44 | Global pooling is effective but sub-optimal |
| + Vision (Mask Pooling) | **21.87** | **38.42** | Local object representation is optimal, with total gain of 27.5% |

| Distillation weight $\alpha$ | 0 | 0.25 | 0.5 | 0.75 | **0.9** | 1 |
|-------------------|---|------|-----|------|---------|---|
| MAE↓ | 26.54 | 23.44 | 23.30 | 22.37 | **21.87** | 23.83 |

### Key Findings

- Simply changing the architecture (matching with learnable prototypes) does not benefit generalization at all (MAE 30.15 vs. baseline); CLIP knowledge distillation is essential.
- The contribution of linguistic knowledge is greater than visual knowledge ($\alpha=0.9$ is optimal) because the CLIP text encoder is already aligned to the visual space, acting intrinsically as a cross-modal bridge.
- V-L representations are complementary: relying only on one modality ($\alpha=0$ or $\alpha=1$) yields inferior results compared to combining both.
- CLIP is not required during inference, achieving identical efficiency to traditional methods—making distillation a "one-off investment".

## Highlights & Insights

- **Simple yet profound core finding**: The key to domain generalization lies in the distribution width of prototypes rather than architectural complexity. CLIP's universal representations naturally address this problem.
- **Decoupled Training and Inference**: Distillation is conducted only during training, and CLIP is removed during inference, achieving zero extra computational overhead at inference time.
- **Hypothesis validated intuitively via t-SNE**: Distinctly demonstrates that narrowly distributed prototypes have blurry boundaries in cross-domain scenarios, whereas CLIP distillation boundary is sharp and clear.

## Limitations & Future Work

- Only verified cross-domain generalization between two datasets; more diverse domains (e.g., satellite imagery, medical images) remain unexplored.
- Reliance on GPT-4 to generate language prompts incurs some cost.
- CLIP distillation assumes target categories have corresponding textual descriptions, which might limit effectiveness for extremely long-tail classes.
- The impact of stronger vision foundation models (such as DINOv2) as distillation teachers remains unexplored.

## Related Work & Insights

- CuPL uses LLM-generated customized prompts to improve CLIP's zero-shot classification; this paper introduces this concept to prototype distillation for counting tasks.
- MaskCLIP modifies the CLIP encoder to obtain pixel-level predictions; this paper employs it to extract local object representations.
- Feature mimicry in CLIP-KD was proven to be the most effective way of CLIP knowledge distillation, which is directly adopted in this paper.
- The paradigm of "distilling universal representations into task-specific prototypes" can be generalized to other visual tasks requiring domain generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ First single domain generalization for few-shot counting, with simple and powerful core findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ Cross-domain, in-domain, and zero-shot settings combined with detailed ablations, though only tested on two datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Highly logical flow from motivation analysis $\rightarrow$ hypothesis validation $\rightarrow$ method design $\rightarrow$ experimental verification.
- Value: ⭐⭐⭐⭐ The paradigm of distilling CLIP into task-specific prototypes holds broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Multimodal Domain Generalization with Few Labels](../../CVPR2026/multimodal_vlm/towards_multimodal_domain_generalization_with_few_labels.md)
- [\[CVPR 2025\] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning](unem_unrolled_generalized_em_for_transductive_few-shot_learning.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] GENIUS: A Generative Framework for Universal Multimodal Search](genius_a_generative_framework_for_universal_multimodal_search.md)

</div>

<!-- RELATED:END -->

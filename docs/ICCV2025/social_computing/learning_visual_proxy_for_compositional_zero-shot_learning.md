---
title: >-
  [Paper Note] Learning Visual Proxy for Compositional Zero-Shot Learning
description: >-
  [ICCV 2025][Social Computing][Compositional Zero-Shot Learning] This paper proposes the concept of Visual Proxy — text-guided visual class centers introduced into CZSL for the first time — and jointly optimizes textual prototypes and visual proxies via Cross-Modal Joint Learning (CMJL), achieving closed-world SOTA on four CZSL benchmarks.
tags:
  - ICCV 2025
  - Social Computing
  - Compositional Zero-Shot Learning
  - Visual Proxy
  - Cross-Modal Learning
  - CLIP
  - VLM
date: 2026-05-08
content_hash: a693bf53bf79d652
---

# Learning Visual Proxy for Compositional Zero-Shot Learning

**Conference**: ICCV 2025
**arXiv**: [2501.13859](https://arxiv.org/abs/2501.13859)
**Code**: [codefish12-09/VP_CMJL](https://github.com/codefish12-09/VP_CMJL)
**Area**: Social Computing
**Keywords**: Compositional Zero-Shot Learning, Visual Proxy, Cross-Modal Learning, CLIP, VLM

## TL;DR

This paper proposes the concept of Visual Proxy — text-guided visual class centers introduced into CZSL for the first time — and jointly optimizes textual prototypes and visual proxies via Cross-Modal Joint Learning (CMJL), achieving closed-world SOTA on four CZSL benchmarks.

## Background & Motivation

Compositional Zero-Shot Learning (CZSL) aims to generalize from seen attribute-object compositions (e.g., "green shirt," "red apple") to unseen ones (e.g., "red shirt," "green apple"). Existing CLIP-based methods perform classification via text-image matching, but suffer from two core issues:

**Persistent Modality Gap**: Although various prompt designs and fusion strategies have partially narrowed the distance between text and visual spaces, complete elimination remains impossible. In top-1 retrieval scenarios, the distance between true cross-modal pairs may exceed that of false positives, causing semantically similar compositions to be confused (e.g., "ripe apple" vs. "unripe apple").

**Lack of Fine-Grained Visual Information in Textual Prototypes**: Each class's textual prototype is derived from a single compositional label, while the corresponding image instances contain rich variation in texture, lighting, and shape. This semantic-visual asymmetry prevents textual prototypes from capturing the fine-grained information needed to distinguish similar compositions.

Core insight: CZSL is fundamentally an image classification task, and the optimal class centers should reside in the visual space. However, directly learning visual centers is difficult due to high variance; the paper therefore leverages the structured text space to guide visual center learning.

## Method

### Overall Architecture

VP-CMJL consists of three modules:
1. **Textual Prototype Learning Module**: Three-branch framework (attribute / object / composition) + cross-modal disentanglement module
2. **Visual Proxy Learning Module**: Text-guided visual proxy learning + MLP disentanglement
3. **Cross-Modal Joint Learning Module**: KL divergence constraint for joint optimization of both modalities

A frozen CLIP ViT-L/14 is used as the visual and text encoder.

### Key Designs

1. **Textual Prototype Learning**:

    - Three-branch learnable prompts: attribute $\theta^a$, object $\theta^o$, composition $\theta^c$, all prefix-initialized with "a photo of"
    - **Cross-modal disentanglement modules (AD-CA / OD-CA)**: Multi-head cross-attention decomposes the global image feature $f_v^{cls}$ into attribute/object features aligned with textual prototypes. Query comes from image features; Key/Value come from textual prototypes; output passes through FFN + LayerNorm + residual connection.
    - **Attention-score-augmented probability computation**: Classification probabilities for the attribute and object branches jointly consider text-image cosine similarity and attention scores $s^a/s^o$:
    $p_t(y_i|x) = \frac{\exp((f_t^y \cdot t_i^y + s_i^y)/\tau_t)}{\sum_k \exp((f_t^y \cdot t_k^y + s_k^y)/\tau_t)}$

2. **Visual Proxy Learning**:

    - **Initialization**: Visual proxies are initialized using CLIP text encoder word embeddings $v_i^a = E_l(w_i^a)$ (experiments confirm CLIP initialization outperforms BERT/GPT).
    - **Compositional proxy construction**: Attribute and object proxies are concatenated and projected via a fully connected layer: $v_{i,j}^c = E_c([v_i^a, v_j^o])$.
    - **MLP disentanglement**: Intra-modal disentanglement within the visual modality uses a simple MLP (rather than cross-attention), as this is intra-modal learning.
    - **Contrastive training**: Intra-class attraction and inter-class repulsion via softmax temperature-scaled cosine similarity.
    - Key theoretical justification: CLIP's optimal class centers lie in the overlap region of visual and text spaces, but due to the modality gap this center remains biased. Visual proxies are learned directly in the visual space, offering a closer approximation to the optimal solution for image classification.

3. **Cross-Modal Joint Learning (CMJL)**:

    - **Training**: The textual prototype distribution serves as the target; the visual proxy distribution is the approximation, constrained by KL divergence:
    $\mathcal{L}_{kl} = D_{KL}(P_t \| P_v)$
      Total loss: $\mathcal{L} = \alpha(\mathcal{L}_t + \mathcal{L}_v) + \beta \mathcal{L}_{kl}$
    - **Inference**: Probabilities from both modalities are fused:
    $p(y_{i,j}|x) = p_t(y_{i,j}|x) + \lambda p_v(y_{i,j}|x)$
      Final prediction is the argmax of the summed probabilities across attribute, object, and composition branches.

### Loss & Training

- Textual path: $\mathcal{L}_t = \gamma_{ao}(\mathcal{L}_t^a + \mathcal{L}_t^o) + \gamma_c \mathcal{L}_t^c$
- Visual path: $\mathcal{L}_v = \gamma_{ao}(\mathcal{L}_v^a + \mathcal{L}_v^o) + \gamma_c \mathcal{L}_v^c$
- Total loss: $\mathcal{L} = \alpha(\mathcal{L}_t + \mathcal{L}_v) + \beta \mathcal{L}_{kl}$
- Trained for 20 epochs with CLIP ViT-L/14 on an NVIDIA A800 GPU.

## Key Experimental Results

### Main Results (Tables)

**Closed-World Results (Best HM / AUC)**

| Method | C-GQA HM | C-GQA AUC | UT-Zappos HM | UT-Zappos AUC | MIT-States HM | MIT-States AUC |
|--------|----------|-----------|--------------|---------------|---------------|----------------|
| Troika (CVPR'24) | 29.4 | 12.4 | 54.6 | 41.7 | 39.3 | 22.1 |
| IMAX (TPAMI'25) | 29.8 | 12.8 | 54.2 | 40.6 | 39.1 | 21.9 |
| CDS-CZSL (CVPR'24) | 28.1 | 11.1 | 52.7 | 39.5 | 39.2 | 22.4 |
| **VP-CMJL (Ours)** | **34.9** | **16.3** | **58.5** | **47.9** | **40.4** | **23.3** |

On C-GQA, HM improves by +5.5% and AUC by +3.9%; on UT-Zappos, HM improves by +3.9% and AUC by +6.2%.

**VAW-CZSL (New Large-Scale Dataset)**

| Method | S | U | HM | AUC |
|--------|---|---|-----|-----|
| CAILA (WACV'24) | 41.6 | 49.2 | 34.6 | 17.2 |
| **VP-CMJL** | **47.8** | **51.1** | **38.2** | **20.7** |

### Ablation Study (Tables)

**Component Ablation (UT-Zappos / MIT-States, Closed-World)**

| TP | VP | UT-Zappos HM | UT-Zappos AUC | MIT-States HM | MIT-States AUC |
|----|-----|--------------|---------------|---------------|----------------|
| ✓ | ✓ | **58.5** | **47.9** | **40.4** | **23.3** |
| ✓ | ✗ | 51.9 | 37.8 | 37.8 | 20.8 |
| ✗ | ✓ | 55.3 | 42.1 | 37.6 | 20.7 |

Removing VP causes a 10.1% drop in UT-Zappos AUC; removing TP causes a 5.8% drop.

**Disentanglement Module Ablation**

| i2t Disentangle | i2v Disentangle | UT-Zappos HM | MIT-States HM |
|-----------------|-----------------|--------------|---------------|
| CA | MLP | **58.5** | **40.4** |
| CA | CA | 54.7 | 39.6 |
| MLP | MLP | 58.5 | 38.8 |
| MLP | CA | 55.7 | 39.6 |

Cross-attention is better suited for cross-modal alignment (textual prototypes), while MLP is better suited for intra-modal learning (visual proxies).

### Key Findings

- **Visual proxy is the critical component**: AUC improves from 37.8 to 47.9 (+26.7%) on UT-Zappos.
- **Implicit mutual enhancement from dual-modal training**: Even when one modality is removed at inference, the performance degradation is smaller than when it is removed during training, indicating that joint optimization promotes mutual representation enhancement.
- **Open-world effectiveness**: C-GQA open-world HM improves by +4.6%; UT-Zappos by +6.7%.
- **t-SNE visualization**: The visual feature space of VP-CMJL is more compact with clearer class boundaries.

## Highlights & Insights

1. **Proposal of the Visual Proxy concept**: This is the first work to introduce text-guided learnable visual class centers in CZSL, addressing the fundamental deficiency of textual prototypes lacking fine-grained visual information.
2. **Cross-modal KL divergence constraint**: A concise yet effective co-optimization strategy that leverages the semantic stability of text to guide visual proxy learning.
3. **Attention-score-augmented classification**: Cross-attention scores are introduced alongside conventional cosine similarity, simultaneously encoding the relationship between the query image and all classes.
4. **Appropriate choice of disentanglement modules**: Cross-attention for cross-modal alignment and MLP for intra-modal transformation, reflecting a principled understanding of the problem.

## Limitations & Future Work

- Open-world performance on MIT-States is only competitive and does not surpass CDS-CZSL, which employs pruning techniques specifically designed for the open-world setting.
- The temperature parameter $\tau_v$ for visual proxies and the fusion weight $\lambda$ require manual tuning.
- Compositional proxies are generated via concatenation and fully connected projection, which may fail to capture nonlinear attribute-object interactions.
- Visual proxies for unseen compositions are obtained by concatenating attribute/object proxies, lacking direct supervision from the visual space.
- The potential of larger-scale VLMs (e.g., SigLIP, EVA-CLIP) has not been explored.

## Related Work & Insights

- **CSP / DFSP / Troika**: A family of CLIP-based CZSL methods; VP-CMJL extends the three-branch framework by adding visual proxies.
- **CDS-CZSL**: An attribute-specificity-aware method with advantages in the open-world setting.
- **Visual center learning**: Traditional methods obtain prototypes by averaging image features, which are susceptible to viewpoint and lighting variations; VP-CMJL avoids this through text-guided learning.
- Insight: In the VLM era, optimization should not be limited to the text-side prompt; jointly learning visual-side class centers may be a generalizable strategy for improving classification performance.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The visual proxy concept is novel, and the cross-modal joint learning strategy is well-designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Four datasets, open/closed-world settings, component ablation, disentanglement module ablation, and visualization analysis.
- **Writing Quality**: ⭐⭐⭐⭐ In-depth motivation analysis (class center theory) with clear method description.
- **Value**: ⭐⭐⭐⭐ Significant gains on CZSL; the visual proxy idea is transferable to other VLM-based classification scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VDRP: Visual Diversity and Region-aware Prompt Learning for Zero-shot HOI Detection](../../NeurIPS2025/social_computing/visual_diversity_and_region-aware_prompt_learning_for_zero-shot_hoi_detection.md)
- [\[ICCV 2025\] Gradient Extrapolation for Debiased Representation Learning](gradient_extrapolation_for_debiased_representation_learning.md)
- [\[AAAI 2026\] Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](../../AAAI2026/social_computing/multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)
- [\[ICCV 2025\] PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination](propvg_end-to-end_proposal-driven_visual_grounding_with_multi-granularity_discri.md)
- [\[NeurIPS 2025\] GraphKeeper: Graph Domain-Incremental Learning via Knowledge Disentanglement and Preservation](../../NeurIPS2025/social_computing/graphkeeper_graph_domain-incremental_learning_via_knowledge_disentanglement_and_.md)

</div>

<!-- RELATED:END -->

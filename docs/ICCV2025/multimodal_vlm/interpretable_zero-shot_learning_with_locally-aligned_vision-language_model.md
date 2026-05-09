---
title: >-
  [Paper Note] Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model
description: >-
  [ICCV 2025][Multimodal VLM][Zero-shot learning] This paper proposes LaZSL, which leverages Optimal Transport (OT) to achieve fine-grained alignment between local visual regions and semantic attributes, constructing an interpretable zero-shot classifier without additional training. LaZSL demonstrates strong accuracy, interpretability, and domain generalization across 9 datasets.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Zero-shot learning
  - interpretability
  - optimal transport
  - local alignment
  - CLIP
  - attributes
date: 2026-05-08
content_hash: 318ee8370347a687
---

# Interpretable Zero-Shot Learning with Locally-Aligned Vision-Language Model

**Conference**: ICCV 2025
**arXiv**: 2506.23822
**Code**: [https://github.com/shiming-chen/LaZSL](https://github.com/shiming-chen/LaZSL)
**Area**: Multimodal VLM
**Keywords**: Zero-shot learning, interpretability, optimal transport, local alignment, CLIP, attributes

## TL;DR

This paper proposes LaZSL, which leverages Optimal Transport (OT) to achieve fine-grained alignment between local visual regions and semantic attributes, constructing an interpretable zero-shot classifier without additional training. LaZSL demonstrates strong accuracy, interpretability, and domain generalization across 9 datasets.

## Background & Motivation

**Limitations of existing CLIP-based methods:**

- CLIP performs zero-shot classification by computing global similarity between entire images and class-level text, which lacks interpretability — the model cannot explain why a particular prediction is made.
- Attribute-based interpretable methods (e.g., DCLIP, CuPL) leverage LLM-generated attribute descriptions to construct classifiers, yet they still match **entire images** against attributes, failing to capture fine-grained relationships between local visual regions and corresponding attributes.

**Core challenge:** How to achieve effective alignment between local visual regions and attributes on top of a frozen pretrained VLM, where the network parameters cannot be redesigned or fine-tuned, making conventional attention-based approaches inapplicable.

**Key insight:** Optimal Transport theory is applied by treating image crops as a set of local patches and class attributes as a semantic set, then using OT to find the optimal matching between the two distributions.

## Method

### Overall Architecture

LaZSL consists of three core modules: (1) construction of semantic and visual sets; (2) OT-based local vision–language alignment; and (3) zero-shot prediction.

### 1. Semantic and Visual Set Construction

**Semantic set construction:** GPT-3 is used to generate multiple attribute descriptions per class. For class $y$, the LLM produces a semantic set:

$$S^y = h(prompt(y)) = \{s_i^y | i=1,...,M\}$$

**Visual set construction:** Multi-scale random crops of the input image are generated to form a local visual region set:

$$V_r^x = \{v_i^x = P_r(x, \gamma_i \min(W,H)) | i=1,...,N\}$$

where $\gamma_i \sim U(\alpha, \beta)$ controls the crop scale and $N$ is typically set to 60–90. Varying $\gamma_i$ ensures that the visual set captures features at multiple scales.

### 2. OT-Based Local Alignment

The visual and semantic sets are encoded via CLIP's visual and text encoders to obtain feature representations $P^x$ and $Q^y$, respectively.

**Vision Selection:** The cosine similarity between each local region and the global image is computed; regions are partitioned into a positively correlated set $P_{pos}^l$ and a negatively correlated set $P_{neg}^l$ using the mean similarity $\delta$ as threshold. Only positively correlated regions participate in OT computation. The visual probability distribution is updated as:

$$r_i^* = \begin{cases} \frac{1}{|P_{pos}^l|} & \text{if } p_i^l \in P_{pos}^l \\ 0 & \text{otherwise} \end{cases}$$

**Region-Global Hybrid Cost:** To mitigate noise introduced by random cropping and knowledge forgetting, global visual information is incorporated into the cost matrix:

$$C_i^* = 1 - (\theta \cdot sim_i + (1-\theta) \cdot p^{g\top} Q^y)$$

where $\theta$ is a mixing coefficient (default 0.8) balancing local and global features.

**Sinkhorn algorithm for OT:** The optimal transport plan $T$ is obtained via iterative updates:

$$T = \text{diag}(\mathcal{U}) \mathcal{M} \text{diag}(\mathcal{V}), \quad \mathcal{M} = \exp(-C^*/\lambda)$$

### 3. Zero-Shot Prediction

The class score is computed as the Frobenius inner product between the OT plan and the hybrid similarity matrix:

$$\psi_y = \langle T, sim^* \rangle_F$$

The predicted class is selected as: $y^* = \arg\max_y \psi_y$

### Loss & Training

LaZSL is a **training-free** method requiring no additional loss functions or optimization. Classification is performed entirely at inference time via OT computation.

## Key Experimental Results

### Main Results: Cross-Dataset Zero-Shot Classification

| Method | ImageNet | CUB | Oxford Pets | Food101 | Place365 | Avg. |
|--------|----------|-----|-------------|---------|----------|------|
| CLIP (ViT-B/16) | 66.7 | 56.0 | 88.1 | 88.4 | 39.3 | 67.7 |
| DCLIP | 67.9 | 57.1 | 86.9 | 88.5 | 40.3 | 68.1 |
| CuPL | 69.6 | 56.4 | 91.1 | 89.0 | 39.8 | 69.2 |
| **LaZSL (Ours)** | **69.2** | **60.3** | **87.4** | **89.7** | **42.0** | **69.7** |

- LaZSL achieves the best **average** performance across all three backbones (ViT-B/32, ViT-B/16, ViT-L/14).
- Gains are most pronounced on the fine-grained dataset CUB (+3.1% over DCLIP on ViT-B/16), where classification relies more heavily on local feature alignment.

### Domain Generalization (ImageNet Variants)

| Method | Training Required | ImageNet-V2 | ImageNet-R | ImageNet-S | ImageNet-A | Avg. |
|--------|------------------|-------------|------------|------------|------------|------|
| CoOp | ✔ | 64.2 | 75.2 | 47.9 | 49.7 | 59.3 |
| MaPLe | ✔ | 64.1 | 77.0 | 49.2 | 50.9 | 60.3 |
| ArGue† | ✔ | 64.6 | 76.6 | 48.9 | 50.9 | 60.3 |
| **LaZSL† (Ours)** | **✗** | **63.3** | **75.6** | **48.2** | **56.2** | **60.9** |

- LaZSL **without any training** outperforms methods requiring additional training (e.g., CoOp, ArGue), with a substantial lead on ImageNet-A (+5.3% over ArGue).

### Ablation Study

| Method | ImageNet | CUB | Place365 |
|--------|----------|-----|----------|
| Baseline (DCLIP) | 67.9 | 57.8 | 40.3 |
| + OT | 68.5 | 59.0 | 41.6 |
| + OT + VS | 69.0 | 60.0 | 41.8 |
| + OT + Hybrid | 69.0 | 59.3 | 41.9 |
| LaZSL (full) | **69.2** | **60.3** | **42.0** |

- OT-based local alignment is the most critical component, contributing the largest performance gains.
- Vision Selection and the Hybrid Cost matrix further improve alignment quality.

### Key Findings

1. Local alignment yields the greatest advantage on fine-grained datasets (CUB +3.1%).
2. A mixing coefficient of $\theta=0.8$ achieves the best balance; excessively high values cause global knowledge forgetting.
3. A crop scale of $\alpha=0.6$ is optimal across all datasets.
4. The additional inference overhead is manageable (0.07s vs. 0.015s per sample for DCLIP).

## Highlights & Insights

1. **Training-free design:** Unlike most CLIP variants that require additional training, LaZSL operates entirely at inference time, achieving local alignment solely through OT.
2. **Introducing OT into interpretable VLM classification:** Image crop regions and attribute descriptions are elegantly modeled as two discrete distributions, with OT finding the optimal matching between them.
3. **Hybrid cost matrix:** Fusing local and global information effectively mitigates the knowledge forgetting problem, representing a concise and effective design choice.
4. **Strong interpretability:** The framework can explicitly visualize which visual regions are matched to which attributes, resembling human cognitive processes.

## Limitations & Future Work

1. **Dependence on LLM-generated attribute quality:** Irrelevant or noisy attribute descriptions generated by the LLM can degrade classification accuracy.
2. **Slower inference speed:** Each image requires 60–90 random crops encoded through CLIP, resulting in inference time approximately 4.7× that of DCLIP.
3. **Stochasticity of random cropping:** Because the visual set is constructed via random crops, repeated inference on the same image may yield slightly inconsistent results.
4. **Underperformance on certain datasets:** LaZSL does not surpass CuPL on Oxford Pets, suggesting that global features may already be sufficient for some datasets.

## Related Work & Insights

- **Classical ZSL → VLM-based ZSL → Interpretable ZSL:** A clear research trajectory from manually annotated attributes to LLM-generated attributes.
- **OT in vision:** Previously applied primarily to prompt tuning (PLOT); this work is the first to apply OT for local feature alignment in zero-shot classification.
- **Inspiration:** Whether this idea can be extended to other tasks requiring local alignment, such as open-vocabulary detection and fine-grained retrieval, remains an open question.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of applying OT to local vision–language alignment is novel and well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — 9 datasets, 3 backbones, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic with intuitive figures and tables.
- **Value**: ⭐⭐⭐⭐ — Training-free with strong interpretability; highly practical.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] NegRefine: Refining Negative Label-Based Zero-Shot OOD Detection](negrefine_refining_negative_label-based_zero-shot_ood_detection.md)
- [\[ICCV 2025\] Dynamic Multimodal Prototype Learning in Vision-Language Models](dynamic_multimodal_prototype_learning_in_vision-language_models.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)
- [\[NeurIPS 2025\] Zero-Shot Robustness of Vision Language Models Via Confidence-Aware Weighting](../../NeurIPS2025/multimodal_vlm/zero-shot_robustness_of_vision_language_models_via_confidence-aware_weighting.md)
- [\[NeurIPS 2025\] TOMCAT: Test-time Comprehensive Knowledge Accumulation for Compositional Zero-Shot Learning](../../NeurIPS2025/multimodal_vlm/tomcat_test-time_comprehensive_knowledge_accumulation_for_compositional_zero-sho.md)

<!-- RELATED:END -->

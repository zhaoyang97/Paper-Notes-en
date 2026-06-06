---
title: >-
  [Paper Note] Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models
description: >-
  [AAAI 2026][Multimodal VLM][CLIP fine-tuning] This paper proposes DiVE, a method that constrains the "difference vectors" between pre-trained and fine-tuned model embeddings to be equal across samples…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "CLIP fine-tuning"
  - "robust fine-tuning"
  - "embedding geometry preservation"
  - "out-of-distribution generalization"
  - "zero-shot performance"
date: 2026-05-08
content_hash: bc2f3cecf5b62cbb
---

# Difference Vector Equalization for Robust Fine-tuning of Vision-Language Models

**Conference**: AAAI 2026
**arXiv**: [2511.09973](https://arxiv.org/abs/2511.09973)  
**Code**: No public code  
**Area**: Multimodal VLM
**Keywords**: CLIP fine-tuning, robust fine-tuning, embedding geometry preservation, out-of-distribution generalization, zero-shot performance

## TL;DR

This paper proposes DiVE, a method that constrains the "difference vectors" between pre-trained and fine-tuned model embeddings to be equal across samples, thereby preserving the geometric structure of the embedding space during CLIP fine-tuning. DiVE achieves comprehensive improvements over existing methods across in-distribution (ID), out-of-distribution (OOD), and zero-shot metrics (averaging 8+ points gain on zero-shot tasks).

## Background & Motivation

Contrastively pre-trained vision-language models such as CLIP exhibit strong zero-shot classification performance, yet often underperform on specific downstream tasks (e.g., satellite imagery that deviates from the pre-training distribution). Standard practice is to fine-tune on downstream data, but this severely degrades generalization—ID performance improves while OOD and zero-shot performance drop substantially.

Existing robust fine-tuning methods (e.g., FLYP, ARF) all adopt contrastive learning for fine-tuning. However, the authors find that these methods **distort the geometric structure of pre-trained embeddings** (i.e., the relative positional relationships among embeddings). Since relative distances between embeddings reflect semantic similarity between inputs, disrupting the geometry directly causes degradation in generalization. This is a critical observation that has been overlooked in prior work.

## Core Problem

**Can preserving the geometric structure of pre-trained embeddings improve robust fine-tuning of vision-language models?**

This is a clear and important research question. Although existing methods employ strategies such as contrastive learning and replay, none explicitly address the preservation of embedding space geometry. The authors use Representation Similarity Analysis (RSA) to quantitatively verify that existing methods indeed distort geometric structure (RSA scores of only 0.825–0.850), and propose a concrete solution.

## Method

### Overall Architecture

DiVE builds upon FLYP (i.e., fine-tuning with contrastive loss while updating both image and text encoders). During fine-tuning, in addition to the contrastive loss on the target data, DiVE introduces a reference dataset (e.g., CC3M) to compute "difference vectors" and impose constraints.

Core pipeline:
1. For each image/text sample in the reference dataset, extract embeddings from both the pre-trained model and the current fine-tuned model.
2. Compute difference vectors: $u(\mathbf{x}) = f_{\theta^{ft}}(\mathbf{x}) - f_{\theta^{pre}}(\mathbf{x})$, $v(\mathbf{t}) = g_{\phi^{ft}}(\mathbf{t}) - g_{\phi^{pre}}(\mathbf{t})$
3. Apply AVL and PVL losses to constrain all difference vectors to be as equal as possible.
4. At inference, only the fine-tuned model is used, with no additional computational overhead.

Intuition: If all difference vectors are equal, fine-tuning amounts to a **uniform translation** in the embedding space, leaving the relative positions (geometric structure) among embeddings entirely intact.

### Key Designs

1. **Difference Vector**: Defined as the difference between the embeddings of the same sample under the fine-tuned and pre-trained models. This is the core concept of DiVE—controlling geometric structure changes indirectly by constraining these difference vectors. This is more flexible than directly constraining embeddings (as in SnD), since it allows non-zero difference vectors, leaving room for adaptation to the target task.

2. **Average Vector Loss (AVL)**: Computes an exponential moving average $\mathbf{m}$ ($\alpha=0.99$) of all difference vectors, then constrains each difference vector to approach this mean. This preserves the **global geometric structure** of the embedding space. Using EMA rather than batch-level mean keeps the average vector stable throughout training.

3. **Pairwise Vector Loss (PVL)**: Constrains the difference vectors of each paired image-text sample to be equal: $\|u(\mathbf{x}_j^{ref}) - v(\mathbf{t}_j^{ref})\|^2$. This preserves **local geometric structure**—ensuring that image-text alignment is not disrupted after fine-tuning, which is critical for classification at inference.

### Loss & Training

$$\mathcal{L}_{final} = \mathcal{L}_{cl} + \lambda \cdot (\mathcal{L}_{avl} + \mathcal{L}_{pvl})$$

- $\mathcal{L}_{cl}$: Standard contrastive loss computed on target data (inherited from FLYP)
- $\mathcal{L}_{avl} = \frac{1}{B'}\sum_{j=1}^{B'}(\|u(\mathbf{x}_j^{ref}) - \mathbf{m}\|^2 + \|v(\mathbf{t}_j^{ref}) - \mathbf{m}\|^2)$
- $\mathcal{L}_{pvl} = \frac{1}{B'}\sum_{j=1}^{B'}\|u(\mathbf{x}_j^{ref}) - v(\mathbf{t}_j^{ref})\|^2$
- $\lambda = 1000$ (selected from {100, 500, 1000, 2500, 5000}, universally applied across all datasets)
- EMA coefficient $\alpha = 0.99$; initial mean vector is the zero vector
- Optimizer: AdamW with cosine learning rate schedule and 500-step warmup

## Key Experimental Results

### ImageNet ID/OOD Performance (ViT-B/16)

| Method | ImageNet (ID) | IN-V2 | IN-R | IN-A | IN-Sketch | ObjectNet | OOD avg |
|--------|-------------|-------|------|------|-----------|-----------|---------|
| Pre-trained | 68.3 | 61.9 | 77.7 | 50.0 | 48.3 | 55.4 | 58.7 |
| Vanilla FT | 81.3 | 71.2 | 66.1 | 37.8 | 46.1 | 53.3 | 54.9 |
| LP-FT | 81.7 | 72.1 | 73.5 | 47.6 | 50.3 | 58.2 | 60.3 |
| FLYP | 82.2 | 73.0 | 71.5 | 48.4 | 49.7 | 54.8 | 59.5 |
| ARF | 82.7 | 72.8 | 75.6 | 50.3 | 51.8 | 55.8 | 61.3 |
| SnD | 82.4 | 73.2 | 74.3 | 50.0 | 51.4 | 54.5 | 60.7 |
| **DiVE** | **82.5** | **73.8** | **77.3** | **54.9** | **52.9** | **56.9** | **63.2** |

### Zero-Shot Performance (ImageNet as Target Task, ViT-B/16)

| Method | Caltech | Flowers | Food | SUN | DTD | Aircraft | Cars | Pets | EuroSAT | UCF | avg |
|--------|---------|---------|------|-----|-----|----------|------|------|---------|-----|-----|
| Pre-trained | 89.5 | 67.0 | 84.5 | 63.0 | 46.3 | 22.6 | 59.0 | 87.7 | 46.3 | 65.7 | 63.2 |
| FLYP | 87.6 | 39.7 | 63.3 | 52.6 | 36.8 | 8.0 | 32.3 | 77.2 | 38.2 | 59.0 | 49.5 |
| ARF | 88.6 | 46.4 | 74.5 | 63.8 | 40.4 | 13.9 | 44.7 | 83.1 | 35.8 | 64.6 | 55.6 |
| SnD | 89.1 | 49.5 | 69.6 | 58.7 | 38.7 | 11.0 | 42.5 | 79.6 | 42.7 | 62.6 | 54.4 |
| **DiVE** | **88.4** | **66.0** | **84.3** | **64.7** | **47.0** | **22.1** | **55.5** | **88.4** | **51.4** | **68.9** | **63.7** |

DiVE's average zero-shot performance surpasses ARF by **8.1 percentage points**, nearly recovering to pre-trained levels (63.7 vs. 63.2), while improving ID performance by 14.2 points.

### iWildCam / FMoW Results

| Method | iWildCam ID | iWildCam OOD | iWildCam ZS | FMoW ID | FMoW OOD | FMoW ZS |
|--------|------------|-------------|------------|---------|---------|---------|
| FLYP | 52.2 | 35.6 | 51.0 | 68.6 | 41.3 | 45.1 |
| FLYP+replay | 48.5 | 35.8 | 62.3 | 68.7 | 41.2 | 63.0 |
| SnD | 50.6 | 37.0 | 60.4 | 67.0 | 41.4 | 56.6 |
| **DiVE** | **53.1** | **37.2** | **65.3** | **69.9** | **42.3** | **65.1** |

### ViT-L/14 Results

| Method | ImageNet (ID) | OOD avg | ZS avg |
|--------|-------------|---------|--------|
| FLYP | 86.0 | 71.5 | 55.4 |
| FLYP+replay | 85.8 | 72.6 | 65.9 |
| SnD | 86.0 | 73.2 | 61.2 |
| **DiVE** | **86.1** | **74.5** | **70.1** |

### RSA Geometric Structure Preservation Evaluation

| Method | RSA Correlation Score |
|--------|-----------------------|
| FLYP | 0.825 |
| FLYP + replay (ARF proxy) | 0.850 |
| SnD | 0.847 |
| FLYP + AVL | 0.978 |
| FLYP + PVL | 0.976 |
| **DiVE** | **0.981** |

### Ablation Study

- **AVL vs. PVL**: AVL contributes more (OOD +3.4, ZS +13.4 vs. PVL's +3.1, +13.2), but the two are complementary and their combination yields the best results (OOD +3.7, ZS +14.2).
- **EMA coefficient α**: Performance degrades at α=0 (OOD 61.7), with α=0.99 being optimal (OOD 62.9)—a stable mean vector is essential.
- **Cosine vs. vector constraint**: Cosine-similarity-based constraints (similar to CyCLIP) achieve RSA of only 0.949, inferior to difference vector constraints (0.981)—cosine similarity captures only angular relationships, lacking directional information.
- **Reference dataset scale**: Performance increases from Flickr8K (8K) → COCO (118K) → CC3M (3M), though even small datasets outperform baselines.
- **Weight ensembling**: DiVE + ensemble further improves performance (82.5→82.6 ID, 63.2→63.5 OOD, 63.7→64.6 ZS).

## Highlights & Insights

- **Novel Starting Point**: Approaching robust fine-tuning from the perspective of embedding geometry preservation provides a clear theoretical intuition—"fine-tuning as uniform translation" suffices to preserve geometric structure.
- **Simple yet Effective Method**: Only two straightforward L2 losses (AVL and PVL) are required, with no complex architectural modifications, making the method easy to implement.
- **Remarkable Zero-Shot Recovery**: After fine-tuning, the average zero-shot performance drops by only 2.3 points (63.7 vs. pre-trained 63.2), whereas FLYP drops by 13.7 points—a substantial practical advantage.
- **Compelling RSA Analysis**: The hypothesis of "preserving geometric structure" is directly validated with a quantitative metric (RSA score of 0.981), rather than relying solely on downstream performance.
- **Comparison with SnD Reveals a Key Insight**: SnD constrains difference vectors to zero (i.e., $f_{\theta^{ft}}(\mathbf{x}) = f_{\theta^{pre}}(\mathbf{x})$), effectively freezing the image encoder's behavior on reference data. DiVE permits non-zero but equal difference vectors, offering a better flexibility–preservation trade-off, and also constrains both image and text encoders, unlike SnD.

## Limitations & Future Work

- **High Computational Cost**: GPU memory increases from 117K MB to 321K MB, and training time grows from 35.9h to 58.9h (due to additional forward/backward passes on the reference dataset), which may become a bottleneck for larger models.
- **Dependence on a Reference Dataset**: The method requires a large-scale image-text dataset such as CC3M, which may be inconvenient to obtain and store in certain application scenarios.
- **Lack of Theoretical Analysis**: The paper provides only intuitive explanations for why equal difference vectors preserve geometric structure; no rigorous theoretical proof is given (the authors acknowledge this limitation in the conclusion).
- **Prompt Learning Methods Not Considered**: Insufficient comparison with prompt learning methods such as CoOp/CoCoOp, which may be more practical in certain settings.
- **Classification Tasks Only**: All experiments involve classification; the effectiveness on more complex downstream tasks such as detection and segmentation remains unverified.
- **$\lambda=1000$ Appears Large**: The regularization loss is weighted 1000× relative to the contrastive loss, implying that structure preservation is far more important than target adaptation; better scheduling strategies may exist.

## Related Work & Insights

**vs. FLYP**: DiVE augments FLYP with geometric structure constraints. FLYP fine-tunes both encoders with contrastive loss alone and has no explicit regularization, resulting in severe geometric distortion (RSA 0.825). DiVE preserves geometry almost perfectly (RSA 0.981), yielding OOD +3.7 and ZS +14.2.

**vs. ARF**: ARF uses contrastive loss on a reference dataset as experience replay, which is essentially an empirical replay strategy. It improves zero-shot performance but to a limited extent (ZS 55.6 vs. DiVE 63.7). The key distinction is that ARF does not explicitly constrain geometric structure (RSA only 0.850), whereas DiVE directly optimizes for its preservation.

**vs. SnD**: SnD constrains difference vectors to zero (i.e., $f_{\theta^{ft}}(\mathbf{x}) = f_{\theta^{pre}}(\mathbf{x})$), which is equivalent to completely freezing the image encoder's behavior on the reference data. This overly strict constraint limits adaptation to the target data. DiVE permits non-zero but equal difference vectors, achieving a better balance between flexibility and preservation. Additionally, SnD constrains only the image encoder, whereas DiVE constrains both image and text encoders.

**Connections and Transferable Ideas**:
- **Relation to Domain Generalization**: Although not a conventional DG method, the core idea—preserving the geometric structure of representations while adapting to a new domain—resonates with learning domain-invariant representations in DG. The equal difference vector constraint is essentially a form of structured knowledge distillation.
- **Transferable Paradigm**: The concept of difference vector equalization may be applicable to other scenarios requiring large model fine-tuning, such as LLM instruction tuning and adapter tuning, where the key idea is "allow movement but preserve relative positions."
- **Connection to Continual Learning**: The proposed method is fundamentally concerned with retaining the structure of prior knowledge during fine-tuning (new task learning), which is highly relevant to knowledge retention in continual learning, yet employs a more flexible mechanism (equality constraint vs. zero constraint).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Addressing robust fine-tuning from the perspective of embedding geometry preservation is novel, though the method is essentially regularization added on top of the FLYP framework and is not particularly groundbreaking in form.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple datasets (ImageNet/iWildCam/FMoW), multiple architectures (ViT-B/16, ViT-L/14), detailed ablation studies, RSA analysis, statistical significance tests, reference dataset ablations, and weight ensembling experiments—extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The paper is clearly structured with thorough motivation; the narrative from problem identification (RSA analysis) to solution to hypothesis validation forms a complete and coherent story.
- **Value**: ⭐⭐⭐⭐ — Substantial progress on the important problem of robust CLIP fine-tuning, with particularly strong practical value from the large zero-shot performance gains; however, computational overhead and the absence of a public code release limit real-world impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Self-Captioning Multimodal Interaction Tuning: Amplifying Exploitable Redundancies for Robust Vision Language Models](../../ICML2026/multimodal_vlm/self-captioning_multimodal_interaction_tuning_amplifying_exploitable_redundancie.md)
- [\[CVPR 2026\] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](../../CVPR2026/multimodal_vlm/trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [\[AAAI 2026\] HeadHunt-VAD: Hunting Robust Anomaly-Sensitive Heads in MLLM for Tuning-Free Video Anomaly Detection](headhunt-vad_hunting_robust_anomaly-sensitive_heads_in_mllm_.md)
- [\[CVPR 2026\] AGFT: Alignment-Guided Fine-Tuning for Zero-Shot Adversarial Robustness of Vision-Language Models](../../CVPR2026/multimodal_vlm/agft_alignment-guided_fine-tuning_for_zero-shot_adversarial_robustness_of_vision.md)
- [\[ICLR 2026\] Directional Embedding Smoothing for Robust Vision Language Models](../../ICLR2026/multimodal_vlm/directional_embedding_smoothing_for_robust_vision_language_models.md)

</div>

<!-- RELATED:END -->

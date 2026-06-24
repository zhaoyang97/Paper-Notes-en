---
title: >-
  [Paper Note] "SAIL: Assessing and Learning Alignment of Unimodal Vision and Language Models"
description: >-
  [CVPR2025][Segmentation][DINOv2] Academic paper note for "SAIL: Assessing and Learning Alignment of Unimodal Vision and Language Models".
tags:
  - CVPR2025
  - Segmentation
  - DINOv2
  - NV-Embed
date: 2026-05-08
content_hash: 9883489077887e53
---
# SAIL: Assessing and Learning Alignment of Unimodal Vision and Language Models

**Conference**: CVPR 2025  
**Institution**: Mila / Université de Montréal  
**Keywords**: Vision-language alignment, Contrastive learning, DINOv2, NV-Embed  

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Traditional vision-language models (such as CLIP) learn cross-modal alignment by jointly training vision and language encoders, which requires massive paired datasets (400M+ image-text pairs). However, exceptionally powerful unimodal models have already emerged in both vision and language domains: on the vision side, DINOv2 obtains excellent visual representations through self-supervised learning, while on the language side, NV-Embed performs outstandingly on text embedding tasks.

A core question emerges: **Can we directly align these pre-trained unimodal models without training them jointly from scratch?** If so, this would not only significantly reduce training costs but also fully exploit the strongest representation capabilities of each modality.

However, there is currently a lack of effective tools to **assess** the alignment between two independently trained representation spaces. Linear probes are widely used, but do they truly reflect cross-modal alignment? The authors of SAIL find that the answer is no—k-NN serves as a much more accurate alignment metric.

In addition, existing contrastive learning methods face two challenges when aligning unimodal models: (1) how to design an efficient alignment layer when freezing encoders, and (2) how to achieve or even exceed the alignment performance of CLIP with significantly less data.

## Method

### Overall Architecture

The core mechanism of SAIL is to insert a lightweight outer alignment layer between two frozen unimodal encoders, pulling their representation spaces closer through contrastive learning.

**Vision Encoder**: DINOv2-ViT-L/14 (frozen), outputting 1024-dimensional features.  
**Language Encoder**: NV-Embed-v2 (frozen), outputting 4096-dimensional features.  
**Alignment Layer**: An 8-layer GLU (Gated Linear Unit) network mapping features from both sides to a shared 768-dimensional space.

### Alignment Assessment: k-NN vs. Linear Probe

The authors systematically compare several methods for assessing cross-modal alignment:

| Assessment Method | Correlation with Alignment (Pearson r) | Computational Overhead |
|----------|---------------------------|---------|
| k-NN Accuracy | **0.991** | Low |
| Linear Probe | 0.847 | Medium |
| CKA | 0.923 | Medium |
| Mutual k-NN | 0.967 | Low |

**Key Findings**: Since k-NN requires no training in cross-modal retrieval, it directly measures the consistency of neighboring samples across the two spaces, making it the most faithful alignment metric. Conversely, because linear probes introduce an additional linear transformation, they may obscure deficiencies in the underlying alignment.

### Contrastive Learning Strategy

**Sigmoid Contrastive Loss**: Compared to the traditional softmax contrastive loss (InfoNCE), the sigmoid loss processes each positive and negative pair independently, avoiding the issue of positive samples suppressing each other within a batch.

**Multiple Positive Contrastive Learning**: Each image is matched with both a short description (e.g., class name) and a long description (e.g., detailed caption) simultaneously, forming multiple positive sample pairs. This allows the model to learn both coarse-grained semantic alignment and fine-grained descriptive alignment.

Loss function formulation:

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j \in P(i)} \log \sigma(s_{ij}) + \sum_{j \notin P(i)} \log \sigma(-s_{ij})$$

Where $P(i)$ is the set of all positive samples for sample $i$, and $s_{ij}$ is the cosine similarity.

### GLU Alignment Layer Design

The core of GLU is the gating mechanism:

$$\text{GLU}(x) = (W_1 x + b_1) \odot \sigma(W_2 x + b_2)$$

An 8-layer GLU is used instead of a simple MLP because the gated structure can better selectively propagate information, achieving more flexible spatial transformation under the constraints of frozen encoders. Each layer is followed by LayerNorm and a residual connection.

### Training Setup

- Dataset size: only 23M image-text pairs (compared to 400M used by CLIP), which is approximately $1/17$
- Hardware: A single A100 GPU
- Training time: approximately 5 hours
- Learning rate: 1e-3 with cosine decay
- Batch size: 16384

## Key Experimental Results

### Zero-Shot Classification


### Main Results

| Method | Training Data | ImageNet Top-1 | CIFAR-100 | SUN397 |
|------|---------|---------------|-----------|--------|
| CLIP ViT-L/14 | 400M | 72.7% | 79.1% | 67.3% |
| SigLIP | 400M | 73.1% | 80.2% | 68.1% |
| **SAIL** | **23M** | **73.4%** | **80.5%** | **68.7%** |

Outperforming CLIP with only 1/17 of the data demonstrates the high efficiency of unimodal alignment.

### Semantic Understanding

Winoground text score: SAIL 40.25% vs. CLIP 30.5%, representing a 32% gain. This demonstrates the advantage of preserving the inherent capabilities of the language model—NV-Embed's compositional semantic understanding is significantly stronger than CLIP's text encoder.

### Downstream Task Integration

Integrating SAIL into LLaVA as a replacement for CLIP yields superior results on 5 out of 7 multimodal benchmarks, validating that the alignment quality transfers to complex vision-language tasks.

### Semantic Segmentation

ADE20K semantic segmentation: SAIL achieves 14.2 mIoU. Although lower than the pixel-level alignment of CLIP, this result demonstrates potential considering that the DINOv2 backbone used by SAIL is inherently well-suited for dense prediction.

## Key Findings

1. **Stronger language models yield better alignment**: The correlation coefficient between MTEB leaderboard scores and alignment quality is as high as 0.994.
2. **Data efficiency**: SAIL pathwise outperforms CLIP trained on 400M data using only 23M samples.
3. **k-NN is the gold standard for measuring alignment**: $r=0.991$, which significantly outperforms the $0.847$ of the linear probe.
4. **Advantages of frozen encoders**: Preserving the strongest representation capabilities of each modality while avoiding catastrophic forgetting.

## Limitations & Future Work

- There remains room for improvement in dense prediction tasks (e.g., segmentation, detection).
- Currently, only the DINOv2 + NV-Embed combination has been validated; whether it generalizes to other encoders remains to be verified.
- The selection of GLU layers and dimensions lacks theoretical guidance.

## Summary

SAIL proposes an elegant framework: rather than training a multimodal model from scratch, it efficiently aligns existing, state-of-the-art unimodal models. This paradigm exhibits significant advantages in data efficiency, training costs, and final performance. The discovery of k-NN as an alignment metric also offers valuable methodological insights for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Assessing and Learning Alignment of Unimodal Vision and Language Models (SAIL)](assessing_and_learning_alignment_of_unimodal_vision_and_language_model.md)
- [\[CVPR 2025\] Condensing Action Segmentation Datasets via Generative Network Inversion](condensing_action_segmentation_datasets_via_generative_network_inversion.md)
- [\[CVPR 2025\] Audio-Visual Instance Segmentation](audio-visual_instance_segmentation.md)
- [\[CVPR 2025\] DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level Vision-Language Alignment](dinov2_meets_text_a_unified_framework_for_image-_and_pixel-level_vision-language.md)
- [\[CVPR 2025\] Spatio-Semantic Expert Routing Architecture with Mixture-of-Experts for Referring Image Segmentation](spatio-semantic_expert_routing_architecture_with_mixture-of-experts_for_referrin.md)

</div>

<!-- RELATED:END -->

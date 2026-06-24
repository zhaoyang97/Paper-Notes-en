---
title: >-
  [Paper Note] Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment
description: >-
  [CVPR 2025][Multimodal VLM][Frozen Encoders] A novel vision-language alignment framework is proposed: by freezing pre-trained unimodal vision (DINOv2) and language (All-Roberta-Large) encoders and only training lightweight MLP projection layers to achieve multimodal alignment, it reaches or exceeds CLIP-level performance with a 20x reduction in data and a 65x reduction in compute.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Frozen Encoders"
  - "Projection Layer Alignment"
  - "CKA"
  - "Concept-Balanced Dataset"
  - "Multimodal Learning"
date: 2026-05-08
content_hash: 8b0633be2c406e7d
---

# Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment

**Conference**: CVPR 2025  
**arXiv**: [2409.19425](https://arxiv.org/abs/2409.19425)  
**Code**: [GitHub](https://github.com/mayug/freeze-align)  
**Area**: Multilingual Translation  
**Keywords**: Frozen Encoders, Projection Layer Alignment, CKA, Concept-Balanced Dataset, Multimodal Learning

## TL;DR

A novel vision-language alignment framework is proposed: by freezing pre-trained unimodal vision (DINOv2) and language (All-Roberta-Large) encoders and only training lightweight MLP projection layers to achieve multimodal alignment, it reaches or exceeds CLIP-level performance with a 20x reduction in data and a 65x reduction in compute.

## Background & Motivation

**Background**:
Contrastive multimodal models like CLIP achieve modal alignment by training vision and text encoders from scratch on 400 million image-text pairs. They have become the standard backbone models for vision-language applications, demonstrating powerful zero-shot capabilities.

**Limitations of Prior Work**:
1. CLIP-style training requires enormous computational resources (approx. 20,000 GPU hours on A100) and data (400 million image-text pairs).
2. Due to the global training objectives, the vision encoders of CLIP perform worse on pixel-level tasks (such as segmentation) compared to unimodal vision models (such as DINOv2).
3. The text encoders of CLIP only support English and have a maximum of 77 tokens, which limits their capability in multilingual and long-text scenarios.
4. Existing efficient methods like LiT still require large-scale computation, while smaller-scale methods like LiLT demonstrate insufficient performance.

**Key Challenge**:
Training powerful multimodal models requires massive computational and data costs, yet many domain-specialized unimodal encoders already exist (such as DINOv2 for visual localization, or multilingual text encoders). How can these capabilities be repurposed at low cost?

**Goal**:
To align existing powerful unimodal encoders and obtain a multimodal model comparable to CLIP at an extremely low cost (by only training projection layers).

**Key Insight**:
Based on the recent discovery that highly trained unimodal vision and language encoders exhibit high semantic similarity (high CKA scores), implying that a simple projection transformation could achieve alignment.

**Core Idea**:
Select encoder pairs with the highest CKA semantic similarity, construct a concept-rich dataset, and train simple projection layers to achieve CLIP-level multimodal alignment.

## Method

### Overall Architecture

The framework consists of three steps: (1) Encoder Pair Selection—using the CKA metric to evaluate the semantic similarity of different vision-language encoder pairs on the COCO validation set, selecting the pair with the highest CKA (DINOv2-Large + All-Roberta-Large-v1, CKA=0.69); (2) Dataset Construction—constructing a concept-dense 20M training set from data sources such as LAION-400M via concept-balanced sampling; (3) Projection Layer Training—training only the projection layers with 11.5M parameters using the InfoNCE contrastive loss.

### Key Designs

1. **CKA-based Encoder Pair Selection**:
    - **Function**: Efficiently screen the most suitable vision-language encoder combinations for alignment.
    - **Mechanism**: Compute CKA scores of different encoder pairs on 5,000 COCO image-text pairs. It is found that CKA is strongly positively correlated with downstream retrieval performance after alignment (Figure 1, Figure 4). Thus, CKA can serve as an efficient metric for encoder selection.
    - **Design Motivation**: Avoid the high cost of training all encoder combinations pair-by-pair; CKA provides a training-free prior.

2. **Concept-Balanced Dataset Construction**:
    - **Function**: Construct a high-quality dataset that supports efficient projection layer training.
    - **Mechanism**:
        - Collect approximately 3,000 unique concepts from datasets like ImageNet.
        - Build few-shot image prototypes for each concept (encoded using CLIP ViT-Large).
        - Perform concept-balanced sampling from LAION-400M with 2,000 samples per concept, prioritizing rare concepts.
        - Combine with highly semantically aligned datasets like CC3M, CC12M, and SBU to form the 20M MIX-CLASS-Collected dataset.
    - **Design Motivation**: High concept coverage ensures dense coverage of various regions in the unimodal space (beneficial for classification), while high semantic alignment is beneficial for retrieval; both are indispensable.

3. **Lightweight Token Projector Architecture**:
    - **Function**: Achieve modal alignment with minimal parameters.
    - **Mechanism**:
        - Vision side: Apply separate Token Projectors (residual structure with linear and non-linear branches) to local tokens and the CLS token. Local tokens share weights, while the CLS token has separate weights.
        - Text side: Use a Token Projector to process tokens, followed by a 2-layer MLP as global projection.
        - The average of all adapted local tokens is added to the adapted CLS token to form the final global embedding.
    - **Design Motivation**: The DINO objective of DINOv2 acts on the CLS token while the iBOT objective acts on patch tokens, requiring separate processing. Since the embedding space of the text encoder is far from the vision space, an additional global projection is needed.

### Loss & Training

- Standard InfoNCE contrastive loss.
- Train only the projection layers (11.5M parameters), keeping DINOv2-Large (300M) and ARL text encoder (355M) frozen.
- 8×A100 GPUs, approximately 50 hours of training (65x reduction compared to CLIP's 21,845 GPU hours).
- 20M training data (20x reduction compared to CLIP's 400M).

## Key Experimental Results

### Main Results

**Zero-Shot Classification Transfer**

| Model | Data Volume | ImageNet | ImageNetv2 | Caltech | Pets | Cars | Average |
|------|--------|----------|------------|---------|------|------|------|
| OpenAI CLIP ViT-L | 400M | 75.3 | 69.8 | 92.6 | 93.5 | 77.3 | - |
| LAION CLIP ViT-L | 400M | 72.7 | 65.4 | 92.5 | 91.5 | 89.6 | - |
| DINOv2-ARL (Ours) | **20M** | **76.3** | **69.2** | **92.8** | 92.1 | 73.9 | - |

**Image-Text Retrieval (Flickr30K / COCO)**

| Model | Flickr I2T | Flickr T2I | COCO I2T | COCO T2I |
|------|-----------|-----------|---------|---------|
| OpenAI CLIP ViT-L | 85.2 | 64.9 | 56.3 | 36.5 |
| LAION CLIP ViT-L | 87.6 | 70.2 | 59.7 | 43.0 |
| DINOv2-ARL (Ours) | **87.5** | **74.1** | **60.1** | **45.1** |

### Ablation Study

**Projector Architecture Ablation (ImageNet Zero-Shot Accuracy)**

| Vision Local | Vision CLS | Text Local | Text Global | ImageNet |
|----------|---------|---------|---------|----------|
| token | identity | identity | identity | 68.84 |
| token | identity | token | mlp | 72.15 |
| identity | token | token | mlp | 75.53 |
| token | token | token | mlp | **76.12** |

**Dataset Ablation**

| Data Source | Data Volume | ImageNet | Flickr I2T | Flickr T2I |
|--------|--------|----------|-----------|-----------|
| LAION-CLASS-Collected | 6M | 76.12 | 52.70 | 42.48 |
| CC3M+CC12M+SBU | 14M | 54.17 | 85.30 | 72.44 |
| Both | 20M | 75.04 | 81.32 | 71.38 |
| Both + longer | 20M | **76.30** | **87.54** | **74.17** |

### Key Findings

1. **Projection layer alignment can match CLIP**: Training only 1% of the parameters (11.5M/670M) achieves 76.3% ImageNet zero-shot accuracy, surpassing OpenAI CLIP (75.3%) and LAION CLIP (72.7%).
2. **CKA is an effective metric for encoder selection**: CKA shows a clear positive correlation with the final retrieval performance. The CKA of DINOv2+ARL is 0.69, the highest among non-CLIP text encoders.
3. **Concept coverage and semantic alignment are both indispensable**: High-coverage-only data leads to good classification but poor retrieval, whereas high-alignment-only data leads to good retrieval but poor classification; combining them yields the best of both worlds.
4. **Unimodal capabilities are preserved after alignment**:
    - DINOv2's localization capability is preserved: Zero-shot segmentation on Pascal VOC achieves a mIoU of 31.37% (CLIP is only 23.46%).
    - MpNet's multilingual capability is preserved: Training only on English data surpasses specialized multilingual models in multilingual retrieval.
    - ARL's long-text capability is preserved: Retrieval performance continuously improves beyond 77 tokens up to 200–300 tokens.

## Highlights & Insights

1. **Paradigm Shift**: Demonstrates that CLIP-level multimodal alignment does not necessarily require training encoders from scratch; freezing existing powerful unimodal encoders and training projection layers is a feasible and highly efficient alternative.
2. **Democratization of Compute**: The 65x compute reduction and 20x data reduction make the development of multimodal models far more accessible to academia.
3. **Prospect of Flexible Combinations**: Unimodal encoders can be flexibly selected based on needs—e.g., multilingual text encoders to multilingual VLMs, long-context encoders to long-text VLMs, or 3D encoders to 3D-language models.
4. **CKA as a Pairing Metric**: Provides a simple and effective prior to evaluate which encoder pairs are easy to align, avoiding trial-and-error.
5. **Superiority of Unimodal Features**: Encoders trained purely on vision, such as DINOv2, outperform CLIP's vision encoders in core visual tasks like localization, and freezing them preserves this advantage.

## Limitations & Future Work

1. The framework assumes that powerful pre-trained unimodal encoders are already available; without such pre-trained encoders, this method is not applicable.
2. CKA selection does not yet account for differences in requirements across different tasks (the highest CKA does not guarantee optimal performance on all downstream tasks).
3. The capacity of the projection layer is limited, which may not bridge the gap between encoder pairs with lower CKA.
4. Training was only conducted on 20M data; whether scaling up can yield further improvements has not been explored.
5. Integration with LLMs (e.g., the LLaVA paradigm) and the performance as a visual feature extractor remain to be validated.

## Related Work & Insights

- **CLIP/ALIGN**: Representative models for contrastive multimodal training from scratch, serving as the primary baseline of comparison in this paper.
- **LiT（Locked Image Tuning）**: Freezes the vision encoder and only trains the text encoder, though the compute remains relatively large.
- **DINOv2**: Unsupervised self-supervised vision encoder with powerful global and local features.
- **Platonic Representation Hypothesis**: Well-trained models of different modalities tend to converge to a shared semantic structure.
- **Insight**: The semantic gap between modalities might be smaller than previously assumed; a simple projection transformation is sufficient to bridge them.

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation](../../ACL2025/multimodal_vlm/flagevalmm_a_flexible_framework_for_comprehensive_multimodal_model_evaluation.md)
- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](multimodal_autoregressive_pre-training_of_large_vision_encoders.md)
- [\[CVPR 2025\] MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders](move-kd_knowledge_distillation_for_vlms_with_mixture_of_visual_encoders.md)
- [\[ACL 2025\] Harnessing PDF Data for Improving Japanese Large Multimodal Models](../../ACL2025/multimodal_vlm/harnessing_pdf_data_for_improving_japanese_large_multimodal_models.md)
- [\[ICML 2025\] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models](../../ICML2025/multimodal_vlm/the_devil_is_in_the_details_tackling_unimodal_spurious_correlations_for_generali.md)

</div>

<!-- RELATED:END -->

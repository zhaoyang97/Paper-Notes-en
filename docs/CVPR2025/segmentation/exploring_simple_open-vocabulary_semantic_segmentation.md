---
title: >-
  [Paper Note] Exploring Simple Open-Vocabulary Semantic Segmentation
description: >-
  [CVPR 2025][Segmentation][Open-vocabulary segmentation] This paper proposes S-Seg, a minimalist open-vocabulary semantic segmentation model. Without relying on CLIP pre-training, annotated masks, or customized grouping encoders, S-Seg trains a MaskFormer using only pseudo-masks (generated from DINO K-Means clustering) and image-text contrastive loss. It achieves comparable performance to complex methods on Pascal VOC, Pascal Context, and COCO…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Open-vocabulary segmentation"
  - "pseudo-mask"
  - "MaskFormer"
  - "image-text pairs"
  - "self-training"
date: 2026-05-08
content_hash: f8ac66afd2121075
---

# Exploring Simple Open-Vocabulary Semantic Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2401.12217](https://arxiv.org/abs/2401.12217)  
**Code**: [https://github.com/zlai0/S-Seg](https://github.com/zlai0/S-Seg)  
**Area**: Segmentation  
**Keywords**: Open-vocabulary segmentation, pseudo-mask, MaskFormer, image-text pairs, self-training

## TL;DR

This paper proposes S-Seg, a minimalist open-vocabulary semantic segmentation model. Without relying on CLIP pre-training, annotated masks, or customized grouping encoders, S-Seg trains a MaskFormer using only pseudo-masks (generated from DINO K-Means clustering) and image-text contrastive loss. It achieves comparable performance to complex methods on Pascal VOC, Pascal Context, and COCO, with self-training further boosting the average mIoU by 5.5%.

## Background & Motivation

**Background**: Open-vocabulary semantic segmentation (OVS) aims to assign semantic labels to every pixel using arbitrary text categories. Current dominant approaches usually rely on a combination of three strategies: (1) adaptation based on image-level vision-language models like CLIP; (2) training on annotated masks to learn pixel-level features; and (3) designing specialized grouping encoders (e.g., GroupViT) to group pixels into semantic regions.

**Limitations of Prior Work**: Each of these three strategies has limitations: CLIP adaptation methods inherit the limits of image-level models in pixel-level tasks (e.g., MaskCLIP generates noisy segmentations); relying on annotated masks restricts scalability; customized grouping encoders increase design complexity. More importantly, it remains unverified whether high-quality OVS can be achieved without relying on any of these strategies.

**Key Challenge**: The key challenge of open-vocabulary segmentation is pixel-level vision-language alignment, while obtaining large-scale pixel-level annotations is impractical. Existing approaches either "downsample" alignment signals from image-level models (causing signal loss), train on restricted annotated categories (limiting generalization), or design complex grouping mechanisms (reducing simplicity).

**Goal**: To demonstrate that a minimalist solution—using pseudo-masks to provide shape supervision and image-text pairs to provide semantic supervision to directly train a standard MaskFormer—can achieve competitive segmentation performance without utilizing CLIP, annotated masks, or customized encoders.

**Key Insight**: The authors observe that self-supervised ViT features from DINO already contain strong priors for object segmentation (high-quality pseudo-masks can be obtained simply via K-Means clustering), while web-crawled texts provide semantic supervision covering long-tail concepts. Combining these two freely available supervision signals allows decoupling mask supervision from semantic supervision to train the segmentation model.

**Core Idea**: Decoupling the supervision for open-vocabulary segmentation into mask supervision (from DINO pseudo-masks) and semantic supervision (from image-text contrastive learning), directly training a standard MaskFormer with these two weak supervision signals to align pixel-level features with language.

## Method

### Overall Architecture

During training, S-Seg takes image-text pairs as input. The image is passed through a MaskFormer to predict $N$ masks and their corresponding mask features. Simultaneously, a pseudo-mask generator produces $K$ pseudo-masks from the image to supervise the mask predictions. The text is encoded into text features by a language model to perform contrastive learning against the average of the mask features. During inference, given a list of candidate category names, the language model encodes the feature of each category, which is then used to compute cosine similarity with the mask features to classify each mask, ultimately yielding the final semantic segmentation map.

### Key Designs

1. **DINO-based Pseudo-Mask Generator**:

    - **Function**: Provides high-quality class-agnostic mask supervision to replace manual annotations.
    - **Mechanism**: Uses a DINO pre-trained ViT-S/8 to extract patch token features of the image, and then applies K-Means clustering ($K=8$) to these features. Each token is assigned to a cluster and reshaped back to the image dimensions to obtain pseudo-masks. The predicted $N$ masks are aligned with the $K$ pseudo-masks using bipartite matching, while the remaining $N-K$ unmatched masks are left unpenalized.
    - **Design Motivation**: Features generated by DINO's self-supervised learning possess natural discontinuities at object boundaries, which K-Means clustering can effectively capture. Experiments indicate that its oracle performance (78.8% VOC mIoU) far exceeds that of GroupViT (73.7%), with a very fast processing speed (0.002s for 128 samples). Not using ImageNet-supervised clustering (68.8%) or raw pixel clustering (49.5%) ensures a completely self-supervised setup.

2. **Semantic Supervision via Image-Text Contrastive Learning**:

    - **Function**: Learns the alignment between mask features and language to empower the model with open-vocabulary classification capabilities.
    - **Mechanism**: The average of the $N$ mask features is used as the global image representation, projected into a shared embedding space via a 2-layer MLP. Text is encoded using a 12-layer Transformer (trained from scratch), and the embedding of the [EOS] token is also projected into the shared space. A standard CLIP-style bidirectional contrastive loss $\mathcal{L}_{I \to T} + \mathcal{L}_{T \to I}$ is applied with a learnable temperature parameter. Negative samples are gathered across GPUs to increase contrastive efficiency.
    - **Design Motivation**: Image-text contrastive learning is an effective paradigm validated by works like CLIP. S-Seg performs contrastive learning directly on the mask features of MaskFormer to align pixel-level features with language naturally, unlike CLIP which aligns only image-level features and requires later adaptation.

3. **Self-Training Enhancement (S-Seg+)**:

    - **Function**: Leverages unlabeled images and category information from the target domain to boost performance further.
    - **Mechanism**: The trained S-Seg is used to generate pseudo-labels for the training set of the target dataset. Then, these pseudo-labels are used to train a UperNet (with an MAE pre-trained ViT backbone) via fully supervised semantic segmentation. This step utilizes two pieces of information available at test time: unlabeled images in the target domain and the list of candidate categories.
    - **Design Motivation**: Self-training exploits the domain distribution and category priors of the target domain, which helps rectify some prediction errors of S-Seg. Experiments show that this self-training boosts the average mIoU by 5.5% (37.1% → 42.6%), and continuously improves with larger training data scale.

### Loss & Training

The total loss is a weighted sum of the mask loss and the contrastive loss: $L = \lambda_{mask}\mathcal{L}_{mask} + \lambda_{contrastive}\mathcal{L}_{contrastive}$, where $\mathcal{L}_{mask} = \lambda_{dice}\mathcal{L}_{dice} + \lambda_{focal}\mathcal{L}_{focal}$. The hyperparameters are set to $\lambda_{mask} = 1.0, \lambda_{contrastive} = 1.0, \lambda_{dice} = 1.0, \lambda_{focal} = 20.0$. S-Seg utilizes a Swin-S backbone, a 6-layer Transformer decoder, and $N=64$ queries. It is trained for 30 epochs with the AdamW optimizer, a base learning rate of $5 \times 10^{-4}$, and a batch size of 4096. Training data includes CC3M + CC12M + RedCaps (up to 26M image-text pairs).

## Key Experimental Results

### Main Results

| Method | Supervision Type | P. VOC | P. Context | COCO | 3-Avg |
|------|---------|--------|-----------|------|-------|
| CLIP | text | 13.5 | 8.1 | 5.9 | 9.2 |
| MaskCLIP | text | 26.8 | 22.8 | 12.8 | 20.8 |
| GroupViT | text | 50.8 | 23.7 | 27.5 | 34.0 |
| SegCLIP | text | 52.6 | 24.7 | 26.5 | 34.6 |
| TCL | text | 55.0 | 30.4 | — | — |
| **S-Seg** | text | 53.2 | 27.9 | 30.3 | **37.1** |
| **S-Seg+** | text | 62.0 | 30.2 | 35.7 | **42.6** |

### Ablation Study—Data Scale and Self-Training

| Data | S-Seg VOC | S-Seg Ctx | S-Seg COCO | S-Seg+ VOC | S-Seg+ COCO |
|------|----------|----------|-----------|-----------|------------|
| 12M | 44.9 | 22.9 | 22.5 | 53.1 | 26.2 |
| 15M | 45.1 | 23.8 | 27.9 | 54.2 | 28.0 |
| 26M | 53.2 | 27.9 | 30.3 | 62.0 | 35.7 |

### Key Findings

- **High-quality OVS can be achieved without relying on CLIP**: S-Seg trains the text encoder and MaskFormer from scratch without any pre-trained VL models, yet still achieves competitive performance.
- **Simple baselines are ineffective**: Pseudo-mask + CLIP classification (6.6% avg) and pseudo-mask ViT (14.9% avg) are far inferior to S-Seg (30.1% avg), showing that joint multi-task learning is crucial.
- **Good data scalability**: From 12M to 26M, S-Seg improves by +8.3% on VOC, indicating the method can fully leverage more data.
- **Self-training is robust and effective**: Across all data scales and dataset settings, self-training yields a consistent improvement (average +5.5%).
- Under the evaluation protocol excluding background classes, S-Seg (81.8% VOC) outperforms ZegFormer (80.7%), an earlier method that required annotated masks.

## Highlights & Insights

- **The "minimalist" research philosophy is highly valuable**: In a field that is becoming increasingly complex, S-Seg verifies what is truly necessary by stripping away all "required" components (CLIP, GT masks, customized encoders). The conclusion is clear: pseudo-masks + language supervision + standard architectures are sufficient.
- **Clever decoupling of supervision signals**: Decoupling mask prediction supervision (from pseudo-masks) from semantic classification supervision (from text) allows both sources of supervision to scale independently. This stands in contrast to traditional coupled "mask + category label" supervision.
- **Surprisingly good quality of the pseudo-mask generator**: DINO K-Means clustering achieves an oracle mIoU of 78.8% on VOC, even outperforming GroupViT (73.7%) which was trained on VL pairs, which is a highly inspiring finding.

## Limitations & Future Work

- Performance on COCO (81 classes) is significantly lower than on VOC (21 classes), indicating that scaling up the number of categories poses a major challenge.
- The pseudo-mask generator may lack accuracy for small objects and highly textured scenes.
- While self-training is effective, it introduces an extra training stage and poses dependency on the target domain.
- The text encoder trained from scratch is limited in size (12 layers); a larger, pre-trained language model could provide better semantic understanding.
- Future work could consider incorporating diffusion models to generate more diverse pseudo-masks, or replacing DINO clustering with SAM as the pseudo-mask source.

## Related Work & Insights

- **vs GroupViT**: GroupViT designs a customized grouping token mechanism to naturally group segments from text supervision, whereas S-Seg achieves a similar goal using a standard MaskFormer under pseudo-mask supervision. S-Seg's approach is simpler and slightly superior.
- **vs ZegFormer**: ZegFormer uses CLIP + GT masks for training, while S-Seg uses neither. Interestingly, S-Seg generalizes better to unseen classes, suggesting that CLIP and GT masks may induce overfitting.
- **vs TCL**: TCL uses CLIP for region-level grounding and contrastive learning, performing slightly better than S-Seg on VOC. However, TCL relies on CLIP pre-training.
- **vs OpenSeg/OVSeg/SAN**: These methods use annotated masks, achieving a higher performance ceiling but requiring expensive annotation costs.

## Rating

- Novelty: ⭐⭐⭐⭐ The core idea is simple yet novel, but the individual components (pseudo-masks, contrastive learning, MaskFormer) are based on existing technologies.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated on three datasets, with multiple evaluation protocols, comparisons with over 16 methods, data-scale ablations, and simple baseline validations.
- Writing Quality: ⭐⭐⭐⭐ The paper is well-organized, and the analysis of simple baselines is highly convincing.
- Value: ⭐⭐⭐⭐ Establishes a strong yet simple baseline for open-vocabulary segmentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DPSeg: Dual-Prompt Cost Volume Learning for Open-Vocabulary Semantic Segmentation](dpseg_dual-prompt_cost_volume_learning_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Effective SAM Combination for Open-Vocabulary Semantic Segmentation](effective_sam_combination_for_open-vocabulary_semantic_segmentation.md)
- [\[CVPR 2025\] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation](semantic_library_adaptation_lora_retrieval_and_fusion_for_open-vocabulary_semant.md)
- [\[CVPR 2025\] Mask-Adapter: The Devil is in the Masks for Open-Vocabulary Segmentation](mask-adapter_the_devil_is_in_the_masks_for_open-vocabulary_segmentation.md)
- [\[CVPR 2025\] Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation
description: >-
  [ICCV 2025][Information Retrieval & RAG][Image-text matching] This paper proposes D2S-VSE, a two-stage training framework (dense-text pretraining + dense-to-sparse feature distillation fine-tuning) that enhances informat…
tags:
  - "ICCV 2025"
  - "Information Retrieval & RAG"
  - "Image-text matching"
  - "visual-semantic embedding"
  - "information capacity"
  - "dense-to-sparse distillation"
  - "cross-modal retrieval"
date: 2026-05-08
content_hash: 9081c3c305b9fb88
---

# Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation

**Conference**: ICCV 2025
**arXiv**: N/A (CVF OpenAccess)  
**Code**: [https://d2s-vse.github.io](https://d2s-vse.github.io)  
**Area**: Information Retrieval
**Keywords**: Image-text matching, visual-semantic embedding, information capacity, dense-to-sparse distillation, cross-modal retrieval

## TL;DR

This paper proposes D2S-VSE, a two-stage training framework (dense-text pretraining + dense-to-sparse feature distillation fine-tuning) that enhances information capacity in visual-semantic embeddings, addressing the core asymmetry in information density between image and text modalities for image-text matching.

## Background & Motivation

### Information Density Asymmetry

The central challenge in image-text matching lies in the inherent information density gap between visual and textual signals. Visual signals constitute a natural, objective recording of real-world phenomena and contain rich fine-grained details, whereas textual signals represent subjective human interpretations that are typically brief and sparse. A single image may be described in multiple ways from different perspectives, a phenomenon referred to as the "ambiguity problem" or "sparse annotation problem."

### Limitations of Prior Work

Existing methods (e.g., PCME, DivE, AVSE) attempt to address ambiguous samples by learning a set of embeddings, but this strategy reduces the **information capacity** of each individual embedding. When text embeddings have limited information capacity, they become susceptible to interference from hard negatives with locally similar semantics, degrading retrieval performance.

### Core Insight

**Why does information capacity matter?** The authors argue that text embeddings derived from dense image descriptions inherently possess greater information capacity. By aligning image embeddings with dense text during pretraining to increase their information capacity, and then distilling the information from dense text embeddings into sparse text embeddings during fine-tuning, both the visual and textual sides can be simultaneously enriched, enabling more precise image-text matching.

## Method

### Overall Architecture

D2S-VSE is a two-stage training framework:

1. **Pretraining stage**: LLaVA-generated dense captions are used to align images with dense text, enhancing the information capacity of visual-semantic embeddings.
2. **Fine-tuning stage**: The model is fine-tuned on image–sparse-text pairs while simultaneously augmenting sparse text embeddings via dense-to-sparse distillation.

During inference, only images and the sparse texts available in the dataset are required; dense texts are not needed, introducing no additional computational overhead.

### Stage 1: Dense-Text Pretraining

**Dense caption generation**: LLaVA is applied to generate detailed textual descriptions for each image in the dataset (prompt: "Describe this image in detail"), with hyperparameters set to Top-P=0.9, temperature=0.2, and max-new-tokens=500. One dense caption is generated per image, corresponding to one-fifth the number of sparse captions.

**Image–dense-text contrastive learning**: A triplet loss is used to align images with dense text:

$$\mathcal{L}_{pretrain} = \sum_{(I,T_d) \in D} [\alpha - S(I, T_d) + S(I, \hat{T}_d)]_+ + [\alpha - S(I, T_d) + S(\hat{I}, T_d)]_+$$

where $\alpha$ is the margin, and $\hat{I}$ and $\hat{T}_d$ denote the hardest negative samples respectively.

**Why does dense text enhance information capacity?** Dense captions provide comprehensive descriptions of object attributes, relations, actions, and contextual information present in the image. Training on data with comparable information density encourages the model to learn embeddings with greater information capacity, enabling a more thorough understanding of visual content.

### Stage 2: Sparse-Caption Fine-Tuning

Although pretraining enhances embedding information capacity, a train-test gap remains (dense text at training vs. sparse text at inference). Fine-tuning solely on sparse text does not enrich the semantic content of sparse text embeddings.

**Core Innovation: Transformer Decoder**

Inspired by masked signal modeling (MAE), sparse text is treated as a **masked version** of dense text. A Transformer decoder is designed to reconstruct dense text embeddings from sparse text embeddings via feature distillation:

$$\hat{t}_s = Decoder([w_s, m]) + t_s$$

where $w_s$ denotes the word embeddings of the sparse text, $m$ represents 100 learnable mask tokens, and $t_s$ is the sparse text embedding. The decoder outputs reconstructed tokens $\bar{m}$, which are averaged and added to $t_s$ to produce the final embedding $\hat{t}_s$.

**Why learnable mask tokens rather than alternative approaches?** These mask tokens serve to predict contextual information absent from the sparse text. By concatenating them with word embeddings and feeding the result into the decoder, they learn to infer the richer semantics contained in the dense text from the limited information in the sparse text, thereby increasing information capacity.

### Loss & Training

The fine-tuning stage jointly optimizes two objectives:

**Distillation loss** (negative cosine similarity):
$$\mathcal{L}_{distill}(t_d, \hat{t}_s) = 1 - \frac{t_d \cdot \hat{t}_s}{\|t_d\| \cdot \|\hat{t}_s\|}$$

**Alignment loss**:
$$\mathcal{L}_{align} = \sum_{(I,T_s) \in D} [\alpha - S(I, T_s) + S(I, \hat{T}_s)]_+ + [\alpha - S(I, T_s) + S(\hat{I}, T_s)]_+$$

**Total loss**: $\mathcal{L} = \mathcal{L}_{align} + \mathcal{L}_{distill}$

Training details: pretraining for 30 epochs (AdamW, lr=0.0005, batch=128); during fine-tuning, the dense text encoder is frozen and only the image and sparse text encoders are updated.

## Key Experimental Results

### Main Results

**Flickr30K (ViT-Base-224 + BERT-base)**:

| Method | Text R@1 | Text R@5 | Image R@1 | Image R@5 | rSum |
|--------|----------|----------|-----------|-----------|------|
| VSE++ | 71.8 | 92.8 | 59.4 | 84.7 | 496.1 |
| LAPS | 74.0 | 93.4 | 62.5 | 87.3 | 507.3 |
| AVSE | 76.0 | 94.6 | 62.7 | 88.4 | 512.3 |
| **D2S-VSE** | **82.8** | **96.1** | **68.5** | **91.3** | **531.9** |

D2S-VSE outperforms AVSE by 6.8% on Text R@1, 5.8% on Image R@1, and 19.6 on rSum.

**Flickr30K (Swin-Base-384 + BERT-base)**:

| Method | Text R@1 | Image R@1 | rSum |
|--------|----------|-----------|------|
| AVSE | 87.1 | 73.6 | 548.2 |
| **D2S-VSE** | **87.8** | **75.7** | **553.2** |

### Ablation Study

**Component-wise contribution of the two-stage framework (Flickr30K, ViT-Base-224)**:

| Pretrain | Align | Distill | Text R@1 | Image R@1 | rSum |
|----------|-------|---------|----------|-----------|------|
| ✓ | | | 56.2 | 41.7 | 425.2 |
| | ✓ | | 76.3 | 61.2 | 509.5 |
| ✓ | ✓ | | 79.2 | 66.8 | 525.7 |
| | ✓ | ✓ | 76.0 | 62.9 | 512.1 |
| ✓ | ✓ | ✓ | **82.8** | **68.5** | **531.9** |

Key finding: pretraining enhances the information capacity of image embeddings (+2.9% Text R@1); distillation further enhances that of sparse text embeddings (+3.6% Text R@1).

**Distillation function comparison**:

| Distillation Function | Text R@1 | Image R@1 | rSum |
|-----------------------|----------|-----------|------|
| L1 distance | 81.9 | 68.0 | 528.9 |
| L2 distance | 82.1 | 67.9 | 530.2 |
| **Negative cosine similarity** | **82.8** | **68.5** | **531.9** |

**Learnable token placement strategy**:

| Placement | Text R@1 | Image R@1 | rSum |
|-----------|----------|-----------|------|
| Prefix | 81.7 | 68.1 | 529.3 |
| Postfix | 82.0 | 67.8 | 529.5 |
| **Surround** | **82.8** | **68.5** | **531.9** |

The surround strategy performs best, as the position of sparse text within the dense text is unknown; surrounding tokens from both sides enables bidirectional contextual inference.

### Key Findings

1. **Information capacity is central**: Experiments clearly validate the independent contributions of pretraining (enhancing image embeddings) and distillation (enhancing text embeddings).
2. **Decoder design**: 100 learnable tokens, 4 layers, and 4 attention heads constitute the optimal configuration.
3. **Difference from MAE**: In MAE, overly deep decoders weaken encoder capacity; however, in D2S-VSE, retrieval performance depends on the decoder's predictive ability, so deeper decoders generally perform better (4 layers optimal).
4. **Backbone generalization**: The method consistently improves performance across diverse backbones including ViT, Swin Transformer, and ResNet+GRU.

## Highlights & Insights

1. **Unique information-capacity perspective**: Reframing image-text matching through the lens of information capacity provides a more fundamental analytical framework than the conventional "ambiguity problem" formulation.
2. **Zero inference overhead**: Dense text and the decoder are not involved during inference, incurring absolutely no additional computational cost.
3. **Novel application of masked signal modeling**: The MAE paradigm is innovatively adapted for cross-modal distillation by treating sparse text as a masked version of dense text.
4. **Concise yet powerful**: The core idea is straightforward, the implementation is efficient, and the empirical gains are substantial.

## Limitations & Future Work

1. The approach relies on LLaVA for dense caption generation; generation quality directly impacts pretraining effectiveness.
2. The current one-to-one image–dense-caption correspondence may lack flexibility; multi-perspective dense descriptions could be more beneficial.
3. Integration with larger-scale pretrained models (e.g., CLIP) has not been explored.
4. The decoder introduces additional training overhead during fine-tuning, despite zero inference overhead.

## Related Work & Insights

- **AVSE**: Proposes asymmetric multi-view embedding matching; serves as the strongest baseline.
- **Long-CLIP / LoTLIP**: Improve CLIP from a long-text perspective, but do not leverage dense-to-sparse distillation.
- **MAE**: The masked autoencoder paradigm is innovatively repurposed here for cross-modal distillation.
- Insight: The information-capacity alignment framework generalizes to other multimodal tasks where information asymmetry between modalities exists.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Aligning Information Capacity Between Vision and Language via Dense-to-Sparse Feature Distillation for Image-Text Matching](aligning_information_capacity_between_vision_and_language_via_dense_to_sparse_feature_distillation.md)
- [\[ICCV 2025\] ViLU: Learning Vision-Language Uncertainties for Failure Prediction](vilu_learning_vision-language_uncertainties_for_failure_prediction.md)
- [\[ICCV 2025\] LangBridge: Interpreting Image as a Combination of Language Embeddings](langbridge_interpreting_image_as_a_combination_of_language_embeddings.md)
- [\[ICCV 2025\] MonSTeR: a Unified Model for Motion, Scene, Text Retrieval](monster_a_unified_model_for_motion_scene_text_retrieval.md)
- [\[ICCV 2025\] External Knowledge Injection for CLIP-Based Class-Incremental Learning](external_knowledge_injection_for_clip-based_class-incremental_learning.md)

</div>

<!-- RELATED:END -->

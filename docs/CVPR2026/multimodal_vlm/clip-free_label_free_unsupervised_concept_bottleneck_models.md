---
title: >-
  [Paper Note] CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models
description: >-
  [CVPR 2026][Multimodal VLM][Concept Bottleneck Model] This paper proposes TextUnlock, a method that aligns the output distribution of an arbitrary frozen visual classifier to a vision-language correspondence space…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Concept Bottleneck Model"
  - "Interpretability"
  - "Knowledge Distillation"
  - "Unsupervised Classification"
  - "Vision-Language Alignment"
  - "Zero-Shot Image Captioning"
date: 2026-05-08
content_hash: c623425ea363985c
---

# CLIP-Free, Label-Free, Unsupervised Concept Bottleneck Models

**Conference**: CVPR 2026
**arXiv**: [2503.10981](https://arxiv.org/abs/2503.10981)
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: Concept Bottleneck Model, Interpretability, Knowledge Distillation, Unsupervised Classification, Vision-Language Alignment, Zero-Shot Image Captioning

## TL;DR

This paper proposes TextUnlock, a method that aligns the output distribution of an arbitrary frozen visual classifier to a vision-language correspondence space, enabling the construction of a fully unsupervised Concept Bottleneck Model (U-F²-CBM) that requires no CLIP, no labels, and no trained linear probes. U-F²-CBM surpasses supervised CLIP-based CBMs across 40+ models.

## Background & Motivation

**Value of Concept Bottleneck Models (CBMs)**: CBMs map dense features to human-interpretable concept activations, which are then linearly combined to predict class labels. They are important interpretability tools, yet existing methods rely heavily on CLIP to provide image-concept annotations.

**Drawbacks of CLIP Dependency**: When CLIP is used to generate concept annotations, the CBM is anchored to the CLIP embedding space. Legacy models must then be explained through CLIP's similarity-based concepts rather than their own learned representations, and CLIP's biases (e.g., typographic bias) are introduced.

**Irreplaceability of Legacy Expert Models**: High-performance task-specific legacy models are common in real-world scenarios. Retraining them on large-scale image-text corpora as CLIP does is impractical due to computational cost and data requirements.

**High Cost of Manual Annotation**: Methods that avoid CLIP require manual annotation of image-concept associations, which is time-consuming and expensive.

**All CBMs Require Training a Linear Probe**: Every existing CBM method requires training a linear classifier on top of concept activations to map concepts to class labels, making fully unsupervised operation impossible.

**Retraining Alters the Model's Decision Distribution**: Further fine-tuning a legacy model changes its original decision process, which is generally undesirable.

## Method

### Overall Architecture

The method consists of two stages:

- **Stage 1 — TextUnlock**: A lightweight MLP is trained to project features from a frozen visual classifier into the text embedding space while preserving the original classification distribution.
- **Stage 2 — U-F²-CBM**: With the MLP frozen, the aligned feature space is used for concept discovery and concept-to-class prediction, requiring no additional training.

### Key Designs

Given a frozen visual classifier $F$ (feature extractor $F_v$ + linear classification head $W$) and an arbitrary text encoder $T$:

1. **MLP Projection**: An MLP is trained to project visual features $f = F_v(I)$ into $\tilde{f} = \text{MLP}(f) \in \mathbb{R}^m$, placing them in the same space as text embeddings.
2. **Text-Based Classification Weights**: The $K$ class names are encoded with the template "an image of a {class}" to produce $U \in \mathbb{R}^{K \times m}$, which serves as the new classification head weights.
3. **Distribution Alignment Training**: The cosine similarity $S = \tilde{f} \cdot U^T$ between projected features and text class names is computed, and cross-entropy loss is used to align the result to the soft distribution $o = \text{softmax}(f \cdot W)$ of the original classifier.

### Loss & Training

$$L = -\sum_{i=1}^{K} o_i \log\left(\frac{e^{s_i}}{\sum_{j=1}^{K} e^{s_j}}\right)$$

This loss is equivalent to the KL divergence between the original distribution $o$ and the predicted distribution (up to a constant entropy term), and can be interpreted as self-distillation — distilling the original model's distribution into its vision-language correspondence distribution. **Key point**: No ground-truth labels are required; only class name texts are needed.

### U-F²-CBM Construction

**Concept Discovery**: A set of $Z = 20K$ common English words is selected as the concept set $\mathcal{Z}$, filtered rigorously to remove terms matching class names, hypernyms/hyponyms, synonyms, etc. These are encoded by the text encoder into $C \in \mathbb{R}^{Z \times m}$. For an image, concept activations are given by $\tilde{f} \cdot C^T \in \mathbb{R}^Z$.

**Unsupervised Concept-to-Class Classifier**: Since $U$ and $C$ are both produced by the same text encoder, the weight matrix is directly computed as $W^{con} = C \cdot U^T \in \mathbb{R}^{Z \times K}$, where each entry represents the textual similarity between a concept and a class name.

**Overall Prediction**:

$$S_{cn} = (\tilde{f} \cdot C^T) \cdot (C \cdot U^T) = \tilde{f} \cdot \underbrace{C^T C}_{\text{Gram matrix}} \cdot U^T$$

Notably, when the Gram matrix $C^T C$ equals the identity matrix, the expression reduces to the original feature classifier $\tilde{f} \cdot U^T$. Thus, the CBM transformation amounts to inserting the concept Gram matrix into the original classifier.

## Key Experimental Results

### Main Results

**TextUnlock Classification Accuracy Preservation** (ImageNet-1K validation set, 17 models):

| Model | TextUnlock Top-1 | Original Top-1 | Δ |
|-------|-----------------|----------------|---|
| ResNet50 | 75.80 | 76.13 | −0.33 |
| EfficientNetv2-M | 84.95 | 85.11 | −0.16 |
| ViT-B/16 | 80.70 | 81.07 | −0.37 |
| Swinv2-Base | 83.72 | 84.11 | −0.39 |
| BeiT-L/16 | 87.22 | 87.34 | −0.12 |
| DINOv2-B | 84.40 | 84.22 | **+0.18** |

Average accuracy drop across 40 models is only approximately **0.2 percentage points**.

**U-F²-CBM vs. Supervised CLIP-based CBM** (ImageNet-1K):

| Method | Model | Top-1 |
|--------|-------|-------|
| LF-CBM (supervised) | CLIP ViT-B/16 | 75.4 |
| DN-CBM (supervised) | CLIP ViT-B/16 | 79.5 |
| DCBM-SAM2 (supervised) | CLIP ViT-L/14 | 77.9 |
| **U-F²-CBM (unsupervised)** | ViT-B/16v2 | **83.2** |
| **U-F²-CBM (unsupervised)** | ConvNeXtV2-B@384 | **86.4** |

Even ResNet50 trained on ImageNet-1K only (73.9) surpasses the CLIP ResNet50 CBM trained on 400M image-text pairs (72.9).

### Cross-Dataset Generalization

| Dataset | Method | Model | Accuracy |
|---------|--------|-------|----------|
| Places365 | CDM (CLIP) | CLIP-RN50 | 52.70 |
| Places365 | **Ours** | DenseNet161 | **53.42** |
| EuroSAT | Baseline (CLIP) | CLIP-ViT-B/16 | 88.57 |
| EuroSAT | **Ours** | ResNet50 | **94.22** |
| DTD | Baseline (CLIP) | CLIP-ViT-B/16 | 61.86 |
| DTD | **Ours** | ResNet50 | **68.88** |

The method is effective on domain-specific (scene/satellite/texture), fine-grained, and small-class datasets alike.

### Ablation Study

- **Training Efficiency**: Only the lightweight MLP is trained (visual encoder, text encoder, and linear classification head are all frozen), making training feasible on standard hardware with far fewer data requirements than CLIP training.
- **Concept Set Flexibility**: The concept set can be replaced on-the-fly at inference time by simply encoding a new concept set with the text encoder.
- **Concept Intervention**: Explicit intervention on bottleneck-layer concepts enables prediction control and bias correction (e.g., textual explanation of arm bias in the "dumbbell" class).
- **Zero-Shot Image Captioning**: Combining TextUnlock with ZeroCap enables any visual classifier to perform zero-shot image captioning. ConvNeXtV2@384 achieves CIDEr=17.9 and SPICE=6.9 on COCO, surpassing CLIP-based methods (CIDEr=14.6, SPICE=5.5).

## Highlights & Insights

- **Triple "Free"**: Simultaneously achieves CLIP-free, label-free, and unsupervised concept-to-class classification — the first fully unsupervised CBM.
- **Elegant Mathematical Insight**: The CBM transformation is equivalent to inserting the concept Gram matrix into the original classifier; when the Gram matrix equals the identity, the model reduces to the original classifier.
- **Strong Generality**: Architecture-agnostic, applicable to 40+ CNN/Transformer/hybrid models.
- **High Data Efficiency**: Training on ImageNet-1K alone surpasses CLIP models trained on 400M image-text pairs.
- **On-the-Fly Concept Set Switching**: The concept set can be replaced without retraining to construct a new CBM.

## Limitations & Future Work

- Concept discovery quality depends on the semantic space quality of the text encoder; a weak text encoder may yield imprecise concept activations.
- Concept redundancy introduced by the Gram matrix may degrade performance when the number of concepts is very large.
- Zero-shot image captioning underperforms CLIP-based methods on n-gram metrics such as BLEU-4 and METEOR (requiring additional compositional captioning strategies to compensate).
- The method has only been validated on classification tasks and has not been extended to more complex tasks such as detection or segmentation.
- Concept filtering relies on handcrafted rules (removing hypernyms, synonyms, etc.), which may miss certain cases of semantic leakage.

## Related Work & Insights

- **Traditional CBM**: Koh et al. [ICML 2020] introduced the original CBM, which requires manual concept annotations.
- **Label-Free CBM (LF-CBM)**: Uses CLIP to provide image-concept annotations, eliminating manual labeling but retaining CLIP dependency.
- **CBMs Built Directly on CLIP**: LaBo, CDM, DN-CBM, and others compute concept activations directly in the CLIP embedding space.
- **Decoding Visual Features into Text**: DeVIL and LIMBER train autoregressive generators to decode visual features into text, but require annotated data and alter the classifier's distribution.
- **T2C**: Trains a linear layer to map arbitrary classifiers to the CLIP visual space, but still depends on CLIP and discards the original class distribution.
- **U-F²-CBM (Ours)**: Requires no CLIP/VLM, no annotated data, does not alter the original decision distribution, and derives the concept-to-class classifier in an entirely unsupervised manner.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — Fully unsupervised CBM construction with triple "Free" is an entirely new contribution
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 40+ models, 4 datasets, ablation studies, concept intervention, and zero-shot captioning
- Writing Quality: ⭐⭐⭐⭐ — Mathematical derivations are clear; Gram matrix insight is elegant
- Value: ⭐⭐⭐⭐⭐ — Eliminates the dependency of interpretable CBMs on CLIP with strong generality

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)
- [\[CVPR 2026\] Label-Free Cross-Task LoRA Merging with Null-Space Compression](label-free_cross-task_lora_merging_with_null-space_compression.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr_turbo_merged_checkpoint_free_teacher.md)
- [\[CVPR 2026\] LLMind: Bio-inspired Training-free Adaptive Visual Representations for Vision-Language Models](llmind_bio-inspired_training-free_adaptive_visual_representations_for_vision-lan.md)
- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](modix_positional_index_scaling.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis
description: >-
  [CVPR 2025][Interpretability][Vision Transformer] Prompt-CAM is proposed to realize almost "free" interpretable fine-grained analysis. By injecting class-specific learnable prompt tokens into a pre-trained ViT, it utilizes the multi-head attention maps of the last layer to identify and localize critical traits that distinguish fine-grained categories.
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Vision Transformer"
  - "Fine-Grained Analysis"
  - "Prompt Learning"
  - "Attention Map"
date: 2026-05-08
content_hash: ec1c04aa22af831f
---

# Prompt-CAM: Making Vision Transformers Interpretable for Fine-Grained Analysis

**Conference**: CVPR 2025  
**arXiv**: [2501.09333](https://arxiv.org/abs/2501.09333)  
**Code**: [https://github.com/Imageomics/Prompt_CAM](https://github.com/Imageomics/Prompt_CAM)  
**Area**: Interpretability  
**Keywords**: Interpretability, Vision Transformer, Fine-Grained Analysis, Prompt Learning, Attention Map

## TL;DR

Prompt-CAM is proposed to realize almost "free" interpretable fine-grained analysis. By injecting class-specific learnable prompt tokens into a pre-trained ViT, it utilizes the multi-head attention maps of the last layer to identify and localize critical traits that distinguish fine-grained categories.

## Background & Motivation

Pre-trained ViTs (such as DINO) have demonstrated powerful capabilities in extracting localized, discriminative features, but existing visualization methods fail to effectively utilize these features for fine-grained analysis:

- **Saliency maps like Grad-CAM**: produce blurry and coarse heatmaps on ViTs, highlighting the entire object rather than discriminative features.
- **Attention maps of the [CLS] token**: although they can attend to local regions (e.g., head, wings, tail), these regions are not "class-specific" — all bird species tend to focus on the same body parts.
- **Interpretable methods like ProtoPNet and INTR**: require designing specialized models and complex training procedures, making it difficult to leverage the latest pre-trained ViTs.

Key Insight: If each class is given its own "query" token, the tokens of each class can attend to image regions that are "specifically meaningful" to that class (i.e., traits) through the attention mechanism. By comparing the attention maps of different class tokens, the key traits that distinguish classes can be precisely localized.

## Method

### Overall Architecture

Prompt-CAM is based on the Visual Prompt Tuning (VPT) framework. Its core modification is changing the prediction head from the [CLS] token output to the injected prompt output. Given a classification task with $C$ classes, $C$ learnable tokens are injected, the entire ViT backbone is frozen, and only these tokens and a shared scoring vector $\boldsymbol{w}$ are trained. During inference, the multi-head attention maps of each class prompt over the image patches directly reveal the traits of that class and their locations.

### Key Designs

1. **Class-Specific Prompt Injection (Prompt-CAM-Deep)**:
    - Function: Introduces class-specific learnable tokens into the ViT to make their attention maps class-discriminative.
    - Mechanism: For an $N$-layer ViT, $C$ class-specific prompts $\boldsymbol{P}_{N-1}$ are input at the last layer $L_N$, while $C$ class-agnostic prompts $\boldsymbol{P}_i$ are input at each of the preceding $N-1$ layers. The output of the last layer $\boldsymbol{Z}_N$ yields class scores via the inner product with the shared vector $\boldsymbol{w}$: $s[c] = \boldsymbol{w}^\top \boldsymbol{z}_N^c$.
    - Design Motivation: Two advantages of the Deep variant: (i) class-specific prompts only attend to high-level features $\boldsymbol{E}_{N-1}$ of the last layer (as early-layer features are too noisy for fine-grained discrimination), and (ii) decoupling "class-specific localization" and "model adaptation" into prompts in different layers avoids a single set of prompts having to perform dual tasks.

2. **Shared Scoring Vector Design**:
    - Function: Constrains the model to distinguish classes through attention maps rather than via feature channels.
    - Mechanism: Traditional classifiers use class-specific weights $\boldsymbol{w}_c$ for prediction, while Prompt-CAM uses a shared $\boldsymbol{w}$ to make a "binary decision" — i.e., determining whether "the traits of class $c$ are present in the image". Formula comparison: traditional $\hat{y} = \arg\max_c \sum_j \alpha^\star[j] \cdot (\boldsymbol{w}_c^\top \boldsymbol{v}^j)$ vs Prompt-CAM $\hat{y} = \arg\max_c \sum_j \alpha^c[j] \cdot (\boldsymbol{w}^\top \boldsymbol{v}^j)$.
    - Design Motivation: Traditional models can "take shortcuts" by encoding global discriminative information in value features, which allows correct classification even if attention maps are meaningless. The shared $\boldsymbol{w}$ eliminates this shortcut — if the values of all patches are identical (no spatial information), the scores for all classes must also be identical. The model is forced to: (i) generate local features that maintain spatial resolution, and (ii) generate distinct attention weights for different classes.

3. **Trait Identification and Localization (Greedy Masking Algorithm)**:
    - Function: Automatically identifies the most discriminative traits for each class.
    - Mechanism: For correctly classified images, the least important attention heads are greedily blurred step-by-step (replaced by a uniform attention $\frac{1}{M}\boldsymbol{1}$) until a classification error occurs. Specifically, in each step, for each unblurred head $r'$, $\boldsymbol{\alpha}_{N-1}^{c,r'}$ is temporarily replaced with a uniform vector and $s[c]$ is recomputed. The head that causes the "minimum drop in $s[c]$ after blurring" is chosen as the least important head. The remaining heads are the most critical traits.
    - Design Motivation: The $R$ attention heads may focus on $R$ different regions, but not all are equally important. Greedy masking automatically filters for "few but essential" discriminative traits.

### Loss & Training

Only the standard cross-entropy loss is used: $-\log \frac{\exp(s[y])}{\sum_c \exp(s[c])}$

- The entire ViT backbone is frozen, and only the prompt tokens $\boldsymbol{P}$ and the shared vector $\boldsymbol{w}$ are learned.
- SGD optimizer is used.
- Default backbone: DINO ViT-B. DINOv2 and BioCLIP are also validated.

## Key Experimental Results

### Main Results (Fidelity Evaluation, CUB-200-2011)

| Method | Insertion ↑ | Deletion ↓ | Description |
|--------|------|------|------|
| Grad-CAM | 0.52 | 0.17 | Post-hoc explanation method |
| Layer-CAM | 0.54 | 0.13 | |
| Eigen-CAM | 0.56 | 0.22 | |
| Attention roll-out | 0.55 | 0.27 | |
| **Prompt-CAM** | **0.61** | **0.09** | Highest insertion + lowest deletion |

### Classification Accuracy vs. Interpretability Trade-off (DINO Backbone)

| Method | CUB | Bird-525 | Dog | Pet |
|------|------|---------|------|------|
| Linear Probing | 78.6 | 99.2 | 82.4 | 92.4 |
| Prompt-CAM | 73.2 | 98.8 | 81.1 | 91.3 |

### Human Evaluation (Trait Identification Quality)

| Method | Trait Identification Rate | Description |
|------|---------|------|
| **Prompt-CAM** | **60.49%** | Significantly superior |
| TesNet | 39.14% | |
| ProtoConcepts | 30.39% | |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Prompt-CAM-Shallow (only first layer) | Slightly lower accuracy | Early-layer features are noisy |
| Prompt-CAM-Deep (last layer) | Superior | Focuses only on high-level features |
| Different backbones (DINO/DINOv2/BioCLIP) | Consistent trait localization | Strong generalizability of the method |
| Removing the red wing patch of Red-winged Blackbird | Classification changes to Boat-tailed Grackle | Counterfactual validation of fidelity |

### Key Findings

- **Prompt-CAM is effective across 13 cross-domain datasets**: including animals (birds, fish, insects), plants (flowers, medicinal leaves), and objects (cars, food), demonstrating exceptional generalizability.
- **Explainable accuracy drop**: Images misclassified by Prompt-CAM but correctly classified by Linear Probing typically suffer from occluded traits or abnormal poses — showing that Prompt-CAM indeed relies on traits rather than global information.
- **Multi-head attention naturally decouples different traits**: For instance, different attention heads of the Scott Oriole focus separately on the yellow belly, black head, and black breast, without requiring extra constraints.
- **Extensible to hierarchical taxonomy**: On fish datasets, Prompt-CAM at different taxonomic levels (Family $\rightarrow$ Genus $\rightarrow$ Species) focuses on features varying from coarse to fine-grained.

## Highlights & Insights

- **Extremely simple design**: Based on VPT, it only requires modifying a few lines of code (changing the prediction head location), without new loss functions, new modules, or changes to the backbone — a "nearly free lunch."
- **Elegant theoretical analysis of shared $\boldsymbol{w}$**: The comparison between Eq. 6 and Eq. 7 clearly explains why attention maps of traditional classifiers fail to provide reliable interpretability — because class-discriminative information can "escape" into value features.
- **Cross-species trait comparison**: Shared traits between two species can be discovered by visualizing the attention of Class B's prompt on Class A's images (Figure 1c), which is a unique capability.

## Limitations & Future Work

- Classification accuracy is somewhat sacrificed (dropping by ~5% on CUB), which represents an inherent trade-off between interpretability and accuracy.
- The number of prompts equals the number of classes $C$; when $C$ is very large, the parameters and computational complexity scale linearly.
- Currently, only the last-layer attention maps are used, and intermediate-layer information is discarded.
- Semantic annotation of traits still requires manual interpretation and is not automatically associated with natural language descriptions.

## Related Work & Insights

- Compared to INTR (encoder-decoder architecture + full fine-tuning), Prompt-CAM is simpler, faster, yields cleaner attention maps, and allows free choice of any ViT backbone.
- Compared to the ProtoPNet family (prototype networks), it does not require maintaining prototype databases or complex training pipelines.
- Insight: The prompt outputs in VPT are "wasted" (original VPT discards them), whereas Prompt-CAM discovers that they contain rich class-specific information — this "waste reduction" strategy is worth exploring in other prompt methods.

## Rating
- Novelty: ⭐⭐⭐⭐ Repurposing VPT prompt outputs to achieve interpretability is clever and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 13 datasets across 3 major domains, human evaluation + counterfactual analysis + hierarchical taxonomy, extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐⭐ Outstanding theoretical analysis (Eq. 6 vs Eq. 7), visualization use cases, and experimental design.
- Value: ⭐⭐⭐⭐⭐ Provides a simple and practical general tool for ViT interpretability, with direct application value for fields like ecology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients information for Zero-Shot NAS on Vision Transformers](lswag_zero_shot_nas.md)
- [\[ACL 2026\] Fine-Grained Analysis of Shared Syntactic Mechanisms in Language Models](../../ACL2026/interpretability/fine-grained_analysis_of_shared_syntactic_mechanisms_in_language_models.md)
- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[CVPR 2025\] Scaling Vision Pre-Training to 4K Resolution](scaling_vision_pre-training_to_4k_resolution.md)
- [\[CVPR 2025\] Probing the Mid-Level Vision Capabilities of Self-Supervised Learning](probing_the_mid-level_vision_capabilities_of_self-supervised_learning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Classification Matters: Improving Video Action Detection with Class-Specific Attention
description: >-
  [ECCV2024][Video Understanding][video action detection] Proposes a class-specific query (class queries) mechanism, which assigns an independent learnable query to each action class, allowing the model to dynamically attend to context regions relevant to each class, significantly improving classification performance in video action detection.
tags:
  - "ECCV2024"
  - "Video Understanding"
  - "video action detection"
  - "class-specific attention"
  - "transformer decoder"
  - "class queries"
  - "spatio-temporal tube"
date: 2026-05-08
content_hash: a575589e767bae75
---

# Classification Matters: Improving Video Action Detection with Class-Specific Attention

**Conference**: ECCV2024  
**arXiv**: [2407.19698](https://arxiv.org/abs/2407.19698)  
**Code**: [jinsingsangsung/ClassificationMatters](https://jinsingsangsung.github.io/ClassificationMatters/)  
**Area**: Video Understanding  
**Keywords**: video action detection, class-specific attention, transformer decoder, class queries, spatio-temporal tube

## TL;DR

Proposes a class-specific query (class queries) mechanism, which assigns an independent learnable query to each action class, allowing the model to dynamically attend to context regions relevant to each class, significantly improving classification performance in video action detection.

## Background & Motivation

Video Action Detection (VAD) requires simultaneously localizing actors and classifying their actions. Since all instances performing actions in VAD are humans, actor localization is relatively straightforward. However, action classification is extremely challenging, as actors of different action classes share highly similar appearances, requiring fine-grained appearance and motion information to distinguish them.

Through experiments, the authors observed that for three state-of-the-art methods (TubeR, EVAD, STMixer), providing ground-truth (GT) class labels yields a much larger performance gain than providing GT bounding boxes. This indicates that the performance bottleneck of VAD primarily lies in classification rather than localization. However, existing Transformer-based methods exhibit a severe bias when constructing classification features: attention converges on the actor's body region, ignoring contextual information critical for classification (e.g., the cigarette in a "smoking" action, or the speaker in a "listen to" action).

## Core Problem

1. **Classification Feature Bias**: Existing methods use a single attention map shared across all action classes to extract context, causing the Transformer weights to bias toward encoding common cross-class semantics (i.e., the actor itself), resulting in attention being highly concentrated within actor regions.
2. **Lack of Class Specificity**: Different action classes require focusing on different contextual regions, but existing methods fail to provide independent attention regions for each class.
3. **Limited Attention Scope**: The attention of prior methods struggles to expand beyond the actor bounding boxes, whereas many key classification cues are located exactly outside these boxes.

## Method

### Overall Architecture

The model consists of three components: backbone, 3D Deformable Transformer Encoder, and Transformer Decoder. It takes a video clip $X \in \mathbb{R}^{T \times H_0 \times W_0 \times 3}$ as input and outputs the spatio-temporal tube and frame-level action classification predictions for each actor.

### 3D Deformable Transformer Encoder

- The multi-scale feature maps $\mathbf{V} = \{\boldsymbol{v}^l \in \mathbb{R}^{T_l \times H_l \times W_l \times D}\}$ from the backbone are fed into the encoder.
- Drawing inspiration from Deformable DETR, the 2D offsets $(\Delta h, \Delta w)$ are extended to 3D offsets $(\Delta t, \Delta h, \Delta w)$, enabling queries to aggregate long-range features along the temporal dimension.
- After encoding, features are aligned to the same spatio-temporal dimension via interpolation.

### Localizing Decoder Layer (LDL)

- Input: actor boxes $A \in \mathbb{R}^{N_a \times 4}$ (spatial part) and actor embeddings $AE \in \mathbb{R}^{N_a \times D}$ (content part).
- Projects $A$ into a $D$-dimensional space to construct actor positional queries $P$.
- Performs actor-conditional aggregation on multi-scale feature maps to generate actor-specific contextual features $\mathbf{x}$.
- Outputs actor features $\mathbf{f} \in \mathbb{R}^{N_a \times D}$.

### Classifying Decoder Layer (CDL) — Key Designs

- **Class queries**: Introduces learnable embeddings $\boldsymbol{q} \in \mathbb{R}^{N_c \times D}$, with one query per action class, encoding class-specific information.
- **Actor positional query appending**: Appends the actor positional query $P_i$ to the class queries to ensure that the class queries attend to the context of the correct actor (addressing the actor-agnostic activation issue).
- **Interaction feature construction**: Broadcasts and sums the actor feature $\mathbf{f}_i$ with the actor-specific context $\mathbf{x}_i$, followed by convolution to obtain the interaction feature map $\mathbf{z}_i$, which represents the interaction between the $i$-th actor and the context.
- **Cross-attention**: The class queries (containing actor position information) serve as queries, and the interaction feature maps serve as keys/values, generating a classification attention map $\mathcal{A}_i \in \mathbb{R}^{N_c \times HW}$.
- Since both queries and keys contain both class and actor information, the variations in attention weight contributions across different classes are far more pronounced than in prior methods.

### Loss & Training

After Hungarian matching, multiple losses are computed: Binary Focal Loss (classification), L1 Loss + GIoU Loss (localization), and BCE Loss (confidence).

## Key Experimental Results

### AVA v2.2 Dataset

| Method | Backbone | Pre-training | mAP |
|------|----------|--------|-----|
| TubeR | CSN-152 | IG65M+K400 | 31.1 |
| STMixer | CSN-152 | IG65M+K400 | 32.8 |
| EVAD | ViT-B | K400 | 32.3 |
| **Ours** | **CSN-152** | **IG65M+K400** | **33.5** |
| **Ours** | **ViT-B** | **K400** | **32.9** |
| EVAD | ViT-B(K710) | K710+K400 | 37.7 |
| **Ours** | **ViT-B(K710)** | **K710+K400** | **38.4** |

### UCF101-24 Dataset

The proposed model achieves f-mAP 85.9 / v-mAP 61.7, outperforming TubeR (83.2/58.4) and EVAD (85.1/58.8).

### Efficiency Comparison (JHMDB 40-frame tube inference)

| Method | Params | FLOPs | Inference Time |
|------|--------|-------|----------|
| EVAD | 185.4M | 10.68T | 8363ms |
| STMixer | 219.2M | 7.64T | 2088ms |
| **Ours** | **117.8M** | **3.26T** | **432ms** |

Requires 37% fewer parameters, only 30% of EVAD's FLOPs, and achieves a 19x faster inference speed.

### Ablation Study

- 3D Deformable Encoder + LDL + CDL complete model: 33.5 mAP, showing a +4.9 improvement over the vanilla baseline (28.6).
- Removing actor positional query appending: AVA 31.7 (-1.8), UCF 82.9 (-3.0).
- Feature aggregation methods: Actor-conditional aggregation (33.5) > Weighted sum (32.9) > Mean pooling (32.0).
- Actor-context feature fusion methods: Summation (33.5) > Concatenation + 1D Convolution (31.8) > Cross-attention (31.3) > Self-attention (30.8).

### GT Box Substitution Experiment Validating Classification Ability

With GT boxes provided, the performance gain of the proposed model (+3.7 to 4.0) is significantly larger than other methods (+2.0 to 2.6), proving that the improvement indeed stems from enhanced classification capabilities.

## Highlights

1. **Deep Problem Insight**: Through GT box/class substitution experiments, it clearly demonstrates that the bottleneck of VAD lies in classification rather than localization.
2. **Elegant Class Query Design**: Assigns an independent query to each class, naturally producing interpretable class attention maps with intuitive visualizations.
3. **Significant Efficiency Advantage**: Generates the entire spatio-temporal tube via a single forward pass, avoiding sliding window strategies and achieving an inference speed far exceeding EVAD (19x).
4. **Thorough Ablation Studies**: Verifies the individual contributions of components such as CDL, LDL, 3D encoder, actor positional queries, and feature aggregation methods.

## Limitations & Future Work

1. **Lack of Inter-Frame Information Interaction**: Due to memory constraints, the current decoder does not exchange information across frames; temporal dynamic modeling relies entirely on the encoder.
2. **Slightly Lower Performance on JHMDB than EVAD**: The authors speculate this is because JHMDB has low class diversity (only 21 classes), making it difficult to leverage the advantages of class queries.
3. **Class Query Count Tied to Class Cardinality**: Scalability to large-scale class sets remains to be validated.
4. Spatially sparse collection of class information could be explored to release memory for temporal dynamic modeling.

## Related Work & Insights

| Dimension | TubeR/EVAD | STMixer | Ours |
|------|-----------|---------|------|
| Classification Features | Single attention map, biased toward actor regions | Multi-scale but still lacks class specificity | Class queries generate class-specific attention |
| Inference Mode | Frame-by-frame / Sliding window | Frame-by-frame | Generates entire spatio-temporal tube in a single pass |
| Attention Scope | Restricted to the vicinity of actor boxes | Restricted to the vicinity of actor boxes | Expands to critical context outside boxes |
| Interpretability | Attention maps lack class differentiation | None | Independent interpretable attention map for each class |
| Efficiency | Moderate | Moderate | High (fewer parameters, lower FLOPs) |

## Related Work & Insights

- **Transferability of the Class Query Concept**: The concept of "assigning an independent query to each class/attribute" can be extended to other detection tasks requiring fine-grained classification (e.g., fine-grained object detection, human pose estimation).
- **Context Modeling Insights**: Action recognition should not only focus on the actor itself; interacting objects and scene context are key classification cues.
- Differs from DAB-DETR's utilization of positional priors—this work uses positional information for actor-specific guidance of class queries, rather than bounding box regression.

## Rating
- Novelty: ⭐⭐⭐⭐ — The class-specific query mechanism addresses the long-overlooked classification bias problem in VAD.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Three benchmarks, extensive ablation studies, efficiency comparisons, GT substitution validation, and attention visualizations.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem analysis with rich and intuitive diagrams.
- Value: ⭐⭐⭐⭐ — Excels in both performance and efficiency with strong interpretability, offering a new paradigm for classification in VAD.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos](actionswitch_class-agnostic_detection_of_simultaneous_actions_in_streaming_video.md)
- [\[ECCV 2024\] Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection](interleaving_one-class_and_weakly-supervised_models_with_adaptive_thresholding_f.md)
- [\[ECCV 2024\] SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders](sa-dvae_improving_zero-shot_skeleton-based_action_recognition_by_disentangled_va.md)
- [\[ECCV 2024\] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition](finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)
- [\[ECCV 2024\] Bayesian Evidential Deep Learning for Online Action Detection](bayesian_evidential_deep_learning_for_online_action_detection.md)

</div>

<!-- RELATED:END -->

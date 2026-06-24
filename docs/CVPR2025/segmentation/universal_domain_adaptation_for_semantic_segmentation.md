---
title: >-
  [Paper Note] Universal Domain Adaptation for Semantic Segmentation
description: >-
  [CVPR 2025][Segmentation][Universal Domain Adaptation] This work introduces the Universal Domain Adaptation for Semantic Segmentation (UniDA-SS) task and proposes the UniMAP framework. By leveraging two core components—Domain-Specific Prototype Differentiation (DSPD) and Target-guided Image Matching (TIM)—UniMAP achieves effective adaptation from synthetic to real-world data without prior knowledge of category configurations, significantly outperforming existing UDA-SS method…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Universal Domain Adaptation"
  - "Semantic Segmentation"
  - "Prototype Learning"
  - "Image Matching"
  - "Pseudo-labeling"
date: 2026-05-08
content_hash: 61506ac24be2b276
---

# Universal Domain Adaptation for Semantic Segmentation

**Conference**: CVPR 2025  
**arXiv**: [2505.22458](https://arxiv.org/abs/2505.22458)  
**Code**: [https://github.com/KU-VGI/UniMAP](https://github.com/KU-VGI/UniMAP)  
**Area**: Semantic Segmentation / Domain Adaptation  
**Keywords**: Universal Domain Adaptation, Semantic Segmentation, Prototype Learning, Image Matching, Pseudo-labeling

## TL;DR

This work introduces the Universal Domain Adaptation for Semantic Segmentation (UniDA-SS) task and proposes the UniMAP framework. By leveraging two core components—Domain-Specific Prototype Differentiation (DSPD) and Target-guided Image Matching (TIM)—UniMAP achieves effective adaptation from synthetic to real-world data without prior knowledge of category configurations, significantly outperforming existing UDA-SS methods.

## Background & Motivation

**Background**: Unsupervised Domain Adaptation for Semantic Segmentation (UDA-SS) aims to leverage labeled synthetic data (source domain) to achieve high-quality segmentation on unlabeled real-world data (target domain). Existing methods are primarily categorized into adversarial learning and self-training. Among them, self-training methods (such as DAFormer, HRDA, and MIC) have made significant progress through pseudo-labeling and domain mixing techniques.

**Limitations of Prior Work**: Existing UDA-SS methods assume that the category relationships between the source and target domains are known, which is impractical in real-world scenarios. Specifically: (1) the target domain may contain classes not present in the source domain (target-private classes); (2) the source domain may contain classes not present in the target domain (source-private classes). When source-private classes exist, models may incorrectly align them with the target domain, leading to negative transfer and significant performance degradation.

**Key Challenge**: When the category configurations are unknown, the confidence of the pseudo-labels relied upon by self-training methods is severely compromised. High feature similarity between source-private classes and certain common classes causes a drop in the pseudo-label confidence of the common classes. Consequently, they are falsely assigned to the target-private class (unknown), failing to effectively learn both common and target-private classes.

**Goal**: To propose the UniDA-SS task to achieve robust domain adaptation segmentation without prior knowledge of category configurations, while correctly classifying common classes and detecting target-private classes.

**Key Insight**: The core problem lies in "increasing the confidence of common classes." The authors observe that while common classes share the same semantics across the two domains, their feature representations differ. Representing them with a single prototype leads to insufficient confidence.

**Core Idea**: Assign two domain-specific prototypes (source + target) to each class, and utilize the relative distances of pixel embeddings to both prototypes to distinguish common classes from private classes. Simultaneously, employ an image matching strategy to increase the exposure of common classes during training.

## Method

### Overall Architecture

Based on the standard self-training UDA-SS framework: the student network $f_\theta$ is trained on labeled source domain data and target domain pseudo-labels, while the teacher network $g_\phi$ is updated via EMA and generates target pseudo-labels. The classification head is set to $C_s+1$ (with an additional "unknown" class). On top of this, DSPD (prototype learning + weight scaling) and TIM (image matching strategy) are integrated, along with the DACS domain mixing technique.

### Key Designs

1. **Domain-Specific Prototype Differentiation (DSPD)**:

    - **Function**: Enhances the confidence of target domain common classes and distinguishes them from private classes by assigning dual (source/target) prototypes for each class.
    - **Mechanism**: Distributes $2C+1$ prototypes (2 domain-specific prototypes per class + 1 unknown prototype) in a fixed Simplex ETF space to guarantee equal pairwise cosine similarity. During training, three losses constrain the relationship between pixel embeddings and prototypes: cross-entropy loss $\mathcal{L}_{CE}$ pulls corresponding prototypes closer, pixel-prototype contrastive learning $\mathcal{L}_{PPC}$ performs push-pull operations in the global space, and pixel-prototype distance optimization $\mathcal{L}_{PPD}$ further reduces distance. Crucially, a weight $w = \frac{2(d_s+1)(d_t+1)}{(d_s+1)+(d_t+1)}$ (harmonic mean) is calculated using the relative distance of the pixel embeddings to the source and target prototypes. Pixels from common classes should be close to both prototypes (yielding a large $w$), whereas private classes only approach one prototype (yielding a small $w$), thereby assigning higher weight to common classes.
    - **Design Motivation**: Traditional methods treat the same class across two domains as completely identical, ignoring domain-specific feature differences. Learning domain-specific feature representations independently using dual prototypes enhances the confidence of target predictions.

2. **Target-guided Image Matching (TIM)**:

    - **Function**: Prioritizes selecting source images that contain the most common-class pixels into the training batch, thereby increasing opportunities to learn common classes.
    - **Mechanism**: First, the pixel proportion $f_c$ for each class in the target pseudo-labels is computed, and a higher weight is assigned to rare classes as $\hat{f_c} = \text{softmax}(\frac{1-f_c}{T})$. For each source image, a matching score is calculated as $S_s = \sum_{c \in c^*} n_c^s \hat{f_c}$ (where $c^*$ represents the overlapping classes in both source labels and target pseudo-labels). The source image with the highest score is paired with the target image to enter the training batch.
    - **Design Motivation**: The presence of source-private classes dilutes the learning ratio of common classes. TIM ensures that each batch contains as many common class pixels as possible through intelligent matching, while mitigating long-tail imbalance via class-wise weighting.

3. **UniDA-SS Benchmark Setup**:

    - **Function**: Defines a complete evaluation protocol for UniDA-SS.
    - **Mechanism**: Based on two cross-domain setups (GTA5 → IDD and Pascal-Context → Cityscapes), evaluation benchmarks for four scenarios—CDA-SS, ODA-SS, PDA-SS, and OPDA-SS—are constructed. Evaluation metrics include the average IoU of common classes (Common), the IoU of private (unknown) classes (Private), and the H-score.
    - **Design Motivation**: Previously, formal definitions and evaluations of UniDA existed only for classification tasks. This work is the first to explore this in the more fine-grained task of semantic segmentation.

### Loss & Training

Total Loss = source segmentation loss $\mathcal{L}_{seg}^s$ + weighted target segmentation loss $w \cdot q_t \cdot \mathcal{L}_{seg}^t$ + prototype loss $\mathcal{L}_{proto} = \mathcal{L}_{CE} + \lambda_1\mathcal{L}_{PPC} + \lambda_2\mathcal{L}_{PPD}$. DACS domain mixing technique is applied. A pseudo-label threshold $\tau_p$ is used to designate low-confidence pixels as "unknown". The teacher network is updated from the student network via EMA.

## Key Experimental Results

### Main Results

Pascal-Context → Cityscapes (OPDA-SS):

| Method | Common mIoU | Private IoU | H-score |
|------|-------------|-------------|---------|
| MIC (CDA-SS SOTA) | 48.67 | 7.85 | 13.51 |
| BUS (ODA-SS) | 57.64 | 20.38 | 30.11 |
| **UniMAP (Ours)** | **60.94** | **31.27** | **41.33** |

GTA5 → IDD (OPDA-SS):

| Method | Common mIoU | Private IoU | H-score |
|------|-------------|-------------|---------|
| DAFormer | 52.05 | 21.07 | 29.99 |
| HRDA | 53.43 | 32.15 | 40.14 |
| **UniMAP** | **55.95** | **39.65** | **46.42** |

### Ablation Study

| Config | Common | Private | H-score |
|------|--------|---------|---------|
| Baseline | 57.64 | 20.38 | 30.11 |
| + DSPD | 59.12 | 27.85 | 37.90 |
| + TIM | 59.78 | 25.42 | 35.68 |
| + DSPD + TIM (UniMAP) | **60.94** | **31.27** | **41.33** |

### Key Findings

- DSPD contributes the most to private class detection (Private IoU from 20.38 to 27.85), indicating that domain-specific prototypes effectively distinguish common from private classes.
- TIM primarily enhances common class performance (Common from 57.64 to 59.78), validating the strategy of increasing common class exposure.
- Both components exhibit complementary effects, improving the H-score from 30.11 to 41.33 (+37%).
- Existing CDA-SS methods (e.g., MIC) suffer from severe degradation under PDA/OPDA scenarios, justifying the necessity of the UniDA-SS task.

## Highlights & Insights

- **First definition of the UniDA-SS task**: Generalizes UniDA from classification to the more challenging pixel-level semantic segmentation task. The contribution of a complete benchmark serves as a guide for the community.
- **Ingenious design of harmonic mean weights**: Utilizes the harmonic mean of the distances between pixel embeddings and the two domain-specific prototypes to distinguish common/private classes. The intuition is straightforward: for common classes, both distances are small, resulting in a large harmonic mean, while for private classes, one distance is small and the other is large, yielding a small harmonic mean.
- **Class-wise weighting strategy of TIM**: Instead of merely selecting source images with abundant common classes, higher weights are assigned to rare common classes, simultaneously addressing both image matching and class imbalance.

## Limitations & Future Work

- ETF prototypes are fixed and cannot be dynamically adjusted during training, which may limit representation capacity.
- TIM relies on the quality of pseudo-labels; the effectiveness of the matching strategy might be limited early in training when pseudo-labels are inaccurate.
- Evaluation is currently limited to two cross-domain setups (GTA5 → IDD and Pascal-Context → Cityscapes). Validating on more domain pairs (e.g., Synthia → Cityscapes) would be more comprehensive.
- More complex category relationships (e.g., hierarchical/containment relationships among classes) have not yet been explored.

## Related Work & Insights

- **vs MIC (CDA-SS)**: SOTA for closed-set domain adaptation, but suffers severe performance drops when source-private classes exist, revealing that closed-set assumptions are unreliable in practice.
- **vs BUS (ODA-SS)**: An open-set domain adaptation method that handles target-private classes but ignores source-private classes. UniMAP extends its capacity to handle source-private classes on top of it.
- **vs UniDA Classification Methods (UAN, UniOT)**: Direct transfer of classification-based UniDA methods to segmentation yields poor results (H-score < 15), proving that pixel-level tasks require dedicated designs.
- The concept of prototype learning (ProtoSeg) and the usage of ETF space can be transferred to other domain adaptation tasks (e.g., object detection, instance segmentation).

## Rating

- Novelty: ⭐⭐⭐⭐ Defining the UniDA-SS task for the first time is pioneering, but the novelty of the specific method (prototype + matching) is moderate.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple scenarios and detailed ablations, though more domain pairs could be included.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and a well-demonstrated motivation.
- Value: ⭐⭐⭐⭐ The new task definition + benchmark possesses lasting impact for the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation](semantic_library_adaptation_lora_retrieval_and_fusion_for_open-vocabulary_semant.md)
- [\[ICCV 2025\] Exploiting Domain Properties in Language-Driven Domain Generalization for Semantic Segmentation](../../ICCV2025/segmentation/exploiting_domain_properties_in_language-driven_domain_generalization_for_semant.md)
- [\[ICML 2025\] Balanced Learning for Domain Adaptive Semantic Segmentation](../../ICML2025/segmentation/balanced_learning_for_domain_adaptive_semantic_segmentation.md)
- [\[ICCV 2025\] Hybrid-TTA: Continual Test-time Adaptation via Dynamic Domain Shift Detection](../../ICCV2025/segmentation/hybrid-tta_continual_test-time_adaptation_via_dynamic_domain_shift_detection.md)
- [\[NeurIPS 2025\] Towards Unsupervised Domain Bridging via Image Degradation in Semantic Segmentation](../../NeurIPS2025/segmentation/towards_unsupervised_domain_bridging_via_image_degradation_in_semantic_segmentat.md)

</div>

<!-- RELATED:END -->

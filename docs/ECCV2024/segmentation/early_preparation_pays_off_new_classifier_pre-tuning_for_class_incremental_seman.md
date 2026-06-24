---
title: >-
  [Paper Note] Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation
description: >-
  [ECCV 2024][Segmentation][Class Incremental Semantic Segmentation] This paper proposes NeST (New claSsifier pre-Tuning), which initializes new classifier weights by learning a linear transformation from all old classifiers to the new ones prior to formal training. It also designs a transformation matrix initialization strategy based on cross-task class similarity. NeST significantly improves the performance of multiple CISS methods on Pascal VOC and ADE20K.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "Class Incremental Semantic Segmentation"
  - "New Classifier Pre-tuning"
  - "Catastrophic Forgetting"
  - "Stability-Plasticity Trade-off"
  - "Knowledge Transfer"
date: 2026-05-08
content_hash: 60a5d5a09738b27a
---

# Early Preparation Pays Off: New Classifier Pre-tuning for Class Incremental Semantic Segmentation

**Conference**: ECCV 2024  
**arXiv**: [2407.14142](https://arxiv.org/abs/2407.14142)  
**Code**: [https://github.com/zhengyuan-xie/ECCV24_NeST](https://github.com/zhengyuan-xie/ECCV24_NeST)  
**Area**: Image Segmentation  
**Keywords**: Class Incremental Semantic Segmentation, New Classifier Pre-tuning, Catastrophic Forgetting, Stability-Plasticity Trade-off, Knowledge Transfer

## TL;DR
This paper proposes NeST (New claSsifier pre-Tuning), which initializes new classifier weights by learning a linear transformation from all old classifiers to the new ones prior to formal training. It also designs a transformation matrix initialization strategy based on cross-task class similarity. NeST significantly improves the performance of multiple CISS methods on Pascal VOC and ADE20K.

## Background & Motivation

**Background**: Class Incremental Semantic Segmentation (CISS) requires models to retain knowledge of old classes while continuously learning new ones, which faces two major challenges: catastrophic forgetting and background shift. Existing methods primarily mitigate forgetting through knowledge distillation (such as MiB and PLOP), pseudo-labeling, and feature regularization.

**Limitations of Prior Work**:

**Improper initialization of new classifiers**: Random initialization causes misalignment between the new classifiers and backbone features. Drastic gradient variations in the early stages of training destroy old knowledge.

**Limitations of background classifier initialization**: MiB initializes new classifiers using the background classifier, but this leads to misclassification of true background pixels.

**Bias in auxiliary classifier training**: SSUL/DKD pre-train auxiliary classifiers, but the absence of true future data leads to bias.

**Neglecting differences among new classes**: Existing methods apply the same initialization strategy to all new classes, ignoring the varied correlations between different new classes and old classes.

**AWT's limitations**: AWT transfers weights from the background classifier via gradient attribution but ignores other old classifiers and incurs a huge memory overhead (>24GB).

**Key Challenge**: How to initialize the new classifier so that it can rapidly adapt to new class data (plasticity) without causing drastic updates to the backbone that disrupt old knowledge (stability)?

**Key Insight**: Instead of directly tuning the parameters of the new classifier, this paper proposes learning a linear transformation from all old classifiers to the new ones—achieved via channel-wise weighting and linear combination of old classifiers using an importance matrix and a projection matrix.

**Core Idea**: A pre-tuning phase is inserted before formal training. By freezing the backbone and training only the transformation matrices, the generated new classifier naturally aligns with the backbone feature space. Meanwhile, the matrix initialization based on cross-task class similarity ensures a balanced trade-off between stability and plasticity.

## Method

### Overall Architecture
NeST inserts a pre-tuning phase before the formal training of each incremental step:
1. Initialize the importance matrix and projection matrix based on cross-task class similarity.
2. Freeze the backbone and old classifiers, and train only the transformation matrices.
3. Generate new classifier weights from the old classifiers using the trained matrices.
4. Initialize the new classifier with the generated weights, remove the extra parameters, and proceed to formal training.

### Key Designs

1. **New Classifier Generation**:
   For each new class $c \in \mathcal{C}_t$, two learnable matrices are allocated:
    - **Importance Matrix** $\mathbf{M}_c \in \mathbb{R}^{d \times n_{old}}$: learns the channel-wise importance weights of each old classifier for the new class.
    - **Projection Matrix** $\mathbf{P}_c \in \mathbb{R}^{n_{old} \times 1}$: learns the linear combination coefficients of the old classifiers.
   
   The formula for generating the new classifier weights is:
    $\mathbf{w}_c = (\mathbf{M}_c \odot \mathbf{W}_{old}) \mathbf{P}_c$
   where $\mathbf{W}_{old} \in \mathbb{R}^{d \times n_{old}}$ represents the weights of all old classifiers. $\odot$ denotes the Hadamard product for channel-wise weighting, and $\mathbf{P}_c$ performs the weighted combination of the old classifiers.
   
   The background class also learns a transformation to adapt to the new task:
    $\hat{\mathbf{w}}_0 = (\mathbf{M}_0 \odot \mathbf{w}_0) \mathbf{P}_0$

2. **Pre-tuning**:

    - Freeze the backbone and old classifier weights, and optimize only $\{\mathbf{M}_c, \mathbf{P}_c\}$.
    - In each forward pass, the new classifier weights are first generated by the matrices, concatenated to the old classifiers, and then used for prediction.
    - Use unbiased cross-entropy loss to prevent overfitting:
    $$\mathcal{L}_{unce} = -\frac{1}{|\mathcal{I}|} \sum_{i \in \mathcal{I}} \log \tilde{q}_x^t(i, y_i)$$
    - After pre-tuning, the final matrices generate the new classifier weights for initialization, and the extra parameters are discarded.

3. **Cross-Task Class Similarity Initialization**:

    - Feed the training images of the current step into the old model. For each pixel embedding $\mathbf{p}_u$ of a new class, decompose the classification process:
    $$\mathbf{H}_u = \mathbf{W}_{old} \odot \mathbf{p}_u', \quad \mathbf{s}_u = \text{softmax}(\text{sum}(\mathbf{H}_u))^\top$$
    - Positive values are masked as contributing channels: $\mathbf{H}_u^{mask}(i,j) = \mathbb{1}[\mathbf{H}_u(i,j) > 0]$.
    - Average across all pixels of the new class to obtain the initial importance matrix:
    $$\mathbf{M}_{c_{new}} = \frac{1}{N} \sum_{\mathbf{p}_u} \mathbf{H}_u^{mask} \odot \mathbf{s}_u'$$
    - The projection matrix is obtained by summing the importance matrix along the channel dimension followed by softmax:
    $$\mathbf{P}_{c_{new}} = (\text{softmax}(\text{sum}(\mathbf{M}_{c_{new}})))^\top$$
    - **Core Idea**: If an old class is more similar to the new class (i.e., the old model's prediction on the new class pixel biases towards that old class), it receives a larger initial weight in the transformation.

### Loss & Training
- **Pre-tuning stage**: uses unbiased cross-entropy. Pre-tuning takes 5 epochs on Pascal VOC (lr=0.001) and 15 epochs on ADE20K.
- **Formal training stage**: follows the loss functions of the baseline methods (e.g., knowledge distillation + unbiased CE in MiB).
- **Optimizer**: SGD optimizer with an initial step lr=0.02, incremental steps lr=0.001, and poly decay.
- **Extremely low overhead**: pre-tuning only adds about 5.4K parameters and 2.7% training time.

## Key Experimental Results

### Main Results

**Pascal VOC 2012 (Overlapped setting, mIoU %)**:

| Method | Backbone | 10-1 (all) | 15-1 (all) | 15-5 (all) | 19-1 (all) |
|------|----------|-----------|-----------|-----------|-----------|
| MiB | Res101 | 10.2 | 38.2 | 70.2 | 69.6 |
| MiB+NeST | Res101 | **37.4** (+27.2) | **51.8** (+13.6) | 70.7 | 69.7 |
| PLOP | Res101 | 32.2 | 56.2 | 70.8 | 74.0 |
| PLOP+NeST | Res101 | **36.9** (+4.7) | **63.1** (+6.9) | **72.4** | **75.7** |
| RCIL | Res101 | 33.1 | 58.9 | 72.5 | 74.5 |
| RCIL+NeST | Res101 | **36.8** (+3.7) | **61.4** (+2.5) | **72.8** | **74.9** |
| MiB | Swin-B | 15.0 | 36.9 | 77.3 | 78.3 |
| MiB+NeST | Swin-B | **51.2** (+36.2) | **71.4** (+34.5) | **77.9** | **78.8** |

**ADE20K (mIoU %)**:

| Method | 100-50 (all) | 100-10 (all) | 100-5 (all) | 50-50 (all) |
|------|-------------|-------------|-------------|-------------|
| MiB+NeST | **35.1** | **33.7** (+4.4) | **32.7** (+6.8) | **33.2** |
| PLOP+NeST | **36.3** (+3.0) | **34.7** (+2.7) | **32.0** (+2.8) | **34.8** |

### Ablation Study

**Comparison of classifier initialization strategies (VOC 15-1)**:

| Initialization Method | 0-15 mIoU | 16-20 mIoU | all mIoU |
|-----------|----------|----------|---------|
| Random Initialization | 43.5 | 4.2 | 34.1 |
| Background Classifier (MiB) | 45.2 | 15.7 | 38.2 |
| Two-Stage Direct Tuning | 46.0 | 15.3 | 38.7 |
| **NeST (Ours)** | **61.7** | **20.4** | **51.8** |

**Component-wise Ablation (VOC 15-1)**:

| Projection Matrix | Importance Matrix | 0-15 | 16-20 | all |
|---------|----------|------|-------|-----|
| ✗ | ✗ | 45.2 | 15.7 | 38.2 |
| ✓ | ✗ | 53.3 | 15.4 | 44.3 |
| ✗ | ✓ | 59.7 | 19.2 | 50.1 |
| ✓ | ✓ | **61.7** | **20.4** | **51.8** |

**Transformation Matrix Initialization Strategy**:

| Initialization | 0-15 | 16-20 | all |
|--------|------|-------|-----|
| Random Matrix Initialization | 53.8 | 7.5 | 42.8 |
| Cross-Task Similarity Initialization | **61.7** | **20.4** | **51.8** |

### Key Findings
- **Pre-tuning significantly reduces the stability gap**: The cosine similarity of features remains higher than the baseline throughout the recovery process, and the loss starts lower from the very beginning of training.
- **The importance matrix contributes more than the projection matrix**: Using the importance matrix alone (50.1) performs significantly better than using the projection matrix alone (44.3), indicating that channel-wise discrimination is more critical than the combination of old classifiers.
- **Matrix initialization strategy is crucial**: Randomly initializing the matrices improves the old classes (53.8) but leads to a collapse in new classes (7.5. In contrast, the cross-task similarity initialization achieves a balance between stability and plasticity.
- **Minimal computational overhead**: NeST adds only ~5.4K parameters, increases training time by 2.7% (3.7 minutes vs 2.3 hours), and requires only 6.47GB memory compared to AWT's >24GB.
- **Generalizable across backbones**: Effective on both ResNet-101 and Swin-B, with even more significant improvements observed on Swin-B.

## Highlights & Insights
- **The concept of 'learning the transformation instead of weights' is elegant**: Rather than directly tuning the parameters of the new classifier, it learns an old-to-new mapping, naturally inheriting prior knowledge.
- **The dual-matrix design (importance + projection)** cleverly decouples the two questions of "which channels are important" and "how to combine old classifiers."
- **Freezing the backbone during pre-tuning** ensures that the old knowledge is not disrupted during the initialization phase while allowing the new classifier to adapt to the data.
- **Plug-and-play versatility**: It can be directly integrated into existing methods like MiB, PLOP, and RCIL without modifying their original training pipelines.

## Limitations & Future Work
- Introduces computational overhead during the extra pre-tuning phase (albeit minimal, it is non-zero).
- The scale of the importance matrix grows linearly with the number of old classes.
- Only validated on semantic segmentation; it could be extended to other incremental learning scenarios such as instance segmentation and panoptic segmentation.
- Matrix initialization relies on the prediction quality of the old model; if the old model is severely degraded, the similarity estimation may be inaccurate.

## Related Work & Insights
- MiB: The first CISS method to systematically address the background shift problem $\rightarrow$ NeST builds upon it by improving classifier initialization.
- AWT: Selecting relevant weights through gradient attribution $\rightarrow$ NeST replaces attribution with learning, which is more flexible and memory-friendly.
- RCIL: Re-parameterization for continual segmentation $\rightarrow$ NeST can be applied on top of it as an orthogonal component.
- EWF: Weight fusion strategy $\rightarrow$ complementary to NeST's pre-tuning philosophy.
- **Inspiration**: The pre-tuning concept can be generalized to other incremental learning scenarios (e.g., classification, detection), where the core idea serves to "first adapt new components to the old feature space."

## Rating
- Novelty: ⭐⭐⭐⭐ Resolves classifier initialization from the perspective of "learning the transformation" rather than "learning weights", featuring a dedicated design for cross-task similarity initialization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Tested on two datasets, multiple settings (10-1/15-1/15-5/19-1, etc.), integrated with three baselines, across ResNet and Swin backbones, with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear logical flow, well-connected from observation to method design and experimental validation.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play, general improvement strategy with minimal overhead and significant efficacy, offering immediate practical value to the CISS community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Cs2K: Class-Specific and Class-Shared Knowledge Guidance for Incremental Semantic Segmentation](cs2k_class-specific_and_class-shared_knowledge_guidance_for_incremental_semantic.md)
- [\[ECCV 2024\] Learning from the Web: Language Drives Weakly-Supervised Incremental Learning for Semantic Segmentation](learning_from_the_web_language_drives_weakly-supervised_incremental_learning_for.md)
- [\[ECCV 2024\] DreamLIP: Language-Image Pre-training with Long Captions](dreamlip_language-image_pre-training_with_long_captions.md)
- [\[ECCV 2024\] CPM: Class-Conditional Prompting Machine for Audio-Visual Segmentation](cpm_class-conditional_prompting_machine_for_audio-visual_segmentation.md)
- [\[ECCV 2024\] Efficient and Versatile Robust Fine-Tuning of Zero-shot Models](efficient_and_versatile_robust_fine-tuning_of_zero-shot_models.md)

</div>

<!-- RELATED:END -->

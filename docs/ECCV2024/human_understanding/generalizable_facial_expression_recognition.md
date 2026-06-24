---
title: >-
  [Paper Note] Generalizable Facial Expression Recognition
description: >-
  [ECCV 2024][Human Understanding][Facial Expression Recognition] This paper proposes the CAFE method, which learns a Sigmoid Mask on fixed CLIP face features to select expression-related features. Combined with channel separation and channel diversity loss, it achieves zero-shot generalization capabilities that significantly outperform SOTA facial expression recognition methods on multiple unseen datasets, using only a single training set.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Facial Expression Recognition"
  - "Zero-shot Generalization"
  - "Sigmoid Mask"
  - "CLIP"
  - "Domain Generalization"
date: 2026-05-08
content_hash: cd1874f0865f9807
---

# Generalizable Facial Expression Recognition

**Conference**: ECCV 2024  
**arXiv**: [2408.10614](https://arxiv.org/abs/2408.10614)  
**Code**: [Yes](https://github.com/zyh-uaiaaaa/Generalizable-FER)  
**Area**: Human Understanding  
**Keywords**: Facial Expression Recognition, Zero-shot Generalization, Sigmoid Mask, CLIP, Domain Generalization

## TL;DR

This paper proposes the CAFE method, which learns a Sigmoid Mask on fixed CLIP face features to select expression-related features. Combined with channel separation and channel diversity loss, it achieves zero-shot generalization capabilities that significantly outperform SOTA facial expression recognition methods on multiple unseen datasets, using only a single training set.

## Background & Motivation

**Problem Definition**: Current SOTA facial expression recognition (FER) methods perform exceptionally well on test sets corresponding to their training sets but suffer from drastic performance drops on test sets with domain shifts. For instance, the EAC model trained on RAF-DB achieves 89.54% accuracy on the RAF-DB test set but only 43.91% on AffectNet.

**Limitations of Prior Work**:

**Poor generalization of SOTA FER methods**: Methods such as SCN, RUL, EAC, and OFER overfit to domain-specific information of the training set, leading to unstable performance on test sets with domain shifts. Stronger methods (e.g., EAC) perform better in the source domain but worse in cross-domain scenarios than weaker methods, indicating an "overfitting-style" performance improvement.

**Domain adaptive FER methods require target domain data**: Existing domain adaptation methods require labeled or unlabeled target domain samples for fine-tuning, which is often impractical for real-world deployment since the distribution of test samples is unknown beforehand.

**Human-Inspired Key Insight**: Human facial expression recognition follows a two-step process of "locating face first $\rightarrow$ extracting expression features second". When facing images with domain shifts, humans filter out domain-related features (e.g., background, illumination) and then focus on expression-related features to make judgments. CAFE mimics this cognitive process:

- Step 1: Extract generalized face features using a pre-trained foundation model (CLIP).
- Step 2: Train an FER model to learn a Mask, selecting expression-related features from the face features.

**Why can't CLIP features be used directly?** CLIP extracts general face features containing non-expression information such as identity, age, and illumination. Applying generic features directly to FER tasks is non-trivial; a mechanism is required to precisely select expression-related feature dimensions while preventing the degradation of generalization capability.

## Method

### Overall Architecture

CAFE consists of three core components:

1. **Fixed CLIP Feature Extraction**: A frozen CLIP ViT-B/32 is used to extract face features $\mathbf{F} \in \mathbb{R}^{N \times C}$ (fixed throughout training).
2. **Sigmoid Mask Learning**: A ResNet-18 backbone learns the mask, which is applied to CLIP features after Sigmoid regularization.
3. **Channel Separation + Channel Diversity**: The masked features are split channel-wise into 7 groups corresponding to the 7 basic expressions to directly generate logits, bypassing FC layer overfitting.

### Key Designs

1. **Sigmoid Mask Learning (Mask on Fixed Face Features)**

   **Function**: Learn a probabilistic mask to filter out expression-related feature channels from fixed CLIP face features.

   **Mechanism**: The FER model (ResNet-18) extracts feature $\mathbf{f}$, which is reshaped to generate Mask $\mathbf{M}$, and then regularized via the Sigmoid function:

    $\mathbf{M}_s = \text{Sigmoid}(\mathbf{M})$
    $\widetilde{\mathbf{F}} = \mathbf{M}_s \mathbf{F}$

   The selected features $\widetilde{\mathbf{F}}$ are fed into the standard classification loss:

    $l_{cls} = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{\mathbf{W}_{y_i} \widetilde{\mathbf{F}}_i}}{\sum_j^L e^{\mathbf{W}_j \widetilde{\mathbf{F}}_i}}$

   **Design Motivation**: (a) Fixing CLIP features prevents the FER model from optimizing face features and overfitting to the training set, maintaining generalization; (b) The Sigmoid function restricts Mask values to $[0, 1]$, providing probabilistic feature selection semantics where the value of each channel represents its selection probability, analogous to how humans select facial expression features; (c) The non-linearity introduced by Sigmoid helps capture non-linear patterns, while its normalization effect reduces the overfitting capacity of the Mask.

2. **Channel-Separation Module**

   **Function**: Evenly split the 512-dimensional masked features into 7 groups along channels, where each group corresponds to a basic expression, directly generating logits via MaxPool and bypassing the FC layer.

   **Mechanism**: Split $\widetilde{\mathbf{F}}$ along channels into $\{\widetilde{\mathbf{F}}_1, ..., \widetilde{\mathbf{F}}_7\}$ (approx. 73 channels each), and apply random channel dropout followed by MaxPool to each group:

    $\overline{\mathbf{F}}^d = \{\max(\widetilde{\mathbf{F}}_1 \mathbf{M}_1), \max(\widetilde{\mathbf{F}}_2 \mathbf{M}_2), ..., \max(\widetilde{\mathbf{F}}_L \mathbf{M}_L)\}$

   Calculate the separation loss $l_{sep}$ based on $\overline{\mathbf{F}}^d$.

   **Design Motivation**: Three-fold considerations: (a) The FC layer has excessive learning capacity and easily overfits training labels; direct mapping from features to logits is simpler; (b) The 512-dimensional features might be too large, whereas a 73-dimensional Mask per group is more likely to focus only on useful instead of redundant information; (c) Analogous to label distribution learning, an image may contain multiple expression characteristics (e.g., compound expressions), supporting the rationale for the 7 groups focusing on 7 individual expressions.

3. **Channel-Diverse Loss**

   **Function**: Force the Masks corresponding to different expression categories to be as diverse as possible, preventing different expression channels from learning similar feature patterns.

   **Mechanism**:

    $l_{div} = 1 - \frac{1}{Nc} \sum_{i=1}^{N} \sum_{j=1}^{L} \widetilde{\mathbf{F}}_{\max_{ij}}$

   Where $\widetilde{\mathbf{F}}_{\max}$ is the result after MaxPool on each group of features, and $c=73$ is the normalization constant.

   **Design Motivation**: If the Masks corresponding to different expressions are highly similar, the model cannot effectively distinguish the feature subspaces of different expressions. Maximizing the maximum value of each feature group forces each Mask group to select different feature channels, enhancing both the diversity and discriminativeness of the Masks.

### Loss & Training

Total training loss:

$$l_{train} = l_{cls} + \lambda \cdot l_{sep} + \beta \cdot l_{div}$$

- $\lambda = 1.5$ (weight for separation loss), $\beta = 5$ (weight for diversity loss)
- During inference, only the module corresponding to $l_{cls}$ is needed; the $l_{sep}$ and $l_{div}$ branches can be discarded.
- Optimizer: Adam, learning rate: 0.0002, scheduler: ExponentialLR (gamma=0.9)

## Key Experimental Results

### Main Results

Results of training on RAF-DB and testing on five datasets (Accuracy %):

| Method | RAF-DB (Source) | FERPlus | AffectNet | SFEW2.0 | MMA | Mean |
|------|-------------|---------|-----------|---------|-----|------|
| SCN | 87.32 | 58.37 | 42.85 | 44.89 | 36.52 | 53.99 |
| EAC | 89.54 | 54.38 | 43.91 | 43.39 | 37.27 | 53.70 |
| OFER | 89.07 | 53.90 | 42.73 | 43.88 | 36.43 | 53.20 |
| **CAFE** | **88.72** | **73.16** | **45.86** | **52.86** | **56.80** | **63.48** |

CAFE's cross-domain mean accuracy reaches 63.48%, achieving a **+9.78%** improvement over the best baseline EAC (53.70%), and consistently leading by a large margin on all unseen test sets.

### Ablation Study

Ablation results on RAF-DB training:

| Mask | Separation | Diverse | FERPlus | AffectNet | SFEW2.0 | MMA | Mean |
|------|------------|---------|---------|-----------|---------|-----|------|
| ✗ | ✗ | ✗ | 58.05 | 43.25 | 42.76 | 42.61 | 46.67 |
| ✓ | ✗ | ✗ | 70.90 | 43.77 | 51.63 | 55.65 | 55.49 |
| ✓ | ✓ | ✗ | 72.01 | 45.17 | 53.31 | 56.69 | 56.80 |
| ✓ | ✓ | ✓ | **73.16** | **45.86** | **52.86** | **56.80** | **57.17** |

- Sigmoid Mask is the most critical component: +8.82% mean improvement
- Channel separation yields an additional +1.31% improvement
- Channel diversity loss further raises performance by +0.37%

### Key Findings

- **The "Overfitting Trap" of SOTA FER Methods**: EAC outperforms SCN in the source domain but matches or performs worse in cross-domain scenarios, indicating that stronger fitting capacity can lead to poorer generalization.
- **The Critical Role of Sigmoid**: The Mask without Sigmoid achieves a mean of only 58.05%, while using Sigmoid reaches 63.48% — the normalization effect is key to preventing Mask overfitting.
- **CLIP+Finetune is Far Inferior to CAFE**: Under the same parameter scale, CLIP+Finetune only achieves 55.73% compared to CAFE's 63.48% (+7.75%), proving that success stems from the Mask design rather than just raw CLIP features.
- **Effectiveness Across Different Backbones**: Improvements are consistently observed across various backbones: MobileNet (+1.64%), ResNet-18 (+8.47%), and ResNet-50 (+3.09%).

## Highlights & Insights

1. **Cognitively-Inspired Design Philosophy**: Mimics the two-step human cognitive process of "locate face first $\rightarrow$ observe expression second", decomposing the problem into decoupled steps of feature extraction and feature selection.
2. **'Frozen + Learning' Paradigm**: Freezing foundation model features preserves generalization, while learning a lightweight Mask maintains accuracy. Decoupling the two is the key to success.
3. **Clever Design to Bypass FC Layers**: Channel separation followed by direct MaxPool to generate logits removes a major source of overfitting.
4. **Low Inference Costs**: Auxiliary modules used during training can be discarded at inference, requiring only CLIP + ResNet-18 + Mask for deployment.

## Limitations & Future Work

- Only evaluates 7 basic expressions; not extended to fine-grained expressions or continuous emotional spaces (valence-arousal value prediction).
- Best performance is reached when the CLIP feature dimension (512) matches the ResNet-18 output dimension; dimension mismatches requiring simple mean pooling may limit performance.
- Has not explored stronger foundation models (e.g., DINOv2, EVA-CLIP) as feature extractors.
- Channel separation assumes uniform splitting without considering that different expressions may require varying numbers of feature channels.

## Related Work & Insights

- **CLIP for downstream tasks**: The paradigm of using fixed CLIP features with a lightweight task head has proven effective in various fields; CAFE provides an elegant adaptation scheme for FER.
- The methodology is generalizable to other fine-grained recognition tasks (e.g., micro-expression recognition, action unit detection) where domain shifts exist between source and target domains.
- The strategy of channel separation + MaxPool to directly generate logits could inspire other classification tasks that require overfitting prevention.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The concept of learning a Sigmoid Mask on fixed foundation model features is novel and intuitive. Bypassing FC layers with channel separation is a clever design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive, featuring cross-validation across 5 datasets, 5 training setups, detailed ablations, multiple backbones, comparison with CLIP+Finetune, analysis of the Sigmoid effect, and hyperparameter sensitivity studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear problem definition, vivid analogy to human cognition, and comprehensive charts and tables.
- **Value**: ⭐⭐⭐⭐ — The first to systematically study the zero-shot generalization of FER. The method is simple yet highly efficient, offering significant reference value for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data](../../ICCV2025/human_understanding/synfer_towards_boosting_facial_expression_recognition_with_synthetic_data.md)
- [\[CVPR 2026\] Dynamic Label Noise Suppression with Optimal Teacher Pool for Facial Expression Recognition](../../CVPR2026/human_understanding/dynamic_label_noise_suppression_with_optimal_teacher_pool_for_facial_expression_.md)
- [\[ECCV 2024\] TF-FAS: Twofold-Element Fine-Grained Semantic Guidance for Generalizable Face Anti-Spoofing](tf-fas_twofold-element_fine-grained_semantic_guidance_for_generalizable_face_ant.md)
- [\[CVPR 2026\] CLEX: Complementary Label Exchange Learning for Noisy Facial Expression Recognition](../../CVPR2026/human_understanding/clex_complementary_label_exchange_learning_for_noisy_facial_expression_recogniti.md)
- [\[CVPR 2026\] D³FER: Dual Channel and Dual Branch Network for Robust Facial Expression Recognition under Dual Challenges](../../CVPR2026/human_understanding/d3fer_dual_channel_and_dual_branch_network_for_robust_facial_expression_recognit.md)

</div>

<!-- RELATED:END -->

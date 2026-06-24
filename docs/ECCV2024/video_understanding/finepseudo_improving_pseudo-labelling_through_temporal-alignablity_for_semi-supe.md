---
title: >-
  [Paper Note] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition
description: >-
  [ECCV 2024][Video Understanding][Fine-Grained Action Recognition] This paper proposes the FinePseudo framework, which utilizes metric learning based on temporal alignability to improve pseudo-label quality. It represents the first systematic approach to semi-supervised fine-grained action recognition, significantly outperforming existing methods on four fine-grained datasets.
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Fine-Grained Action Recognition"
  - "Pseudo-Labelling"
  - "Temporal Alignment"
  - "Semi-Supervised Learning"
  - "Metric Learning"
date: 2026-05-08
content_hash: 24f4e437f22e088d
---

# FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition

**Conference**: ECCV 2024  
**arXiv**: [2409.01448](https://arxiv.org/abs/2409.01448)  
**Code**: [Available](https://daveishan.github.io/finepsuedo-webpage/)  
**Area**: Video Understanding / Semi-Supervised Learning  
**Keywords**: Fine-Grained Action Recognition, Pseudo-Labelling, Temporal Alignment, Semi-Supervised Learning, Metric Learning

## TL;DR

This paper proposes the FinePseudo framework, which utilizes metric learning based on temporal alignability to improve pseudo-label quality. It represents the first systematic approach to semi-supervised fine-grained action recognition, significantly outperforming existing methods on four fine-grained datasets.

## Background & Motivation

Fine-Grained Action Recognition (FGAR) is crucial in practical applications such as sports analysis, surgical videos, and AR/VR. Unlike coarse-grained actions (e.g., "playing guitar" vs. "throwing javelin"), differences between fine-grained actions (e.g., different types of diving) are manifested only in subtle variations of action phases—for example, among the three phases of takeoff, flight, and entry, a difference in only the entry posture can change the action category.

However, annotating fine-grained actions is extremely expensive, requiring experts to repeatedly watch videos to make accurate annotations. This makes semi-supervised learning a natural choice for FGAR. Nevertheless, existing semi-supervised video methods face two core challenges:

**Scene Bias Dependency**: Existing methods (such as augmentation strategies like token-mix and CutMix) primarily rely on scene context to distinguish actions. Because fine-grained actions typically occur in identical scenes (e.g., all dives take place at a diving platform), these strategies fail.

**Insufficient Temporal Granularity**: Video-level self-supervised methods (e.g., TimeBal) learn representations that lack frame-level phase information, which is critical for FGAR.

The authors conducted a key preliminary experiment: after extracting embeddings using a frame-level video encoder, they compared the capability of different distance metrics in distinguishing intra-class vs. inter-class fine-grained action pairs. The results indicated that:
- Cosine distance after temporal pooling loses temporal details.
- Frame-by-frame cosine distance cannot handle variations in the duration of different action phases.
- **Dynamic Time Warping (DTW) alignment distance** allows phase-to-phase comparisons, significantly improving discriminative capability.

This finding has never been previously explored under the label-limited setting of FGAR.

## Method

### Overall Architecture

FinePseudo is a co-training framework based on pseudo-labeling, consisting of two branches:

- **Action Encoder** $f_E$: Learns video-level high-level semantic features (action classification).
- **Alignability Encoder** $f_A$: A frame-level video encoder (VTN) that learns low-level intra-frame representations based on action phases.

The training workflow consists of three stages: (1) self-supervised pre-training on unlabeled data; (2) alignability-verification metric learning on labeled data; and (3) collaborative pseudo-label self-training.

### Key Designs

1. **Alignability-Verification Metric Learning**

   Core Hypothesis: Fine-grained videos of the same class are more "alignable" than those of different classes.

   For a video pair $U, V$, frame-level embeddings $\mathbf{u}, \mathbf{v} \in \mathbb{R}^{T \times F}$ are extracted via $f_A$ to construct a cost matrix $\mathbb{C}(i,j) = h(\mathbf{u}(i), \mathbf{v}(j))$ (cosine distance). Then, softDTW is used to compute the differentiable alignment distance:

    $D(\mathbf{u}, \mathbf{v}) = \mathbb{C}(i,j) + \gamma\text{-smooth-min}(\Pi_{\text{cost}}(i,j))$

   Based on this alignment distance, triplet loss is applied for metric learning:

    $\mathcal{L}_{AT} = \sum_{i=1}^{N} [D(\mathbf{v}^{(i)}, \mathbf{v}^{(j)}) - D(\mathbf{v}^{(i)}, \mathbf{v}^{(k)}) + m]$

   where positive pairs belong to the same class, negative pairs belong to different classes, and hard-negative mining is utilized.

2. **Learnable Alignability Score**

   The alignment distance $D$ is mapped to $[0,1]$ through a non-linear scaling function and a sigmoid function:

    $S(\mathbf{u}, \mathbf{v}) = \varsigma(f_S(D(\mathbf{u}, \mathbf{v})))$

   This score function is trained using binary cross-entropy:

    $\mathcal{L}_{Score} = -[y_A \log(S) + (1-y_A)\log(1-S)]$

   The overall optimization objective for alignability is $\mathcal{L}_{AV} = \mathcal{L}_{AT} + \omega \mathcal{L}_{Score}$. This score provides better class discriminative power compared to the raw DTW distance.

3. **Collaborative Pseudo-Labeling**

   For each unlabeled video $U$, predictions are obtained simultaneously from both encoders:
    - $\mathbf{p}_E$: Output directly by the classification head of $f_E$.
    - $\mathbf{p}_A$: Based on a non-parametric classifier, which is calculated as the average alignability score $\bar{S}_c$ between $U$'s embedding and labeled samples of each class, followed by a softmax with temperature:

    $\mathbf{p}_A(c) = \frac{\exp(\bar{S}_c / \tau)}{\sum_j \exp(\bar{S}_j / \tau)}$

   The final prediction is $\mathbf{p} = \mathbf{p}_A + \mathbf{p}_E$. Samples exceeding a confidence threshold $\theta$ are assigned pseudo-labels and added to the training set. The two branches provide complementary information (video-level vs. alignability) to iteratively update the pseudo-labels.

### Loss & Training

- $f_A$ is first pre-trained with GITDL in a self-supervised manner to learn inter-frame dynamics, and then trained with $\mathcal{L}_{AV}$ on labeled data.
- $f_E$ is trained using the standard cross-entropy loss $\mathcal{L}_{CE}$.
- The self-training phase proceeds iteratively: generating collaborative pseudo-labels $\rightarrow$ expanding the labeled set $\rightarrow$ retraining both encoders.

## Key Experimental Results

### Main Results

Across four fine-grained datasets using the R2plus1D-18 backbone:

| Dataset | Label Ratio | FinePseudo | TimeBal (Prev. SOTA) | Gain |
|--------|---------|------------|-------------------|------|
| Diving48 | 5% | **20.9** | 15.8 | +5.1 |
| Diving48 | 10% | **37.6** | 33.7 | +3.9 |
| Diving48 | 20% | **60.4** | 56.3 | +4.1 |
| FineGym99 | 5% | **49.2** | 44.4 | +4.8 |
| FineGym99 | 10% | **69.9** | 65.9 | +4.0 |
| FineGym288 | 5% | **41.7** | 37.3 | +4.4 |
| FineDiving | 5% | **28.4** | 25.1 | +3.3 |

Using AIM-ViTB (with CLIP initialization) on Diving48:

| Method | 5% | 10% | 20% |
|------|-----|------|------|
| SVFormer | 38.00 | 56.02 | 76.20 |
| TimeBal | 38.12 | 55.80 | 76.01 |
| **FinePseudo** | **43.02** | **60.79** | **80.02** |

### Ablation Study

| Configuration | 10% Acc | 20% Acc | Description |
|------|---------|---------|------|
| Only $f_E$ + PL | 33.40 | 54.00 | Baseline pseudo-labeling |
| Only $f_A$ | 32.82 | 51.05 | Alignment encoder alone is insufficient |
| $f_E$ + $\mathcal{L}_{AT}$ only | 33.73 | 55.67 | Contribution of triplet loss |
| $f_E$ + $\mathcal{L}_{Score}$ only | 36.11 | 59.32 | Score loss contributes more |
| W/o SSL pre-training | 35.23 | 58.64 | SSL pre-training is helpful |
| **Full FinePseudo** | **37.64** | **60.40** | Joint collaboration of all components is optimal |

### Key Findings

- FinePseudo consistently outperforms prior methods by 4-5% in absolute accuracy across all fine-grained datasets.
- It also exhibits competitive or slightly improved performance on coarse-grained datasets (K400, SSv2).
- Under the open-world setting (where unlabeled data contains out-of-distribution classes), the non-parametric classifier effectively filters out unknown classes, demonstrating robustness.
- The alignability score loss $\mathcal{L}_{Score}$ contributes more than the triplet loss $\mathcal{L}_{AT}$ when used individually, but their combination achieves the best performance.

## Highlights & Insights

- **Elegant Core Insight**: Extends "alignability" from traditional intra-class alignment assumptions to an inter-class discriminative metric specifically designed for the semi-supervised setting.
- **Strong Complementarity**: Video-level semantic prediction and frame-level alignability prediction provide orthogonal sources of information.
- **Generality**: The method is applicable to both fine-grained and coarse-grained action recognition.
- **Open-World Robustness**: The non-parametric classifier is naturally suited for handling out-of-distribution categories.

## Limitations & Future Work

- The computational complexity of softDTW is $O(T^2)$, which may raise efficiency issues for long videos.
- It relies on SSL pre-trained weights (TCLR/Kinetics400), making it sensitive to pre-training quality.
- The alignability assumption might see diminished effectiveness for unstructured actions (e.g., daily activities without explicit phases).
- The impact of class imbalance in unlabeled data remains unexplored.

## Related Work & Insights

- Unlike alignment methods like LAV (Learning by Aligning Videos), the "alignability verification" in this work does not assume that videos are necessarily alignable; instead, it learns to assess alignability difficulty.
- The alignability score can serve as a general video similarity metric, with potential applications in retrieval, quality assessment, and other tasks.
- The collaborative pseudo-labeling idea can be extended to other semi-supervised scenarios requiring complementary sources of information.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing alignability into semi-supervised FGAR is an original and effective insight.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation on 4 fine-grained and 2 coarse-grained datasets, comprehensive ablation, and a novel open-world setting.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation starts from preliminary experiments with smooth logic.
- **Value**: ⭐⭐⭐⭐ — First systematic study on semi-supervised FGAR, presenting practical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] SA-DVAE: Improving Zero-Shot Skeleton-Based Action Recognition by Disentangled Variational Autoencoders](sa-dvae_improving_zero-shot_skeleton-based_action_recognition_by_disentangled_va.md)
- [\[AAAI 2026\] FineTec: Fine-Grained Action Recognition Under Temporal Corruption via Skeleton Decomposition and Sequence Completion](../../AAAI2026/video_understanding/finetec_fine-grained_action_recognition_under_temporal_corruption_via_skeleton_d.md)
- [\[ECCV 2024\] Leveraging Temporal Contextualization for Video Action Recognition](leveraging_temporal_contextualization_for_video_action_recognition.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](../../CVPR2026/video_understanding/ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[ECCV 2024\] Referring Atomic Video Action Recognition](referring_atomic_video_action_recognition.md)

</div>

<!-- RELATED:END -->

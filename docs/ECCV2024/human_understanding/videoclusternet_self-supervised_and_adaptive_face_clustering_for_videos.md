---
title: >-
  [Paper Note] VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos
description: >-
  [ECCV 2024][Human Understanding][Face Clustering] VideoClusterNet proposes a fully self-supervised video face clustering method: adapting a generic face recognition model via a self-distillation mechanism, and designing a parameter-free clustering algorithm based on a learned loss metric, achieving SOTA performance in movie/TV show scenarios.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Face Clustering"
  - "Self-Supervised Learning"
  - "Self-Distillation"
  - "Video Understanding"
  - "Parameter-Free Clustering"
date: 2026-05-08
content_hash: f360f38e7b597c99
---

# VideoClusterNet: Self-Supervised and Adaptive Face Clustering for Videos

**Conference**: ECCV 2024  
**arXiv**: [2407.12214](https://arxiv.org/abs/2407.12214)  
**Code**: None  
**Area**: Human Understanding / Video Face Clustering  
**Keywords**: Face Clustering, Self-Supervised Learning, Self-Distillation, Video Understanding, Parameter-Free Clustering

## TL;DR

VideoClusterNet proposes a fully self-supervised video face clustering method: adapting a generic face recognition model via a self-distillation mechanism, and designing a parameter-free clustering algorithm based on a learned loss metric, achieving SOTA performance in movie/TV show scenarios.

## Background & Motivation

**Background**: Video face clustering aims to group detected faces in videos by identity, which serves as a crucial foundation for tasks like video scene description, video question answering, and video understanding. With the growth of film and television content production, the demand for face clustering in the movie/TV show domain is increasing.

**Limitations of Prior Work**:
   - Generic pretrained face recognition (Face ID) models have limited effectiveness in movie scenes due to high dynamic range, unique cinematic styles, and extreme variations in pose, lighting, and expression.
   - **Difficulty in negative sample mining**: Most methods rely on positive and negative pairs in contrastive learning, where negative samples are derived from temporal constraints of co-occurring tracks. However, incorrect negative samples lead to sub-optimal solutions.
   - **Dependence on clustering hyperparameters**: Bottom-up methods require predefined distance thresholds, while top-down methods require a predefined number of clusters—both of which are non-intuitive manual parameters.
   - Lack of high-quality movie face clustering benchmark datasets.

**Key Challenge**: Faces of the same character in movie scenes vary dramatically (pose, lighting, makeup), but generic models are not adapted to specific video styles, and high annotation costs make supervised finetuning unfeasible.

**Goal**: (1) Adaptively finetune Face ID models to specific videos without manual annotations; (2) Design a clustering algorithm that requires no user-input parameters.

**Key Insight**: Apply self-supervised learning with a self-distillation (DINO-style) mechanism to learn positive pairs from video face tracks, completely skipping negative sample mining; and use the training loss function itself as the distance metric for clustering.

**Core Idea**: Self-distillation finetuning using only positive pairs + using training loss directly as the clustering distance metric = fully automated video face clustering.

## Method

### Overall Architecture

VideoClusterNet consists of three stages:
- **Stage 1 — Track Preprocessing**: Scene boundary detection $\to$ Face detection $\to$ Quality filtering $\to$ Motion tracking to generate face tracks
- **Stage 2 — Self-Supervised Model Finetuning**: Finetuning the Face ID model on intra-track and inter-track positive pairs using teacher-student self-distillation
- **Stage 3 — Parameter-Free Clustering**: Hierarchical clustering using the embeddings and training loss of the finetuned model as distance metrics to adaptively calculate matching thresholds for each track

### Key Designs

1. **Self-Supervised Model Finetuning**:

   **Function**: Adapt generic Face ID models to face variations observed in specific videos.

   **Mechanism**: Utilizing a teacher-student self-distillation mechanism. A randomly initialized MLP head is attached to a pretrained Face ID model, duplicating it into teacher and student branches. Positive pairs are sampled from the same or matching tracks and trained using a cross-entropy-style similarity loss:

    $L_{ssl} = -1 \times softmax\left(\frac{embed_t - c}{temp}\right) \times \log(softmax(embed_s))$

   Where $embed_t$ and $embed_s$ are the outputs of the teacher and student respectively, and $c$ is the running mean of the teacher's embeddings. Gradients only backpropagate through the student branch, and teacher weights are updated via an exponential moving average of the student.

   **Design Motivation**: 
    - **Positive pairs only**: Completely bypasses negative sample mining to avoid sub-optimal solutions caused by false negatives (e.g., the same character in different scenes being mistakenly treated as different people).
    - **Multi-stage training**: Freeze the base model first to train only the head (which is randomly initialized), followed by joint finetuning, ensuring training stability.
    - **Data augmentation**: Horizontal flipping, rotation, and color temperature changes are used to reduce reliance on natural video variations.

2. **Coarse Track Matching**:

   **Function**: Match different tracks across shot boundaries that likely belong to the same identity, providing more sample pairs for finetuning.

   **Mechanism**: Fit a multivariate Gaussian distribution $N_{t_j}(\mu_{t_j}, \Sigma_{t_j})$ to the face embeddings of each track. The adaptive threshold is determined by taking the mean of the bottom 25% PDF values of the embeddings within that track. If the PDF value of the mean embedding of another track $\ge$ the threshold, the two tracks are considered a match.

   **Design Motivation**: A single track is limited to a single shot with restricted lighting/appearance variations. By matching across shots, the model gets exposed to the full range of variation a character exhibits throughout the film. Finetuning and coarse matching are executed alternately to progressively improve matching quality.

3. **Parameter-Free Track Clustering**:

   **Function**: Cluster all tracks by identity without requiring any user-specified parameters.

   **Mechanism**: 
    - **Distance Metric**: Directly utilize the SSL training loss $L_{ssl}$ as the track similarity metric (asymmetric, taking the average of both directions).
    - **Adaptive Threshold**: Compute pairwise $L_{ssl}$ for all sampled faces within each track, taking the mean as the matching threshold for that track.
    - **Hierarchical Aggregation**: Initially, each track is treated as a cluster $\to$ pairwise comparison is performed $\to$ if the match value is lower than the threshold of either party, they are merged $\to$ transitive merging is applied $\to$ iterate until no new merges occur.

   **Design Motivation**: 
    - The training loss naturally optimizes the embedding space during the finetuning process, making it a more suitable distance metric than Euclidean or Cosine distance.
    - Each track has an independent threshold to adapt to the model's differing confidence levels for different identities.

### Track Quality Estimation

SER-FIQ is employed to evaluate the model's certainty regarding faces using dropout. Low-quality tracks are labeled as Unknown and excluded from clustering. MAD (Median Absolute Deviation) is used for outlier detection.

### Loss & Training

- Training Strategy: Freeze base model $\to$ Train MLP head $\to$ Alternate joint finetuning + Coarse matching.
- All hyperparameters (epochs, batch size, learning rate) remain constant across datasets.
- Image Augmentations: Horizontal flipping, rotation, color temperature variation.

## Key Experimental Results

### Main Results — BBT and BVS TV Show Benchmarks

| Method | BBT S01 Combined WCP(%) | BVS S05 Combined WCP(%) |
|------|------------------------|------------------------|
| SCTL | 66.48 (E1 only) | - |
| TSiam | 96.4 (E1 only) | 92.46 (E2 only) |
| SSiam | 96.2 (E1 only) | 90.87 (E2 only) |
| MLR | 83.71 | 66.37 |
| BCL | 89.63 | 83.62 |
| CCL | 98.2 (E1 only) | 92.1 (E2 only) |
| VCTRSF | 94.20 | - |
| **Ours** | **98.70** | **96.10** |

### Ablation Study — MovieFaceCluster: The Hidden Soldier

| Ablation Dimension | Configuration | Clustering Accuracy (%) | Clustering Ratio (Pred/GT) |
|---------|------|-------------|-----------------|
| Model Finetuning | Un-finetuned + HAC | 86.10 | - |
| Model Finetuning | **Finetuned + HAC** | **91.52** | - |
| Clustering Algorithm | HAC + Cosine | 91.52 | 1.43 (30/21) |
| Clustering Algorithm | Ours + Cosine | 93.70 | 2.0 (42/21) |
| Clustering Algorithm | Ours + Euclidean | 96.50 | 3.5 (74/21) |
| Clustering Algorithm | **Ours + Loss Func.** | **98.50** | **1.04 (22/21)** |
| Base Model | FaRL-P16 (baseline/ours) | 78.7 $\to$ 90.2 | - |
| Base Model | VGGFace2-R50 (baseline/ours) | 84.2 $\to$ 95.7 | - |
| Base Model | ArcFace-R100 (baseline/ours) | 86.1 $\to$ 98.5 | - |
| Base Model | AdaFace-R100 (baseline/ours) | 86.9 $\to$ 98.4 | - |

### Key Findings

- **Self-supervised finetuning yields a ~6% improvement in clustering accuracy** (86.1 $\to$ 91.5), and achieves a 5-12% performance boost across all evaluated Face ID models (CNN/Transformer).
- **Using the loss function as the distance metric far outperforms predefined metrics**: Loss Func. (98.5%, ratio 1.04) vs Cosine (91.5%, ratio 1.43) vs Euclidean (96.5%, ratio 3.5). The loss metric not only yields the highest accuracy but also predicts a cluster number closest to the ground truth.
- **MovieFaceCluster dataset** is more challenging: average quality score of 0.706 (lower than BBT's 0.714 and BVS's 0.712), and an average of 23.2 characters (far exceeding BBT's 6.33 and BVS's 14.5).
- Achieves up to 99.70% on a single episode of BBT (S1E1) and 98.70% Combined.
- Achieves up to 99.10% on a single episode of BVS (S5E2) and 96.10% Combined.
- Comprehensively outperforms all methods on the MovieFaceCluster dataset across 9 movies.

## Highlights & Insights

1. **Elegant "positive-only" approach**: Completely bypassing negative sample mining directly eliminates the difficulty of constructing negative samples in video face clustering (temporal co-occurrence does not guarantee different identities).
2. **Elegant closed-loop of "Training Loss = Clustering Metric"**: Since the embedding space is optimized precisely for this loss metric, using it for clustering is naturally optimal without requiring any prior assumptions about the embedding space.
3. **Adaptive thresholds eliminating hyperparameters**: Each track has its own matching threshold, which dynamically adapts to the model's varying confidence for different identities.
4. **Strong generalizability**: Significant improvements are consistently demonstrated across both CNN-based (ArcFace-R100) and Transformer-based (FaRL-P16) architectures.

## Limitations & Future Work

1. **Propagation of pre-training bias**: If the generic Face ID model incorrectly assigns high similarity to two different individuals, self-distillation may reinforce this error.
2. **Computational efficiency**: Each video requires independent model finetuning, which is unsuitable for real-time applications. Future work could explore test-time adaptation or few-shot approaches.
3. **Lack of contextual utilization (clothing, body, etc.)**: The method relies solely on facial features, ignoring richer context such as clothing, body style, etc.
4. **Limited scale of the MovieFaceCluster dataset**: It only comprises 9 movies, which does not cover larger-scale scenarios.
5. **Convergence speed of iterative clustering**: The number of iterations in hierarchical aggregation was not analyzed, and efficiency could become a bottleneck for large track sets.

## Related Work & Insights

- **Self-distillation (DINO/iBOT style)**: This paper effectively adapts image-level self-distillation to positive-pair learning in video face tracks, showcasing the potential of SSL methods in domain-specific finetuning.
- **TSiam/SSiam**: These methods require complex negative sample mining strategies (co-occurrence constraints, pseudo-relevance feedback), whereas this work proves that relying solely on positive samples can outperform them.
- **BCL**: Requires the pre-defined target number of clusters as input, whereas this method is completely automated.
- **VCTRSF**: A video-centric Transformer method that leverages temporal modeling but similarly relies on manual parameters.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The closed-loop design of "positive-only + training loss as distance metric" is simple and highly novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across three major datasets (BBT/BVS/MovieFaceCluster) and analyzed via ablation studies across three dimensions (finetuning, clustering, base models).
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation, with a complete presentation of the algorithmic pseudo-logic.
- **Value**: ⭐⭐⭐ The lack of open-source code and the requirement for per-video finetuning limit its practical application.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Pose-Aware Self-Supervised Learning with Viewpoint Trajectory Regularization](pose-aware_self-supervised_learning_with_viewpoint_trajectory_regularization.md)
- [\[ICCV 2025\] Bi-Level Optimization for Self-Supervised AI-Generated Face Detection](../../ICCV2025/human_understanding/bi-level_optimization_for_self-supervised_ai-generated_face_detection.md)
- [\[ECCV 2024\] AdaDistill: Adaptive Knowledge Distillation for Deep Face Recognition](adadistill_adaptive_knowledge_distillation_for_deep_face_rec.md)
- [\[CVPR 2025\] FSFM: A Generalizable Face Security Foundation Model via Self-Supervised Facial Representation Learning](../../CVPR2025/human_understanding/fsfm_a_generalizable_face_security_foundation_model_via_self-supervised_facial_r.md)
- [\[ECCV 2024\] Avatar Fingerprinting for Authorized Use of Synthetic Talking-Head Videos](avatar_fingerprinting_for_authorized_use_of_synthetic_talking-head_videos.md)

</div>

<!-- RELATED:END -->

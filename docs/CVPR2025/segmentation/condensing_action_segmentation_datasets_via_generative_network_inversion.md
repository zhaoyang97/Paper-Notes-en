---
title: >-
  [Paper Note] "Condensing Action Segmentation Datasets via Generative Network Inversion"
description: >-
  [CVPR2025][Segmentation][Paper Note] Academic paper note for "Condensing Action Segmentation Datasets via Generative Network Inversion".
tags:
  - CVPR2025
  - Segmentation
date: 2026-05-08
content_hash: 48fae9a8a67544d4
---
# Condensing Action Segmentation Datasets via Generative Network Inversion

**Conference**: CVPR 2025  
**Institution**: National University of Singapore (NUS)  
**Subject**: Dataset Distillation / Temporal Action Segmentation  

## Background & Motivation

### Background

**Background**: Temporal Action Segmentation aims to assign each frame in a long video sequence to its corresponding action category, which is an important task in video understanding. However, training high-performance action segmentation models requires large amounts of long video data and dense frame-by-frame annotations, incurring massive storage and transmission costs.

**Limitations of Prior Work:**

**Massive Data Scale**: Action segmentation datasets typically contain long continuous videos. For example, the Breakfast dataset contains approximately 28GB of feature data, and the 50Salads dataset also contains about 4.5GB.

**Extremely High Annotation Cost**: Action category annotations are required for every frame of the video, which is significantly more expensive than image classification annotations.

**Privacy and Distribution Issues**: In certain fields (such as healthcare and industry), original video data may contain sensitive information and cannot be freely distributed.

**Limitations of Existing Distillation Methods**: Image-level dataset distillation methods (e.g., Dataset Distillation) are difficult to apply directly to temporal data, since action segmentation involves long-sequence temporal structures and action transition patterns.

**Temporal Consistency**: Distilled data must maintain the temporal structure of actions; simple frame-level sampling destroys action continuity and transition patterns.

The key insight of this paper is that **generative models can be utilized to learn data distributions, and network inversion can then be employed to find compact latent representations that represent the entire dataset.**

## Method

### Overall Architecture

The proposed method consists of two core phases: (1) training a TCA generative model to learn the data distribution; (2) optimizing the latent codes through network inversion to find compact representations that can replace the original dataset.

### TCA Generative Model

TCA (Temporally Coherent Action) is a generative model specifically designed for action segmentation data, capable of generating temporally coherent action sequences.

**Model Architecture:**

The generator $G$ takes the latent code $z$, the action label sequence $a$, and the conditional information $c$ as input, and generates the corresponding feature sequence:

$$x_{\text{gen}} = G(z, a, c)$$

**Temporal Consistency Design:**

TCA ensures that the generated action sequences are temporally coherent through the following mechanisms:

- The action label sequence $a$ provides frame-level action guidance
- The conditional information $c$ encodes the global action transition patterns
- Temporal convolutions are used inside the generator to maintain local temporal continuity

### Network Inversion and Latent Code Optimization

The core step is to optimize the latent code to find representations that can reconstruct the original data:

$$z^* = \arg\min_{z} \| D(z, a, c) - x \|^2$$

where $D$ is the decoder and $x$ is the original data. The optimized latent code $z^*$ is much more compact than the original data, requiring only a low-dimensional vector to store each sample.

### Diverse Sequence Sampling

To ensure that the distilled dataset covers the diversity of the original data, this paper designs a diversity selection strategy based on edit distance and farthest point sampling:

| Strategy | Objective | Method |
|------|------|------|
| Edit Distance | Measure structural differences between action sequences | Calculate the edit distance between action label sequences |
| Farthest Point Sampling | Select the most scattered samples in the latent space | Greedily select the point furthest from the already selected samples |
| Class Balancing | Ensure sufficient coverage for each action category | Allocate sampling quotas proportionally according to action categories |

### Training After Distillation

Training action segmentation models using the distilled compact dataset:

1. Randomly sample from the stored latent codes
2. Decode into feature sequences through the frozen generator
3. Train the segmentation model using the decoded features and corresponding labels

## Key Experimental Results

### Compression Ratio

| Dataset | Original Size | Distilled Size | Compression Ratio |
|--------|----------|----------|--------|
| Breakfast | 28GB | 44MB | 636× |
| 50Salads | 4.5%GB | 3.9%MB | ~1150× |

### Segmentation Performance

| Dataset | Method | MS-TCN Accuracy |
|--------|------|-------------|
| Breakfast | Distilled Data | 55.5% |
| Breakfast | Full Data | 67.2% |
| 50Salads | Distilled Data | 74.4% |
| 50Salads | Full Data | 80.6% |

Although there is some performance loss with the distilled data, this performance retention is impressive considering the hundreds-fold compression ratio.

### Ablation Study

- The temporal consistency design of the TCA generative model is critical; using common generative models results in unnatural jumps at action transitions in the generated sequences.
- The diversity sampling strategy significantly outperforms random sampling, validating the importance of covering the diversity of the data distribution.
- There is a trade-off between accuracy and compression ratio regarding the dimension of the latent code.

## Summary & Outlook

This paper is the first to apply dataset distillation technology to the temporal action segmentation task, achieving extreme data compression through the TCA generative model and network inversion. It achieves a 636× compression ratio on the Breakfast dataset and approximately a 1150× compression ratio on the 50Salads dataset, while retaining respectable segmentation performance. This method opens up a new direction for data-efficient video understanding research and can be extended to other sequential tasks and larger-scale datasets in the future.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Audio-Visual Instance Segmentation](audio-visual_instance_segmentation.md)
- [\[ICML 2025\] ActionPiece: Contextually Tokenizing Action Sequences for Generative Recommendation](../../ICML2025/segmentation/actionpiece_contextually_tokenizing_action_sequences_for_generative_recommendati.md)
- [\[CVPR 2025\] Scale Efficient Training for Large Datasets](scale_efficient_training_for_large_datasets.md)
- [\[CVPR 2025\] RSONet: Region-guided Selective Optimization Network for RGB-T Salient Object Detection](rsonet_region-guided_selective_optimization_network_for_rgb-t_salient_object_det.md)
- [\[CVPR 2025\] Generative Video Propagation](generative_video_propagation.md)

</div>

<!-- RELATED:END -->

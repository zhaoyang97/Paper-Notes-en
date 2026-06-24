---
title: >-
  [Paper Note] Consistency-aware Self-Training for Iterative-based Stereo Matching
description: >-
  [CVPR 2025][3D Vision][Stereo Matching] This paper proposes the first consistency-aware self-training framework (CST-Stereo) for iterative-based stereo matching. It evaluates pseudo-label reliability through multi-resolution prediction consistency filtering and iterative prediction consistency filtering, and combines them with a soft-weighted loss to leverage unlabeled real-world data effectively, thereby improving model performance and generalization.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Stereo Matching"
  - "Self-Training"
  - "Pseudo-label Filtering"
  - "Consistency-aware"
  - "Iterative Optimization"
date: 2026-05-08
content_hash: 2a2f7a6fdc6be355
---

# Consistency-aware Self-Training for Iterative-based Stereo Matching

**Conference**: CVPR 2025  
**arXiv**: [2503.23747](https://arxiv.org/abs/2503.23747)  
**Code**: None  
**Area**: 3D Vision / Stereo Matching  
**Keywords**: Stereo Matching, Self-Training, Pseudo-label Filtering, Consistency-aware, Iterative Optimization

## TL;DR

This paper proposes the first consistency-aware self-training framework (CST-Stereo) for iterative-based stereo matching. It evaluates pseudo-label reliability through multi-resolution prediction consistency filtering and iterative prediction consistency filtering, and combines them with a soft-weighted loss to leverage unlabeled real-world data effectively, thereby improving model performance and generalization.

## Background & Motivation

### Background

**Background**: Iterative methods (e.g., RAFT-Stereo, IGEV-Stereo) have become the mainstream in stereo matching, but they rely heavily on annotated data.

### Limitations of Prior Work

**Limitations of Prior Work**: Obtaining high-quality stereo annotations is extremely expensive. Existing annotated datasets are mostly synthetic, leading to poor generalization in real-world scenes.

### Key Challenge

**Key Challenge**: Existing self-training methods are only suitable for cost-volume-based methods and cannot be directly applied to iterative methods that lack a complete cost volume.

### Proposed Solution

**Proposed Solution**: Existing pseudo-label filtering strategies adopt hard-threshold binary selection, which both discards valuable hard samples and fails to distinguish the reliability differences among pseudo-labels within the threshold.

### Additional Notes

**Additional Notes**: Core observation: **regions with larger errors exhibit more pronounced oscillation characteristics in model predictions.**

## Method

### Overall Architecture

CST-Stereo adopts a teacher-student self-training framework: the teacher model generates pseudo-labels on unlabeled data; the Consistency-aware Soft Filtering (CSF) module evaluates the reliability of the pseudo-labels; the student model learns with a soft-weighted loss under strongly augmented inputs; and the teacher is updated via EMA.

### Key Designs

1. **Multi-Resolution Prediction Consistency Filter (MRPCF)**:
    - Function: Evaluates pseudo-label reliability from the spatial dimension.
    - Mechanism: Feeds the input images into the teacher model after upsampling, maintaining the original scale, and downsampling, respectively. It then computes the pixel-level variance $\sigma_{i,j}$ of the predictions across the three resolutions, which is converted to a reliability weight $w_{rc} = 1/(1 + e^{-\varepsilon_1(\sigma - \tau_1)})$ via sigmoid mapping.
    - Design Motivation: Pixels with inconsistent predictions across different resolutions tend to be unreliable (e.g., edges, occluded regions), whereas consistent pixels are more reliable.

2. **Iterative Prediction Consistency Filter (IPCF)**:
    - Function: Evaluates pseudo-label reliability from the temporal dimension.
    - Mechanism: Calculates the average prediction difference between adjacent iterations in the second half of the iterative process, i.e., $\Delta_{i,j} = \frac{1}{\lceil n/2 \rceil} \sum_{k} |P^{k+1}_{i,j} - P^k_{i,j}|$, which is similarly mapped to a weight $w_{ic}$ via sigmoid mapping.
    - Design Motivation: Pixels that still oscillate in the late stages of iteration indicate inconsistent multi-source information where the model fails to converge to a deterministic value, making the predictions unreliable.

3. **Consistency-aware Soft-Weighted Loss**:
    - Function: Guides student model training by combining the weights of both filters.
    - Mechanism: $w_{soft} = w_{rc} \odot w_{ic}$, and the loss is $L_{st} = w_{soft} \odot |\hat{P} - P^O|$. The multiplication ensures that only pixels deemed reliable by both filters receive high weights.
    - Design Motivation: Avoids discarding valuable hard samples due to hard thresholding, while mitigating the impact of noisy pseudo-labels.

### Loss & Training

- Pre-train on annotated datasets to obtain initial weights.
- Self-training phase: The student receives strongly augmented images while the teacher receives non-augmented ones.
- Update the teacher via EMA: $\theta_T \leftarrow \lambda \theta_T + (1-\lambda) \theta_S$
- Applicable to various iterative baselines: RAFT-Stereo, CREStereo, IGEV-Stereo, Selective-Stereo, etc.

## Key Experimental Results

### Main Results

| Method | KITTI2015 D1-all ↓ | Middlebury bad1.0 ↓ | ETH3D bad1.0 ↓ |
|------|-------------------|---------------------|----------------|
| RAFT-Stereo | 1.96 | 9.37 | 2.44 |
| IGEV-Stereo | 1.59 | 9.41 | 1.12 |
| Selective-IGEV | 1.55 | 6.53 | 1.23 |
| **CST-Stereo** | **1.50** | **6.23** | **1.02** |

### Ablation Study

| Configuration | Key Effect | Description |
|------|---------|------|
| Baseline without filtering | Performance drop | Direct usage of pseudo-labels leads to noise accumulation |
| MRPCF only | Gain | Captures error regions sensitive to spatial resolution |
| IPCF only | Gain | Captures error regions related to iterative oscillation |
| MRPCF + IPCF (Hard Threshold) | Suboptimal | Discards hard samples |
| MRPCF + IPCF (Soft Weights) | **Optimal** | Blends reliability evaluation and hard sample utilization |

### Key Findings

- The method is highly versatile and can be applied in a plug-and-play manner to boost multiple iterative baselines (e.g., reducing error by 35% for RAFT-Stereo and 21% for CREStereo).
- It is effective in all three scenarios: in-domain, domain adaptation, and domain generalization.
- It achieves SOTA performance among published methods on multiple public leaderboards, including KITTI2015, Middlebury, and ETH3D.
- Multi-resolution consistency and iterative consistency are complementary: MRPCF is effective in edge regions, whereas IPCF excels in blurry areas.

## Highlights & Insights

- The observation of "oscillation characteristics in error regions" is highly intuitive and thoroughly verified through visualization.
- The soft-threshold design is significantly more elegant than hard binary filtering, preserving the training value of hard samples.
- Direct of cost volumes is avoided, making this method the first to successfully extend self-training to the entire family of iterative stereo matching.
- The two consistency filters evaluate reliability from spatial and temporal dimensions, which are orthogonal to each other.

## Limitations & Future Work

- Multi-resolution prediction requires three forward passes, increasing the training computational overhead.
- The soft threshold parameters $\tau_1, \tau_2$ and scaling factors $\varepsilon_1, \varepsilon_2$ in the filters require manual tuning.
- More complex weight fusion strategies (such as learned fusion weights) have not yet been explored.
- Generalizing the observation to other dense prediction tasks, such as optical flow estimation, could be explored.

## Related Work & Insights

- Continuative of self-training stereo matching works such as StereoBase and PCT-Stereo, but targeted specifically at iterative-based methods for the first time.
- The concept of soft filtering can be borrowed for other pseudo-label learning scenarios, such as semi-supervised semantic segmentation.
- The consistency observation may hold deeper significance for understanding the uncertainty of iterative models.

## Rating

- Novelty: ⭐⭐⭐⭐ The observation of oscillation characteristics is novel, and the design of the two complementary filters is well-founded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across multiple baselines, scenarios, and datasets, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with a complete logical flow from observation to design.
- Value: ⭐⭐⭐⭐ Provides plug-and-play performance gains for iterative-based stereo matching, offering high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2026\] PIP-Stereo: Progressive Iterations Pruner for Iterative Optimization based Stereo Matching](../../CVPR2026/3d_vision/pip-stereo_progressive_iterations_pruner_for_iterative_optimization_based_stereo.md)
- [\[CVPR 2025\] FoundationStereo: Zero-Shot Stereo Matching](foundationstereo_zero-shot_stereo_matching.md)
- [\[CVPR 2025\] SpatialDreamer: Self-supervised Stereo Video Synthesis from Monocular Input](spatialdreamer_self-supervised_stereo_video_synthesis_from_monocular_input.md)
- [\[NeurIPS 2025\] U-CAN: Unsupervised Point Cloud Denoising with Consistency-Aware Noise2Noise Matching](../../NeurIPS2025/3d_vision/u-can_unsupervised_point_cloud_denoising_with_consistency-aware_noise2noise_matc.md)

</div>

<!-- RELATED:END -->

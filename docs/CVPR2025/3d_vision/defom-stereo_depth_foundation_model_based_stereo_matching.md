---
title: >-
  [Paper Note] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching
description: >-
  [CVPR 2025][3D Vision][Stereo Matching] Integrates the monocular depth foundation model (Depth Anything V2) into the recurrent stereo matching framework RAFT-Stereo. By incorporating combined feature encoders and a scale update module, this approach achieves state-of-the-art stereo matching performance across multiple benchmarks while preserving strong generalization capabilities.
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Stereo Matching"
  - "Depth Foundation Models"
  - "Zero-shot Generalization"
  - "Monocular Depth"
  - "Scale Update"
date: 2026-05-08
content_hash: 298e5c497caaddc3
---

# DEFOM-Stereo: Depth Foundation Model Based Stereo Matching

**Conference**: CVPR 2025  
**arXiv**: [2501.09466](https://arxiv.org/abs/2501.09466)  
**Code**: [https://insta360-research-team.github.io/DEFOM-Stereo](https://insta360-research-team.github.io/DEFOM-Stereo)  
**Area**: 3D Vision  
**Keywords**: Stereo Matching, Depth Foundation Models, Zero-shot Generalization, Monocular Depth, Scale Update

## TL;DR

Integrates the monocular depth foundation model (Depth Anything V2) into the recurrent stereo matching framework RAFT-Stereo. By incorporating combined feature encoders and a scale update module, this approach achieves state-of-the-art stereo matching performance across multiple benchmarks while preserving strong generalization capabilities.

## Background & Motivation

Stereo matching is a crucial technology for metric depth estimation in computer vision, yet it faces challenges such as occlusion, textureless regions, and blur. Recently, monocular relative depth estimation methods (such as Depth Anything V2), powered by vision foundation models, have demonstrated excellent zero-shot generalization capabilities, but they provide only relative depth instead of metric depth. Distinct observations indicate that:

- Although subsequent recurrent optimization methods improve in-domain fitting, their zero-shot generalization performance is inferior to the baseline RAFT-Stereo.
- Depth Anything V2 recovers fine details and robust relative depth but suffers from severe **scale inconsistency**—the depth predictions across different regions within the same image do not conform to a unified scale.
- Simply aligning DEFOM depth via a least-squares affine transformation still results in significant disparity errors, particularly on the synthetic Scene Flow dataset.

The core motivation is to introduce the robust monocular cues of DEFOM into stereo matching while resolving its critical scale inconsistency issue.

## Method

### Overall Architecture

DEFOM-Stereo is built on the RAFT-Stereo framework and integrates Depth Anything V2 (DEFOM). It consists of two stages:
1. **Feature Extraction Stage**: Enhances the CNN feature encoders with DEFOM pre-trained features and uses DEFOM depth to initialize disparity.
2. **Disparity Update Stage**: Operates a Scale Update (SU) module first to recover pixel-wise scales, followed by a traditional Delta Update (DU) to restore local details.

### Key Designs

1. **Combined Feature Encoder (CFE + CCE)**:
    - Function: Fuses pre-trained ViT features from DEFOM with CNN features to construct stronger matching and context feature encoders.
    - Mechanism: Initializes a new, trainable DPT head to extract features from the frozen ViT backbone (the original DPT remains frozen for depth prediction). Channel dimensions are aligned through convolutional blocks before performing element-wise addition with CNN features. The context encoder extracts multi-scale features from $Reassemble_4$, $Reassemble_8$, and $Reassemble_{16}$ of the DPT.
    - Design Motivation: The DPT features from DEFOM are highly correlated with depth tasks, whereas CNN features preserve local matching capabilities. Simple additive fusion allows them to complement each other effectively.

2. **Monocular Depth Initialization (DI)**:
    - Function: Initializes the disparity map using the relative depth predicted by DEFOM, replacing traditional zero initialization.
    - Mechanism: Since disparity magnitude is generally proportional to the image width $w$, the depth estimation is normalized as $\mathbf{d}_0 = \frac{\eta w \cdot \mathbf{z}}{\max(\mathbf{z})} + \epsilon$, where $\eta = 1/2$.
    - Design Motivation: The scale of the relative depth output by DEFOM varies significantly across datasets and backbone sizes. Normalizing with the image width provides a reasonable initialization.

3. **Scale Update Module (SU)**:
    - Function: Resolves scale-inconsistent initial disparities into geometrically accurate disparities through recurrent pixel-wise dense scale updates.
    - Mechanism: The update is formulated multiplicatively as $\mathbf{d}_n = \mathbf{s} \cdot \mathbf{d}_{n-1}$ (unlike traditional additive updates), operating in conjunction with **Scale Lookup (SL)** to retrieve information from the correlation volume. SL multiplies the current disparity by a series of predefined scale factors $\{1,2,4,6,8,10,12,16\}/8$, samples the corresponding correlation values, spanning a global search range.
    - Design Motivation: The maximum search range of traditional pyramid lookup is limited to 128 pixels, which is insufficient for handling large disparities. SL achieves global matching through scale scaling; SU first obtains a geometrically consistent scale, and then DU recovers local details.

### Loss & Training

- Employs an L1 loss with exponentially increasing weights: $\mathcal{L} = \sum_{n} \gamma^{N-n} \|\mathbf{d}_{gt} - \mathbf{d}_n\|_1$, where $\gamma=0.9$.
- Pre-trained on Scene Flow for 200k steps with a crop size of $320 \times 736$, utilizing the AdamW optimizer with a one-cycle learning rate of 2e-4.
- 8 SU iterations + 10 DU iterations = 18 total iterations during training, and 32 total iterations during evaluation.
- The ViT backbone and the original DPT of DEFOM remain frozen throughout.

## Key Experimental Results

### Main Results (Zero-shot Generalization, Scene Flow → Real-world Datasets)

| Dataset | Metric | DEFOM-Stereo (ViT-L) | RAFT-Stereo | Gain |
|--------|------|------|----------|------|
| KITTI 2012 | Bad 3.0 | 3.76 | 4.35 | 13.6% |
| KITTI 2015 | Bad 3.0 | 4.99 | 5.74 | 13.1% |
| Middlebury-full | Bad 2.0 | 11.95 | 18.33 | 34.8% |
| Middlebury-half | Bad 2.0 | 5.91 | 12.59 | 53.1% |
| ETH3D | Bad 1.0 | 2.35 | 3.28 | 28.4% |

### Ablation Study

| Configuration | Scene Flow EPE | Middlebury-half Bad 2.0 | Description |
|------|---------|------|------|
| Baseline (Simplified RAFT-Stereo) | 0.56 | 10.67 | 2-level correlation pyramid |
| +CCE | 0.49 | 8.42 | Combined context encoder |
| +CFE | 0.50 | 10.45 | Combined feature encoder |
| +DI+SU | 0.50 | 8.15 | Depth initialization + Scale update |
| Full Model (ViT-S) | 0.46 | 6.76 | All components |
| Full Model (ViT-L) | 0.42 | 5.91 | Large backbone |

### Key Findings

- DEFOM-Stereo ranks first on multiple online benchmarks and comprehensively outperforms previous approaches in the Robust Vision Challenge joint evaluation.
- Reduces the zero-shot error rate by over 50% on Middlebury-half and outperforms the previous state-of-the-art method DLNR by 29% on the high-resolution Middlebury-full.
- The SU module introduces 5.4M additional parameters but only increases inference time by about 10%, as inference time is dominated by recurrent iterations.
- The global search capability of the scale lookup is key to the success of SU, as the traditional pyramid lookup range is insufficient.

## Highlights & Insights

- **Scale Inconsistency Analysis of DEFOM Depth**: The authors point out that the scale inconsistency of DEFOM stems from training with an affine-invariant loss on images with varying fields of view (FoV), which is a profound and important observation.
- **Multiplicative vs. Additive Update**: Updating the disparity initialized by depth through scale multiplication is conceptually simple yet highly effective.
- **Effectiveness of Simple Fusion**: Merely performing element-wise addition of CNN and ViT features is remarkably effective, bypassing the need for complex attention-based adapters.
- **Freezing Strategy**: Freezing DEFOM's backbone and DPT while training only the new DPT head and the SU module retains generalization ability while reducing training costs.

## Limitations & Future Work

- The parameter count of the ViT-L model is 47.3M with an inference time of 0.316s ($960 \times 540$), depicting a 42% increase compared to the baseline.
- Depth initialization relies on the quality of DEFOM; thus, performance might be constrained in extreme scenarios where DEFOM fails.
- The set of scale factors is manually determined; adaptive selection could be explored.
- Currently, only Depth Anything V2 has been validated; other depth foundation models, such as Marigold, have not been evaluated.

## Related Work & Insights

- Difference from concurrent works: Other works integrating DEFOM into stereo matching (e.g., Foundation Stereo) typically initialize disparity using a 4D cost volume and 3D convolutions, followed by alignment with DEFOM. In contrast, this work directly replaces zero initialization with DEFOM depth and iteratively refines it via the SU module.
- The choice of RAFT-Stereo as the baseline offers unique insights: although subsequent methods achieve superior in-domain fitting, RAFT-Stereo ironically exhibits the best zero-shot generalization.
- Insight: The paradigm of combining foundation models with task-specific modules holds immense promise for dense prediction tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The integration of depth foundation models into stereo matching is intuitive yet effective; the scale update module is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering zero-shot generalization, online benchmark submissions, the RVC joint evaluation, and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Features a clear structure with a complete logical chain of motivation and design.
- Value: ⭐⭐⭐⭐⭐ Ranking first on multiple benchmarks, it practically validates the immense potential of empowering stereo matching with foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Consistency-aware Self-Training for Iterative-based Stereo Matching](consistency-aware_self-training_for_iterative-based_stereo_matching.md)
- [\[CVPR 2025\] FoundationStereo: Zero-Shot Stereo Matching](foundationstereo_zero-shot_stereo_matching.md)
- [\[CVPR 2025\] Helvipad: A Real-World Dataset for Omnidirectional Stereo Depth Estimation](helvipad_a_real-world_dataset_for_omnidirectional_stereo_depth_estimation.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](../../ICCV2025/3d_vision/zerostereo_zero-shot_stereo_matching_from_single_images.md)

</div>

<!-- RELATED:END -->

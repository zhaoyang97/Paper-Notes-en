---
title: >-
  [Paper Note] MVSAnywhere: Zero-Shot Multi-View Stereo
description: >-
  [CVPR 2025][3D Vision][Multi-View Stereo] This paper proposes MVSAnywhere (MVSA), a general-purpose multi-view stereo matching architecture. By using a Cost Volume Patchifier, cost volume information is efficiently tokenized and fused with monocular ViT features (Mono/Multi Cue Combiner). Combined with a view-count and scale-agnostic metadata encoding and a cascaded adaptive depth range estimation, MVSA achieves zero-shot SOTA on the Robust MVS Benchmark…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Multi-View Stereo"
  - "Zero-shot Generalization"
  - "Adaptive Cost Volume"
  - "Monocular Depth Prior"
  - "ViT"
  - "Depth Estimation"
date: 2026-05-08
content_hash: 98cb9229e38c691e
---

# MVSAnywhere: Zero-Shot Multi-View Stereo

**Conference**: CVPR 2025  
**arXiv**: [2503.22430](https://arxiv.org/abs/2503.22430)  
**Code**: [https://github.com/nianticlabs/mvsanywhere](https://github.com/nianticlabs/mvsanywhere)  
**Area**: 3D Vision/Multi-View Stereo  
**Keywords**: Multi-View Stereo, Zero-shot Generalization, Adaptive Cost Volume, Monocular Depth Prior, ViT, Depth Estimation

## TL;DR

This paper proposes MVSAnywhere (MVSA), a general-purpose multi-view stereo matching architecture. By using a Cost Volume Patchifier, cost volume information is efficiently tokenized and fused with monocular ViT features (Mono/Multi Cue Combiner). Combined with a view-count and scale-agnostic metadata encoding and a cascaded adaptive depth range estimation, MVSA achieves zero-shot SOTA on the Robust MVS Benchmark, while supporting an arbitrary number of source views and arbitrary depth ranges.

## Background & Motivation

**Background**: Learning-based multi-view stereo (MVS) has achieved excellent performance, but existing methods (such as MVSNet, SimpleRecon) usually only generalize within specific domains (indoor/outdoor/driving) and require known depth ranges and a fixed number of source views. Monocular depth estimation models (such as Depth Anything V2, Depth Pro) generalize well but lack the accuracy brought by multi-view geometric signals.

**Limitations of Prior Work**: (1) Depth ranges vary drastically across different scenarios (DTU: ~1m, KITTI: ~80m), making fixed depth bins ungeneralizable across domains; (2) SimpleRecon requires exactly 8 source frames, limiting flexibility; (3) CNN-based cost volume processors cannot leverage the strong representation capabilities of ViTs; (4) Limited domains in training data lead to poor zero-shot performance.

**Key Challenge**: Building a general-purpose MVS system requires simultaneously addressing generalization across four dimensions—domain generalization, depth range generalization, source frame count generalization, and 3D consistency assurance—where these goals may conflict in terms of architectural design.

**Goal**: Design a single MVS model that can generalize in a zero-shot manner to any domain, any depth range, and an arbitrary number of source frames, while predicting 3D-consistent depth maps.

**Key Insight**: (1) Train using diverse large-scale synthetic data; (2) Inject monocular ViT priors into the MVS pipeline; (3) Design a view-count and scale-agnostic metadata mechanism and an adaptive cost volume depth range.

**Core Idea**: Elegantly fuse cost volume with monocular ViT features through a Cost Volume Patchifier, coupled with view-count-agnostic metadata aggregation and cascaded adaptive depth range, to achieve a truly general-purpose MVS.

## Method

### Overall Architecture

The inputs are the reference image $I_r$ and $N$ source frames $I_i$, along with their relative poses and camera intrinsics. The feature extractor (the first two blocks of ResNet18) extracts $H/4 \times W/4$ feature maps to construct the cost volume. The reference image encoder (ViT-Base from Depth Anything V2) extracts $H/16 \times W/16$ monocular features. The cost volume is converted into token sequences via the Cost Volume Patchifier, then fused with monocular features in the Mono/Multi Cue Combiner ViT. Finally, a DPT-style decoder progressively upsamples the features to output full-resolution depth maps.

### Key Designs

1. **View-count and Scale-agnostic Metadata Encoding**:
    - **Function**: Makes cost volume construction independent of a fixed number of source frames and insensitive to scene scale.
    - **Mechanism**: An MLP processes metadata (feature matching score, ray direction, depth hypotheses, etc.) independently for each source frame to predict a score and a weight. $N$ source frames produce $N$ pairs of scores and weights. The weights are normalized via softmax and then used to compute a weighted sum of the scores, yielding the final value for each $(u,v,k)$ position in the cost volume. Simultaneously, maximum normalization is applied to relative poses and depth hypotheses to eliminate scale dependency.
    - **Design Motivation**: The fixed 8-frame concatenation MLP in SimpleRecon limits flexibility. Processing frames independently followed by weighted aggregation allows the model to automatically learn how to allocate attention across varying quantities and qualities of source frames.

2. **Cost Volume Patchifier + Mono/Multi Cue Combiner**:
    - **Function**: Efficiently converts the $|D| \times H/4 \times W/4$ cost volume into ViT tokens and fuses them with monocular features.
    - **Mechanism**: Instead of simply downsampling the cost volume with strided convolutions, features from the first two blocks of the reference image encoder (projected to 1/4 and 1/8 resolution via transpose) are concatenated before the two strided convolutions, allowing monocular context to guide the downsampling of the cost volume. The resulting $H/16 \times W/16$ token sequence is linearly projected and element-wise added to the monocular ViT features, repeatedly injecting multi-level monocular cues at blocks 2, 5, 9, and 11 of the ViT.
    - **Design Motivation**: Naive patchification loses crucial matching information of the cost volume. Guiding the downsampling using monocular features informs the network "which depth hypotheses are visually more reasonable", and multiple subsequent fusions in the ViT ensure sufficient interaction between the two types of signals.

3. **Cascaded Adaptive Depth Range**:
    - **Function**: Automatically determines the appropriate depth range at inference time without prior knowledge of the scene's depth distribution.
    - **Mechanism**: Utilizing known intrinsics and extrinsics, the minimum and maximum matchable depths between $I_r$ and all $I_i$ are calculated. Simple prediction is performed by placing 64 depth bins with logarithmic uniform spacing within this coarse range, and then the min/max values of this initial depth map are used to reconstruct the cost volume for the final prediction. During training, robustness is enhanced by randomly perturbing the ground-truth depth range.
    - **Design Motivation**: The effective depth ranges of different datasets vary by more than 100 times (see Fig. 2); a fixed range will inevitably fail in some domains. A two-step cascaded coarse-to-fine scheme gradually locks in the correct range.

### Loss & Training

- L1 loss in log-depth space + gradient loss + normal loss, applied to the 4 output scales of the decoder.
- Depth prediction uses sigmoid to map to the cost volume depth range: $$\hat{D}_r = \exp(\log(d_{\min}) + \log(d_{\max}/d_{\min}) \cdot \sigma(x))$$

## Key Experimental Results

### Main Results

**Zero-shot Evaluation on Robust MVS Benchmark (Average of 5 Datasets)**:

| Method | GT Poses | GT Range | Average rel↓ | Average τ↑ |
|------|----------|----------|------------|-----------|
| DeMoN | ✗ | ✗ | 16.0 | 18.3 |
| MAST3R (raw output) | ✗ | ✗ | 3.3 | 71.8 |
| SimpleRecon | ✓ | ✓ | 2.2 | 83.2 |
| **MVSA** | ✓ | ✗ | **1.8** | **87.0** |

MVSA outperforms SimpleRecon, which requires the ground-truth depth range, without requiring it, demonstrating the effectiveness of the adaptive depth range strategy.

### Ablation Study

**Contribution of Each Component (Average of KITTI + ScanNet + ETH3D)**:

| Configuration | rel↓ | τ↑ |
|------|------|-----|
| Baseline (CNN cost volume) | ~3.5 | ~72 |
| + ViT Mono/Multi Combiner | ~2.5 | ~80 |
| + Adaptive depth range | ~2.0 | ~85 |
| + Scale-agnostic metadata | **~1.8** | **~87** |

### Key Findings

- In the zero-shot setting (unseen test domains), MVSA completely outperforms specially trained MVS methods and the latest monocular methods across 5 different datasets.
- The monocular prior is crucial for handling low overlap between source/reference frames—gracefully degrading to high-quality monocular estimation when multi-view signals are weak.
- Training is conducted only on 8 synthetic datasets (pure RGB-D, no real annotations) but generalizes to real-world data.
- The view-count-agnostic design makes the model robust when the number of source frames varies from 1 to 16.
- The generated depth maps exhibit excellent 3D consistency, performing better for direct mesh reconstruction than monocular methods such as Depth Pro.

## Highlights & Insights

1. The design of **Cost Volume Patchifier** is highly ingenious: using monocular features to guide the downsampling of the cost volume means the tokenization process itself is "semantics-aware" rather than blindly compressing.
2. **View-count-agnostic metadata aggregation** solves a long-standing engineering problem: there is no longer a need to fix the number of source frames for each application scenario, making the model truly flexible.
3. The 8 synthetic datasets cover diverse scenarios such as indoor, outdoor, aerial, and driving. This **data strategy** is the crucial foundation for its generalization capability.
4. Initializing the ViT feature extractor with pre-trained weights from Depth Anything V2 **leverages monocular depth pre-training knowledge** to enhance MVS, exemplifying "standing on the shoulders of giants".

## Limitations & Future Work

- Trained only on synthetic data; whether it remains robust in large-scale real-world scenarios (e.g., city-scale SfM) has not been fully verified.
- The cascaded adaptive depth range introduces two forward inferences, increasing computational overhead.
- Robustness to actual challenging scenarios, such as extreme lighting changes and dynamic object occlusions, still needs improvement.
- The parameter size of ViT-Base is considerable (~86M for the reference image encoder alone), making on-device deployment difficult.
- Integration with emerging 3D representations, such as 3D Gaussian Splatting, has not been explored.

## Related Work & Insights

- **SimpleRecon** [Sayed et al.]: Proposed incorporating geometric metadata into the cost volume. This work extends its design to make it view-count-agnostic.
- **Depth Anything V2** [Yang et al.]: Provides powerful pre-trained monocular depth ViT weights, utilized as the reference image encoder in this work.
- **MAST3R** [Leroy et al.]: A pose-free dense matching method; this work significantly outperforms it when poses are provided.
- **CasMVSNet** [Gu et al.]: The coarse-to-fine strategy of cascaded cost volumes inspired the adaptive depth range design in this work.
- **Insight**: Injecting monocular priors as a "safety net" into the MVS framework is a powerful paradigm—automatically degrading to monocular estimation when multi-view signals are insufficient, thus guaranteeing the lower bound.

## Rating

⭐⭐⭐⭐ — Systematically addresses the four major generalization challenges of general-purpose MVS (domain, depth range, view-count, 3D consistency) with an elegant architectural design and comprehensive experiments. The zero-shot SOTA on the Robust MVS Benchmark is highly convincing. Open-sourcing the code is a definite plus.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FoundationStereo: Zero-Shot Stereo Matching](foundationstereo_zero-shot_stereo_matching.md)
- [\[CVPR 2025\] MVGD: Zero-Shot Novel View and Depth Synthesis with Multi-View Geometric Diffusion](zero-shot_novel_view_and_depth_synthesis_with_multi-view_geometric_diffusion.md)
- [\[CVPR 2025\] Cross-View Completion Models are Zero-shot Correspondence Estimators](cross-view_completion_models_are_zero-shot_correspondence_estimators.md)
- [\[ICCV 2025\] RobuSTereo: Robust Zero-Shot Stereo Matching under Adverse Weather](../../ICCV2025/3d_vision/robustereo_robust_zero-shot_stereo_matching_under_adverse_weather.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](../../ICCV2025/3d_vision/zerostereo_zero-shot_stereo_matching_from_single_images.md)

</div>

<!-- RELATED:END -->

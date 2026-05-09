---
title: >-
  [Paper Note] Generalized Geometry Encoding Volume for Real-time Stereo Matching
description: >-
  [AAAI 2026][3D Vision][Stereo Matching] This paper proposes GGEV, which integrates depth priors from a monocular depth foundation model (Depth Anything V2) into the cost aggregation process in a lightweight manner. Through Depth-aware Dynamic Cost Aggregation (DDCA), GGEV adaptively enhances matching relationships across different disparity hypotheses, achieving strong generalization at real-time inference speed.
tags:
  - AAAI 2026
  - 3D Vision
  - Stereo Matching
  - Real-time Inference
  - Zero-shot Generalization
  - Monocular Depth Foundation Model
  - Dynamic Cost Aggregation
date: 2026-05-08
content_hash: d1de77228d59c86f
---

# Generalized Geometry Encoding Volume for Real-time Stereo Matching

**Conference**: AAAI 2026
**arXiv**: [2512.06793](https://arxiv.org/abs/2512.06793)
**Code**: [https://github.com/JiaxinLiu-A/GGEV](https://github.com/JiaxinLiu-A/GGEV)
**Area**: 3D Vision / Stereo Matching
**Keywords**: Stereo Matching, Real-time Inference, Zero-shot Generalization, Monocular Depth Foundation Model, Dynamic Cost Aggregation

## TL;DR

This paper proposes GGEV, which integrates depth priors from a monocular depth foundation model (Depth Anything V2) into the cost aggregation process in a lightweight manner. Through Depth-aware Dynamic Cost Aggregation (DDCA), GGEV adaptively enhances matching relationships across different disparity hypotheses, achieving strong generalization at real-time inference speed.

## Background & Motivation

### State of the Field

Stereo matching is a classic computer vision task that estimates dense disparity maps from rectified stereo image pairs. Practical applications such as autonomous driving and 3D reconstruction impose strict requirements on both generalization capability and inference latency. Existing methods fall into two camps:

- **Real-time methods** (RT-IGEV, Fast-ACVNet, etc.): Achieve fast inference via downsampled cost volumes, lightweight aggregation, and replacing 3D convolutions with 2D convolutions, but exhibit fragile matching in unseen scenes (occlusions, textureless regions, repetitive textures, thin structures).
- **Generalization-oriented methods** (FoundationStereo, MonSter, etc.): Leverage monocular foundation models (MFMs) to improve generalization, but rely on large backbones (ViT-L) and complex iterative mechanisms to handle scale-shift issues, resulting in high inference latency.

### Root Cause

How to design a stereo matching network that is **both real-time and highly generalizable**?

### Limitations of Prior Work

The authors identify two key limitations of existing Geometry Encoding Volumes (GEV):

**Critical regions differ substantially across disparity hypotheses**: Uniform processing of all disparity hypotheses leads to incorrect matches.

**Matching relationships in these regions are extremely fragile in unseen scenes**: Textureless regions, occlusions, and repetitive textures cause matching failures.

### Starting Point

Rather than using MFMs to construct cost volumes as in FoundationStereo (which introduces scale-shift issues), GGEV employs depth features to **guide cost aggregation** — avoiding scale-shift while remaining lightweight.

## Method

### Overall Architecture

GGEV consists of four stages:
1. **Multi-cue feature extraction**: Texture features (MobileNetV2) + depth features (frozen Depth Anything V2 Small)
2. **Cost volume construction**: Group-wise correlation cost volume based on texture features
3. **Depth-aware Dynamic Cost Aggregation (DDCA)**: Adaptively enhances the cost volume using depth priors
4. **Depth-aware iterative refinement**: GRU-based iterative disparity refinement

### Key Designs

#### 1. Multi-cue Feature Extraction and Selective Channel Fusion (SCF)

**Function**: Extracts texture and depth features, then fuses them into depth-aware prior features in a lightweight manner.

**Mechanism**:
- **Texture branch**: Multi-scale texture features $\mathbf{f}_{l,i}, \mathbf{f}_{r,i}$ ($i \in \{4,8,16\}$) extracted from left and right images using ImageNet-pretrained MobileNetV2.
- **Depth branch**: Multi-scale depth features $\mathbf{f}_{d,i}$ extracted from the left image only using a **frozen** Depth Anything V2 Small.
- **SCF module**: Fuses concatenated texture and depth features via 1×1 convolutions to produce depth-aware prior features $\mathbf{f}_{da,i}$.

**Design Motivation**:
- Using a frozen MFM avoids training overhead while exploiting domain-invariant structural priors learned from large-scale real data.
- MobileNetV2 rather than ViT is used as the texture backbone to maintain real-time performance.
- 1×1 convolution fusion avoids spatial blurring and preserves structural details.

#### 2. Depth-aware Dynamic Cost Aggregation (DDCA)

**Function**: Adaptively injects depth structural priors into the cost volume for each disparity hypothesis, enhancing fragile matching relationships.

**Mechanism**:

**Step 1 — Disparity-level depth structure representation**: Computes an affinity matrix between each disparity hypothesis and the depth features:

$$\mathbf{Q} = \text{Re}(W_q \mathbf{C}_d), \quad \mathbf{K} = \text{Re}(W_k \text{Pool}(\mathbf{f}_{da}))$$
$$\mathbf{A} = \mathbf{Q}^T \mathbf{K}$$

Analogous to multi-head attention, $\mathbf{A}^g$ is computed by partitioning along the channel dimension into $G$ groups.

**Step 2 — Disparity-level adaptive cost aggregation**: Dynamic convolution kernels are generated from the affinity matrix:

$$\mathbf{M}^g = \text{softmax}(\mathbf{A}^g W_m)$$
$$\mathbf{C}_d' = \mathbf{C}_d * \mathbf{M}^g_{\text{dynamic}}(\mathbf{C}_d, \mathbf{f}_{da})$$

**Design Motivation**:
- Different disparity hypotheses correspond to different foreground/background regions and require **different aggregation strategies**.
- Conventional hourglass aggregation networks process all disparity hypotheses uniformly and cannot treat them differentially.
- Dynamic convolution kernels provide per-pixel, per-disparity-plane adaptive filter weights.
- Combining large and small kernels captures complementary low- and high-frequency information.
- A sliding window approach (analogous to standard 2D convolution) keeps the module lightweight and real-time.

#### 3. Depth-aware Iterative Refinement

**Function**: Iteratively refines the disparity map via GRU, injecting depth priors into the initial hidden state.

**Mechanism**:
- Initial disparity $\mathbf{d}_0$ is regressed from GGEV via soft-argmin.
- GRU hidden state $h_0$ is initialized with depth-aware features $\mathbf{f}_{da,4}$ to inject structural priors.
- Each iteration: index geometric features from GGEV → concatenate with current disparity → GRU update → decode residual disparity.
- During upsampling, GRU features are concatenated with depth features to generate weighting maps.

### Loss & Training

$$\mathcal{L} = |\mathbf{d}_0 - \mathbf{d}_{gt}|_{smooth} + \sum_{i=1}^{N} \gamma^{N-i} \|\mathbf{d}_i - \mathbf{d}_{gt}\|_1$$

- Smooth L1 loss for initial disparity; L1 loss for iterative disparities.
- Decay factor $\gamma = 0.9$; trained for 11 iterations and tested at 8 iterations.
- AdamW optimizer with gradient clipping $[-1,1]$ and one-cycle learning rate schedule.

## Key Experimental Results

### Main Results

#### Zero-shot Generalization (Trained on Scene Flow Only)

| Method | Type | KITTI 2012 | KITTI 2015 | Middlebury | ETH3D |
|--------|------|-----------|-----------|------------|-------|
| RT-IGEV | Real-time | 5.8 | 6.6 | 7.8 | 5.8 |
| Fast-ACVNet | Real-time | 12.4 | 10.6 | 13.5 | 7.9 |
| RAFT-Stereo | Accuracy | 4.5 | 5.7 | 9.3 | 3.2 |
| DEFOM-Stereo(ViT-S) | Accuracy | 4.2 | 5.3 | 6.3 | 2.6 |
| **GGEV (Ours)** | **Real-time** | **4.1** | **5.5** | **6.5** | **2.8** |

Error rate reduction vs. RT-IGEV: KITTI 2012 ↓29%, KITTI 2015 ↓16%, ETH3D ↓51%.

#### Benchmark Accuracy (After Fine-tuning)

| Method | KITTI 2012 3-noc | KITTI 2015 D1-all | ETH3D Bad 1.0 | Inference (ms) |
|--------|-----------------|-------------------|---------------|----------------|
| RT-IGEV | 1.29 | 1.79 | — | 40 |
| BANet-3D | 1.27 | 1.77 | — | 30 |
| **GGEV** | **1.10** | **1.70** | **1.19** | **47** |

### Ablation Study

| Configuration | Scene Flow EPE | KITTI 2015 D1 | ETH3D Bad 1.0 | Params (M) | Inference (ms) |
|---------------|---------------|---------------|---------------|------------|----------------|
| Baseline | 0.54 | 8.01 | 3.60 | — | 30 |
| +DFE(ViT-S) | 0.52 | 6.32 | 3.57 | — | 37 |
| +DFE+SCF | 0.49 | 7.58 | 3.65 | — | 38 |
| +DCA only | 0.47 | 6.75 | 3.63 | — | 39 |
| Full(ViT-S) | **0.46** | **5.56** | **2.84** | 3.68 | 47 |

#### Evaluation on Reflective Regions (KITTI 2012 Reflective)

| Method | 2-noc | 3-noc |
|--------|-------|-------|
| RAFT-Stereo | 8.41 | 5.40 |
| RT-IGEV | 9.56 | 5.76 |
| **GGEV** | **7.33** | **4.04** |

### Key Findings

1. **Adding DFE alone improves generalization but offers limited in-domain gains**: Depth priors generalize well but require adaptive fusion to be fully effective.
2. **SCF improves in-domain fitting but with mixed generalization effects**: Simple feature fusion is insufficient for complex scenes.
3. **DCA alone improves in-domain accuracy but generalizes poorly**: Texture features remain sensitive to textureless regions and appearance shifts.
4. **All three components working together yield simultaneous gains in accuracy and generalization**: SCF introduces MFM generalization capacity; DDCA provides adaptive fusion for comprehensive improvement.
5. **Performance is particularly strong in reflective and challenging regions**: Error rates are more than 30% lower than all competing methods.

## Highlights & Insights

1. **Elegant avoidance of scale-shift issues**: Rather than using MFMs to generate disparity initialization (which introduces scale-shift), the paper uses depth features to guide cost aggregation, fundamentally circumventing the alignment problem.
2. **Novel design of "depth features as dynamic convolution kernels"**: The pipeline of affinity matrix → dynamic convolution kernels → adaptive aggregation is lightweight yet effective.
3. **Frozen MFM + trainable fusion**: Preserves pretrained knowledge while enabling task-specific adaptation.
4. **Successful balance between real-time speed and generalization**: At 47 ms (~21 fps), the method fully meets real-time requirements while achieving generalization performance close to accuracy-oriented methods.
5. **Highly intuitive DDCA visualizations**: Comparisons between the original cost volume and the post-DDCA cost volume clearly demonstrate the suppression of erroneous matches.

## Limitations & Future Work

1. **The ViT-L variant requires 110 ms, losing real-time capability**: Larger backbones trade speed for accuracy.
2. **Difficulty handling very large disparity ranges during training**: Constrained by GPU memory during cost volume construction.
3. **Generalization to extreme weather and nighttime scenes is not demonstrated**: Evaluation is conducted primarily on standard benchmarks.
4. DDCA could potentially be extended to other tasks requiring cost aggregation, such as optical flow estimation.

## Related Work & Insights

- **RT-IGEV** (Xu et al., 2025): The current leading real-time method and the direct baseline for this work.
- **Depth Anything V2** (Yang et al., 2024): Provides the frozen depth feature encoder.
- **OverLoCK** (Lou & Yu, 2025): Inspires the dynamic convolution design in DDCA.
- **FoundationStereo** (Wen et al., 2025): A generalization-oriented method using larger models at the cost of real-time performance.
- **MonSter** (Cheng et al., 2025): A dual-branch architecture for handling scale-shift, but with slow inference.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The DDCA module is cleverly designed; the approach to avoiding scale-shift issues is valuable.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Five benchmarks, zero-shot and fine-tuned evaluations, comprehensive ablations, and reflective region analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation analysis is excellent; visualizations are convincing.
- **Value**: ⭐⭐⭐⭐⭐ — The demand for real-time and generalizable stereo matching is strong; this paper provides an elegant solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Domain Generalized Stereo Matching with Uncertainty-guided Data Augmentation](domain_generalized_stereo_matching_with_uncertainty-guided_data_augmentation.md)
- [\[CVPR 2026\] Lite Any Stereo: Efficient Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/lite_any_stereo_efficient_zero-shot_stereo_matching.md)
- [\[AAAI 2026\] RTGaze: Real-Time 3D-Aware Gaze Redirection from a Single Image](rtgaze_real-time_3d-aware_gaze_redirection_from_a_single_image.md)
- [\[ICCV 2025\] Diving into the Fusion of Monocular Priors for Generalized Stereo Matching](../../ICCV2025/3d_vision/diving_into_the_fusion_of_monocular_priors_for_generalized_stereo_matching.md)
- [\[AAAI 2026\] Cheating Stereo Matching in Full-Scale: Physical Adversarial Attack against Binocular Depth Estimation](cheating_stereo_matching_in_full-scale_physical_adversarial_attack_against_binoc.md)

</div>

<!-- RELATED:END -->

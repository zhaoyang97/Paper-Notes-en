---
title: >-
  [Paper Note] FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth
description: >-
  [ICML 2026][3D Vision][Image-to-Point Cloud Registration] This paper abstracts the human observation process of "first scanning broadly, then examining in detail" into a two-stage Focus-Sweep paradigm. It replaces Transf…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Image-to-Point Cloud Registration"
  - "Mamba/SSM"
  - "Reinforcement Learning for Layer Selection"
  - "Focus-Sweep"
  - "Multi-Scale Interaction"
date: 2026-05-08
content_hash: f546cd9180a58082
---

# FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth

**Conference**: ICML 2026  
**arXiv**: [2605.07607](https://arxiv.org/abs/2605.07607)  
**Code**: None  
**Area**: 3D Vision / Cross-Modal Registration  
**Keywords**: Image-to-Point Cloud Registration, Mamba/SSM, Reinforcement Learning for Layer Selection, Focus-Sweep, Multi-Scale Interaction

## TL;DR
This paper abstracts the human observation process of "first scanning broadly, then examining in detail" into a two-stage Focus-Sweep paradigm. It replaces Transformer with Mamba for image-point cloud interaction and uses reinforcement learning to dynamically determine the number of interaction layers at each scale, achieving SOTA in I2P registration on RGB-D Scenes V2 and 7-Scenes.

## Background & Motivation

**Background**: The mainstream approach for image-to-point cloud (I2P) registration has shifted from "detect-then-match" to a "detection-free" coarse-to-fine framework, such as 2D3D-MATR, B2-3D, and CA-I2P. These methods rely on multi-scale features and Transformer cross-attention to establish patch-level correspondences, followed by PnP+RANSAC for pose estimation.

**Limitations of Prior Work**: The authors identified two overlooked issues through experiments: (1) Stacking too many cross-attention layers leads to "attention drift," where small early-layer deviations are amplified (Matthew effect), increasing MMD. (2) While multi-scale designs alleviate some scale differences, they still suffer from scale ambiguity in repetitive texture scenes, causing mismatches.

**Key Challenge**: Cross-modal alignment inherently requires long-range interaction, necessitating deep layers. However, deeper layers exacerbate drift, and determining the number of layers is a discrete, non-differentiable decision, making it unlearnable via standard gradient descent. This creates dual trade-offs: "deep interaction vs. drift" and "learnable depth vs. discrete decisions."

**Goal**: (1) Design a cross-modal interaction mechanism more stable than Transformer cross-attention and capable of suppressing scale ambiguity. (2) Assign each scale an adaptive interaction depth, enabling the model to "stop when it has seen enough," akin to human behavior.

**Key Insight**: Cognitive psychology suggests that humans perform cross-modal matching in two steps: first, global scale estimation and coarse scanning (Focus), followed by detailed block-wise comparison (Sweep). This sequential, directional, and memory-retentive process aligns naturally with SSM (Mamba)'s scanning mechanism. The decision of "how many times to look" is essentially a strategy optimizable via RL.

**Core Idea**: Replace fixed-depth Transformer cross-attention with Mamba-based alternating Focus-Sweep interactions and RL-learned iterative depth at each scale.

## Method

### Overall Architecture
The input consists of an RGB image $I\in\mathbb{R}^{H\times W\times 3}$ and a point cloud $P\in\mathbb{R}^{N\times 3}$ from the same scene, with the goal of outputting the rigid transformation $[R,\mathbf{t}]$. The pipeline follows a coarse-to-fine approach: ResNet+FPN extracts 2D multi-scale features $F_{Ia},F_{Ib},F_{Ic}$, and KPFCNN extracts point cloud features $F_P$. Initial self/cross-attention layers establish preliminary connections. The core Hierarchical Focus-Sweep Interaction Module then alternates Focus (global scale alignment) and Sweep (block-wise fine interaction) on the three image feature scales. The number of FS-Layer iterations at each scale is determined by the RL-based Dynamic Layer Allocation Strategy. Finally, the multi-scale image features are concatenated, and cosine similarity with the three scales of point cloud features is computed, followed by element-wise max to obtain a score map. Top-k patch matching is refined to the pixel level, and PnP+RANSAC is used for pose estimation.

### Key Designs

1. **Focus (Norm-Adapted Global Coarse Alignment)**:

    - **Function**: Modulates image features using the "overall scale" of the entire point cloud, establishing coarse correspondences akin to "scanning broadly."
    - **Mechanism**: The point cloud features $F_P$ are globally average pooled and linearly projected to produce three channel-wise factors $[\alpha,\beta,\gamma]=\text{Linear}(\text{AvgPool}(F_P))$. These factors adjust the image features' statistics via $F'_i=\gamma\cdot\text{VSSM}(\alpha\cdot F_i+\beta)+F_i$, where VSSM is the visual SSM feedforward layer in VMamba. This operation avoids explicit cross-attention matrices, reducing overhead while aligning multi-scale image features with point cloud scales.
    - **Design Motivation**: The authors observed that Transformers amplify early-layer attention biases under scale mismatches. Performing coarse alignment as a one-time norm modulation prevents subsequent SSM layers from being repeatedly misled by incorrect scales.

2. **Sweep (Partition-Scan-Recover Block-Wise Interaction)**:

    - **Function**: Performs detailed block-wise comparisons after Focus, akin to "examining in detail," and is the primary module for accurate FS-I2P matching.
    - **Mechanism**: First, Partition the image $F_i\in\mathbb{R}^{h\times w\times C}$ into $P=hw/o^2$ non-overlapping patches $[F_i^1,\dots,F_i^t]$ of size $o$. Then, Scan by constructing a mixed sequence $F_H=[F_i^1 F_P, F_i^2 F_P,\dots, F_i^t F_P]$, where the point cloud sequence is repeatedly inserted after each image patch, followed by a VSSM layer. Finally, Recover the scanned sequence back into image features (via direct rearrangement), and compute a weighted average of the point cloud features $F_P^{re}=\sum_u \lambda_u F_P^t/t$ using learnable weights $\lambda=[\lambda_1,\dots,\lambda_t]$. SSM's token proximity sensitivity ensures that the point cloud sequence is "reminded" at each new patch, forcing repeated alignment of the current region with the global point cloud.
    - **Design Motivation**: Traditional cross-attention lacks sequential constraints between patches, leading to attention drift. Repeatedly inserting the point cloud between image patches and leveraging SSM's directional scanning preserves both local fine alignment and global receptive fields. Mamba's linear complexity makes dense interactions feasible.

3. **Dynamic Layer Allocation (RL-Driven Adaptive Depth)**:

    - **Function**: Dynamically selects FS-Layer iteration counts $\{n_1,n_2,n_3\}$ for the three scales (allowing zero layers to skip scale interaction), replacing fixed depths.
    - **Mechanism**: The state $s$ is constructed by concatenating mean+max pooling of image and point cloud tokens. A lightweight policy network $g_\theta$ outputs action logits $\mathbf{z}=g_\theta(s)$, yielding a categorical distribution $\pi_\theta(n\mid s)=\text{Softmax}(\mathbf{z})$. During training, actions $a\sim\pi_\theta(\cdot\mid s)$ are sampled, and $\log p=\log\pi_\theta(a\mid s)$ is recorded. During inference, actions are greedily selected as $a=\arg\max \mathbf{z}$. Rewards are derived from global registration constraints (e.g., Inlier Ratio / FMR / RR), and policy gradients update the strategy.
    - **Design Motivation**: Layer selection is discrete and non-differentiable, making it unlearnable via standard gradients. RL naturally aligns with the human behavior of "stopping when the goal is reached"—too few layers are insufficient, while too many introduce noise. Using global constraints as rewards is more direct than ad-hoc heuristics.

### Loss & Training
The training objective combines standard I2P registration losses (patch-level correspondence supervision + refinement-level supervision) with the policy gradient $\mathcal{L}_{RL}=-\mathbb{E}[R\cdot\log p]$, where $R$ is derived from inlier counts or distance errors. The maximum allowable depth $l_{\max}$ is a hyperparameter, and the three scales independently select depths from 0 to $l_{\max}$.

## Key Experimental Results

### Main Results
Two public benchmarks: RGB-D Scenes V2 (4 scenes) and 7-Scenes (7 scenes), evaluated on three common metrics: Inlier Ratio (IR), Feature Matching Recall (FMR), and Registration Recall (RR).

| Dataset | Metric | FS-I2P (Ours) | Flow-I2P | 2D3D-MATR | Notes |
|---------|--------|---------------|----------|-----------|-------|
| RGB-D Scenes V2 (mean) | IR | **42.9** | 40.1 | 32.4 | +2.8 vs previous best |
| RGB-D Scenes V2 (mean) | FMR | **94.4** | 93.3 | 90.8 | Matches B2-3D's best |
| 7-Scenes (mean) | IR | **53.9** | 52.0 | 50.1 | Average across all 7 scenes |
| 7-Scenes (mean) | FMR | 92.4 | 91.6 | 92.1 | Matches SOTA |

Significant improvements were observed in Scene-11 / Scene-12 (highly repetitive textures), validating the mitigation of scale ambiguity.

### Ablation Study

| Configuration | RGB-D V2 mean IR | Notes |
|---------------|------------------|-------|
| Full FS-I2P | 42.9 | Complete model |
| w/o Focus (Sweep only) | Significant drop | Lacks global scale alignment, causing multi-scale interference |
| w/o Sweep (Focus only) | Large drop | Lacks fine block-wise interaction; norm modulation alone is insufficient |
| w/o Dynamic Layer (Fixed 4 layers) | Slightly lower | Fixed depth cannot adapt to different scenes; Figure 3 shows MMD increases with deeper Transformers |
| Mamba → Transformer | Drop | Demonstrates that Mamba's benefits extend beyond component replacement, structurally mitigating the Matthew effect |

### Key Findings
- Transformer depth increases initially reduce MMD (image-point cloud feature distribution distance) but later cause it to rise. FS-I2P avoids this drift using SSM + RL adaptive depth, as evidenced by T-SNE visualizations showing tighter clustering.
- The learned Dynamic Layer Allocation strategy is interpretable: it deepens specific scales in high-scale-difference scenes and skips some scales in simpler scenes, validating the "observe as needed" hypothesis.
- Focus and Sweep are individually insufficient; their alternation is the primary driver of performance, highlighting the need for both global scale priors and fine block-wise comparisons in cross-modal alignment, akin to human two-stage perception.

## Highlights & Insights
- Mamba's "order sensitivity + linear complexity" naturally aligns with the two-stage observation theory in cognitive science, creating an elegant "motivation → backbone choice" linkage.
- Repeatedly inserting the point cloud sequence after each image patch is a clever engineering trick: leveraging SSM's sensitivity to recent tokens, it achieves repeated alignment without explicit cross-attention. This approach is transferable to any cross-modal matching task involving sequences and sets (e.g., text-to-point cloud, speech-to-image).
- Transforming "interaction depth" from a manually tuned hyperparameter into an RL-learned strategy is the first work in detection-free frameworks to dynamically determine "how many times to look," with potential applications in any coarse-to-fine architecture.
- The paper provides concrete evidence of the Matthew effect (MMD curves at different depths), moving beyond generic claims of "cross-attention overfitting," and offers data-driven insights for future SSM vs. Transformer trade-offs.

## Limitations & Future Work
- The authors acknowledge that RL training requires differentiable/semi-differentiable global rewards, and the transferability of reward design to larger I2P datasets (e.g., outdoor large-scale scenes like KITTI) remains unverified.
- Self-assessment: The policy network's state $s$ uses only mean+max pooling, which is coarse. In highly complex geometries, the learned strategy may still be overly conservative.
- The paper does not provide results on cross-dataset generalization (e.g., training on RGB-D V2 → testing on 7-Scenes), leaving the strategy's overfitting to specific benchmark scale distributions uncertain.
- Comparisons with LiDAR-based outdoor large-scale registration (KITTI, NuScenes) are missing; current validation is limited to indoor RGB-D scenes, and generalizability requires further experiments.
- Future work could extend Focus-Sweep to multi-view I2P (multiple images registered to the same point cloud), using RL to jointly select layer depth and viewpoints.

## Related Work & Insights
- **vs 2D3D-MATR**: Both are detection-free coarse-to-fine frameworks, but 2D3D-MATR uses Transformer cross-attention for fixed-depth interaction. This paper employs Mamba + RL for dynamic depth, offering greater robustness to repetitive textures.
- **vs B2-3D**: B2-3D uses hierarchical cross-attention to address scale ambiguity. This paper replaces attention with norm-adaptation Focus and block-wise SSM Sweep, highlighting the Matthew effect in stacked cross-attention.
- **vs Flow-I2P / Diff2I2P**: Flow-I2P uses Beltrami flow, and Diff2I2P employs depth-conditioned diffusion. This paper takes a cognitive engineering approach based on human perception + SSM, avoiding reliance on additional depth/diffusion priors and enabling single-pass inference.
- **Transferable Insights**: (1) The use of SSM's token order to construct cross-modal alignment anchors can generalize to any heterogeneous sequence fusion task. (2) The RL-based layer selection paradigm can be applied to any task where backbone depth is a hyperparameter (e.g., dynamic Transformers, dynamic diffusion steps).

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of Focus-Sweep paradigm, Mamba interaction, and RL dynamic depth is novel in the I2P domain, though each individual component is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, three metrics, comprehensive comparisons with the latest five baselines, and experimental evidence for the Matthew effect; lacks outdoor large-scale scenes.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation-to-method linkage, with cognitive psychology analogies making the architecture choice convincing. Equations and illustrations are well-integrated.
- Value: ⭐⭐⭐⭐ Advances SOTA in the relatively niche but practical I2P domain, with the RL depth selection idea offering transferable value to other dynamic architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection](../../ICCV2025/3d_vision/ca-i2p_channel-adaptive_registration_network_with_global_optimal_selection.md)
- [\[CVPR 2026\] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](../../CVPR2026/3d_vision/cmhanet_a_crossmodal_hybrid_attention_network_for.md)
- [\[NeurIPS 2025\] DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints](../../NeurIPS2025/3d_vision/dualfocus_depth_from_focus_with_spatio-focal_dual_variational_constraints.md)
- [\[CVPR 2026\] Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting](../../CVPR2026/3d_vision/hierarchical_visual_relocalization_with_nearest_view_synthesis_from_feature_gaus.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)

</div>

<!-- RELATED:END -->

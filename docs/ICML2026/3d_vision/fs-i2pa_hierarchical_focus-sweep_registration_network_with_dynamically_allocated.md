---
title: >-
  [Paper Note] FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth
description: >-
  [ICML 2026][3D Vision][Image-to-PointCloud registration] This paper abstracts the human observation process of "scanning first then examining piece by piece" into a two-stage Focus-Sweep paradigm. It replaces Transformer…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Image-to-PointCloud registration"
  - "Mamba/SSM"
  - "RL layer selection"
  - "Focus-Sweep"
  - "Multi-scale interaction"
date: 2026-05-08
content_hash: 7485adce778d8c07
---

# FSI2P: A Hierarchical Focus–Sweep Registration Network with Dynamically Allocated Depth

**Conference**: ICML 2026  
**arXiv**: [2605.07607](https://arxiv.org/abs/2605.07607)  
**Code**: None  
**Area**: 3D Vision / Cross-modal Registration  
**Keywords**: Image-to-PointCloud registration, Mamba/SSM, RL layer selection, Focus-Sweep, Multi-scale interaction

## TL;DR
This paper abstracts the human observation process of "scanning first then examining piece by piece" into a two-stage Focus-Sweep paradigm. It replaces Transformer with Mamba for image-point cloud interaction and employs reinforcement learning to dynamically determine the number of interaction layers at each scale, achieving SOTA on RGB-D Scenes V2 and 7-Scenes for I2P registration.

## Background & Motivation

**Background**: The mainstream approach for Image-to-PointCloud (I2P) registration has transitioned from "detect-then-match" to "detection-free" coarse-to-fine frameworks, such as 2D3D-MATR, B2-3D, and CA-I2P. These methods rely on multi-scale features combined with Transformer cross-attention to establish patch-level correspondences, followed by PnP+RANSAC for pose estimation.

**Limitations of Prior Work**: The authors observe two overlooked issues through experiments: first, stacking too many cross-attention layers leads to "attention drift," where small deviations in early layers are repeatedly amplified (the Matthew effect), causing the MMD to actually increase; second, while multi-scale designs mitigate some scale discrepancies, scale ambiguity still occurs in repetitive texture scenes due to similar textures across different resolutions, leading to misalignments.

**Key Challenge**: Cross-modal alignment inherently requires long-range interaction, necessitating many layers; however, more layers increase the risk of drift. Furthermore, deciding the number of layers is a discrete, non-differentiable decision that cannot be learned via standard gradient descent—creating a double trade-off between "requiring deep interaction" vs "deep interaction causing drift" and "learnable depth" vs "discrete decisions."

**Goal**: (1) Design a cross-modal interaction mechanism that is more stable than Transformer cross-attention and capable of suppressing scale ambiguity; (2) allocate a data-adaptive interaction depth for each scale, allowing the model to "stop once it has seen enough," much like a human.

**Key Insight**: Cognitive psychology indicates that humans perform two steps during cross-modal matching: first, a global scale estimation and coarse scan (Focus), followed by region-wise detailed comparison (Sweep). This sequential, directional process maintaining long-term memory aligns naturally with the scanning mechanism of SSM (Mamba). Moreover, deciding "how many looks are enough" is essentially a strategy that can be optimized by RL.

**Core Idea**: Replace fixed-depth cross-attention in Transformers with Mamba-based alternating "Focus-Sweep" interactions and use RL to learn the iterative depth at each scale.

## Method

### Overall Architecture
The input consists of an RGB image $I\in\mathbb{R}^{H\times W\times 3}$ and a point cloud $P\in\mathbb{R}^{N\times 3}$ from the same scene, with the goal of outputting the rigid transformation $[R,\mathbf{t}]$. The overall pipeline is coarse-to-fine: ResNet+FPN extracts 2D multi-scale features $F_{Ia},F_{Ib},F_{Ic}$, and KPFCNN extracts point cloud features $F_P$. A preliminary bridge is established via one layer of self/cross-attention. Then, features enter the core Hierarchical Focus-Sweep Interaction Module, which performs alternating Focus (global scale alignment) and Sweep (fine-grained patch interaction) on image features across three scales. The number of FS-Layer iterations at each scale is determined by the RL policy network of the Dynamic Layer Allocation Strategy. Finally, multi-scale image features are concatenated and element-wise maximized with point cloud features of three scales using cosine similarity to obtain a score map. Top-k patch matching is performed and refined to the pixel level, with the final pose calculated via PnP+RANSAC.

### Key Designs

1.  **Focus (Global Coarse Alignment via Norm Adaptation)**:
    - **Function**: Uses the "overall scale" of the entire point cloud to modulate image features in one go, establishing coarse correspondence, analogous to a human "glancing first."
    - **Mechanism**: Global average pooling of $F_P$ is followed by a linear layer to project three sets of channel-wise factors $[\alpha,\beta,\gamma]=\text{Linear}(\text{AvgPool}(F_P))$. Image features statistics are then modified as $F'_i=\gamma\cdot\text{VSSM}(\alpha\cdot F_i+\beta)+F_i$, where VSSM is the visual SSM feed-forward layer from VMamba. This operation lacks an explicit cross-attention matrix, incurring minimal overhead while rearranging image channel mean/variance "according to point cloud characteristics" to align multi-scale image features with the point cloud scale.
    - **Design Motivation**: The authors found that Transformers tend to amplify early attention deviations when scales are inconsistent. Using a one-time norm modulation for coarse alignment prevents subsequent SSMs from being repeatedly misled by incorrect scales.

2.  **Sweep (Local Fine Interaction via Partition-Scan-Recover)**:
    - **Function**: Performs fine-grained region-wise comparison after Focus, analogous to "looking back and forth at parts," acting as the primary module for accurate matching in FS-I2P.
    - **Mechanism**: First, **Partition**—divide the image $F_i\in\mathbb{R}^{h\times w\times C}$ into $P=hw/o^2$ non-overlapping patches $[F_i^1,\dots,F_i^t]$ of size $o$. Next, **Scan**—construct a hybrid sequence $F_H=[F_i^1 F_P, F_i^2 F_P,\dots, F_i^t F_P]$ by repeatedly inserting the point cloud sequence after each image patch, then pass it through a VSSM layer. Finally, **Recover**—split the scanned sequence back into image features (direct reshuffling), while point cloud features are weighted averaged using learnable weights $\lambda=[\lambda_1,\dots,\lambda_t]$ as $F_P^{re}=\sum_u \lambda_u F_P^t/t$. Utilizing the SSM property where "tokens closer to the current time receive more focus," the point cloud sequence "reminds" the model each time it enters a new patch, forcing the model to repeatedly align the current region with the global point cloud.
    - **Design Motivation**: Traditional cross-attention lacks sequence constraints between different patches, making attention prone to drifting. By inserting point clouds between image patches and using the directional scanning of SSM, the model maintains local fine alignment while preserving a global receptive field. Mamba's linear complexity makes such dense interaction feasible.

3.  **Dynamic Layer Allocation (RL-Driven Adaptive Depth)**:
    - **Function**: Dynamically selects the number of FS-Layer iterations for the three scales $\{n_1,n_2,n_3\}$ (allowing 0 to skip interaction at a scale) rather than using a fixed depth.
    - **Mechanism**: A state $s$ is formed by concatenating the mean and max pooling of image and point cloud tokens. A lightweight policy network $g_\theta$ outputs action logits $\mathbf{z}=g_\theta(s)$, yielding a categorical distribution over candidate depths $\pi_\theta(n\mid s)=\text{Softmax}(\mathbf{z})$. During training, $a\sim\pi_\theta(\cdot\mid s)$ is sampled and $\log p=\log\pi_\theta(a\mid s)$ is recorded; during inference, the greedy action $a=\arg\max \mathbf{z}$ is taken. Rewards are derived from global registration constraints (Inlier Ratio / FMR / RR, etc.), and the policy is updated via policy gradient.
    - **Design Motivation**: Layer selection is discrete and non-differentiable, making standard gradient learning impossible. RL naturally fits the human behavior of "stopping once the target is seen"—too few layers are inaccurate, while too many introduce noise. Using global constraints as rewards is more direct than ad-hoc heuristics.

### Loss & Training
The training objective = standard I2P registration loss (patch-level correspondence supervision + refinement-level supervision) + policy gradient $\mathcal{L}_{RL}=-\mathbb{E}[R\cdot\log p]$, where $R$ is constructed from the number of registration inliers or distance error. The maximum allowed depth $l_{\max}$ is a hyperparameter, and the three scales can independently choose from $0..l_{\max}$.

## Key Experimental Results

### Main Results
Two public benchmarks: RGB-D Scenes V2 (4 scenes) and 7-Scenes (7 scenes), with three common metrics: Inlier Ratio (IR), Feature Matching Recall (FMR), and Registration Recall (RR).

| Dataset | Metric | FS-I2P (Ours) | Flow-I2P | 2D3D-MATR | Remarks |
|--------|------|--------------|---------|-----------|------|
| RGB-D Scenes V2 (mean) | IR | **42.9** | 40.1 | 32.4 | +2.8 vs. previous best |
| RGB-D Scenes V2 (mean) | FMR | **94.4** | 93.3 | 90.8 | Best, tied with B2-3D |
| 7-Scenes (mean) | IR | **53.9** | 52.0 | 50.1 | Average of all 7 scenes |
| 7-Scenes (mean) | FMR | 92.4 | 91.6 | 92.1 | Tied with SOTA |

Improvements are particularly significant in Scene-11 / Scene-12 (scenes with severe repetitive textures), validating the mitigation of scale ambiguity.

### Ablation Study

| Configuration | RGB-D V2 mean IR | Description |
|------|------------------|------|
| Full FS-I2P | 42.9 | Full model |
| w/o Focus (Sweep only) | Significant drop | Lack of global scale alignment; multi-scales interfere |
| w/o Sweep (Focus only) | Large drop | Lack of fine patch interaction; norm modulation alone is insufficient |
| w/o Dynamic Layer (Fixed 4) | Slightly lower | Fixed depth cannot adapt to different scenes; Fig. 3 shows MMD rises as Transformer deepens |
| Mamba → Transformer | Drop | Proves Mamba is more than a replacement; it offers structural gains against the Matthew effect |

### Key Findings
- When Transformer depth increases, MMD (Image-PointCloud feature distribution distance) first drops then rises, while FS-I2P avoids this drift using SSM + RL adaptive depth; T-SNE visualization shows better clustering.
- Strategies learned by Dynamic Layer Allocation are explainable: it tends to increase depth for specific scales in scenes with large scale discrepancies and skip some scales in simple structural scenes, validating the "observation on demand" hypothesis.
- Focus and Sweep alone are not strong enough; their alternation is the primary performance driver—this indicates that cross-modal alignment requires both global scale priors and fine-grained patch comparison, similar to two-stage perception in the human brain.

## Highlights & Insights
- Matching "sequential sensitivity + linear complexity" of Mamba with two-stage observation theory from cognitive science aligns architecture choice with human perception, creating an elegant "motivation → backbone selection" pipeline.
- The "repeatedly inserting the point cloud sequence after each image patch" is a clever engineering trick: capitalizing on the SSM property of being more sensitive to recent tokens allows for repeated alignment without explicit cross-attention. This can be directly transferred to any "sequence vs. set" cross-modal matching tasks (e.g., text-to-point cloud).
- Transforming "interaction depth"—previously a manually selected hyperparameter—into an RL strategy makes this the first work in the detection-free series to explicitly dynamize "how many looks." This idea can be generalized to any coarse-to-fine framework.
- The paper provides specific evidence of the Matthew effect (MMD curves at different depths), moving beyond vague claims of "cross-attention overfitting" with data-driven support for the SSM vs. Transformer discussion.

## Limitations & Future Work
- The authors acknowledge that RL training requires differentiable or semi-differentiable global rewards; the transferability of reward designs to more I2P datasets (e.g., large-scale outdoor KITTI) is unverified.
- Self-evaluation: The state $s$ of the policy network only uses mean+max pooling, which is relatively coarse; in highly complex geometries, the learned strategy might still be conservative.
- The paper does not provide RL policy transfer results across datasets (e.g., trained on RGB-D V2 → tested on 7-Scenes), making it difficult to judge if the policy itself overfits to the scale distribution of a specific benchmark.
- Lack of comparison with outdoor large-scale LiDAR registration (KITTI, NuScenes); currently only validated in indoor RGB-D scenes.
- Future work could extend Focus-Sweep to multi-view I2P (matching multiple images to one point cloud), using RL to select both the number of layers and the views.

## Related Work & Insights
- **vs. 2D3D-MATR**: Also detection-free coarse-to-fine, but 2D3D-MATR uses Transformer cross-attention for fixed-depth interaction; Ours uses Mamba + RL dynamic depth, proving much more robust to repetitive textures.
- **vs. B2-3D**: B2-3D uses hierarchical cross-attention for scale ambiguity; Ours replaces attention with norm-adapted Focus + partitioned SSM Sweep and highlights the Matthew effect in cross-attention stacking.
- **vs. Flow-I2P / Diff2I2P**: Flow-I2P uses Beltrami flow, and Diff2I2P uses depth-conditioned diffusion; Ours follows a "Cognitive Psychology + SSM" cognitive engineering route, independent of extra depth/diffusion priors, requiring only a single pass during inference.
- **Transferable Insights**: (1) Using SSM token order to construct cross-modal alignment anchors can be extended to any heterogeneous sequence fusion; (2) The RL-based layer selection paradigm can be applied to any task where backbone depth is a hyperparameter (dynamic transformers, dynamic diffusion steps).

## Rating
- Novelty: ⭐⭐⭐⭐ Focus-Sweep paradigm + Mamba interaction + RL dynamic depth; the combination is a first in the I2P field, though individual components are known.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks, three metrics, comprehensive comparison with 5 recent baselines, plus experimental evidence for the Matthew effect; missing outdoor large-scale scenes.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation-to-method pipeline, cognitive psychology analogy makes the architecture choice convincing, with good coordination between formulas and diagrams.
- Value: ⭐⭐⭐⭐ Solidly pushes the SOTA in the niche but practical I2P direction; the RL depth selection idea is valuable for other dynamic architectures.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CMHANet: A Cross-Modal Hybrid Attention Network for Point Cloud Registration](../../CVPR2026/3d_vision/cmhanet_a_crossmodal_hybrid_attention_network_for.md)
- [\[NeurIPS 2025\] DualFocus: Depth from Focus with Spatio-Focal Dual Variational Constraints](../../NeurIPS2025/3d_vision/dualfocus_depth_from_focus_with_spatio-focal_dual_variational_constraints.md)
- [\[ICCV 2025\] CA-I2P: Channel-Adaptive Registration Network with Global Optimal Selection](../../ICCV2025/3d_vision/ca-i2p_channel-adaptive_registration_network_with_global_optimal_selection.md)
- [\[CVPR 2026\] Hierarchical Visual Relocalization with Nearest View Synthesis from Feature Gaussian Splatting](../../CVPR2026/3d_vision/hierarchical_visual_relocalization_with_nearest_view_synthesis_from_feature_gaus.md)
- [\[ICCV 2025\] Depth AnyEvent: A Cross-Modal Distillation Paradigm for Event-Based Monocular Depth Estimation](../../ICCV2025/3d_vision/depth_anyevent_a_cross-modal_distillation_paradigm_for_event-based_monocular_dep.md)

</div>

<!-- RELATED:END -->

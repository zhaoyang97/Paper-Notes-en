---
title: >-
  [Paper Note] FoundationStereo: Zero-Shot Stereo Matching
description: >-
  [CVPR 2025][3D Vision][Zero-shot stereo matching] Presents FoundationStereo, a large-scale foundation model for stereo depth estimation. By leveraging a million-scale high-fidelity synthetic dataset, fusing monocular depth priors via a Side-Tuning Adapter, and adopting a hybrid cost volume filtering mechanism (incorporating Axial-Planar Convolution and Disparity Transformer), this method achieves strong zero-shot generalization performance without requiring target-domain fine…
tags:
  - "CVPR 2025"
  - "3D Vision"
  - "Zero-shot stereo matching"
  - "foundation model"
  - "monocular depth prior"
  - "synthetic data"
  - "cost volume filtering"
date: 2026-05-08
content_hash: d54dae341b772a9b
---

# FoundationStereo: Zero-Shot Stereo Matching

**Conference**: CVPR 2025  
**arXiv**: [2501.09898](https://arxiv.org/abs/2501.09898)  
**Code**: [https://nvlabs.github.io/FoundationStereo/](https://nvlabs.github.io/FoundationStereo/)  
**Area**: 3D Vision / Stereo Matching  
**Keywords**: Zero-shot stereo matching, foundation model, monocular depth prior, synthetic data, cost volume filtering  

## TL;DR
Presents FoundationStereo, a large-scale foundation model for stereo depth estimation. By leveraging a million-scale high-fidelity synthetic dataset, fusing monocular depth priors via a Side-Tuning Adapter, and adopting a hybrid cost volume filtering mechanism (incorporating Axial-Planar Convolution and Disparity Transformer), this method achieves strong zero-shot generalization performance without requiring target-domain fine-tuning.

## Background & Motivation

**Background**: Deep stereo matching has reached saturation on top benchmarks under per-domain fine-tuning settings. Dominant methodologies include cost volume filtering (e.g., GwcNet, IGEV) and iterative refinement (e.g., RAFT-Stereo), but all rely heavily on target-domain fine-tuning to yield competitive results.

**Limitations of Prior Work**: While other vision tasks (such as segmentation with SAM or monocular depth estimation with DepthAnything) have demonstrated robust zero-shot generalization capabilities, a true "foundation model" has remained absent in the stereo matching domain. Existing cross-domain generalization methods are primarily trained on Scene Flow (only 40K pairs), lacking sufficient data scale and diversity. Architecturally, 3D CNNs are limited by small kernel sizes, making it difficult to capture global context under large disparity settings.

**Key Challenge**: Zero-shot generalization in stereo matching is constrained by the scale and diversity of training data, as well as the representational capacity of network architectures—existing structures cannot effectively exploit large-scale training data.

**Goal**: Construct a foundation model for stereo matching that achieves or exceeds the accuracy of fine-tuned methods in diverse scenes without requiring target-domain fine-tuning.

**Key Insight**: Simultaneously address three dimensions: (1) resolve data bottlenecks via a million-scale high-fidelity synthetic dataset; (2) bridge the sim-to-real gap by adapting rich priors from monocular depth foundation models; and (3) design scalable architectural components to enhance contextual reasoning across disparity and spatial dimensions.

**Core Idea**: Promote stereo matching to foundation-model-level zero-shot generalization through a three-pronged approach: large-scale data, monocular prior adaptation, and long-range cost volume filtering.

## Method

### Overall Architecture
Given left and right stereo image pairs as input, multi-scale features fused with DepthAnythingV2 priors are extracted via the Side-Tuning Adapter (STA) to construct a hybrid cost volume (group-wise correlation + feature concatenation). The cost volume is then filtered using Attentive Hybrid Cost Filtering (AHCF), which incorporates an APC hourglass network and a Disparity Transformer. A soft-argmin operation generates the initial disparity, which is further refined iteratively through a multi-scale GRU to output the final dense disparity map.

### Key Designs

1. **Side-Tuning Adapter (STA)**:

    - Function: Adapts the rich semantic and geometric priors of a pre-trained monocular depth model (DepthAnythingV2) to the stereo matching task.
    - Mechanism: Freezes the ViT backbone of DepthAnythingV2 to extract features, downsamples the DPT head output, and concatenates it with multi-scale features of the same level from a CNN (EdgeNeXt-S) to form 1/4-scale mixed features. The CNN branch learns to adapt the ViT features for the stereo matching task. The authors compared three fusion strategies and found that the simplest "downsampling + concatenation" significantly outperforms ViT-Adapter-style interactions and directly utilizing ViT features.
    - Design Motivation: DepthAnythingV2, trained on massive real-world images, contains rich semantic and geometric priors that can bridge the gap between synthetic training data and real-world scenes. Freezing the ViT avoids damaging learned priors, while CNN side-tuning allows the model to learn how to translate monocular priors into features suitable for stereo matching.

2. **Axial-Planar Convolution (APC)**:

    - Function: Expands the receptive field in cost volume hourglass filtering, especially under large disparity settings.
    - Mechanism: Decouples standard $3\times3\times3$ 3D convolutions into a spatial convolution of size $K_s \times K_s \times 1$ and a disparity convolution of size $1 \times 1 \times K_d$, resembling a 3D version of depthwise separable convolution but without splitting channels. This enables larger kernel sizes (e.g., $K_s=5, K_d=7$) without exhausting GPU memory.
    - Design Motivation: Traditional $3\times3\times3$ convolutions have insufficient receptive fields at large disparities, and directly expanding to $5\times5\times5$ causes out-of-memory (OOM) errors on 80GB GPUs. The decoupled design of APC greatly enhances representational capacity under equivalent VRAM constraints, allowing the model to better exploit large-scale training data.

3. **Disparity Transformer (DT)**:

    - Function: Performs global self-attention reasoning along the disparity dimension within the cost volume.
    - Mechanism: First downsamples the cost volume using a 3D convolution with a stride of $4\times4\times4$, reshapes the tensor, and performs multi-head self-attention using FlashAttention along the disparity dimension (using 4 transformer encoder blocks). The resolution is then restored via trilinear interpolation and added back to the hourglass output. This models global dependencies among different disparity levels at each spatial location.
    - Design Motivation: The disparity dimension of a cost volume encodes matching probability distributions, where long-range dependencies are crucial for resolving repetitive textures and large textureless regions. Even with APC, 3D CNNs can only capture local disparity context; DT fills this global reasoning gap.

### Loss & Training
The loss function consists of two parts: a smooth L1 loss for the initial disparity, and an exponentially weighted L1 loss ($\gamma=0.9$) for the sequence of iteratively refined disparities. Training is conducted on 32 A100 GPUs with a total batch size of 128 for 200K steps using the AdamW optimizer with a learning rate of 1e-4. Inputs are cropped to 320×736, with 22 GRU iterations. The dataset is a mixture of their self-developed FSD and multiple public synthetic datasets. An iterative self-curation pipeline is utilized: the current model is evaluated on FSD, and samples with BP-2 > 60% are deemed ambiguous and regenerated, alternating for two rounds.

## Key Experimental Results

### Main Results

| Dataset | Metric | FoundationStereo | Prev. SOTA | Gain |
|--------|------|-----------------|---------|------|
| Middlebury | BP-2↓ | **1.1** | 7.5 (NMRF) | -85% |
| ETH3D | BP-1↓ | **0.5** | 1.8 (Scene Flow version of Ours) | -72% |
| KITTI-12 | D1↓ | **2.3** | 3.2 (S-IGEV*) | -28% |
| KITTI-15 | D1↓ | **2.8** | 4.5 (S-IGEV*) | -38% |

### Ablation Study

| Configuration | Middlebury BP-2 | ETH3D BP-1 | Description |
|------|----------------|-----------|------|
| Full model | **1.1** | **0.5** | Complete model |
| W/o STA | Degraded significantly | Degraded significantly | No monocular priors, poor prediction in ambiguous regions |
| W/o AHCF (with 3D CNN) | Degraded | Degraded | Performance decays in fine structures and repetitive texture regions |
| STA design (a) Direct ViT | Poor | Poor | ViT features are not adequately adapted to the stereo task |
| STA design (b) ViT-Adapter | Moderate | Moderate | Interactive fusion performs worse than simple concatenation |

### Key Findings
- Even when trained solely on Scene Flow, FoundationStereo consistently outperforms all baseline methods, demonstrating the effectiveness of introducing monocular priors via STA.
- STA delivers the most prominent improvements in illumination-inconsistent regions (such as shadows of light fixtures) and geometrically ambiguous regions (such as guitar soundholes).
- AHCF yields the most notable improvements in thin, repetitive structures.
- The self-curation pipeline effectively identifies ambiguous samples in the dataset (e.g., highly repetitive textures, solid color uninformative regions), enhancing training stability.

## Highlights & Insights
- **Inspirations of the Side-Tuning Strategy**: Freezing pre-trained foundation models as "knowledge sources" and using a lightweight CNN for task adaptation is a paradigm that can be extended to any scenario where large model priors need to be utilized but task differences are substantial (e.g., adapting CLIP to detection, or SAM to tracking).
- **Engineering Value of APC**: Spatial-disparity decoupling of 3D separable convolutions is a practical technique that considerably expands the receptive field while controlling GPU memory consumption. It is highly applicable to any scenario requiring large-kernel 3D convolutions (e.g., video understanding, 4D reconstruction).
- **Self-Curation Training Pipeline**: A closed-loop design that refiles and cleans training data using the model itself is extremely practical under large-scale synthetic data scenarios. This is worth referencing in other domains reliant on synthetic data (e.g., 6DoF pose estimation, optical flow).

## Limitations & Future Work
- The million-scale synthetic dataset is generated using NVIDIA Omniverse, setting a high barrier to reproduction that is difficult for external researchers to replicate.
- DepthAnythingV2 is kept frozen; the adaptation capability of STA may be constrained by representation bottlenecks of ViT features.
- Using 32 GRU iterations during inference entails significant computational overhead. Adaptive iteration schemes could be explored.
- Potential improvements: Support variable-resolution inference to avoid accuracy loss caused by resizing; explore lightweight versions of DepthAnything to reduce inference costs.

## Related Work & Insights
- **vs RAFT-Stereo / IGEV Series**: Traditional approaches are highly robust after domain-specific fine-tuning but show weak zero-shot capabilities. FoundationStereo bridges this generalization gap through data scale and architectural improvements, directly rivaling or exceeding fine-tuned results out-of-the-box.
- **vs Monocular Depth Models (DepthAnythingV2)**: FoundationStereo absorbs the priors of monocular depth models while preserving the sub-pixel precision and absolute scale capability of stereo matching.
- **vs Concurrent Works**: Concurrent monocular prior-enhanced methods also exploit monocular priors to assist the correlation volume. However, FoundationStereo achieves more robust overall performance through the combination of STA, AHCF, and large-scale data.

## Rating
- Novelty: ⭐⭐⭐⭐ Individual components are not entirely novel, but their combination and engineering optimization are executed to perfection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 5 benchmarks, thorough ablation studies, and qualitative results in-the-wild.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, detailed methodology description, and high-quality charts/figures.
- Value: ⭐⭐⭐⭐⭐ A milestone task in the stereo matching field, realizing the first truly zero-shot foundation model.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MVSAnywhere: Zero-Shot Multi-View Stereo](mvsanywhere_zero-shot_multi-view_stereo.md)
- [\[CVPR 2026\] Fast-FoundationStereo: Real-Time Zero-Shot Stereo Matching](../../CVPR2026/3d_vision/fast-foundationstereo_real-time_zero-shot_stereo_matching.md)
- [\[ICCV 2025\] ZeroStereo: Zero-shot Stereo Matching from Single Images](../../ICCV2025/3d_vision/zerostereo_zero-shot_stereo_matching_from_single_images.md)
- [\[CVPR 2025\] DEFOM-Stereo: Depth Foundation Model Based Stereo Matching](defom-stereo_depth_foundation_model_based_stereo_matching.md)
- [\[CVPR 2026\] PromptStereo: Zero-Shot Stereo Matching via Structure and Motion Prompts](../../CVPR2026/3d_vision/promptstereo_zero-shot_stereo_matching_via_structure_and_motion_prompts.md)

</div>

<!-- RELATED:END -->

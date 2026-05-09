---
title: >-
  [Paper Note] FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases
description: >-
  [ICCV 2025][Video Understanding][optical flow estimation] FlowSeek integrates the prior knowledge of a depth foundation model (Depth Anything V2) and classical low-dimensional motion parameterization (motion bases) into an optical flow network, achieving state-of-the-art cross-dataset generalization while training on a single consumer-grade GPU.
tags:
  - ICCV 2025
  - Video Understanding
  - optical flow estimation
  - depth foundation model
  - motion bases
  - low-resource training
  - cross-dataset generalization
date: 2026-05-08
content_hash: 21b24562f91e5d32
---

# FlowSeek: Optical Flow Made Easier with Depth Foundation Models and Motion Bases

**Conference**: ICCV 2025
**arXiv**: [2509.05297](https://arxiv.org/abs/2509.05297)
**Code**: [https://flowseek25.github.io/](https://flowseek25.github.io/)
**Area**: Video Understanding / Optical Flow Estimation
**Keywords**: optical flow estimation, depth foundation model, motion bases, low-resource training, cross-dataset generalization

## TL;DR

FlowSeek integrates the prior knowledge of a depth foundation model (Depth Anything V2) and classical low-dimensional motion parameterization (motion bases) into an optical flow network, achieving state-of-the-art cross-dataset generalization while training on a single consumer-grade GPU.

## Background & Motivation

Optical flow estimation is a classical problem in computer vision, and iterative deep networks exemplified by RAFT have achieved remarkable progress in recent years. However, a common dependency among current state-of-the-art methods (e.g., SEA-RAFT, FlowFormer) is the **requirement for large numbers of high-end GPUs during training**—FlowFormer uses 4×V100, and SEA-RAFT uses 8×3090.

This hardware dependency gives rise to two core problems:

**Academic inequality**: Research groups with limited GPU resources cannot reproduce or compete with these methods.

**Methodological stagnation**: Over-reliance on hardware to push accuracy may obscure more fundamental methodological innovations.

Inspired by DeepSeek (a paradigm for low-resource training in NLP), the authors argue that **reusing knowledge from existing visual foundation models is more efficient than training from scratch**. Optical flow and depth are geometrically closely related—given fixed camera motion, the pixel displacement of a point is proportional to the inverse depth of its corresponding 3D point. Therefore, the rich geometric priors embedded in depth foundation models can be "transplanted" into the optical flow task.

**Core Idea**: Transform features and depth outputs from Depth Anything V2 into optical flow priors via classical 6-DOF motion bases, inject them into the SEA-RAFT framework, and achieve state-of-the-art accuracy at minimal training cost.

## Method

### Overall Architecture

FlowSeek is built upon the SEA-RAFT backbone and integrates three levels of techniques:
1. **Modern optical flow network design** (the iterative refinement framework of SEA-RAFT)
2. **Depth foundation model** (Depth Anything V2, with frozen parameters)
3. **Classical motion parameterization** (low-dimensional motion bases theory from 30 years ago)

### Key Designs

#### 1. Depth Foundation Model Feature Injection

Given an image pair $\mathbf{I}_0, \mathbf{I}_1$, FlowSeek uses the frozen Depth Anything V2 to extract:
- **Inverse depth maps** $\mathbf{D}_0, \mathbf{D}_1$
- **Decoder end features** $\mathbf{\Phi}_0, \mathbf{\Phi}_1$ (intermediate representations highly correlated with depth)

These features are downsampled to 1/8 resolution via a BottleNeck network (three 3×3 convolutions with stride 2), then concatenated with the output of the original feature extractor to form enhanced features:

$$\mathbf{F}_0^{\mathbf{\Phi}} = \text{FeatNet}(\mathbf{I}_0) \oplus \text{BottNeck}(\mathbf{\Phi}_0)$$

The enhanced features are used to construct the 4D correlation volume, improving matching quality.

#### 2. Low-Dimensional Motion Bases

This is the most central innovation of FlowSeek. Classical theory states that for a static scene with known depth, optical flow can be decomposed as a linear combination of 6 basis vectors corresponding to the 6 degrees of freedom of 3D motion.

The authors extend the original 6 bases to **8 focal-length-free bases**:

$$\mathcal{B}_{\text{motion}} = \{\Delta_{\mathbf{T}x}, \Delta_{\mathbf{T}y}, \Delta_{\mathbf{T}z}, \Delta_{\mathbf{R}^1 x}, \Delta_{\mathbf{R}^2 x}, \Delta_{\mathbf{R}^1 y}, \Delta_{\mathbf{R}^2 y}, \Delta_{\mathbf{R} z}\}$$

where the translation bases depend on depth $\mathbf{D}_0$ and the rotation bases depend only on pixel coordinates. The key technique is to eliminate the dependency on camera focal length by splitting the rotation bases and assuming $f_x = f_y$.

These bases are processed through a **BasesNet** network to extract features, which are concatenated with context features to provide an initial motion prior for iterative refinement.

**Design Motivation**: Although motion bases theory holds strictly only for rigid body motion, the initial estimate it provides substantially reduces the burden on iterative refinement, enabling the model to converge more quickly to the correct optical flow.

#### 3. Context Enhancement

The depth maps $\mathbf{D}_0, \mathbf{D}_1$ can also be fed into the ContextNet together with the images to extract stronger context features:

$$\mathbf{C}, \mathbf{H}^0 = \text{ContexNet}(\mathbf{I}_0 \oplus \mathbf{D}_0 \oplus \mathbf{I}_1 \oplus \mathbf{D}_1)$$

### Loss & Training

The mixed Laplace distribution modeling from SEA-RAFT is retained. The flow update at each iteration is parameterized as a mixture of two Laplace distributions, supervised by minimizing the negative log-likelihood:

$$\mathcal{L}_{\mathcal{F}} = \sum_{j=0}^{\text{iters}} \gamma^{N-j}(-\log \mathcal{F}^j)$$

All models are trained on a **single RTX 3090 GPU** with a batch size of 4–6.

## Key Experimental Results

### Main Results

| Method | Extra Data | Sintel Clean↓ | Sintel Final↓ | KITTI Fl-EPE↓ | KITTI Fl-all↓ |
|------|-----------|--------------|--------------|--------------|--------------|
| RAFT | - | 1.43 | 2.71 | 5.04 | 17.4 |
| FlowFormer | - | 1.01 | 2.40 | 4.09 | 14.7 |
| SEA-RAFT (L) | - | 1.19 | 4.11 | 3.62 | 12.9 |
| FlowSeek (S) | - | 1.04 | 2.43 | 3.36 | 11.5 |
| FlowSeek (L) | - | 1.07 | 2.21 | 3.82 | 12.5 |
| SEA-RAFT (L) | Tartan | 1.23 | 3.37 | 3.73 | 12.7 |
| **FlowSeek (L)** | **Tartan** | **1.03** | **2.18** | **3.31** | **11.2** |

FlowSeek (L) achieves a **35%** relative improvement over SEA-RAFT (L) on Sintel Final and **12%** on KITTI, while using a training batch size only 1/8 of that of its counterpart.

### Ablation Study

| Prior Combination | Φ | D | BaseNet | TartanAir EPE↓ | KITTI Fl-All↓ |
|---------|---|---|---------|---------------|--------------|
| SEA-RAFT (S) baseline | - | - | - | 1.38 | 6.31 |
| Φ only | ✓ | - | - | 1.30 | 5.69 |
| D only | - | ✓ | - | 1.15 | 4.67 |
| **BaseNet only** | - | - | ✓ | **1.04** | **4.23** |
| **Φ + BaseNet (best)** | ✓ | - | ✓ | **1.03** | **4.16** |
| All three | ✓ | ✓ | ✓ | 1.03 | 4.19 |

Key finding: BaseNet (motion bases) is the most critical component for performance improvement; used alone, it reduces EPE from 1.38 to 1.04.

### Key Findings

1. **Motion bases are central**: BasesNet contributes most to accuracy gains, demonstrating that converting depth priors into motion bases is more effective than directly using depth maps.
2. **Newer depth models yield better results**: Performance improves monotonically from DPT → Depth Anything V1 → V2.
3. **Strong generalizability**: Adding BaseNet to different backbones such as CRAFT and FlowFormer consistently yields significant improvements.
4. **Advantage on LayeredFlow**: On the LayeredFlow dataset containing transparent/reflective surfaces, FlowSeek (L) reduces EPE by more than 2 pixels compared to SEA-RAFT (L).

## Highlights & Insights

- **"Doing more with less" paradigm**: Reusing existing foundation model knowledge avoids training from scratch, achieving 1-GPU competitiveness against 8-GPU methods.
- **Bridging classical and modern**: Motion bases theory from 30 years ago synergizes with the latest depth foundation models across a wide temporal gap.
- **Systematic exploration of design space**: The comprehensive ablation over prior combinations, model scales, depth model choices, and backbone compatibility is highly rigorous.

## Limitations & Future Work

- The approach depends on pretrained depth models, which may themselves require substantial compute to train, effectively "transferring" computational costs elsewhere.
- Motion bases theory holds strictly only for rigid body motion, limiting effectiveness in scenes with severe non-rigid motion.
- Training data remains a bottleneck—better synthetic data could further improve performance.

## Related Work & Insights

- **SEA-RAFT** [Wang et al., 2024]: The paper directly builds upon this work, retaining its iterative refinement and mixed Laplace loss.
- **Depth Anything V2** [Yang et al., 2024]: Used as the source of depth priors with frozen weights.
- **Stereo Anywhere** [Bartolomei et al., 2025]: A similar approach applying depth foundation models to stereo matching.
- Inspiration: Other geometric vision tasks (scene flow, pose estimation) may similarly benefit from the "foundation model prior + classical parameterization" paradigm.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of motion bases and depth foundation models is a distinctive idea.
- Technical Depth: ⭐⭐⭐⭐ — Comprehensive ablations and well-considered design choices.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Systematic evaluation across multiple datasets, backbones, and model scales.
- Practicality: ⭐⭐⭐⭐⭐ — Single-GPU training makes this highly practical.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Flow4Agent: Long-form Video Understanding via Motion Prior from Optical Flow](flow4agent_long-form_video_understanding_via_motion_prior_from_optical_flow.md)
- [\[ICCV 2025\] Unsupervised Joint Learning of Optical Flow and Intensity with Event Cameras](unsupervised_joint_learning_of_optical_flow_and_intensity_with_event_cameras.md)
- [\[ICCV 2025\] PriOr-Flow: Enhancing Primitive Panoramic Optical Flow with Orthogonal View](prior-flow_enhancing_primitive_panoramic_optical_flow_with_orthogonal_view.md)
- [\[ICCV 2025\] MEMFOF: High-Resolution Training for Memory-Efficient Multi-Frame Optical Flow Estimation](memfof_high-resolution_training_for_memory-efficient_multi-frame_optical_flow_es.md)
- [\[NeurIPS 2025\] MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models](../../NeurIPS2025/video_understanding/mimeqa_towards_socially-intelligent_nonverbal_foundation_models.md)

</div>

<!-- RELATED:END -->

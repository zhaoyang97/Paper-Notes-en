---
title: >-
  [Paper Note] LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors
description: >-
  [AAAI 2026][Autonomous Driving][LiDAR reconstruction] This paper proposes LiDAR-GS++, which introduces a **controllable LiDAR diffusion generative model** as a prior to perform **extended reconstruction** of a neural 2DGS field. The method addresses the severe degradation in reconstruction quality under extrapolated viewpoints (e.g., lane-change scenarios) encountered in single-pass LiDAR scanning, achieving state-of-the-art performance on both interpolated and extrapolated views across multiple public benchmarks.
tags:
  - AAAI 2026
  - Autonomous Driving
  - LiDAR reconstruction
  - Gaussian splatting
  - diffusion priors
  - novel view synthesis
  - autonomous driving simulation
date: 2026-05-08
content_hash: 0a79c51eeca46004
---

# LiDAR-GS++: Improving LiDAR Gaussian Reconstruction via Diffusion Priors

**Conference**: AAAI 2026
**arXiv**: [2511.12304](https://arxiv.org/abs/2511.12304)
**Code**: [github](https://github.com/CN-ADLab/LiDAR_GS_plus)
**Area**: Autonomous Driving
**Keywords**: LiDAR reconstruction, Gaussian splatting, diffusion priors, novel view synthesis, autonomous driving simulation

## TL;DR

This paper proposes LiDAR-GS++, which introduces a **controllable LiDAR diffusion generative model** as a prior to perform **extended reconstruction** of a neural 2DGS field. The method addresses the severe degradation in reconstruction quality under extrapolated viewpoints (e.g., lane-change scenarios) encountered in single-pass LiDAR scanning, achieving state-of-the-art performance on both interpolated and extrapolated views across multiple public benchmarks.

## Background & Motivation

### Problem Definition
Reconstruction-based autonomous driving simulators leverage Gaussian Splatting to reconstruct scenes from real driving data for closed-loop simulation testing. However, these methods are constrained by the viewpoint distribution of the original driving trajectory, and their performance degrades significantly when rendering under **extrapolated viewpoints** (e.g., lateral offsets during lane-change maneuvers).

### Core Challenges

**Imbalanced progress between camera and LiDAR simulation**: While camera-based simulators such as FreeSim have addressed extrapolation, LiDAR re-simulation has not yet explicitly tackled extrapolation quality, impeding the development of multi-modal sensor L4 driving agents.

**Fidelity of generated data**: LiDAR data generated via cross-modal prompts (text, maps, bounding boxes) exhibits a notable domain gap and is unsuitable for novel view synthesis.

**Consistency of generated data**: Naively blending generated scans with real scans introduces hallucinations and contradictions in already-converged regions.

### Mechanism
Controllable LiDAR-to-LiDAR generation is identified as the most suitable scene extension strategy — coarse extrapolated renderings serve as conditioning signals to guide a diffusion model in generating geometrically consistent LiDAR scans, which are then selectively integrated via a depth-distortion-aware distillation strategy.

## Method

### Overall Architecture

LiDAR-GS++ operates in three stages:
1. **Initial Reconstruction**: A neural 2DGS field is used to reconstruct the scene from a single-pass driving segment (5,000 iterations).
2. **Controllable LiDAR Generation**: Coarse LiDAR scans are rendered at extrapolated viewpoints and fed into a pretrained diffusion model to generate geometrically consistent additional supervision signals.
3. **Extended Reconstruction**: Generated data is distilled into the GS representation via the DDAD strategy (2,000 iterations).

### Key Designs

#### 1. Neural 2DGS Field

**Core Idea**: Given the inherent range- and direction-dependent attenuation characteristics of LiDAR signals and the geometric fidelity advantages of 2DGS, the paper proposes a neural-network-enhanced 2D Gaussian field for LiDAR scene modeling.

Each 2D Gaussian $\xi$ comprises: center position $\mathbf{x}$, rotation quaternion $\mathbf{R}$, scale $\mathbf{S}$, intensity $\rho$, ray-drop probability $r$, opacity $\alpha$, and a 32-dimensional learnable feature token $\mathbf{v}_\xi$.

The key innovation is the introduction of four lightweight MLPs that take the feature token, local ray direction $\mathbf{d}'$, and flight distance $d$ as inputs to predict per-Gaussian attributes, enabling the network to capture direction- and distance-dependent properties.

**Range View Rendering**: 3D point clouds are projected onto a range view (three channels: intensity, depth, ray-drop). Rendering employs 2DGS ray–splat intersection computation and volume rendering integration:

$$[\bar{\rho}, \bar{d}, \bar{r}] \leftarrow \sum_{i \in \mathbf{N}} [\rho_{\xi_i}, d_{\xi_i}, r_{\xi_i}] \alpha_{\xi_i} G_{\xi_i} \prod_{j=1}^{i-1}(1-\alpha_{\xi_j} G_{\xi_j})$$

**Design Motivation**: Compared to the 3DGS used in LiDAR-GS, 2DGS provides superior geometric fidelity (flatter surfaces), and neural network conditioning captures the physical characteristics of LiDAR signals.

#### 2. Controllable LiDAR Generation Model

**Core Idea**: Rather than using sparse cross-modal prompts such as semantic maps or bounding boxes, the paper proposes controllable LiDAR-to-LiDAR generation, where coarse extrapolated renderings condition the diffusion model to produce high-quality LiDAR scans.

**Training Pair Construction**: Since single-pass segments lack ground-truth extrapolated views, training pairs are constructed by applying perturbations with variance $\sigma=0.2$ to the inputs of the neural 2DGS field and randomly dropping Gaussian primitives at rate $\tau=0.1$ to simulate low-quality renderings, yielding approximately 27k training pairs.

**Diffusion Model Architecture**: Built upon LiDM (Latent LiDAR Diffusion Model), a pretrained VAE encoder encodes both input and condition into latent space. Fourier positional encodings and wavelet-based up/downsampling modules are added to enhance generation detail. The training objective is:

$$\mathcal{L}_{diff} = \mathbb{E}_{z_0^L, \epsilon, c^{\bar{L}}, t} [\|\epsilon - \epsilon_\theta(z_t^L, t, c^{\bar{L}})\|_2^2]$$

**Generation Pipeline**: The trained generative model is frozen; coarse LiDAR scans rendered at extrapolated viewpoints are used as conditioning signals to generate geometrically consistent additional supervision.

#### 3. Depth Distortion-Aware Distillation Strategy (DDAD)

**Core Idea**: Fully injecting generated data introduces hallucinations and caps performance; selective correction targeting only under-fitted regions is necessary.

**Distortion Region Identification**: Under-fitted regions are identified by comparing the median depth $d_m$ (depth at which transmittance $T$ is closest to 0.5, recorded during rendering) with the rendered depth $\bar{d}$:

$$M = \{|\bar{d}_m - \bar{d}| > \delta\}$$

where $\delta = \text{median}\{\max(s_u, s_v)\}$ is the median of the longest-axis scale coefficients across all Gaussians. A large discrepancy indicates that the Gaussian attributes in that region have not yet converged.

**Selective Loss**: During extended reconstruction, generated and real scans are mixed at a 1:1 ratio, but loss is computed only over distortion regions: $\mathcal{L}_e = M \cdot \mathcal{L}$.

**Design Motivation**: Ablation experiments (Table 2) confirm that injecting generated data without DDAD degrades performance in already-converged regions; DDAD mitigates this by precisely identifying under-fitted regions.

### Loss & Training

The total loss for the initial reconstruction stage is:
$$\mathcal{L} = \mathcal{L}_d + \mathcal{L}_\rho + \mathcal{L}_r + \mathcal{L}_S$$

where $\mathcal{L}_d$ is the L1 depth loss, $\mathcal{L}_\rho = (1-\lambda_\rho) \cdot \mathcal{L}_1 + \lambda_\rho \cdot \mathcal{L}_\text{D-SSIM}$ ($\lambda_\rho=0.2$), $\mathcal{L}_r$ is the L2 ray-drop loss, and $\mathcal{L}_S$ is a scale regularization term.

Training configuration: 7,000 iterations on an RTX 3090 (5,000 for initial reconstruction + 2,000 for extended reconstruction), 500K initial GS anchors, Adam optimizer. The diffusion model is trained once on Waymo + Para-Lane (not per-scene) on 8×A100 for 50K iterations.

## Key Experimental Results

### Main Results

| Method | Para-Lane Extrap. CD↓ | F-score↑ | PSNR↑ | Waymo Extrap. FRID↓ | FPVD↓ | Interp. CD↓ | Train (min) | Infer. (fps) |
|---|---|---|---|---|---|---|---|---|
| LiDAR4D | 1.518 | 0.785 | 29.464 | 48.503 | 52.651 | 0.112 | 426 | 1.7 |
| LiDAR-RT | 0.482 | 0.806 | 30.430 | 41.330 | 57.551 | 0.159 | 213 | 20.7 |
| GS-LiDAR | 0.305 | 0.843 | 29.279 | 31.967 | 78.84 | 0.086 | 129 | 10.8 |
| LiDAR-GS | 0.270 | 0.865 | 30.742 | 39.095 | 34.018 | 0.090 | 18 | 15.8 |
| **LiDAR-GS++** | **0.102** | **0.923** | **31.843** | **11.669** | **15.134** | **0.079** | 26 | 16.2 |

Extrapolation performance improves substantially: CD on Para-Lane drops from 0.270 to 0.102 (↓62%); FRID on Waymo drops from 31.967 to 11.669 (↓63%).

### Ablation Study

| Configuration | Extrap. CD↓ | F-score↑ | Extrap. PSNR↑ | Interp. CD↓ | Note |
|---|---|---|---|---|---|
| w/o NGF (vanilla 2DGS) | 0.417 | 0.825 | 29.878 | 0.095 | Neural 2DGS field is foundational for high-quality reconstruction |
| w/o Diff (no diffusion prior) | 0.264 | 0.869 | 30.777 | 0.079 | Diffusion prior substantially improves extrapolation quality |
| w/o DDAD (full injection) | 0.163 | 0.905 | 30.701 | 0.085 | Without DDAD, interpolation performance also degrades |
| **Full LiDAR-GS++** | **0.102** | **0.923** | **31.843** | **0.079** | All three components work synergistically |

Generalizability of the diffusion prior is verified by integrating the diffusion prior + DDAD into GS-LiDAR, reducing extrapolation CD from 0.305 to 0.116.

### Key Findings

1. **Neural 2DGS outperforms both 3DGS and vanilla 2DGS**: Accounting for view/distance dependence alongside 2D geometric advantages reduces extrapolation CD from 0.417 to 0.102.
2. **LiDAR-to-LiDAR conditional generation outperforms cross-modal conditioning**: A generative model conditioned on semantic maps + bounding boxes achieves FRID of 46.74, whereas the proposed rendering-conditioned approach achieves 28.39.
3. **DDAD is critical**: It prevents generated data from negatively affecting converged regions while selectively correcting under-fitted areas.
4. **Computational efficiency**: Training requires only 26 minutes with inference at 16.2 fps, balancing quality and real-time capability.

## Highlights & Insights

1. The **coarse-to-fine progressive extrapolation strategy** is worth adopting: coarsely render extrapolated views from the existing reconstruction → complete with a generative model → selectively distill back into the reconstruction. This iterative approach generalizes to other scene extension tasks.
2. The **distortion region detection mechanism is elegant**: the discrepancy between median depth and rendered depth serves as an under-fitting indicator without requiring additional annotations, constituting a self-supervised quality assessment mechanism.
3. **Same-modality conditional generation** is more suitable than cross-modal generation for reconstruction tasks, avoiding geometric inconsistencies caused by semantic sparsity.

## Limitations & Future Work

1. Non-rigid dynamic objects (e.g., pedestrians) are not handled; NSG is used for instance decomposition and separate reconstruction.
2. The generative model does not account for temporal consistency, potentially causing flickering across consecutive frames.
3. Future directions include: leveraging more advanced video generative models to improve temporal consistency, and handling non-rigid motion.

## Related Work & Insights

- **LiDAR-GS** (the foundation of this work): the first differentiable laser-beam splatting LiDAR reconstruction using 3DGS.
- **LiDM**: Latent LiDAR Diffusion Model, serving as the backbone of the proposed generative model.
- **FreeSim / ReconDreamer**: camera-based simulators addressing extrapolation with analogous motivation but different modalities.
- Insight: diffusion priors can serve as "knowledge infill" mechanisms applicable to various incomplete reconstruction tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ — First to introduce diffusion priors into LiDAR GS reconstruction; DDAD strategy is cleverly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Dual-dataset validation, generalizability experiments for diffusion priors, comprehensive ablation study.
- Writing Quality: ⭐⭐⭐⭐ — Clear problem formulation and well-motivated methodology.
- Value: ⭐⭐⭐⭐ — Addresses an important practical problem in autonomous driving simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] LiNeXt: Revisiting LiDAR Completion with Efficient Non-Diffusion Architectures](linext_revisiting_lidar_completion_with_efficient_non-diffusion_architectures.md)
- [\[CVPR 2026\] LR-SGS: Robust LiDAR-Reflectance-Guided Salient Gaussian Splatting for Self-Driving Scene Reconstruction](../../CVPR2026/autonomous_driving/lr-sgs_robust_lidar-reflectance-guided_salient_gaussian_splatting_for_self-drivi.md)
- [\[ICCV 2025\] GS-Occ3D: Scaling Vision-only Occupancy Reconstruction with Gaussian Splatting](../../ICCV2025/autonomous_driving/gs-occ3d_scaling_vision-only_occupancy_reconstruction_with_gaussian_splatting.md)
- [\[ICCV 2025\] GS-LIVM: Real-Time Photo-Realistic LiDAR-Inertial-Visual Mapping with Gaussian Splatting](../../ICCV2025/autonomous_driving/gs-livm_real-time_photo-realistic_lidar-inertial-visual_mapping_with_gaussian_sp.md)
- [\[AAAI 2026\] LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences](lidarcrafter_dynamic_4d_world_modeling_from_lidar_sequences.md)

</div>

<!-- RELATED:END -->

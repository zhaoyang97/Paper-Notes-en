---
title: >-
  [Paper Note] Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis
description: >-
  [ECCV 2024][3D Vision][Dynamic Novel View Synthesis] Proposes Generative Camera Dolly (GCD), which fine-tunes the Stable Video Diffusion model to generate synchronized dynamic novel-view videos from any viewpoint using a monocular video, supporting extreme camera transitions up to 180° without requiring depth input or explicit 3D modeling.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "Dynamic Novel View Synthesis"
  - "Video Diffusion Models"
  - "Stable Video Diffusion"
  - "Camera Control"
  - "Monocular Video"
date: 2026-05-08
content_hash: 4075372bbbede203
---

# Generative Camera Dolly: Extreme Monocular Dynamic Novel View Synthesis

**Conference**: ECCV 2024  
**arXiv**: [2405.14868](https://arxiv.org/abs/2405.14868)  
**Code**: [https://gcd.cs.columbia.edu](https://gcd.cs.columbia.edu) (Yes, project page)  
**Area**: 3D Vision  
**Keywords**: Dynamic Novel View Synthesis, Video Diffusion Models, Stable Video Diffusion, Camera Control, Monocular Video

## TL;DR

Proposes Generative Camera Dolly (GCD), which fine-tunes the Stable Video Diffusion model to generate synchronized dynamic novel-view videos from any viewpoint using a monocular video, supporting extreme camera transitions up to 180° without requiring depth input or explicit 3D modeling.

## Background & Motivation

**Background**: Although there is significant work on Dynamic Novel View Synthesis (DVS), most methods rely on synchronized multi-view video inputs (e.g., HexPlane, 4D-GS) or only support narrow viewpoint changes (e.g., DynIBaR is limited to a few degrees), restricting practical applications.

**Limitations of Prior Work**: (a) High cost of multi-view synchronized video acquisition, severely limiting in-the-wild usage; (b) per-scene optimization-based methods (such as the NeRF series) cannot generalize across scenes and struggle to reason about occluded areas; (c) existing monocular approaches can only handle tiny viewpoint changes.

**Key Challenge**: Monocular dynamic novel view synthesis is extremely under-constrained—inferring one viewpoint from another requires strong prior knowledge, whereas existing methods either lack such priors (per-scene optimization) or fail to achieve precise camera control (video generation models).

**Goal**: Given a monocular video of any scene, generate a synchronized dynamic novel-view video seen from any given target camera pose.

**Key Insight**: Leverage large-scale video diffusion models (such as SVD), which contain rich 3D geometric and dynamic scene priors, and "teach" the model to perform video-to-video translation with precise camera control by fine-tuning on synthetic multi-view data.

**Core Idea**: Formulate DVS as a conditional video generation problem, and fine-tune SVD on synthetic data to learn end-to-end video-to-video translation controlled by camera poses.

## Method

### Overall Architecture

GCD is an end-to-end video-to-video translation pipeline. Given a source viewpoint RGB video $\boldsymbol{x} \in \mathbb{R}^{T \times H \times W \times 3}$ and relative camera extrinsic parameters $\Delta\mathcal{E} = \{\mathcal{E}_{src,t}^{-1} \cdot \mathcal{E}_{dst,t}\}_{t=0}^{T-1}$, the output target viewpoint video is $\boldsymbol{y}$:

$$\boldsymbol{y} = f(\boldsymbol{x}, \Delta\mathcal{E})$$

It modifies and fine-tunes the image-to-video architecture of Stable Video Diffusion (SVD).

### Key Designs

1. **Camera Viewpoint Control**: The relative extrinsic matrix $\Delta\mathcal{E}_t \in \text{SE}(3)$ is decomposed into rotation $R_t \in \text{SO}(3)}$ and translation $T_t \in \mathbb{R}^3$. The flattened information is projected into an embedding via an MLP $m$ and merged with the micro-conditioning mechanism of SVD—added to the feature vectors of each convolutional layer in the network (similar to SV3D). The new camera embedder $m$ is randomly initialized, while the remaining weights are loaded from the pre-trained SVD checkpoint, maximally preserving the video priors learned by SVD.

2. **Video Conditioning**: The original SVD architecture processes two signaling paths: CLIP embedding for cross-attention, and VAE encoding followed by channel concatenation. GCD retains this mechanism but makes a key modification—extending the conditioning from only using the first frame $\boldsymbol{x}_0$ to using the **entire input video** $\boldsymbol{x}$ to enable the model to observe the complete scene dynamics. Specifically, at each timestep $t$, the synchronized input frames are concatenated to the output samples. The U-Net receives an input of $2D \times T \times \frac{H}{F} \times \frac{W}{F}$ and yields an output of $D \times T \times \frac{H}{F} \times \frac{W}{F}$. During inference, classifier-free guidance is used:

$$\hat{\boldsymbol{y}}_{u-1} = w\epsilon(\hat{\boldsymbol{y}}_u \| \boldsymbol{x}, \Delta\mathcal{E}) - (w-1)\epsilon(\hat{\boldsymbol{y}}_u)$$

where $w \in [1, \infty)$ is the guidance scale. SVD's factorized 3D U-Net establishes spatio-temporal attention between input and output frames across spatial and temporal blocks, and each frame has a corresponding CLIP embedding $c(\boldsymbol{x}_t)$ acting as cross-attention conditioning.

3. **Camera Trajectory Choice**: Contrast two trajectory modes through ablation studies:

$$\mathcal{E}_{dst,t} = \begin{cases} g(\alpha \mathcal{P}_{dst} + (1-\alpha)\mathcal{P}_{src}), & \text{gradual} \\ g(\mathcal{P}_{dst}), & \text{direct} \end{cases}$$

where $\alpha = \frac{t}{T-1}$. The study finds: **(a)** gradual interpolation outperforms direct jumps (+1.17 dB PSNR on average); **(b)** a training range of max 90° is better than max 180° (+0.55 dB); **(c)** fine-tuning from SVD checkpoints outperforms training from scratch (+1.34 dB). The final configuration adopts **gradual, max 90°, finetuned**.

### Dataset Construction

- **Kubric-4D**: 3000 scenes are generated using the Kubric simulator, with 7-22 objects per scene, 16 fixed virtual cameras, and 60 frames @ 24FPS. The training data is rendered from arbitrary viewpoints via a backprojection-reprojection data augmentation strategy.
- **ParallelDomain-4D**: High-fidelity driving scenes with 1533 scenes @ 10FPS, 19 virtual cameras, including multi-modal annotations such as RGB, semantic labels, and depth.

### Loss & Training

The SVD variant predicts $T=14$ frames at a resolution of $384 \times 256$. It is trained on Kubric-4D using 7×A100 GPUs for 10k iterations (batch size 56, taking approximately 3 days). v-parameterization preconditioning is adopted, with 25-step EDM sampler inference, and the classifier-free guidance range is set to $[1, 1.5]$.

## Key Experimental Results

### Main Results

**Kubric-4D Benchmark Comparison (averaging 13 frames, monocular RGB input)**:

| Method | PSNR(all)↑ | SSIM(all)↑ | LPIPS(all)↓ | PSNR(occ.)↑ | SSIM(occ.)↑ |
|------|------------|------------|-------------|-------------|-------------|
| HexPlane | 15.38 | 0.428 | 0.568 | 14.71 | 0.428 |
| 4D-GS | 14.92 | 0.388 | 0.584 | 14.55 | 0.392 |
| DynIBaR | 12.86 | 0.356 | 0.646 | 12.78 | 0.358 |
| Vanilla SVD | 13.85 | 0.312 | 0.556 | 13.66 | 0.326 |
| ZeroNVS | 15.68 | 0.396 | 0.508 | 14.18 | 0.368 |
| **GCD (Ours)** | **20.30** | **0.587** | **0.408** | **18.60** | **0.527** |

**ParallelDomain-4D (RGB)**: GCD PSNR 25.04 vs ZeroNVS 18.88, leading by a large margin.

### Ablation Study

**Kubric-4D Ablation (evaluated on the last frame)**:

| Variant | PSNR(all)↑ | SSIM(all)↑ | LPIPS(all)↓ |
|------|------------|------------|-------------|
| direct, max 90°, scratch | 15.96 | 0.450 | 0.575 |
| gradual, max 90°, scratch | 16.92 | 0.486 | 0.542 |
| direct, max 90°, finetuned | 17.23 | 0.494 | 0.507 |
| **gradual, max 90°, finetuned** | **17.88** | **0.521** | **0.486** |
| gradual, max 180°, finetuned | 17.81 | 0.521 | 0.488 |

### Key Findings

- Per-scene optimization methods (HexPlane/4D-GS/DynIBaR) fail catastrophically under single-view input.
- Gradual trajectories outperform direct trajectories (+1.17 dB), which stems from better alignment with the SVD pre-training distribution.
- Although trained only on synthetic data, it exhibits zero-shot generalization capabilities in real-world scenarios such as driving, robotic manipulation, and indoor videos.
- The model possesses the ability of "object permanence" reasoning—correctly predicting the position and appearance of occluded objects after occlusion occurs.

## Highlights & Insights

- **Paradigm Innovation**: It shifts DVS from the traditional per-scene optimization paradigm to a conditional video generation paradigm, achieving extreme viewpoint (up to 180°) novel view synthesis from monocular video for the first time.
- **Leveraging Large Model Priors**: Cleverly leverages SVD's video priors to achieve precise 6-DoF camera control through lightweight fine-tuning.
- **Multi-modal Capability**: Not only enables RGB novel view synthesis but also supports semantic segmentation view synthesis (ParallelDomain mIoU 43.4%), proving the generalizability of the method.
- **Inference Efficiency**: Generating a video takes only about 10 seconds, which is several orders of magnitude faster than per-scene optimization methods.

## Limitations & Future Work

- Primarily trained on synthetic data, limiting its generalization capabilities to out-of-distribution samples (such as videos containing moving humans).
- The output resolution is limited ($384 \times 256$), making it difficult to meet high-resolution demands.
- The correspondence between input and output objects is not always clear, and rigid bodies sometimes suffer from incorrect distortion.
- Lack of explicit 3D geometry modeling may lead to outputs with poor geometric consistency.
- Future improvements can be achieved through better pre-trained models, larger-scale datasets, and more computational resources.

## Related Work & Insights

- **Stable Video Diffusion (SVD)**: The core backbone network, providing powerful video generation priors.
- **SV3D**: Concurrent work that utilizes a similar micro-conditioning concept for 3D generation.
- **DynIBaR**: A volume rendering-based monocular DVS approach, but limited to narrow viewpoint changes.
- **Sora**: Demonstrates the potential of video models as world simulators, inspiring the use of video diffusion models for 3D/4D understanding.
- **Insight**: Video diffusion models are not just generative tools, but also powerful engines for 3D/4D scene understanding.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ A completely new paradigm—using video diffusion models for dynamic view synthesis, pioneering work
- Experimental Thoroughness: ⭐⭐⭐⭐ Two synthetic datasets + real-world generalization + comprehensive ablation studies, but lacks more quantitative evaluations on real-world data
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous structure, with a deep and thorough ablation analysis of trajectory selection
- Value: ⭐⭐⭐⭐⭐ Great potential for applications in robotics, autonomous driving, and VR/AR, opening up new research directions

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dynamic Novel View Synthesis in High Dynamic Range](../../ICLR2026/3d_vision/dynamic_novel_view_synthesis_in_high_dynamic_range.md)
- [\[ECCV 2024\] CoherentGS: Sparse Novel View Synthesis with Coherent 3D Gaussians](coherentgs_sparse_novel_view_synthesis_with_coherent_3d_gaussians.md)
- [\[NeurIPS 2025\] Reconstruct, Inpaint, Test-Time Finetune: Dynamic Novel-View Synthesis from Monocular Videos](../../NeurIPS2025/3d_vision/reconstruct_inpaint_test-time_finetune_dynamic_novel-view_synthesis_from_monocul.md)
- [\[ECCV 2024\] A Compact Dynamic 3D Gaussian Representation for Real-Time Dynamic View Synthesis](a_compact_dynamic_3d_gaussian_representation_for_realtime_dy.md)
- [\[ECCV 2024\] Thermal3D-GS: Physics-induced 3D Gaussians for Thermal Infrared Novel-view Synthesis](thermal3d-gs_physics-induced_3d_gaussians_for_thermal_infrared_novel-view_synthe.md)

</div>

<!-- RELATED:END -->

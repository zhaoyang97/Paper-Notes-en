---
title: >-
  [Paper Note] FE2E: From Editor to Dense Geometry Estimator
description: >-
  [CVPR 2026][3D Vision][Depth Estimation] This paper systematically analyzes the fine-tuning behavior of image editing models versus generative models for dense geometry estimation. It finds that editing models possess inherent structural prior advantages, and proposes the FE2E framework — the first to adapt a DiT-based image editing model as a joint depth and normal estimator — achieving substantial zero-shot improvements over existing SOTA (35% AbsRel reduction on ETH3D).
tags:
  - CVPR 2026
  - 3D Vision
  - Depth Estimation
  - Normal Estimation
  - Image Editing Models
  - Diffusion Models
  - DiT
date: 2026-05-08
content_hash: 135faf8c5c29aaae
---

# FE2E: From Editor to Dense Geometry Estimator

**Conference**: CVPR 2026
**arXiv**: [2509.04338](https://arxiv.org/abs/2509.04338)
**Code**: N/A
**Area**: 3D Vision
**Keywords**: Depth Estimation, Normal Estimation, Image Editing Models, Diffusion Models, DiT

## TL;DR
This paper systematically analyzes the fine-tuning behavior of image editing models versus generative models for dense geometry estimation. It finds that editing models possess inherent structural prior advantages, and proposes the FE2E framework — the first to adapt a DiT-based image editing model as a joint depth and normal estimator — achieving substantial zero-shot improvements over existing SOTA (35% AbsRel reduction on ETH3D).

## Background & Motivation

1. **Background**: Monocular dense geometry estimation (depth and normals) is a core task in 3D vision. Recent methods such as Marigold leverage pretrained generative priors from Stable Diffusion to achieve impressive zero-shot predictions with limited data. Another line of work, represented by the DepthAnything series, follows a data-driven approach, training general-purpose estimators on large-scale data (62.6M images).

2. **Limitations of Prior Work**: Text-to-image generative models are designed to synthesize images from text; their internal features do not naturally align with geometric structure. Fine-tuning such models requires restructuring features from scratch, leading to unstable training and performance bottlenecks. Data-driven methods, while effective, are limited in generalizability due to their reliance on large-scale annotated data.

3. **Key Challenge**: Dense geometry estimation is inherently an image-to-image task, yet existing methods fine-tune T2I generative models — a fundamental mismatch between task paradigm and model paradigm.

4. **Goal**: (1) Verify whether image editing models are better suited than generative models for dense geometry estimation; (2) Address training objective, numerical precision, and computational efficiency issues encountered when adapting editing models into deterministic predictors.

5. **Key Insight**: The authors are motivated by the intuition that image editing models inherently understand the structure of input images while retaining the capabilities of generative models, making them more suitable for dense prediction than T2I models. This hypothesis is validated through systematic analysis of feature evolution and training dynamics.

6. **Core Idea**: Replace generative models with image editing models as the backbone for dense geometry estimation, and adapt the editor into an estimator via three technical contributions: consistent velocity flow matching, logarithmic quantization, and zero-cost joint estimation.

## Method

### Overall Architecture
FE2E is built upon Step1X-Edit, a state-of-the-art DiT-based image editing model. The input is an RGB image, and the output consists of the corresponding depth map and normal map. The pipeline is as follows: a VAE encoder maps the input image and geometry annotations into latent space; the DiT learns a constant-velocity straight-line path from a fixed starting point to the target latent representation; and a VAE decoder converts predictions back to pixel space. The key innovation lies in exploiting the parallel output regions of the DiT editing model to simultaneously predict depth and normals without additional computational overhead.

### Key Designs

1. **Systematic Analysis: Editing Models vs. Generative Models**

   - **Function**: Validate the superiority of editing models as backbones for dense geometry estimation.
   - **Mechanism**: Step1X-Edit (editor) and FLUX (generator) are fine-tuned under identical settings. Feature evolution across different DiT layers (Block 1/20/35) and training loss curves are compared via visualization. The editing model's initial features are already aligned with image geometric structure, and fine-tuning merely refines and focuses existing capabilities. In contrast, the generative model's features must be restructured from a chaotic state, leading to oscillatory training and a loss plateau of approximately 0.08.
   - **Design Motivation**: Provide a theoretical foundation for the "From Editor to Estimator" paradigm and explain why editing models achieve better results with less data.

2. **Consistent Velocity Flow Matching**

   - **Function**: Transform the stochastic flow matching objective of editing models into a training objective suitable for deterministic prediction.
   - **Mechanism**: In standard flow matching, the model learns the instantaneous velocity field over all possible paths, resulting in a nonlinear global velocity field and curved integration trajectories that accumulate discretization errors. FE2E introduces two simplifications: (1) the velocity direction and magnitude are enforced to remain constant along the entire path, making the training objective $\mathcal{L} = \mathbb{E}[\|\mathbf{v} - f_\theta(\mathbf{z}^x)\|^2]$ completely independent of timestep $t$; (2) the stochastic Gaussian starting point is fixed to the zero vector $\mathbf{z}_0^y = \mathbf{0}$, eliminating sampling randomness. At inference, the prediction is obtained directly as $\mathbf{z}_1^y = f_\theta(\mathbf{z}^x)$ in a single step, without iterative solving.
   - **Design Motivation**: Geometry estimation is a deterministic task with a unique ground truth, requiring no generative diversity. Constant-velocity straight-line paths fundamentally eliminate discretization errors from curved trajectories while substantially accelerating inference.

3. **Logarithmic Annotation Quantization**

   - **Function**: Resolve the contradiction between BF16-precision training and the high numerical precision required for depth estimation.
   - **Mechanism**: Modern editing models are trained in BF16 precision, which is sufficient for RGB output (1/256 precision) but introduces severe quantization errors when applied to depth annotations. For example, on the Virtual KITTI dataset (depth range 0–80 m), uniform quantization to $[-1, 1]$ yields an AbsRel error of 1.6 at 0.1 m. Inverse-depth (disparity) quantization achieves high near-range precision but completely fails at far distances (39 m and 78 m map to the same value). FE2E adopts logarithmic quantization $D_{log} = \ln(D_{GT} + 1e{-6})$, followed by percentile normalization $\mathbf{y}_D = \langle((D_{log} - D_{log,2})/(D_{log,98} - D_{log,2}) - 0.5) \times 2\rangle$, achieving uniformly low error at both near and far distances (AbsRel ≈ 0.013).
   - **Design Motivation**: Enable high-precision depth estimation within a BF16-only model, avoiding the increased cost and degraded prior inheritance caused by forcing FP32 in prior methods.

### Loss & Training
- The primary loss is the consistent velocity flow matching loss computed in latent space.
- For joint estimation, the left output region supervises depth and the right supervises normals: $\mathcal{L}_{fm} = \mathbb{E}(\|\mathbf{v}_D - p_l\|^2 + \|\mathbf{v}_N - p_r\|^2)$
- An auxiliary dispersion loss encourages latent features of different samples to spread apart in latent space.
- Training uses LoRA (rank=64, α=32) with all parameters outside the DiT frozen; AdamW optimizer with learning rate 1e-4 for 30 epochs.
- Training is feasible on a single RTX 4090 (with gradient checkpointing) and completes in approximately 1.5 days on H20 GPUs.

## Key Experimental Results

### Main Results

| Dataset | Metric (AbsRel↓) | FE2E | Lotus-D (Prev. SOTA) | DepthAnything V2 | Gain |
|---------|------------------|------|----------------------|------------------|------|
| NYUv2 | AbsRel | **4.1** | 5.1 | 4.5 | 19.6% vs Lotus-D |
| KITTI | AbsRel | **6.6** | 8.1 | 7.4 | 18.5% |
| ETH3D | AbsRel | **3.8** | 6.1 | 13.1 | **37.7%** |
| ScanNet | AbsRel | **4.4** | 5.5 | — | 20.0% |
| DIODE | AbsRel | **22.8** | 22.8 | 26.5 | On par |

Training data comprises only 71K images, far fewer than DepthAnything V2's 62.6M.

### Ablation Study

| ID | Configuration | KITTI AbsRel | ETH3D AbsRel | Notes |
|----|--------------|--------------|--------------|-------|
| 2 | DirectAdapt (Step1X-Edit) | 9.5 | 5.6 | Baseline |
| 3 | + Consistent Velocity | 8.8 | 5.0 | CV contribution: −7%/−10% |
| 4 | + Fixed Start | 8.6 | 4.8 | Further improvement |
| 6 | + Logarithmic Quant | **6.8** | **3.9** | Log quant contribution: −19%/−13% |
| 7 | FLUX + proposed methods | 7.1 | 4.5 | Editing model still outperforms generator |
| 8 | FE2E full (+ joint estimation) | **6.6** | **3.8** | Joint estimation yields additional gains |
| 9 | FLUX-Kontext + full method | 6.7 | 3.6 | Validates generalizability to other editors |

### Key Findings
- Logarithmic quantization is the single largest contributor, accounting for a 19% error reduction on KITTI alone.
- The performance gap between editing and generative models is consistent across all settings (ID2 vs. ID1, ID6 vs. ID7).
- Joint estimation incurs zero additional cost and yields visible quality improvements in challenging scenes (planar butterfly structures, distant buildings).
- The method generalizes to other editors such as FLUX-Kontext (ID9), with even better performance.

## Highlights & Insights
- **Systematic justification of the "From Editor to Estimator" paradigm**: Beyond simply substituting the backbone, the paper provides a mechanistic explanation for why editing models are more suitable for dense estimation through feature evolution visualization and training dynamics analysis. This finding may inspire broader adoption of editing model priors for image-to-image tasks.
- **Elegant zero-cost joint estimation design**: The input concatenation mechanism of DiT editing models causes 50% of the output to be discarded; FE2E repurposes this "wasted" output for a second task, achieving multi-task learning at zero additional computational cost.
- **Logarithmic quantization for BF16 precision**: This solution elegantly resolves the practical engineering challenge of performing high-precision depth estimation within a BF16-only model, and the approach is broadly applicable.

## Limitations & Future Work
- Training data is limited to synthetic sources (Hypersim + Virtual KITTI), with no real-world data used.
- The improvement in normal estimation is less pronounced compared to depth estimation.
- The method only supports affine-invariant depth prediction and has not been extended to metric depth.
- While logarithmic quantization provides balanced precision, it is not optimal at extreme near or far distances; piecewise adaptive quantization could be explored.

## Related Work & Insights
- **vs. Marigold/Lotus**: These methods are based on Stable Diffusion (T2I), whereas FE2E is based on Step1X-Edit (editing model). FE2E substantially outperforms them under comparable data volumes, demonstrating that backbone selection is more critical than algorithmic improvements.
- **vs. DepthAnything V2**: The latter uses 100× more training data, yet FE2E still significantly outperforms it on datasets such as ETH3D, demonstrating that editing priors can compensate for data scarcity.
- **vs. Diffusion-E2E-FT**: E2E-FT proposes end-to-end fine-tuning of denoising architectures; FE2E advances this direction further with an editing model combined with consistent velocity flow matching.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic justification and implementation of the "editing model → estimator" paradigm shift.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five depth benchmarks, four normal benchmarks, and detailed ablations form a complete evidence chain.
- **Writing Quality**: ⭐⭐⭐⭐ Outstanding visualizations and in-depth analysis.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm and theoretical foundation for backbone selection in dense prediction tasks.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Unblur-SLAM: Dense Neural SLAM for Blurry Inputs](unblur-slam_dense_neural_slam_for_blurry_inputs.md)
- [\[CVPR 2026\] Fast3Dcache: Training-free 3D Geometry Synthesis Acceleration](fast3dcache_training-free_3d_geometry_synthesis_acceleration.md)
- [\[AAAI 2026\] Free-Form Scene Editor: Enabling Multi-Round Object Manipulation like in a 3D Engine](../../AAAI2026/3d_vision/free-form_scene_editor_enabling_multi-round_object_manipulation_like_in_a_3d_eng.md)
- [\[CVPR 2026\] GGPT: Geometry-Grounded Point Transformer](ggpt_geometry_grounded_point_transformer.md)
- [\[CVPR 2026\] LongStream: Long-Sequence Streaming Autoregressive Visual Geometry](longstream_long-sequence_streaming_autoregressive_visual_geometry.md)

<!-- RELATED:END -->

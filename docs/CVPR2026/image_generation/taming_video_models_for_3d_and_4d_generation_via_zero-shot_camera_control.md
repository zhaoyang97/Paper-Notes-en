---
title: >-
  [Paper Note] Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control
description: >-
  [CVPR 2026][Image Generation][Video Diffusion Models] WorldForge proposes a fully training-free inference-time guidance framework that adapts pretrained video diffusion models into precise camera-trajectory-controllable 3D/4D generation tools via three synergistic components—Intra-Step Recursive Refinement (IRR), Flow-Gated Latent Fusion (FLF), and Dual-Path Self-Corrective Guidance (DSG)—simultaneously surpassing both training-based and inference-based baselines in trajectory accuracy and perceptual quality.
tags:
  - CVPR 2026
  - Image Generation
  - Video Diffusion Models
  - 3D Generation
  - 4D Generation
  - Camera Control
  - Training-Free Inference
date: 2026-05-08
content_hash: 2464e74851c5962e
---

# Taming Video Models for 3D and 4D Generation via Zero-Shot Camera Control

**Conference**: CVPR 2026
**arXiv**: [2509.15130](https://arxiv.org/abs/2509.15130)
**Code**: [https://worldforge-agi.github.io](https://worldforge-agi.github.io) (project page)
**Area**: Diffusion Models / 3D Vision
**Keywords**: Video Diffusion Models, 3D Generation, 4D Generation, Camera Control, Training-Free Inference

## TL;DR
WorldForge proposes a fully training-free inference-time guidance framework that adapts pretrained video diffusion models into precise camera-trajectory-controllable 3D/4D generation tools via three synergistic components—Intra-Step Recursive Refinement (IRR), Flow-Gated Latent Fusion (FLF), and Dual-Path Self-Corrective Guidance (DSG)—simultaneously surpassing both training-based and inference-based baselines in trajectory accuracy and perceptual quality.

## Background & Motivation

1. **Background**: Video diffusion models (VDMs) trained on large-scale video data encode rich spatiotemporal priors and can generate highly realistic visual content. Researchers have begun leveraging VDMs for 3D/4D tasks such as novel view synthesis, scene generation, and dynamic re-rendering.

2. **Limitations of Prior Work**:
    - **Poor controllability**: VDMs struggle to precisely follow 6-DoF camera trajectories, leading to spatial inconsistencies.
    - **Scene–camera motion coupling**: Changing viewpoints introduces undesired object deformation and scene instability.
    - **High cost of fine-tuning approaches**: Fine-tuning on motion-conditioned data (e.g., LoRA, ControlNet) is computationally expensive, generalizes poorly, and may corrupt pretrained priors.
    - **Fragility of warp-and-repaint approaches**: Projecting frames along a new camera path and inpainting with a generative model is unreliable because the warped inputs are out-of-distribution (OOD) for the pretrained model, producing visible artifacts.

3. **Key Challenge**: There is a fundamental tension between fine-grained camera controllability and generation quality/generalizability—injecting control signals disrupts model priors, while preserving the priors precludes precise control.

4. **Goal**: Inject precise trajectory control at inference time while fully preserving the VDM's world priors, without any training or fine-tuning.

5. **Key Insight**: The warp-and-repaint pipeline is adopted to obtain trajectory-guided frames, but three carefully designed inference-time intervention mechanisms address its inherent OOD issues. A core observation is that different channels of the VAE latent space encode distinct information (motion vs. appearance), enabling selective injection of control signals.

6. **Core Idea**: Achieve training-free precise camera control by selectively injecting trajectory signals into motion-relevant channels, applying micro-corrections at each denoising step, and leveraging the divergence between guided and unguided paths for self-correction.

## Method

### Overall Architecture
Given a single image or video frame, a visual foundation model (depth estimation) reconstructs a scene point cloud, which is warped along a user-specified camera trajectory to produce a guidance video. The input image is simultaneously converted into a text prompt and a latent representation and fed into an Image-to-Video diffusion model. During denoising, the IRR, FLF, and DSG modules inject trajectory control signals. The output is a high-quality video precisely aligned with the target trajectory.

### Key Designs

1. **Intra-Step Recursive Refinement (IRR)**:

    - **Function**: Embeds a micro-level "predict-correct" loop within each denoising step to continuously inject trajectory information into the generation process.
    - **Mechanism**: At each denoising step $t$, the model first obtains an intermediate estimate $\hat{\mathbf{x}}_0^{(t)}$ via standard denoising. A fusion operator $\mathbf{F}$ then injects the trajectory latent $\mathbf{x}_{traj}$ (warp frames encoded into latent space) into observable regions: $\mathbf{F}(\hat{\mathbf{x}}_0^{(t)}, \mathbf{x}_{traj}) = \mathbf{M} \cdot \mathbf{x}_{traj} + (1-\mathbf{M}) \cdot \hat{\mathbf{x}}_0^{(t)}$, where $\mathbf{M}$ is the valid-pixel mask produced by warping. The fused result is re-noised and fed into the next denoising step. A key distinction is that fusion is performed in the clean space $\hat{\mathbf{x}}_0^{(t)}$ (rather than in the noise space $\mathbf{x}_{t-1}$ as in prior inpainting works), laying the foundation for the subsequent FLF module.
    - **Design Motivation**: Step-wise correction ensures the trajectory signal exerts continuous influence throughout sampling, avoiding signal attenuation caused by one-shot injection.

2. **Flow-Gated Latent Fusion (FLF)**:

    - **Function**: Identifies channels in the latent space that are highly motion-correlated and injects trajectory information only into those channels, protecting appearance channels from interference.
    - **Mechanism**: For each channel $c$ of $\hat{\mathbf{x}}_0^{(t)}$, inter-frame optical flow $\mathcal{F}_{pred}^{(t,c)}$ is computed, and similarly reference flow $\mathcal{F}_{gt}^{(t,c)}$ is computed for $\mathbf{x}_{traj}$. Comparing the two using three optical flow metrics (M-EPE, M-AE, Fl-all) yields a motion similarity score $S^{(t,c)}$. A dynamic threshold $\delta^{(t)} = \mu_S^{(t)} - \lambda^{(t)}\sigma_S^{(t)}$ selects motion-relevant channels. High-scoring channels receive trajectory information; low-scoring channels retain the original prediction. The schedule of $\lambda^{(t)}$ relaxes to tighten progressively—retaining more channels early to establish structure and gradually tightening later to protect fine details.
    - **Design Motivation**: Experiments reveal significant functional specialization across VAE channels—e.g., channel 13 is almost always filtered out (low motion relevance), while channel 8 is never filtered (high motion relevance). Indiscriminately overwriting all channels severely degrades visual quality.

3. **Dual-Path Self-Corrective Guidance (DSG)**:

    - **Function**: Suppresses artifacts introduced by depth errors, occlusions, and misalignment in warped frames, balancing control precision with generation quality.
    - **Mechanism**: Inspired by CFG but addressing a different problem. Two paths are maintained per step: (1) an unguided path $\mathbf{v}_t^{ori}$ (denoised from original $\mathbf{x}_t$), which is high-quality but lacks trajectory control; (2) a guided path $\mathbf{v}_t^{traj}$ (denoised from corrected $\mathbf{x}_t'$), which follows the trajectory but may contain noise. A key finding is that the cosine similarity between the two paths is only 0.3–0.6 (angular difference 50°–70°), far greater than the divergence between conditional and unconditional paths in CFG (near 1). The standard CFG formula therefore cannot be applied directly. The solution extracts the orthogonal component of the guided direction with respect to the unguided direction for correction: $\mathbf{v}_t^{corr} = \mathbf{v}_t^{traj} + \rho \cdot \beta_t(\mathbf{v}_t^{traj} - \alpha_t \cdot \mathbf{v}_t^{ori})$, where $\beta_t = \sqrt{1-\alpha_t^2}$ provides adaptive scaling—strongly correcting when path divergence is large and preserving natural predictions when paths are consistent.
    - **Design Motivation**: Applying the CFG formula directly causes severe artifacts due to the large angular difference between the two paths (confirmed by ablation studies). The orthogonal-component projection elegantly avoids the adverse effects of large-angle divergence.

### Inference Details
WorldForge uses the Wan2.1 I2V-14B model with 50-step UniPC sampling. IRR is applied during the first 20 steps (approximately 35–45%). FLF is disabled in the early phase (first 5 steps) to preserve structural integrity and progressively tightens selection thereafter. The method runs on a single GPU (≥69 GB VRAM) and supports video generation up to 1280×720.

## Key Experimental Results

### Main Results (3D Static Scene Generation)

| Method | Training-Free | FID ↓ | CLIP_sim ↑ | ATE ↓ | RPE-T ↓ | RPE-R ↓ |
|--------|--------------|------|-----------|------|---------|---------|
| ViewCrafter | ✗ | 117.50 | 0.930 | 0.236 | 0.315 | 0.728 |
| TrajectoryCrafter | ✗ | 111.49 | 0.910 | 0.090 | 0.152 | 0.267 |
| NVS-Solver | ✓ | 118.64 | 0.937 | 0.224 | 0.268 | 1.056 |
| See3D | ✗ | 123.26 | 0.941 | 0.091 | 0.089 | 0.250 |
| **WorldForge** | **✓** | **96.08** | **0.948** | **0.077** | **0.086** | **0.221** |

### 4D Dynamic Scene Control

| Method | Training-Free | FVD ↓ | CLIP-V_sim ↑ | ATE ↓ | RPE-T ↓ |
|--------|--------------|------|-------------|------|---------|
| ViewExtrapolator | ✓ | 108.48 | 0.913 | 1.040 | 1.208 |
| TrajectoryCrafter | ✗ | 97.31 | 0.923 | 0.431 | 1.078 |
| **WorldForge** | **✓** | **93.17** | **0.938** | 0.527 | **0.826** |

### Ablation Study

| Configuration | FID ↓ (3D) | CLIP_sim ↑ (3D) | FVD ↓ (4D) | CLIP-V_sim ↑ (4D) |
|---------------|-----------|----------------|-----------|-------------------|
| w/o DSG | 109.43 | 0.943 | 95.69 | 0.937 |
| w/o FLF | 112.69 | 0.945 | 99.79 | 0.932 |
| w/o DSG & FLF | 113.12 | 0.943 | 103.17 | 0.931 |
| DSG (w/ CFG formula) | 120.91 | 0.936 | 109.10 | 0.919 |
| **Full Model** | **96.08** | **0.948** | **93.17** | **0.938** |

### Key Findings
- As a training-free method, WorldForge simultaneously surpasses all training-based baselines in both generation quality and trajectory accuracy (FID 96.08 vs. second-best 111.49).
- All three components are indispensable—removing IRR entirely eliminates trajectory control; removing FLF produces unnatural outputs (FID rises from 96.08 to 112.69); removing DSG introduces warp artifacts (FID rises to 109.43).
- Applying the CFG formula directly for DSG yields the worst performance (FID 120.91), validating the analysis that CFG fails under large angular divergence.
- The method generalizes across models, demonstrating effectiveness on both SVD and LongCat-Video.
- Channel roles are stable: channel 13 is almost always filtered as low motion-relevant, while channel 8 is consistently retained as high motion-relevant; 4D scenes exhibit more diverse channel score distributions than 3D scenes.

## Highlights & Insights
- **Elegance of selective channel injection**: The core insight—that different channels in the VAE latent space encode distinct information—is leveraged by FLF, which uses optical flow as a direct measure of motion-channel relevance (rather than statistical methods such as PCA), achieving motion–appearance disentanglement without gradient optimization.
- **Orthogonal projection resolves large-angle CFG failure**: Standard CFG assumes near-zero angular difference between conditional and unconditional directions; in this setting the two paths diverge by 50–70°, causing direct subtraction to fail catastrophically. The orthogonal-component approach draws inspiration from APG and validates its effectiveness in an entirely different task.
- **Training-free plug-and-play generality**: A single framework supports 12+ applications including 3D scene generation, 4D re-rendering, video editing, video stabilization, and virtual try-on, and is model-agnostic (compatible with Wan2.1, SVD, and LongCat-Video).

## Limitations & Future Work
- The iterative guidance process precludes real-time execution.
- The method depends on depth estimation quality—although VDM priors can partially compensate for depth errors, extreme inaccuracies still affect results.
- Single-channel scan analysis may increase computational overhead when VDM context lengths are long.
- Future directions include distilling the guidance process into fewer steps, integrating stronger generative models, and improving output resolution.

## Related Work & Insights
- **vs. ViewCrafter/See3D (training-based)**: These methods train on warp-and-fine-tuning data, incurring high computational costs and potentially corrupting VDM priors; WorldForge fully preserves the priors and generalizes more broadly.
- **vs. NVS-Solver (training-free)**: Also performs inference-time guidance but lacks motion–appearance disentanglement and self-correction, resulting in poor trajectory accuracy (ATE 0.224 vs. 0.077).
- **vs. ReCamMaster (training-based 4D)**: Uses a T2V model with a specially trained camera control module, but cannot accept the same warp inputs, limiting controllability and generalizability.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The three inference-time guidance components are highly original; the design rationale for FLF and DSG is well-motivated and novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 70+ scenes, 50+ videos, both 3D and 4D tasks, cross-model validation, and comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ The framework description is clear, though the heavy notation across three components creates a non-trivial reading barrier.
- **Value**: ⭐⭐⭐⭐⭐ Training-free plug-and-play generality confers extremely high practical value; the demonstration of 12+ applications is impressive.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[ICCV 2025\] AnyPortal: Zero-Shot Consistent Video Background Replacement](../../ICCV2025/image_generation/anyportal_zero-shot_consistent_video_background_replacement.md)
- [\[CVPR 2026\] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration](v-bridge_bridging_video_generative_priors_to_versatile_few-shot_image_restoratio.md)
- [\[CVPR 2026\] Taming Sampling Perturbations with Variance Expansion Loss for Latent Diffusion Models](taming_sampling_perturbations_with_variance_expansion_loss_for_latent_diffusion_.md)
- [\[CVPR 2026\] Learning by Neighbor-Aware Semantics, Deciding by Open-form Flows: Towards Robust Zero-Shot Skeleton Action Recognition](learning_by_neighbor-aware_semantics_deciding_by_open-form_flows_towards_robust_.md)

<!-- RELATED:END -->

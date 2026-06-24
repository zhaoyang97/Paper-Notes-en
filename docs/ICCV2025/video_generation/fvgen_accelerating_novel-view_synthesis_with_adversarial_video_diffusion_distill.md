---
title: >-
  [Paper Note] FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation
description: >-
  [ICCV 2025][Video Generation][Novel-view synthesis] This paper proposes FVGen, a framework that distills a multi-step video diffusion model (VDM) into a student model requiring only 4 sampling steps. Through GAN-based student initialization and softened reverse KL divergence optimization, FVGen reduces sampling time by over 90% while maintaining or even surpassing the visual quality of the teacher model.
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Novel-view synthesis"
  - "video diffusion distillation"
  - "adversarial training"
  - "softened reverse KL divergence"
  - "few-step sampling"
date: 2026-05-08
content_hash: d408b0838aede025
---

# FVGen: Accelerating Novel-View Synthesis with Adversarial Video Diffusion Distillation

**Conference**: ICCV 2025
**arXiv**: [2508.06392](https://arxiv.org/abs/2508.06392)  
**Code**: [https://wbteng9526.github.io/fvgen/](https://wbteng9526.github.io/fvgen/)  
**Area**: Video Generation
**Keywords**: Novel-view synthesis, video diffusion distillation, adversarial training, softened reverse KL divergence, few-step sampling

## TL;DR

This paper proposes FVGen, a framework that distills a multi-step video diffusion model (VDM) into a student model requiring only 4 sampling steps. Through GAN-based student initialization and softened reverse KL divergence optimization, FVGen reduces sampling time by over 90% while maintaining or even surpassing the visual quality of the teacher model.

## Background & Motivation

**Background**: NeRF- and 3D Gaussian Splatting-based novel-view synthesis achieves high-quality 3D scene reconstruction under dense viewpoint inputs, but suffers from artifacts under sparse-view settings (2–5 input images). Recent work has employed video diffusion models (VDMs) for sparse-view synthesis—by generating continuous view sequences between input views to fill unobserved regions. For example, ViewCrafter leverages DUSt3R to construct initial point clouds for guiding VDM-based novel-view generation.

**Limitations of Prior Work**: The core bottleneck of VDMs is **extremely slow sampling**—ViewCrafter requires approximately 13.2 seconds to generate 16 frames (50 denoising steps), making VDM-based methods impractical for real-time 3D reconstruction, dynamic scene reconstruction, or large-scale scenes requiring numerous novel views. Diffusion distillation methods (e.g., DMD, DMD2) perform well in the image domain but suffer from severe **training instability and mode collapse** when directly applied to multi-view video generation, likely because multi-view video datasets are far smaller than general-purpose video datasets.

**Key Challenge**: VDMs achieve excellent quality in sparse-view synthesis but are prohibitively slow; distillation-based acceleration is a natural solution, yet existing distillation methods are unstable on small-scale multi-view data. The DMD loss is fundamentally a reverse KL divergence minimization, and the mode-seeking property of reverse KL easily causes the student to capture only a subset of the teacher's distribution modes.

**Goal**: Design a stable and efficient video diffusion distillation framework that compresses the 50-step ViewCrafter into a 4-step student model without degrading visual quality.

**Key Insight**: The authors identify two key observations: (1) initializing the student model with a GAN objective is more effective and faster than the conventional ODE solver + regression loss approach; (2) "softening" the reverse KL divergence preserves the mode-seeking property while avoiding mode collapse.

**Core Idea**: First initialize the student model via GAN adversarial training (with the teacher as discriminator), then refine distribution matching using a softened reverse KL divergence—a two-stage training pipeline that enables stable and efficient 4-step video generation.

## Method

### Overall Architecture

FVGen training consists of two stages. The input is a set of sparse-view images and a target camera trajectory; the output is a novel-view video sequence generated in 4 steps. **Stage 1 (GAN Initialization)**: The student model acts as a generator and the teacher model acts as a discriminator; adversarial training provides effective weight initialization. **Stage 2 (Distribution Matching Distillation)**: A softened reverse KL divergence optimizes the distributional gap between student and teacher, while the fake score function is updated dynamically. All three networks—teacher, student, and fake score function—are initialized from ViewCrafter's sparse model.

### Key Designs

1. **GAN-Based Student Initialization**:

    - **Function**: Provides effective weight initialization for the student model so that it can generate samples close to the real data distribution before DMD training begins.
    - **Mechanism**: The student model $G_\theta$ is treated as a generator and the teacher model serves as a discriminator $D$. The student starts from noise $\mathbf{z} \sim \mathcal{N}(0, \mathbf{I})$, generates videos via few-step denoising, injects noise, and feeds the result into the teacher model to extract intermediate-layer features, which are then classified by a 3D convolutional classifier $f$. The loss is a standard GAN adversarial objective: $\mathcal{L}_D = \mathbb{E}[\log f(D(F(\mathbf{x}, t)))] - \mathbb{E}[\log f(D(F(G_\theta(\mathbf{z}), t)))]$. Key design choices: (a) **real samples** (rather than teacher-generated samples) are used as positives, preventing the student from being upper-bounded by teacher quality; (b) intermediate features from the teacher UNet serve as the discriminator backbone, saving computation while leveraging the teacher's semantic representations; (c) classifier $f$ uses 3D convolutions to accommodate the spatiotemporal structure of video data.
    - **Design Motivation**: Prior distillation methods (e.g., CausVid) initialize via ODE-solver-generated noise–sample pairs followed by regression loss training, which is time-consuming and caps student quality at the teacher's level. GAN initialization is faster and directly targets the real data distribution.

2. **Softened Reverse KL Divergence**:

    - **Function**: Replaces the standard reverse KL divergence for distribution matching distillation, resolving training instability and mode collapse.
    - **Mechanism**: Standard DMD loss minimizes $D_{\text{KL}}(p_{\text{fake}} \| p_{\text{real}})$ (reverse KL), which is mode-seeking—the student can minimize the loss by ignoring less prominent modes in the teacher distribution. The softened variant instead minimizes $D_{\text{KL}}(\frac{1}{2}p_{\text{real}} + \frac{1}{2}p_{\text{fake}} \| p_{\text{real}})$, i.e., comparing the uniform mixture of real and fake distributions against the real distribution. This preserves the mode-seeking property of reverse KL, but since the mixture already contains half of the real distribution, the student cannot entirely neglect any real mode. The gradient takes the form $\nabla \mathcal{L} = -\mathbb{E}_t[\frac{1}{r(\mathbf{x},t)}(s_{\text{real}} - s_{\text{fake}}) \frac{dG_\theta(\mathbf{z})}{d\theta}]$, where the density ratio $r = p_{\text{real}} / p_{\text{fake}}$ is estimated directly by the discriminator trained in the GAN stage.
    - **Design Motivation**: In small-data settings such as multi-view video, the mode-collapse tendency of reverse KL is exacerbated—because the distribution modes themselves are not strongly represented, the student easily concentrates on a few modes. The softened KL's penalization mechanism effectively mitigates this issue.

3. **Two-Stage Decoupled Training Strategy**:

    - **Function**: Ensures training stability and accurate density ratio estimation.
    - **Mechanism**: Unlike DMD2, which jointly optimizes GAN training and DMD training end-to-end, FVGen strictly decouples GAN initialization (4,000 iterations) and DMD distillation (5,000 iterations) into two separate stages. The GAN stage trains only the student and the discriminator classification head; the DMD stage continues training the student and the fake score function. This decoupling ensures the two objectives do not interfere: the GAN stage yields accurate density ratio estimates and a good initialization, and the DMD stage leverages the stabilized density ratio for fine-grained distillation.
    - **Design Motivation**: End-to-end training as in DMD2 exhibits significant training variance and instability on multi-view video data; ablation experiments confirm that decoupled training is critical for stability.

### Loss & Training

GAN stage: standard adversarial loss + two-timescale update rule. DMD stage: softened reverse KL divergence gradient + diffusion loss for updating the fake score function. The full pipeline is trained on 8× NVIDIA H100 GPUs for approximately one day, with batch size 4 and resolution 512×320. Training data consists of 20,000 multi-view video–point cloud pairs constructed from DL3DV-10K.

## Key Experimental Results

### Main Results

| Dataset | Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ | Time (s) ↓ |
|---------|--------|--------|--------|---------|-------|------------|
| MipNeRF360 | DepthSplat | 11.23 | 0.213 | 0.715 | 32.45 | 4.2 |
| MipNeRF360 | MVSplat360 | 12.28 | 0.285 | 0.682 | 25.69 | 87.2 |
| MipNeRF360 | ViewCrafter | 16.35 | 0.346 | 0.433 | 16.28 | 66.3 |
| MipNeRF360 | **FVGen** | **16.28** | **0.352** | **0.429** | 17.44 | **5.1** |
| TNT | DepthSplat | 12.43 | 0.263 | 0.677 | 35.88 | 4.3 |
| TNT | MVSplat360 | 14.18 | 0.301 | 0.532 | 25.23 | 87.3 |
| TNT | ViewCrafter | 18.69 | 0.402 | 0.208 | 23.94 | 65.9 |
| TNT | **FVGen** | **18.72** | **0.411** | **0.210** | 23.64 | **5.0** |

### Ablation Study

| GAN Init | DMD | Soften KL | PSNR ↑ | SSIM ↑ | LPIPS ↓ | FID ↓ |
|----------|-----|-----------|--------|--------|---------|-------|
| ✗ | ✓ | ✓ | 8.62 | 0.154 | 0.880 | 40.17 |
| ✓ | ✗ | ✗ | 16.23 | 0.369 | 0.375 | 21.48 |
| ✓ | ✓ | ✗ | 16.85 | 0.385 | 0.337 | 21.05 |
| ✓ | ✓ | ✓ | **17.50** | **0.382** | **0.320** | **20.54** |

### Key Findings

- **FVGen achieves quality on par with or superior to the teacher ViewCrafter** while delivering a **13× speedup** (66s → 5s). On MipNeRF360, FVGen surpasses ViewCrafter on both SSIM and LPIPS.
- **GAN initialization is indispensable**—removing it causes PSNR to drop sharply from 17.50 to 8.62, demonstrating that DMD alone cannot train the student model from scratch.
- **Distribution matching distillation (DMD) further improves upon the initialization**—using GAN alone (without DMD) yields PSNR of 16.23, which improves to 16.85 after adding DMD, indicating that DMD provides additional refinement of distributional alignment.
- **The contribution of softened KL is consistent and significant**—PSNR improves from 16.85 to 17.50 and FID decreases from 21.05 to 20.54, confirming the effectiveness of softening in alleviating mode collapse.
- Compared to other distillation methods: DMD2 is unstable when applied to ViewCrafter, achieving only PSNR 9.29–10.27; CausVid is more stable but still suffers from mode collapse, achieving PSNR 15.77–17.33. FVGen outperforms both comprehensively.

## Highlights & Insights

- **The "teacher as discriminator" design serves a dual purpose**—it eliminates the overhead of an additional discriminator network while cleverly exploiting the semantic representational capacity of the teacher's intermediate features. This trick is transferable to any diffusion distillation scenario.
- **GAN initialization provides not only good weights but also accurate density ratio estimates**—an underappreciated yet critically important insight in the paper. The density ratio $r(x,t) \approx f(D(x,t))/(1-f(D(x,t)))$ is obtained directly from the discriminator, making the computation of the softened KL effectively free.
- **The difference between softened and standard reverse KL is amplified in small-data regimes**—suggesting that softened KL may outperform standard KL in any distillation scenario with limited data, such as domain-specific diffusion models.
- **A 90% speedup with negligible quality loss** represents enormous efficiency gains for practical production pipelines.

## Limitations & Future Work

- **Inherits the intrinsic limitations of ViewCrafter**—structural integrity and consistency still degrade under extremely sparse inputs (e.g., single image); FVGen cannot exceed the capability ceiling of its teacher.
- **Only trained for short 16-frame video generation**—due to computational constraints, very large scenes are not covered. Extending to longer sequences may require segment-wise generation and cross-segment temporal consistency mechanisms.
- **Three video diffusion models** (teacher, student, fake score function) impose high memory requirements, limiting training at higher resolutions or with longer videos.
- **Downstream 3D reconstruction quality is not evaluated**—while novel-view quality is comparable, whether the subtle distributional differences introduced by distillation are amplified after 3DGS reconstruction remains to be verified.
- Future directions include: (1) extending distillation to stronger VDM teachers (e.g., CogVideoX); (2) end-to-end joint optimization with 3DGS; (3) adaptive step distillation—using 2 steps for simple scenes and 4 steps for complex ones.

## Related Work & Insights

- **vs. ViewCrafter**: Teacher model with 50-step DDIM sampling. FVGen distills it into 4 steps, achieving 13× speedup with comparable quality. ViewCrafter uses DUSt3R point clouds to guide generation.
- **vs. DMD2**: DMD2 jointly optimizes the GAN, student, and fake score function, but end-to-end training is unstable on multi-view data. FVGen's key improvements are decoupled training and the 3D discriminator.
- **vs. CausVid**: CausVid initializes via ODE-solver-generated pairs followed by standard DMD distillation. FVGen's GAN initialization is more efficient, and softened KL is more stable than standard KL.
- **vs. MVSplat360**: MVSplat360 refines 3DGS renderings with a VDM but only supports low resolution. FVGen directly accelerates VDM sampling and is more general.
- This work demonstrates the viability of video diffusion distillation for 3D vision tasks, laying the groundwork for future real-time VDM-driven 3D reconstruction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of GAN initialization and softened KL divergence is proposed for the first time in video distillation, though individual components draw on ideas from image distillation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Two standard benchmarks, multiple baselines, dedicated comparisons against distillation methods, and comprehensive ablations constitute very thorough experimentation.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear, mathematical derivations are complete, and figures/tables are information-dense.
- **Value**: ⭐⭐⭐⭐⭐ A 90% speedup with no quality loss is of enormous practical value, and the framework is generalizable to other VDM tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adversarial Distribution Matching for Diffusion Distillation Towards Efficient Image and Video Synthesis](adversarial_distribution_matching_for_diffusion_distillation_towards_efficient_image_and_video_synthesis.md)
- [\[ECCV 2024\] SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion](../../ECCV2024/video_generation/sv3d_novel_multi-view_synthesis_and_3d_generation_from_a_single_image_using_late.md)
- [\[CVPR 2025\] StreetCrafter: Street View Synthesis with Controllable Video Diffusion Models](../../CVPR2025/video_generation/streetcrafter_street_view_synthesis_with_controllable_video_diffusion_models.md)
- [\[ICCV 2025\] V.I.P.: Iterative Online Preference Distillation for Efficient Video Diffusion Models](vip_iterative_online_preference_distillation_for_efficient_video_diffusion_model.md)
- [\[AAAI 2026\] Phased One-Step Adversarial Equilibrium for Video Diffusion Models](../../AAAI2026/video_generation/phased_one-step_adversarial_equilibrium_for_video_diffusion_models.md)

</div>

<!-- RELATED:END -->

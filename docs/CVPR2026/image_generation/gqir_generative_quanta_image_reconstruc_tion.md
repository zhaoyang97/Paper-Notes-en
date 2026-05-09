---
title: >-
  [Paper Note] gQIR: Generative Quanta Image Reconstruction
description: >-
  [CVPR 2026][Image Generation][Single-photon sensor] This work adapts a large-scale text-to-image latent diffusion model to the extreme photon-limited imaging regime of single-photon avalanche diodes (SPADs) via a three-stage framework—Quanta-aligned VAE → adversarially fine-tuned LoRA U-Net → FusionViT spatiotemporal fusion—enabling high-quality RGB image reconstruction from sparse binary photon detections and significantly outperforming all existing methods under extreme conditions of 10K–100K fps.
tags:
  - CVPR 2026
  - Image Generation
  - Single-photon sensor
  - diffusion model
  - image reconstruction
  - burst imaging
  - VAE alignment
date: 2026-05-08
content_hash: 3218b238d94933e7
---

# gQIR: Generative Quanta Image Reconstruction

**Conference**: CVPR 2026
**arXiv**: [2602.20417](https://arxiv.org/abs/2602.20417)
**Code**: [GitHub](https://github.com/Aryan-Garg/gQIR)
**Area**: Image Generation / Image Reconstruction / Computational Imaging
**Keywords**: Single-photon sensor, diffusion model, image reconstruction, burst imaging, VAE alignment

## TL;DR

This work adapts a large-scale text-to-image latent diffusion model to the extreme photon-limited imaging regime of single-photon avalanche diodes (SPADs) via a three-stage framework—Quanta-aligned VAE → adversarially fine-tuned LoRA U-Net → FusionViT spatiotemporal fusion—enabling high-quality RGB image reconstruction from sparse binary photon detections and significantly outperforming all existing methods under extreme conditions of 10K–100K fps.

## Background & Motivation

**State of the Field**: SPAD single-photon sensors can capture images under extremely low illumination and at high speeds where conventional cameras fail, but each pixel records only a binary photon detection event (photon arrived / not arrived), making single-frame data extremely sparse and noise-dominated. Existing methods fall into two categories: (1) classical vision methods such as QBP, which reconstruct via block matching + frame alignment + Wiener filtering; and (2) learning-based methods such as QUIVER and QuDI, which process burst sequences via optical flow estimation combined with recurrent fusion or temporally conditioned U-Nets.

**Limitations of Prior Work**:
- Classical methods produce unreliable motion estimates when photons are extremely scarce and lack semantic priors, resulting in loss of high-frequency detail.
- Learning-based methods, while incorporating learned modules, are task-specific and trained from scratch, failing to leverage the structural and semantic knowledge embedded in large-scale pretrained generative models.
- All existing methods degrade severely under extreme deformation or ultra-high-speed motion (>10K fps).
- The Bayer mosaic of color SPADs further exacerbates sparsity, as each color channel receives even fewer photon events.

**Root Cause**: Large-scale T2I diffusion models possess strong natural image priors, but they assume continuous Gaussian noise, whereas SPAD data follows a discrete Bernoulli distribution with noise far exceeding that of conventional photography—naive fine-tuning leads to encoder collapse (catastrophic forgetting) and shortcut learning.

**Starting Point**: Rather than training from scratch, the paper proposes a carefully designed multi-stage adaptation strategy that transfers the structural priors of Stable Diffusion to the quanta domain. A key observation is that directly fine-tuning the VAE encoder degrades to constant output due to the extreme noise of SPAD inputs; a latent space alignment constraint must therefore be introduced to prevent collapse.

**Core Idea**: A three-stage modular framework: first align the latent space to handle Bernoulli statistical characteristics; then distill the diffusion prior into a single-step generator for enhanced perceptual quality; finally perform burst-level spatiotemporal fusion in the latent space.

## Method

### Overall Architecture

The input consists of binary photon frames from a SPAD sensor (or their aggregated 3-bit nano-bursts), and the output is a high-quality RGB image. The pipeline comprises three sequentially trained stages: Stage 1 trains a quanta-aligned VAE encoder for denoising and demosaicing; Stage 2 adversarially trains a LoRA U-Net to enhance perceptual fidelity; Stage 3 trains a FusionViT for burst-level spatiotemporal fusion in the latent space. Each stage freezes the parameters of all preceding modules.

### Key Designs

1. **Image Formation Model (Bernoulli SPAD Simulator)**

    - **Function**: Establishes a physically consistent simulation pipeline from clean sRGB images to SPAD observations.
    - **Mechanism**: A clean image $x_{gt}$ is mapped to linear radiance space via gamma correction as $x_{lin} = x_{gt}^{2.2}$; photon detection at each pixel then follows a Bernoulli distribution $x_{spad} = \text{Bern}(1 - e^{-\alpha \cdot x_{lin}})$, where $\alpha$ controls the expected photons per pixel (PPP). For color SPADs, a random Bayer pattern is applied to produce a mosaiced binary frame. Averaging $N$ frames yields a $\log_2(N+1)$-bit observation.
    - **Design Motivation**: The Bernoulli statistics, fundamentally different from conventional Gaussian noise models, serve as the starting point for all technical innovations in this work. A value of $\alpha = 1.0$ (PPP ≈ 3.5) is used during training.

2. **Stage 1: Quanta-Aligned VAE**

    - **Function**: Fine-tunes the pretrained SD VAE encoder so that it produces latent codes aligned with clean images from extremely noisy SPAD inputs.
    - **Mechanism**: The decoder $\mathcal{D}$ is frozen while the encoder $\mathcal{E}_{\phi^*}$ is fine-tuned. Two key modifications prevent encoder collapse:
        - **Deterministic mean encoding**: Rather than sampling from the posterior distribution, the mean $\mu_\phi(x_{lq})$ is used directly, avoiding variance amplification under Bernoulli noise.
        - **Latent Space Alignment (LSA) Loss**: $\mathcal{L}_{lsa} = \|\mu_{\phi^*}(x_{lq}) - \mu_\phi(x_{gt})\|_2^2$, where the second term encodes the GT image using the **frozen pretrained encoder**—unlike existing methods such as SUPIR/DiffBIR that use the same trainable encoder for both, which causes collapse via predegradation removal loss.
    - Total loss: $\mathcal{L} = \lambda_{lsa}\mathcal{L}_{lsa} + \lambda_{MSE}\mathcal{L}_{MSE} + \lambda_{perc}\mathcal{L}_{perc}$
    - **Design Motivation**: Directly applying the fine-tuning objectives of existing methods (e.g., SUPIR's predegradation removal loss) causes the encoder to learn a shortcut—producing approximately the same latent code regardless of input—because the trainable encoder controls both the supervision and prediction ends simultaneously, converging rapidly to a degenerate solution under extreme noise.

3. **Stage 2: Adversarially Fine-tuned LoRA U-Net (Perceptual Enhancement)**

    - **Function**: Distills the SD diffusion U-Net into a single-step generator to enhance high-frequency detail and perceptual quality of reconstructions.
    - **Mechanism**: A LoRA adapter initializes the generator $\mathcal{G}_{lora}$ (inheriting diffusion weights to ensure small initial gradients), and a multi-scale ConvNext-Large discriminator $\mathcal{V}_\theta$ is designed. Adversarial training proceeds via a standard min-max GAN objective: $\min_\phi \max_\theta \mathbb{E}[\log \mathcal{V}_\theta(x)] + \mathbb{E}[\log(1 - \mathcal{V}_\theta(\mathcal{G}(x)))]$
    - Total loss: $\mathcal{L} = \mathcal{L}_{adv} + \mathcal{L}_{perc} + \|\mathcal{D}(\mathcal{G}_{lora}(\mu_{\phi^*}(x_{lq}))) - x_{gt}\|_2^2$
    - **Design Motivation**: The ultra-high frame rates of SPADs (10K–100K fps) generate massive data volumes, rendering multi-step diffusion sampling impractical. Adversarial distillation compresses the diffusion prior into single-step inference, preserving pretrained knowledge while meeting real-time requirements.

4. **Stage 3: FusionViT (Latent Burst Spatiotemporal Fusion)**

    - **Function**: Exploits temporal redundancy across burst sequences to align and dynamically fuse multiple frames in the latent space, improving reconstruction fidelity and mitigating temporal artifacts.
    - **Mechanism**:
        - All frames are first reconstructed using Stages 1+2: $Y = \mathcal{D}(\mathcal{G}_{lora}(\mathcal{E}_{\phi^*}(X_{lq})))$; optical flow is then estimated on the reconstructed images using pretrained RAFT (directly estimating flow on noisy SPAD frames fails severely due to domain gap).
        - All burst latent maps are warped to the center frame according to the estimated flow.
        - A pseudo-3D miniViT $\mathcal{F}$ with sub-quadratic-complexity windowed attention performs dynamic fusion across temporal and spatial axes (naïve averaging produces motion blur).
        - The FusionViT output is added residually to the center frame latent $z_{T/2}$ via a learnable scalar $\delta$ (initialized to 0.05).
    - Loss: $\mathcal{L}_{fusion} = \|\mathcal{F}(\mu_{\phi^*}(X_{lq})) - \mu_\phi(x_{gt})\|_2^2 + \text{pixel MSE} + \mathcal{L}_{perc}$
    - **Design Motivation**: Naïve flow-warp-and-average produces severe blurring in motion scenes. FusionViT uses attention mechanisms to adaptively weight contributions from different frames based on motion magnitude and temporal distance to the center frame.

### Loss & Training

Progressive three-stage training with preceding modules frozen at each stage:
- **Stage 1**: 8× A100, 600K steps, $\lambda_{lsa}=0.1, \lambda_{MSE}=10^3, \lambda_{perc}=2$
- **Stage 2**: Single RTX 4090, 100K iterations, 256×256, $\lambda_{adv}=0.5, \lambda_{MSE}=500, \lambda_{perc}=5$
- **Stage 3**: FusionViT trained for only 20K steps using RAFT optical flow (pretrained on FlyingThings3D)

Training data: 2.81 million images + 44,575 videos, covering diverse sources including image super-resolution, face restoration, and video deblurring. 3-bit nano-bursts are obtained by averaging 7 binary frames; burst mode uses 11 3-bit nano-bursts (77 binary frames in total).

## Key Experimental Results

### Main Results

**Single-frame 3-bit reconstruction (334 test images, 384×384)**:

| Method | PSNR↑ | SSIM↑ | LPIPS↓ | ManIQA↑ | ClipIQA↑ | MUSIQ↑ |
|--------|-------|-------|--------|---------|----------|--------|
| InstantIR | 10.79 | 0.178 | 0.651 | 0.187 | 0.346 | 36.65 |
| ft-Restormer | 28.73 | 0.816 | 0.294 | 0.262 | 0.435 | 40.44 |
| ft-NAFNet | 28.28 | 0.830 | 0.261 | 0.300 | 0.473 | 39.13 |
| qVAE (Stage 1) | 28.18 | **0.863** | 0.327 | 0.299 | 0.487 | 44.90 |
| **gQIR (S1+S2)** | 27.28 | 0.839 | **0.318** | **0.331** | **0.547** | **45.61** |

**Burst reconstruction (extreme motion scenes)**:

| Dataset (fps) | QBP PSNR/SSIM | QUIVER PSNR/SSIM | **Burst-gQIR PSNR/SSIM** |
|---------------|---------------|-------------------|--------------------------|
| XVFI (1000) | 12.01 / 0.370 | 23.00 / 0.751 | **25.82 / 0.712** |
| I2-2000fps (2000) | 16.04 / 0.549 | 25.06 / 0.874 | **31.21 / 0.878** |
| XD (2K-100K) | 12.78 / 0.409 | 20.10 / 0.790 | **30.33 / 0.895** |
| **Overall** | 13.38 / 0.448 | 22.43 / 0.814 | **29.83 / 0.856** |

**I2-2000fps full test set**: Burst-gQIR achieves 30.81 dB PSNR / 0.868 SSIM, surpassing the previous SOTA QuDI (28.64 dB) by **+2.17 dB**.

### Ablation Study

**Stage 1 design choice ablation**:

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|---------------|-------|-------|--------|
| w/o deterministic encoding (A) | 20.56 | 0.435 | 0.167 |
| w/o LSA loss (B) | 10.39 | 0.222 | 0.139 |
| w/o (A) + (B) | 10.30 | 0.218 | 0.136 |
| **Full Stage 1** | **24.78** | **0.665** | **0.194** |

**Progressive three-stage improvement — fidelity vs. temporal stability**:

| Stage | PSNR↑ | SSIM↑ | $E^*_{warp}$↓ |
|-------|-------|-------|--------------|
| Alignment (S1) | 20.04 | 0.759 | 9.088 |
| Perceptual (S2) | 24.11 | 0.846 | 8.508 |
| **Fidelity (S3)** | **27.63** | **0.869** | **8.005** |

### Key Findings

- **LSA loss is critical to Stage 1 success**: Removing LSA causes PSNR to plummet from 24.78 to 10.39 (near random output), with the encoder degenerating to a constant mapping—validating the necessity of latent space alignment under Bernoulli noise.
- Deterministic encoding also matters, with its removal causing a ~4 dB drop in PSNR, though its impact is smaller than that of LSA.
- Adversarial training in Stage 2 slightly increases content drift while improving perceptual quality, but this is effectively mitigated by the spatiotemporal fusion in Stage 3.
- On the extreme motion dataset XD, gQIR achieves a **+17.5 dB** gain over QBP, demonstrating the substantial advantage of generative priors in out-of-distribution scenarios.
- On real color SPAD data, photo-realistic quality is achieved without dark count or hot pixel correction, requiring only gray-world white balancing.

## Highlights & Insights

- **First successful transfer of large-scale T2I diffusion priors to single-photon imaging**: This constitutes a domain with fundamentally different noise statistics (Bernoulli vs. Gaussian); the key breakthrough lies in identifying and resolving the encoder collapse problem.
- **The design of using a frozen pretrained encoder as the LSA alignment target is particularly elegant**: In contrast to SUPIR/DiffBIR, which use the same trainable encoder to generate both supervision and prediction simultaneously, the proposed approach severs the shortcut path by using a frozen copy under extreme noise conditions.
- **The three-stage decoupled design assigns clear responsibilities**: S1 addresses domain adaptation (structure and color), S2 enhances perceptual quality (high-frequency detail), and S3 exploits temporal information (stability and fidelity), with each stage independently optimizable.
- **Estimating optical flow in the reconstructed domain rather than the noisy domain** via pre-denoising circumvents the failure of flow estimation on raw SPAD data.

## Limitations & Future Work

- Training is fixed at PPP = 3.5; robustness under extremely low illumination (PPP ≤ 1) is limited—explicitly conditioning on PPP as a signal may improve generalization.
- The pretrained VAE decoder's 8-bit output constrains the native HDR capability of SPADs; developing an HDR-capable decoder is an important direction.
- Content drift from Stage 2 adversarial training is mitigated but not fundamentally resolved by Stage 3; video-level diffusion priors may further improve temporal consistency.
- Stage 3 relies on RAFT optical flow, which may still fail under extremely large motion; implicit alignment within the latent space warrants further exploration.
- Quantitative evaluation is conducted exclusively on synthetic data, as real SPAD data lacks ground-truth references.

## Related Work & Insights

- **vs. QBP**: QBP is a classical align-and-merge pipeline (block matching + Wiener filtering) that works well when photons are sufficient but lacks semantic priors. gQIR generalizes the align-and-merge philosophy to the latent space, replacing simple averaging with FusionViT, yielding a substantial advantage under extreme motion (+17.5 dB).
- **vs. QUIVER/QuDI**: QUIVER employs SpyNet optical flow with recurrent fusion; QuDI uses a temporally conditioned U-Net. Both are task-specific and trained from scratch without leveraging pretrained generative priors. gQIR surpasses QuDI by +2.17 dB on I2-2000fps.
- **vs. SUPIR/DiffBIR (general image restoration)**: These methods also fine-tune the VAE encoder to accommodate degraded inputs but collapse directly under Bernoulli noise. The LSA loss combined with the frozen encoder target constitutes the critical improvement specifically addressing the extreme noise of SPADs.
- The three-stage decoupled training paradigm is transferable to sensor restoration tasks with other non-standard noise models, such as event cameras and quantum sensors.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First adaptation of T2I diffusion to quanta imaging, with identification and resolution of the encoder collapse problem as a key contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers synthetic and real data, single-frame and burst modes, multiple fps levels, and complete ablations; also contributes the first color SPAD burst dataset and the XD benchmark.
- **Writing Quality**: ⭐⭐⭐⭐ Physical model and method motivation are clearly articulated; the analysis and visualization of encoder collapse are intuitive.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for introducing generative priors into computational imaging; code and datasets are publicly released.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] VOSR: A Vision-Only Generative Model for Image Super-Resolution](vosr_a_vision_only_generative_model_for_image_super_resolution.md)
- [\[CVPR 2026\] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](slice_semantic_latent_injection_via_compartmentali.md)
- [\[CVPR 2026\] FG-Portrait: 3D Flow Guided Editable Portrait Animation](fg-portrait_3d_flow_guided_editable_portrait_animation.md)
- [\[CVPR 2026\] 2ndMatch: Finetuning Pruned Diffusion Models via Second-Order Jacobian Matching](2ndmatch_finetuning_pruned_diffusion_models_via_second-order_jacobian_matching.md)
- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)

<!-- RELATED:END -->

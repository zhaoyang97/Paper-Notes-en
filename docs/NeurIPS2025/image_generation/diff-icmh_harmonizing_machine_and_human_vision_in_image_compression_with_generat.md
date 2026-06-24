---
title: >-
  [Paper Note] Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior
description: >-
  [Image Generation] This paper proposes Diff-ICMH, a diffusion-based generative image compression framework that preserves semantic integrity via a Semantic Consistency (SC) loss and activates generative priors via a Tag Guidance Module (TGM). Using a single encoder-decoder and a single bitstream, the framework simultaneously serves 10+ machine intelligence tasks and human visual perception without any task-specific adaptation.
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 35e250988b359879
---

# Diff-ICMH: Harmonizing Machine and Human Vision in Image Compression with Generative Prior

## Basic Information
- **arXiv**: 2511.22549
- **Conference**: NeurIPS 2025
- **Authors**: Ruoyu Feng, Yunpeng Qi, Jinming Liu, Yixin Gao, Xin Li, Xin Jin, Zhibo Chen
- **Affiliation**: USTC, Eastern Institute of Technology (Ningbo)
- **Code**: https://github.com/RuoyuFeng/Diff-ICMH

## TL;DR
This paper proposes Diff-ICMH, a diffusion-based generative image compression framework that preserves semantic integrity via a Semantic Consistency (SC) loss and activates generative priors via a Tag Guidance Module (TGM). Using a single encoder-decoder and a single bitstream, the framework simultaneously serves 10+ machine intelligence tasks and human visual perception without any task-specific adaptation.

## Background & Motivation
Image compression faces a fundamental split between two optimization objectives:

1. **Human-oriented compression**: Optimizes pixel fidelity (PSNR) or perceptual quality (LPIPS/FID), but offers poor support for machine vision tasks.
2. **Image Coding for Machines (ICM)**:
    - Traditional codec approaches: Adapted via quantization parameter tuning or bit allocation, constrained by non-differentiable fidelity-driven designs.
    - Task-driven end-to-end methods: Strong on specific tasks but poor cross-task generalization.
    - Feature compression methods: Directly compress intermediate features, tightly coupled to specific models, and incompatible with human viewing.

**Key Insight**: Semantic integrity and perceptual realism are jointly required by both machine intelligence and human perception — these two objectives are not inherently opposed.

## Core Problem
How to design a universal image codec that efficiently serves multiple downstream machine intelligence tasks and human visual perception simultaneously from a single bitstream?

## Method

### 1. Design Philosophy
Fidelity-driven compression introduces two primary sources of information loss:
- **Semantic distortion**: Loss of core semantic information → directly impairs task analysis.
- **Perceptual mismatch**: Textures and details deviate from the natural distribution → domain shift causes accumulated errors in feature extraction.

Experimental validation (Figure 3): At deep layers of ResNet50 (layer4), fidelity-driven codecs (VTM, ELIC) exhibit far greater feature deviation than generative codecs (MS-ILLM), demonstrating that **realistic textures effectively mitigate error accumulation in deep layers**.

### 2. Overall Architecture
- **Encoding side**: Input image $\mathbf{x}$ is compressed into latent features $\hat{\mathbf{z}}$, targeting the VAE latent space of Stable Diffusion ($8\times$ spatial downsampling).
- **Tag extraction**: Recognize Anything extracts word-level semantic tags $\mathbf{c}$.
- **Bitstream**: Compressed latents + tag IDs (losslessly encoded, ~100 bits/image).
- **Decoding side**: $\hat{\mathbf{z}}$ is fed as a condition into a ControlNet-style control module, jointly with a frozen Stable Diffusion model for generative reconstruction.

### 3. Semantic Consistency Loss (SC Loss)
The pretrained diffusion model's feature extraction capability is leveraged as a semantic space:
$$\mathcal{L}_\text{sem} = -\mathbb{E}_{\mathbf{z}, \hat{\mathbf{z}}} \left[ \frac{1}{N} \sum_{n=1}^N \text{sim}(f(\mathbf{z})_n, f(\hat{\mathbf{z}})_n) \right]$$

where $f(\cdot)$ denotes the forward pass of the frozen diffusion model and $\text{sim}$ is cosine similarity:
$$\text{sim}(\mathbf{z}, \hat{\mathbf{z}}) = \frac{\mathbf{z}^T \hat{\mathbf{z}}}{|\mathbf{z}|_2 |\hat{\mathbf{z}}|_2}$$

Key design choices:
- Applying the loss at the **middle block** of the U-Net yields the best performance — deep features better capture abstract semantics.
- Using **noise-free input** ($t=0$) — the diffusion model's semantic feature extraction is optimal on clean signals.

### 4. Tag Guidance Module (TGM)
- A pretrained tag extractor $\mathcal{E}_t$ (Recognize Anything) generates image-level tags.
- Tags are mapped to numerical indices in a predefined dictionary and losslessly encoded.
- At decoding, indices are converted back to text strings and used as conditions for the diffusion model and control module.
- **Classifier-Free Guidance** (CFG scale = 5.0) is applied at inference to enhance semantic clarity.
- Overhead is minimal: approximately **100 bits/image**.

### 5. Complete Loss Function
$$\mathcal{L}_\text{final} = \lambda_\text{rate} \mathcal{L}_\text{rate} + \lambda_\text{dist} \mathcal{L}_\text{dist} + \lambda_\text{diff} \mathcal{L}_\text{diff} + \lambda_\text{sem} \mathcal{L}_\text{sem}$$

Components:
- $\mathcal{L}_\text{rate}$: Rate loss (estimated entropy of quantized latents and hyperprior).
- $\mathcal{L}_\text{dist} = \|\mathcal{E}_\text{VAE}(\mathbf{x}) - \mathcal{D}_c(\hat{\mathbf{y}})\|_2^2$: Latent-space reconstruction loss.
- $\mathcal{L}_\text{diff} = \mathbb{E}[\|\boldsymbol{\epsilon} - \boldsymbol{\epsilon}_\theta(\mathbf{z}_t, \hat{\mathbf{z}}, \mathbf{c}, t)\|_2^2]$: Diffusion noise prediction loss.
- Weights: $\lambda_\text{dist} = \lambda_\text{diff} = 1, \lambda_\text{sem} = 2, \lambda_\text{rate} \in \{2, 4, 8, 16, 32\}$.

### 6. Loss & Training
- Dataset: LSDIR, randomly cropped to $512\times512$.
- Base model: Stable Diffusion 2.1 (frozen).
- Two-stage training: 200K steps at high bitrate → 200K steps of multi-rate fine-tuning.
- DDIM sampling with 50 steps at inference.

## Key Experimental Results

### Machine Intelligence Task Performance (10+ Tasks, Single Codec/Bitstream)
- **COCO object detection / instance segmentation / panoptic segmentation**: Consistently outperforms VTM, ELIC, and most perception-oriented methods.
- **Flickr30K cross-modal retrieval**: Significant advantage at ultra-low bitrates (0.01–0.05 bpp).
- **ADE20K open-vocabulary segmentation**: Substantially outperforms competing methods at 0.02–0.1 bpp.
- All results are achieved without any task-specific fine-tuning.

### Perceptual Quality (Kodak / Tecnick / CLIC2020)

| Metric Type | Diff-ICMH vs. Competitors |
|-------------|--------------------------|
| PSNR/MS-SSIM (fidelity) | Below VTM/ELIC (inherent to generative methods); on par with DiffEIC |
| LPIPS ↓ | Outperforms all methods |
| **FID ↓** | **State of the Art** |
| **DISTS ↓** | **State of the Art** |

- Perceptual advantages are most pronounced at ultra-low bitrates.

### Ablation Study (COCO Object Detection mAP)
- SC loss + TGM combination: ~4 mAP improvement over the baseline at ~0.025 bpp.
- Optimal SC loss configuration: $\lambda_\text{sem}=2.0$, middle block, noise-free input.

## Highlights & Insights
1. **One codec, multiple uses**: A single codec and bitstream support 10+ downstream tasks and human viewing without any adaptation.
2. **Dual guarantee of semantics and perception**: SC loss preserves semantic integrity; the generative framework ensures perceptual realism.
3. **TGM with negligible overhead**: Only ~100 bits of tag information suffice to substantially activate the generative prior.
4. **Latent-space compression design**: Decoding to the VAE latent space rather than pixel space naturally filters out semantically irrelevant redundancy.
5. **Ultra-low bitrate advantage**: Performance gains are most significant under extreme conditions of 0.01–0.05 bpp.

## Limitations & Future Work
1. **Decoding speed**: Diffusion denoising requires 50 iterative forward passes, making decoding substantially slower than traditional codecs.
2. **Fidelity loss**: PSNR/MS-SSIM are below fidelity-driven methods, making the approach unsuitable for scenarios requiring pixel-accurate reconstruction.
3. **VAE bottleneck**: The $8\times$ downsampled latent space may discard fine spatial details (e.g., pose estimation).
4. **Training cost**: Two-stage training requires GPU resources commensurate with Stable Diffusion.

## Related Work & Insights
- **vs. VTM/ELIC (fidelity-driven)**: Diff-ICMH achieves lower PSNR but substantially outperforms on machine intelligence tasks and perceptual quality.
- **vs. DiffEIC (diffusion-based compression)**: Comprehensively surpasses on machine tasks; on par or better on perceptual quality.
- **vs. TransTIC/Adapter-ICMH (task-adaptive)**: Diff-ICMH achieves superior performance without any adaptation.
- **vs. feature compression methods**: Diff-ICMH operates in the image domain, simultaneously supporting human viewing.
- **Semantic preservation is the key to universal compression**: The success of SC loss demonstrates that semantic information is the shared foundation of both machine task performance and human understanding.
- **A new role for generative compression**: Beyond pursuing perceptual quality, generative compression represents a pathway toward universal intelligent codecs.
- **CFG in compression**: Classifier-Free Guidance is extended from generation to conditional enhancement in compression.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First work to design a universal codec for both machine and human use from a unified semantic-perceptual perspective.
- Technical Depth: ⭐⭐⭐⭐⭐ — SC loss design is well-motivated and rigorously validated through ablation studies.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 10 tasks + 3 perceptual datasets + comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐☆ — Framework is clearly presented, though the extensive reference list slightly reduces readability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] DEXTER: Diffusion-Guided EXplanations with TExtual Reasoning for Vision Models](dexter_diffusion-guided_explanations_with_textual_reasoning_for_vision_models.md)
- [\[ICCV 2025\] Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities](../../ICCV2025/image_generation/rethinking_the_embodied_gap_in_vision-and-language_navigation_a_holistic_study_o.md)
- [\[NeurIPS 2025\] Denoising Weak Lensing Mass Maps with Diffusion Model and Generative Adversarial Network](denoising_weak_lensing_mass_maps_with_diffusion_model_and_generative_adversarial.md)
- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](detecting_generated_images_by_fitting_natural_image_distributions.md)
- [\[ICCV 2025\] REGEN: Learning Compact Video Embedding with (Re-)Generative Decoder](../../ICCV2025/image_generation/regen_learning_compact_video_embedding_with_re-generative_decoder.md)

</div>

<!-- RELATED:END -->

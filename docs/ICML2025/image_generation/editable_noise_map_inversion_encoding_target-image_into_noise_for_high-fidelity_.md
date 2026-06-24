---
title: >-
  [Paper Note] Editable Noise Map Inversion: Encoding Target-image into Noise For High-Fidelity Image Manipulation
description: >-
  [ICML 2025][Image Generation][Diffusion model inversion] Proposes Editable Noise Map Inversion (ENM Inversion), which simultaneously optimizes reconstruction error and edit alignment error during the inversion process. This "engraves" both source and target image information into the noise map, achieving an optimal balance between content preservation and editing fidelity.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Diffusion model inversion"
  - "image editing"
  - "noise map"
  - "DDIM inversion"
  - "attention control"
date: 2026-05-08
content_hash: 56b3277f293eaa2e
---

# Editable Noise Map Inversion: Encoding Target-image into Noise For High-Fidelity Image Manipulation

**Conference**: ICML 2025  
**arXiv**: [2509.25776](https://arxiv.org/abs/2509.25776)  
**Code**: None  
**Area**: Image Generation  
**Keywords**: Diffusion model inversion, image editing, noise map, DDIM inversion, attention control

## TL;DR

Proposes Editable Noise Map Inversion (ENM Inversion), which simultaneously optimizes reconstruction error and edit alignment error during the inversion process. This "engraves" both source and target image information into the noise map, achieving an optimal balance between content preservation and editing fidelity.

## Background & Motivation

Diffusion-model-based image editing workflows typically consist of two steps: (1) Inversion—encoding the input image into a sequence of noise maps; and (2) Editing—manipulating the image using attention control (e.g., Prompt-to-Prompt, MasaCtrl) based on the target prompt.

**Key Challenge**: Existing inversion methods (such as DDIM Inversion, Null-Text Inversion, PNP Inversion, and Fixed-Point Iteration) aim for faithful reconstruction of the source image. Although the resulting noise maps reconstruct the original image precisely, they severely restrict editing flexibility. In other words, target image information is not "engraved" into the noise map, leading to poor editing performance and artifacts (e.g., failing to transform a butterfly into a parrot).

**Key Observation**: Analysis by the authors on the AFHQ dataset reveals that a smaller discrepancy between the noise map reconstructed using the source prompt and the noise map edited using the target prompt correlates with better editing quality (lower LPIPS, higher CLIP Score). This pattern consistently holds across different inversion steps.

## Method

### Overall Architecture

ENM Inversion introduces an **Editable Noise Refinement** step on top of standard DDIM Inversion. The overall pipeline:

1. **DDIM Forward Inversion**: Iteratively inverts the source image $z_0$ into a sequence of noise maps $\{z_1, z_2, \dots, z_T\}$.
2. **Editable Noise Refinement**: Iteratively optimizes $z_t$ after each inversion step to simultaneously satisfy reconstruction and editing requirements.
3. **Attention-Controlled Editing**: Feeds the optimized noise map into the editing pipeline (P2P/MasaCtrl/PnP), leveraging the transfer of attention maps from the reconstruction path to the editing path.

### Key Designs

**Core Idea**: At each inversion step, the noise map is required not only to precisely reconstruct the source image but also to minimize the discrepancy between the noise maps produced by the reconstruction and editing paths.

#### 1. Edit Alignment Loss $L_{edit}$

Measures the difference between denoising outputs at $z_t$ using the source prompt $C_{src}$ and the target prompt $C_{tgt}$, respectively:

$$L_{edit} = \|f(z_t, t, C_{src}) - f(z_t, t, C_{tgt})\|^2$$

Where $f(z_t, t, C)$ is the DDIM sampling function (one-step denoising). Minimizing this term encourages the noise map to yield similar denoising results for both source and target prompts, thereby encoding ("engraving") target image information into the noise.

#### 2. Reconstruction Preservation Loss $L_{prev}$

Ensures that the noise map still accurately reconstructs the source image:

$$L_{prev} = \|z_{t-1} - f(z_t, t, C_{src})\|^2$$

This term prevents the optimized noise map from drifting too far from the source image.

#### 3. Joint Optimization Objective

$$\arg\min_{z_t} L = L_{prev} + \lambda \cdot L_{edit}$$

Where $\lambda$ is a hyperparameter weight controlling the strength of edit alignment.

#### 4. Threshold Truncation and Iterative Refinement

To improve efficiency, a predefined threshold $\tau$ is introduced to early-stop the iterations when $L$ is below $\tau$. At most $\mathcal{K}$ refinement iterations are performed at each step.

### Algorithmic Pipeline (Algorithm 1)

```text
Input: Source image z_0, inversion steps T, source/target prompts, refinement steps K, threshold τ
Output: Optimized noise maps {z_T, ..., z_1}

for t = 1 to T:
    z_t ← f_inv(z_{t-1}, t-1, C_src)         # Standard DDIM inversion
    z_{t-1}^e ← f(z_t, t, C_tgt)              # Denoise using target prompt
    for k = 1 to K:
        Calculate L = L_prev + λ·L_edit
        if L < τ: break
        z_t ← z_t - lr · ∇_{z_t} L           # Gradient descent update
```

### Compatibility with Existing Editing Methods

As a plug-and-play inversion module, ENM Inversion can directly replace the inversion steps in the following editing methods:
- **Prompt-to-Prompt (P2P)**: Replaces cross-attention maps.
- **MasaCtrl**: Mutual self-attention control for non-rigid editing.
- **Plug-and-Play (PnP)**: Modifies spatial features and self-attention maps.

### Extension to Video Editing

Integrating ENM Inversion into the Video-P2P framework: first fine-tune the image diffusion model for video modeling, then perform ENM Inversion on each frame, and finally ensure temporal consistency through cross-frame attention control.

### Loss & Training

This method requires **no training** and fully optimizes the noise map at inference time. Key hyperparameters:
- $\lambda = 10$ (editing weight, default)
- $T = 50$ (DDIM sampling steps)
- CFG = 7.5
- Base model: Stable Diffusion V1-4 (P2P/MasaCtrl) or V1-5 (PnP/Video-P2P)

## Key Experimental Results

### Main Results

Evaluated on the PIE-Bench dataset (700 images × 9 editing tasks) in combination with Prompt-to-Prompt:

| Method | Structure Dist.↓ | PSNR↑ | LPIPS↓ | MSE↓ | SSIM↑ | CLIP-Whole↑ | CLIP-Edited↑ |
|---|---|---|---|---|---|---|---|
| DDIM+P2P | 69.43 | 17.87 | 208.80 | 219.88 | 71.14 | 25.01 | 22.44 |
| NTI+P2P | 13.44 | 27.03 | 60.67 | 35.86 | 84.11 | 24.75 | 21.86 |
| PNPInv+P2P | 11.65 | 27.22 | 54.55 | 32.86 | 84.76 | 25.02 | 22.10 |
| **Ours+P2P** | **10.13** | **28.19** | **45.26** | **27.02** | **86.29** | **25.30** | **22.12** |

Integration with MasaCtrl and Plug-and-Play similarly outperforms all baselines. Comparison with Fixed-Point methods:

| Method | Structure Dist.↓ | PSNR↑ | LPIPS↓ | SSIM↑ | CLIP-Whole↑ |
|---|---|---|---|---|---|
| AIDI+P2P | 12.19 | 26.96 | 57.92 | 84.17 | 24.96 |
| FPI+P2P | 14.71 | 26.61 | 61.97 | 83.52 | 23.93 |
| ReNoise | 22.60 | 25.19 | 85.29 | 82.30 | 23.78 |
| **Ours+P2P** | **10.13** | **28.19** | **45.26** | **86.29** | **25.30** |

Inference time comparison (combined with P2P, on a single RTX 3090):

| Method | Time (s) |
|---|---|
| DDIM | 18.22 |
| NTI | 148.48 |
| StyleD | 382.98 |
| PNPInv | 28.17 |
| **Ours** | **38.87** |

### Ablation Study

**Impact of Hyperparameter $\lambda$ (Editing Weight)**:

| $\lambda$ | Structure Dist.↓ | PSNR↑ | SSIM↑ | CLIP-Whole↑ | Description |
|---|---|---|---|---|---|
| 5 | 10.10 | 28.22 | 86.23 | 25.17 | Slightly weaker editing capability |
| **10 (Default)** | **10.13** | **28.19** | **86.29** | **25.30** | **Optimal balance** |
| 15 | 10.38 | 28.13 | 86.15 | 25.32 | Enhanced editing but reconstructed fidelity drops |
| 20 | 12.41 | 28.01 | 86.12 | 25.34 | Over-editing, structural loss |

**Impact of DDIM Steps $T$**: Fewer steps (20 steps) favor preservation, while more steps (75/100 steps) benefit editing; 50 steps achieves the optimal balance.

**Robustness to Source Prompt**: Replacing the source prompt with a null text shows no significant difference in reconstruction quality, demonstrating that the method is insensitive to the source prompt.

### Key Findings

1. **Strong Correlation Between Noise Map Discrepancy and Edit Quality**: Lower discrepancy between reconstruction and editing noise maps leads to better editing quality. This key observation forms the theoretical foundation of the method.
2. **Cross-Attention Alignment Analysis**: ENM Inversion maintains a high and stable cross-attention alignment score throughout the entire denoising process, whereas DDIM shows declining alignment in later stages. While DDPM shows some improvement, it remains unstable.
3. **Extensibility to Flow-Based Models**: When integrated with RF Inversion (Ours+RFInv), it significantly improves reconstruction quality on the Flux model (LPIPS: 232.88 $\rightarrow$ 185.86).

## Highlights & Insights

1. **Elegant Entry Point**: Instead of focusing solely on more accurate source image reconstruction, it "encodes the target image into the noise map," fundamentally resolving the trade-off that "better reconstruction leads to worse editing."
2. **Plug-and-Play**: Serves as an inversion module compatible with mainstream editing methods such as P2P, MasaCtrl, and PnP, without requiring modifications to their editing pipelines.
3. **Training-Free**: Requires no extra training and optimizes purely at inference time, lowering the threshold for deployment.
4. **Comprehensive Experiments**: Covers both image editing (PIE-Bench, 9 task types) and video editing (DAVIS), with thorough comparisons against 7+ baselines.
5. **Simple and Effective Loss Design**: Weighting just two loss terms addresses both core requirements elegantly and efficiently.

## Limitations & Future Work

1. **Dependence on Base Model Capability**: If the target image exceeds the generation domain of Stable Diffusion, the method will fail.
2. **Requirement of Re-Inversion for Each Edit**: Unlike traditional methods where the noise map can be reused for multiple target prompts, ENM requires individual inversion for each source-target pair, increasing computational overhead.
3. **Inference Speed**: Although 38.87s is far faster than NTI (148s) and StyleD (383s), it is still about twice as slow as DDIM (18s).
4. **Only Validated on SD v1.4/v1.5**: Has not been validated on SDXL or newer large models.
5. **Optimization Instability**: Gradient optimization of the noise map may fail to converge in extreme editing scenarios.

## Related Work & Insights

- **Null-Text Inversion (NTI)**: Optimizes null-text embeddings to improve reconstruction quality but compromises editability.
- **PNP Inversion**: Improves efficiency by separating source and target branches, but editing capabilities remain limited.
- **Fixed-Point Iteration (AIDI/FPI/ReNoise)**: Iteratively solves implicit equations to reduce approximation errors, but is highly reconstruction-oriented.
- **DDPM Inversion (Edit-Friendly)**: Introduces stochasticity to enhance editability, but yields unstable reconstruction.
- **Pix2pix-Zero**: Calculates edit directions using changes in attention maps, which inspired this paper's methodology of analyzing noise map discrepancies.

**Insight**: In generative model editing, "reconstruction quality" and "editing flexibility" represent an inherent trade-off. The joint optimization framework of ENM offers a general paradigm for such trade-off problems—simultaneously optimizing both objectives rather than focusing on only one.

## Rating

- Novelty: ⭐⭐⭐⭐ — The core observation (discrepancy between noise maps relates to editing quality) is insightful, though the optimization framework itself is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Highly comprehensive, featuring multiple editing methods x multiple baselines x image/video datasets x ablation studies x hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-explained motivation, and rich figures and tables.
- Value: ⭐⭐⭐⭐ — High practical value and plug-and-play capability, though bound within the SD v1 ecosystem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Exploring Position Encoding in Diffusion U-Net for Training-free High-resolution Image Generation](exploring_position_encoding_in_diffusion_u-net_for_training-free_high-resolution.md)
- [\[CVPR 2025\] Noise Diffusion for Enhancing Semantic Faithfulness in Text-to-Image Synthesis](../../CVPR2025/image_generation/noise_diffusion_for_enhancing_semantic_faithfulness_in_text-to-image_synthesis.md)
- [\[ICML 2025\] Taming Rectified Flow for Inversion and Editing](taming_rectified_flow_for_inversion_and_editing.md)
- [\[ECCV 2024\] Diffusion-based Image-to-Image Translation by Noise Correction via Prompt Interpolation](../../ECCV2024/image_generation/diffusion-based_image-to-image_translation_by_noise_correction_via_prompt_interp.md)
- [\[ICCV 2025\] Improved Noise Schedule for Diffusion Training](../../ICCV2025/image_generation/improved_noise_schedule_for_diffusion_training.md)

</div>

<!-- RELATED:END -->

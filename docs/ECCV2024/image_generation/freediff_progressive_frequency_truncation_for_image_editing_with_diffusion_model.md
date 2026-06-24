---
title: >-
  [Paper Note] FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models
description: >-
  [ECCV 2024][Image Generation][Diffusion Models] Revisiting the image editing process of diffusion models from a frequency perspective, this work reveals that the denoising network preferentially restores low-frequency components, leading to a misalignment between editing guidance and the target region. The authors propose progressive frequency truncation (FreeDiff) to refine guidance signals in frequency space, achieving tuning-free, general image editing.
tags:
  - "ECCV 2024"
  - "Image Generation"
  - "Diffusion Models"
  - "Image Editing"
  - "Frequency Truncation"
  - "Guidance Refinement"
  - "Tuning-free"
date: 2026-05-08
content_hash: dc5981c6e08dfe16
---

# FreeDiff: Progressive Frequency Truncation for Image Editing with Diffusion Models

**Conference**: ECCV 2024  
**arXiv**: [2404.11895](https://arxiv.org/abs/2404.11895)  
**Code**: [GitHub](https://github.com/Thermal-Dynamics/FreeDiff)  
**Area**: Image Generation  
**Keywords**: Diffusion Models, Image Editing, Frequency Truncation, Guidance Refinement, Tuning-free

## TL;DR

Revisiting the image editing process of diffusion models from a frequency perspective, this work reveals that the denoising network preferentially restores low-frequency components, leading to a misalignment between editing guidance and the target region. The authors propose progressive frequency truncation (FreeDiff) to refine guidance signals in frequency space, achieving tuning-free, general image editing.

## Background & Motivation

Text-driven image editing is a fundamental task in computer vision. Although large-scale T2I models exhibit powerful generation capabilities, they face a key challenge of **misalignment between guidance signals and target edit regions** during precise editing—for instance, editing "a hat" often leads to undesired changes in non-target areas.

Existing solutions fall into two categories:

**Tuning Paradigm** (e.g., SVDiff, InstructPix2Pix): Requires additional training data, and upgrading the base model requires retraining.

**Tuning-free Paradigm** (e.g., P2P, PNP, MasaCtrl): Refines guidance through attention map manipulation, but has serious limitations—being **highly tailored to specific edit types**, demanding different attention manipulation strategies for different tasks, and having hyperparameters sensitive to individual images.

Key Observations:
- Due to the **power law** of natural images (energy is concentrated in low frequencies) and the **decaying noise schedule** (large noise in early timesteps), the denoising network mainly restores low-frequency components in the early stages.
- Different edit types correspond to different frequency ranges: pose/shape corresponds to low frequencies, identity replacement/texture to high frequencies, and color to the lowest frequencies.
- Directly applying guidance introduces **excessive low-frequency components**, which disturb the non-target regions.

## Method

### Overall Architecture

The pipeline of FreeDiff consists of two core steps:
1. **Fixed-Point DDIM Inversion**: Obtains the accurate inverted latent $\hat{x}_T$ from the source image.
2. **Progressive Frequency Truncation**: Incrementally truncates and refines the guidance signal $g_t$ in the frequency domain during the generation process.

The key innovation is that it **does not delve into the internal structure of the network** (does not manipulate attention maps) and only filters the output of the denoising network (the guidance signal) in the frequency space, thereby unifying the handling of both rigid and non-rigid editing tasks.

### Key Designs

#### 1. Analyzing Diffusion Priors from a Frequency Perspective

The authors reveal key mechanisms through detailed frequency analysis:

**Power Law of Natural Images**: The amplitude spectrum of natural images peaks at low frequencies and decays as $1/f^\beta$ ($\beta \approx 1.1$) as the frequency increases.

**Signal-to-Noise Ratio (SNR) Analysis**: The noise added at timestep $t$ is Additive White Gaussian Noise (AWGN), whose power spectrum is uniformly distributed across all frequencies as $\sigma_t^2 = 1-\alpha$. Since image energy is concentrated in low frequencies, the SNR at low frequencies is much higher than that at high frequencies. Consequently, the denoising network can only effectively restore low-frequency components in the early stages (large noise) and high-frequency components in the later stages.

**Guidance Weight Decay**: The weight coefficient of guidance $g_t$ in the final output $x_0$ is:
$$w_{g_t} = -\gamma\sqrt{\alpha_1}(\sqrt{\frac{1}{\alpha_t}-1} - \sqrt{\frac{1}{\alpha_{t-1}}-1})$$

Under a typical 50-step DDIM: $w_{g_{981}}=1.25$, $w_{g_{681}}=0.23$, $w_{g_{181}}=0.046$, showing a decaying trend—guidance weight is largest in the early stage, but the guidance at this point mainly contains low-frequency components.

**Frequency Difference Verification**: By comparing the frequency-domain difference $\mathcal{F}_{diff}$ of the source image vs. direct editing vs. attention-based editing, it is verified that successful edits indeed introduce less power in the low-frequency components.

#### 2. Progressive Frequency Truncation

The core truncation operation is achieved through frequency-domain filtering:

$$\hat{g}_t = \text{IFFT}(\text{FFT}(g_t) \circ \mathcal{M}_t^H(r) \circ \mathcal{M}_t^L(r))$$

where:
- $\mathcal{M}_t^H(r) = \mathcal{I}(r > r_t^H)$ is a high-pass filter to remove excessive low frequencies.
- $\mathcal{M}_t^L(r) = \mathcal{I}(r < r_t^L)$ is a low-pass filter to remove undesired high frequencies.
- The filtering radii $r_t^H, r_t^L$ progressively change with timesteps.

It also includes two steps of spatial refinement:
1. **Change-rate Truncation**: Removes pixels that change excessively after frequency truncation (as the energy of these pixels mainly originates from low frequencies):
$$\mathcal{M}_t^S = \mathcal{I}(\frac{|{\hat{g}_t - g_t}|}{|{g_t}|} < \kappa), \quad \kappa = 0.6$$

2. **$\eta$-Truncation**: Removes the lowest 80% of values, retaining only the most significant edit signals:
$$\mathcal{M}_t^V = \mathcal{I}(\tilde{g}_t > \eta_{0.8}(\tilde{g}_t))$$

#### 3. Response Period Design

Based on two hypotheses:
- During generation, the guidance functions through a **single continuous response period** (corresponding to an atomic editing command).
- Guidance outside the response period is irrelevant and should be set to zero.

The response period is defined by $T_{st}$ (start timestep) and $T_{ed}$ (end timestep), and guidance outside this period is zeroed out.

#### 4. Edit Type Adaptation

Classification of different edit types from a frequency perspective:
- **Identity Replacement** (e.g., dog $\rightarrow$ lion): Similar to object removal, involving high spatial frequency information.
- **Shape/Pose Change**: Involves low spatial frequency information.
- **Color Change/Environment Adjustment**: Involves the lowest spatial frequency components.

Special handling for color editing (two-step method):
1. First, generate a coarse mask of the target object through frequency truncation at specific timesteps.
2. Only utilize this mask to perform guidance truncation for color editing.

#### 5. Fixed-Point DDIM Inversion

Fixed-point iteration is used with $N=3\sim5$ to solve the implicit equation:
$$x_{t+1}^{i+1} = f(x_{t+1}^i), \quad x_{t+1}^0 = f(x_t)$$

Compared to standard DDIM inversion, fixed-point iteration achieves near-perfect reconstruction even under a large guidance scale.

### Loss & Training

FreeDiff is a **completely tuning-free** method and involves no training loss. The core operations are performed entirely during the inference stage, requiring only a pre-trained T2I model.

Inference configuration:
- Base model: SD v1.4 / v1.5
- DDIM sampling: 50 steps
- Guidance Scale: 7.5
- Inversion method: Fixed-point iteration, N=5
- Default hyperparameter sets $(T_{st}, T_{ed}, r_t^H, \tau_i)$ provided for different edit types.

## Key Experimental Results

### Main Results

Quantitative evaluation on around 200 filtered images from the PIE benchmark dataset:

| Method | CLIP Score (Full) ↑ | Background LPIPS ↓ |
|------|-------------------|-------------------|
| P2P | 24.75 | 11.83 |
| PNP | 25.47 | 15.01 |
| MasaCtrl | 24.66 | 13.97 |
| **FreeDiff** | **25.51** | **11.14** |

FreeDiff outperforms all attention-based methods in both semantic consistency (CLIP Score) and background preservation (LPIPS).

Detailed results for each subcategory (CLIP Score$\uparrow$ / LPIPS$\downarrow$):

| Method | Cat:1 (n:77) | Cat:2 (n:50) | Cat:3 (n:27) | Cat:5 (n:11) | Cat:7 (n:38) |
|------|------------|------------|------------|------------|------------|
| MasaCtrl | 24.57/.166 | 24.83/.100 | 25.58/.181 | 26.92/.104 | 25.01/.119 |
| PNP | 25.30/.173 | 26.03/.105 | 25.77/.200 | 26.92/.129 | 26.45/.133 |
| P2P | 24.78/.134 | 25.11/.089 | 24.02/.177 | 27.14/.084 | 25.76/.094 |
| **FreeDiff** | 24.97/.125 | **26.49/.080** | 24.17/.143 | **27.47/.134** | 25.74/.097 |

### Ablation Study

1. **Impact of Frequency Truncation Range** (fixing $r_t^H \in \{0,4,8,12,16,20\}$):
    - Progressively expanding the truncation range $\rightarrow$ edited images become increasingly closer to the source image.
    - Expected editing effects (e.g., eyeglasses) become increasingly inconspicuous.
    - Validates the existence of the response period and the effective frequency band.

2. **Edit Prompt Sensitivity**:
    - A simple target prompt like "a pizza" causes fewer background changes compared to a detailed description like "white plate with pizza on it".
    - Conclusion: **Edit prompts should avoid describing objects and regions unrelated to the editing target.**

3. **Effect of $\eta$-Truncation**:
    - $\eta$-truncation is not the primary driver but helps preserve details in non-edited regions.
    - It sustains fine characteristics such as light reflection on hair and facial shapes.

### Key Findings

1. **Generality of the Frequency Perspective**: Unifies the explanation of why different editing methods perform differently across various task types from a frequency standpoint.
2. **Low-Frequency Signals as the Root of Edit Failures**: Denoising network prior preferences and weight scheduling both lean towards low frequencies, causing edit guidance to spatially "spill over" into non-target areas.
3. **Frequency Truncation as an Alternative to Attention Manipulation**: Achieves precise editing without needing complex internal operations within the network.
4. **First Frequency-Domain Guidance Refinement Method**: Opens up a new direction for image editing research.

## Highlights & Insights

1. **Prominent Theoretical Contribution**: Systematically analyzes the editing mechanisms of diffusion models from a frequency perspective, offering an intuitive understanding of the relationship between guidance signals and editing effects, which explains prior empirical findings.
2. **Simple and Elegant Method**: Only performs frequency filtering on the network output without modifying the network architecture, maintaining high generality.
3. **Unified Framework**: Deals with both rigid (object insertion/replacement) and non-rigid (pose changes) edits within a single framework, whereas prior works required distinct methods.
4. **Compatibility**: Compatible with various Stable Diffusion versions and theoretically applicable to any guidance-based diffusion model.

## Limitations & Future Work

1. **Bottlenecked by Base Model Priors**: If the denoising network cannot generate the desired layout (e.g., changing pose), the editing fail.
2. **Sensitivity to Prompts**: Full prompts containing descriptions of non-editing targets impair structural preservation.
3. **Two-Step Processing Needed for Color Edits**: Direct editing of color information via frequency truncation has limited effectiveness.
4. **Manual Hyperparameter Optimization**: Though default values are provided, achieving optimal results still requires fine-tuning based on user aesthetics.
5. **Cascading Effects of Inversion Failures**: Fixed-point DDIM inversion does not guarantee absolute correctness, and its failure directly impairs edit quality.

## Related Work & Insights

- **P2P / PNP / MasaCtrl**: Attention manipulation methods have distinct advantages but are hard to unify; FreeDiff offers a more general alternative.
- **Edit Friendly DDPM / AIDI**: Advancements in inversion technologies lay the foundation for precise editing.
- **Power Law of Natural Images (Field, 1997)**: Findings in classical vision science find new utility in modern generative models.
- **Insights**: Frequency-domain analysis could be equally valuable for other diffusion tasks (e.g., 3D generation, video editing).

## Rating

- **Novelty**: ★★★★☆ — Deep and novel analysis from a frequency perspective, opening up a new direction for editing research.
- **Value**: ★★★★☆ — Tuning-free, processes multiple edit types with a single model, offering high practical value.
- **Experimental Thoroughness**: ★★★☆☆ — The PIE dataset has flaws and required subset filtering; lacks larger-scale evaluation.
- **Writing Quality**: ★★★★★ — Rigorous theoretical analysis, excellent visualization, clear logic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] RegionDrag: Fast Region-Based Image Editing with Diffusion Models](regiondrag_fast_region-based_image_editing_with_diffusion_models.md)
- [\[ECCV 2024\] Unveiling Advanced Frequency Disentanglement Paradigm for Low-Light Image Enhancement](unveiling_advanced_frequency_disentanglement_paradigm_for_low-light_image_enhanc.md)
- [\[ECCV 2024\] FouriScale: A Frequency Perspective on Training-Free High-Resolution Image Synthesis](fouriscale_a_frequency_perspective_on_training-free_high-resolution_image_synthe.md)
- [\[ECCV 2024\] Lazy Diffusion Transformer for Interactive Image Editing](lazy_diffusion_transformer_for_interactive_image_editing.md)
- [\[ECCV 2024\] Robust-Wide: Robust Watermarking against Instruction-driven Image Editing](robust-wide_robust_watermarking_against_instruction-driven_image_editing.md)

</div>

<!-- RELATED:END -->

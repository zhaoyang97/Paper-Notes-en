---
title: >-
  [Paper Note] Zero-Reference Joint Low-Light Enhancement and Deblurring via Visual Autoregressive Modeling with VLM-Derived Modulation
description: >-
  [AAAI 2026][Multimodal VLM][Low-light enhancement] This paper proposes VAR-LIDE, a fully unsupervised visual autoregressive framework that jointly addresses low-light enhancement and deblurring through three modules guided by VLM perceptual priors: adaptive illumination modulation, spatial-frequency RoPE, and recursive phase-domain modulation. Without paired training data, the method achieves perceptual quality comparable to or exceeding supervised approaches.
tags:
  - AAAI 2026
  - Multimodal VLM
  - Low-light enhancement
  - deblurring
  - visual autoregressive model
  - VLM perceptual prior
  - unsupervised
  - frequency-domain phase modulation
date: 2026-05-08
content_hash: 10c9e7b20daa0fe3
---

# Zero-Reference Joint Low-Light Enhancement and Deblurring via Visual Autoregressive Modeling with VLM-Derived Modulation

**Conference**: AAAI 2026  
**arXiv**: [2511.18591](https://arxiv.org/abs/2511.18591)  
**Authors**: Wei Dong, Han Zhou, Junwei Lin, Jun Chen  
**Code**: [LowLevelAI/VAR-LIDE](https://github.com/LowLevelAI/VAR-LIDE)  
**Area**: Multimodal VLM  
**Keywords**: Low-light enhancement, deblurring, visual autoregressive model, VLM perceptual prior, unsupervised, frequency-domain phase modulation

## TL;DR

This paper proposes VAR-LIDE, a fully unsupervised visual autoregressive framework that jointly addresses low-light enhancement and deblurring through three modules guided by VLM perceptual priors: adaptive illumination modulation, spatial-frequency RoPE, and recursive phase-domain modulation. Without paired training data, the method achieves perceptual quality comparable to or exceeding supervised approaches.

## Background & Motivation

### Problem Definition

Images captured in real-world low-light conditions suffer from **triple degradation**: low visibility/contrast, complex sensor noise, and motion blur introduced by long exposure. Formally, the degradation process is described as:

$$\mathbf{x}_{LQ} = \gamma \, f(\mathbf{x}_{HQ}, \mathbf{k}) + \mathbf{n}$$

where $\gamma$ models dynamic range compression and underexposure, $f$ denotes convolution with blur kernel $\mathbf{k}$, and $\mathbf{n}$ represents sensor noise. While long exposure increases photon capture, it inevitably exacerbates motion blur and noise, severely degrading image quality.

### Three Bottlenecks in Prior Work

**Cascade failure from separate processing**: Conventional approaches treat low-light image enhancement (LLIE) and deblurring as independent tasks. LLIE methods (Zero-DCE, SCI, etc.) focus solely on brightness without addressing blur; deblurring methods (Blur2Blur, MambaIR, etc.) assume sufficient illumination. In cascaded pipelines, the enhancement stage may destroy blur cues, while the deblurring stage cannot recover motion details due to insufficient visibility—each constraining the other.

**Paired data bottleneck in supervised joint methods**: Joint methods such as LEDNet and DarkIR achieve strong results but rely heavily on paired training data, which is nearly impossible to acquire in real-world settings. This limits their generalization capability.

**Efficiency limitations of diffusion models**: The unsupervised diffusion method FourierDiff avoids the need for paired data but requires approximately 1,000 sampling iterations, making inference highly inefficient and impractical for real deployment.

### Two Key Pilot Experiments

The core motivation of this paper stems from two carefully designed pilot experiments:

**Experiment 1 — Denoising/deblurring potential of VAR**: Feeding images with noise and blur directly into a pretrained VARSR model reveals that it naturally possesses some noise suppression and blur mitigation capability. However, VAR cannot significantly improve image visibility nor recover fine structural details. This observation suggests that the VAR backbone is promising, but requires additional task-specific modules to compensate for its deficiencies in illumination enhancement and fine structure recovery.

**Experiment 2 — Adaptive failure of fixed-iteration Zero-DCE**: Zero-DCE adjusts illumination through iterative curve transformations, but fixed iteration counts (e.g., the default $N=8$) yield inconsistent results across varying lighting conditions: for very dark images, $n=4$ or $n=6$ iterations are far insufficient, causing underenhancement; for moderately lit images, $n=8$ iterations produce severe overexposure. This reveals a fundamental flaw—**a single fixed enhancement strength cannot adapt to diverse real-world lighting conditions**, necessitating an adaptive mechanism to dynamically determine the enhancement degree based on actual image visibility.

These two observations jointly motivate the three core module designs of VAR-LIDE.

## Method

### Overall Architecture

VAR-LIDE builds upon the pretrained VAR model (VARSR) as its backbone, augmented with three complementary plug-and-play modules:

1. **VLM-Informed Conditioning Module (VICM)**: Addresses adaptive illumination
2. **Spatial-Frequency RoPE (SF-RoPE)**: Enhances VAR's modeling of blur-degraded structures
3. **VLM-Guided Phase Modulation (VGPM)**: Eliminates phase-repetition artifacts caused by blur in the frequency domain

The full inference pipeline proceeds as follows: a low-quality image is fed in → the VLM perceptual prior pipeline extracts visibility score $v$ and blur score $b$ → VICM adaptively enhances illumination based on $v$ as a conditional input to VAR → the VAR backbone with SF-RoPE performs multi-scale autoregressive generation → VGPM recursively modulates the FFT phase domain guided by $b$ to eliminate residual blur → the high-quality image is output.

### Key Designs 1: VLM-Informed Conditioning Module (VICM)

The core idea of VICM is to **dynamically control the number of illumination enhancement iterations using VLM global perceptual scores**, replacing fixed parameter settings.

Zero-DCE models enhancement as iterative curve transformations:

$$\bm{E}_N(\mathbf{x}) = \mathbf{x} + \sum_{n=1}^{N} \bm{\mathcal{A}}_n(\mathbf{x}) \cdot \bm{E}_{n-1}(\mathbf{x}) \cdot (1 - \bm{E}_{n-1}(\mathbf{x}))$$

VICM introduces an **adaptive truncation** mechanism: the visibility score $v$ is first obtained from the VLM via the GPP-LLIE pipeline, then mapped to the optimal iteration count $n_v$ through a lightweight MLP $\boldsymbol{\Theta}_v$. The curve estimator $\bm{\Psi}_I$ still produces $N$ sets of curve parameters $\{\bm{\mathcal{A}}_n(\mathbf{x})\}_{n=1}^N$, but parameters beyond $n_v$ are set to zero:

$$\bm{\mathcal{A}}_j(\mathbf{x}) = 0, \quad \forall j > n_v$$

This adaptive truncation ensures that very dark images receive sufficient enhancement (larger $n_v$) while moderately lit images avoid overexposure (smaller $n_v$), keeping enhancement results within a perceptually valid range. The enhanced image is then embedded and tokenized as the conditional input to the VAR model. Compared to Zero-DCE's fixed pipeline, the conditioning signal provided by VICM is spatially more adaptive, significantly improving downstream generation quality.

### Key Designs 2: Content-Aware Spatial-Frequency RoPE (SF-RoPE)

SF-RoPE aims to enhance the sensitivity of the attention mechanism in the VAR backbone to blur-degraded regions. The standard RoPE used in VARSR relies solely on fixed positional-index rotation matrices, lacking awareness of content degradation. SF-RoPE incorporates **frequency-domain phase information** into positional encoding, enabling content-adaptive attention modulation.

**Frequency RoPE**: At scale $K$, frequency-domain phase information $\Phi(u,v) = \arg(\text{FFT}(\mathbf{x}_{K-1}))$ is extracted from the embedding $\mathbf{x}_{K-1}$ via FFT, constructing a phase-driven rotation matrix $\mathbf{R}_{\Phi(u,v)}$ so that each token's positional encoding directly reflects local blur-sensitive frequency features.

**Spatial RoPE**: In parallel, standard RoPE is computed using scale-normalized spatial coordinates, ensuring positional consistency across multiple resolutions.

**Adaptive fusion**: The two encodings are fused via a learnable mixing coefficient $\lambda$:

$$\text{RoPE}_{\text{fused}} = \lambda \cdot \text{RoPE}_{\text{freq}} + (1-\lambda) \cdot \text{RoPE}_{\text{spa}}$$

This fusion enables the attention module to maintain global positional alignment while applying finer attention to blur-degraded regions. Experiments demonstrate that SF-RoPE yields sharper edge recovery and improved spatial coherence.

### Key Designs 3: VLM-Guided Phase Modulation (VGPM)

VGPM targets the **repetitive edge artifacts** that appear in the FFT phase domain of blurry images. Phase information is more robust than spatial features to occlusion ambiguity and illumination noise, making it particularly suited for blur suppression in this domain.

The phase is first normalized as $\hat{\phi} = (\phi + \pi) / 2\pi \in [0,1]$, followed by recursive enhancement:

$$\bm{M}_T(\hat{\phi}) = \hat{\phi} + \sum_{t=1}^{T} \bm{\mathcal{F}}_t(\hat{\phi}) \cdot M_{t-1}(\hat{\phi}) \cdot (1 - \bm{M}_{t-1}(\hat{\phi}))$$

where $T=8$ is the number of modulation steps and $\bm{\mathcal{F}}_t \in [0,1]$ is predicted by the phase estimator $\bm{\Psi}_p$. The VLM blur score $b$ adaptively guides the modulation strength through MLP $\boldsymbol{\Theta}_b$. The refined phase is then inverse-transformed back to the spatial domain to produce the output image. This module effectively eliminates ghosting artifacts caused by blur.

### Loss & Training

Fully unsupervised training with joint optimization of all parameters:

$$\mathcal{L} = \mathcal{L}_{ex} + \lambda_{en}\mathcal{L}_{en} + \lambda_{con}\mathcal{L}_{con} + \lambda_{tv}\mathcal{L}_{tv}$$

- **Adaptive exposure control loss** $\mathcal{L}_{ex}$: Uses $E=0.45$ as a reference, dynamically adjusting $E_d \in [-0.1, 0.1]$ to constrain the mean brightness of the output
- **Structural entropy loss** $\mathcal{L}_{en}$: Computes Shannon entropy over the inverse FFT magnitude of the refined phase to promote structural fidelity
- **Structural contrastive loss** $\mathcal{L}_{con}$: Computes the negative mean of patch-wise variance across 16 patches to enhance local structural discriminability
- **Total variation loss** $\mathcal{L}_{tv}$: Suppresses spatial noise and artifacts

## Key Experimental Results

### Quantitative Comparison on LOLBlur Dataset

| Method | Type | PSNR↑ | NIQE↓ | LPIPS↓ | FID↓ | CLIPIQA↑ |
|--------|------|-------|-------|--------|------|----------|
| EnlightenGAN + Blur2Blur | L+D | 18.16 | 5.02 | 0.396 | 45.73 | 0.206 |
| SSFlow* | w/o R | 19.24 | 5.93 | 0.307 | 42.05 | 0.183 |
| FourierDiff | w/o R | 20.22 | 4.97 | 0.441 | 50.59 | 0.161 |
| LEDNet | w R | 24.36 | 5.37 | 0.227 | 25.19 | 0.207 |
| DarkIR-L | w R | 26.14 | 5.15 | 0.146 | 14.27 | 0.291 |
| LIEDNet-L | w R | 26.42 | 5.17 | 0.127 | 11.38 | 0.305 |
| **Ours** | **w/o R** | **23.39** | **4.80** | **0.191** | **26.04** | **0.262** |

**Key Findings**: Among unsupervised methods, VAR-LIDE substantially outperforms SSFlow (PSNR +4.15 dB) and FourierDiff (PSNR +3.17 dB), and achieves the best NIQE score across all methods (4.80). Despite using no ground-truth supervision, PSNR approaches supervised LEDNet (23.39 vs. 24.36), and perceptual quality metrics even surpass several supervised methods.

### Generalization Evaluation on Real-LOLBlur Dataset

| Method | Type | NIQE↓ | CLIPIQA↑ | MUSIQ↑ | MANIQA↑ |
|--------|------|-------|----------|--------|---------|
| SCI + Blur2Blur | L+D | 5.13 | 0.185 | 33.87 | 0.129 |
| SSFlow | w/o R | 5.94 | 0.190 | 30.93 | 0.148 |
| FourierDiff | w/o R | 5.59 | 0.187 | 32.01 | 0.122 |
| JUDE | w R | 4.92 | 0.236 | 50.29 | 0.223 |
| DarkIR-L | w R | 4.90 | 0.262 | 48.72 | 0.216 |
| **Ours** | **w/o R** | **5.16** | **0.226** | **47.53** | **0.223** |

**Key Findings**: VAR-LIDE demonstrates strong generalization on the real unpaired dataset. MUSIQ reaches 47.53 (vs. SSFlow 30.93, FourierDiff 32.01), approaching or matching supervised methods DarkIR-L (48.72) and JUDE (50.29), demonstrating that VLM perceptual priors confer exceptional cross-dataset generalization.

## Highlights & Insights

1. **VLM as an adaptive modulator paradigm**: Unlike using VLMs for semantic understanding or image captioning, this paper reduces VLM scores to scalar visibility/blur values that serve as control signals for low-level image processing modules. This "VLM → scalar score → module parameters" pipeline is lightweight yet effective, and merits generalization to other image restoration tasks.

2. **Elegant solution via Zero-DCE iteration truncation**: Rather than designing a new enhancement network, the authors retain the Zero-DCE curve estimator and achieve adaptability solely through dynamic iteration truncation—a minimalist modification strategy that balances simplicity and effectiveness.

3. **FFT phase domain as the entry point for deblurring**: Operating in the FFT phase domain more directly addresses the repetitive edge phenomenon caused by blur than recovering sharp structures in the spatial domain. The robustness of phase representations to illumination variation also makes them particularly suited to low-light scenarios.

4. **Unsupervised performance approaching supervised**: On Real-LOLBlur, the method's MANIQA score (0.223) matches supervised JUDE (0.223), and the MUSIQ gap is only approximately 5%, fully validating the feasibility of replacing paired ground truth with VLM priors.

## Limitations & Future Work

1. **Additional overhead from VLM dependency**: Although VLM score extraction requires only a single forward pass at inference time, the GPP-LLIE pipeline internally invokes a VLM (e.g., CLIP-based models), introducing additional model loading and computational overhead that is unfriendly to resource-constrained deployment.

2. **Remaining PSNR gap**: Compared to the best supervised method (LIEDNet-L, 26.42 dB), the proposed method (23.39 dB) still trails by approximately 3 dB. Performance may be insufficient for applications requiring precise pixel-level reconstruction.

3. **Absence of truly extreme scenario testing**: Experiments are validated only on LOLBlur/Real-LOLBlur; robustness under more extreme degradation combinations such as high-speed nighttime motion or ultra-long exposure remains unexplored.

4. **Missing ablation study details**: A complete ablation study quantifying the independent contribution of each module should be present in the paper; this section is absent from the available materials.

## Related Work & Insights

- **GPP-LLIE**: Provides the VLM perceptual prior extraction pipeline; this paper directly reuses its scoring framework
- **VARSR**: Pioneering work applying VAR to image super-resolution, providing the SA-RoPE and pretrained backbone
- **Zero-DCE / Zero-DCE++**: Lightweight reference-free LLIE baseline; VICM is its adaptive extension
- **FourierDiff**: The primary unsupervised diffusion baseline; the VAR-based approach holds a significant efficiency advantage (no 1,000-step sampling required)
- **Transferable directions**: VLM-score-guided adaptive parameter control is transferable to other degradation scenarios such as deraining and dehazing; the frequency-aware positional encoding concept of SF-RoPE can be extended to inter-frame video alignment

## Rating

⭐⭐⭐⭐ (4/5)

**Rationale**: The method is clearly motivated, the module combination is well-reasoned, and the integration of VLM perceptual priors is elegant and practical. Achieving near-supervised performance under a fully unsupervised setting demonstrates significant practical value. One point is deducted for the remaining PSNR gap and the added system complexity introduced by VLM dependency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[AAAI 2026\] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semantic-sensitive_underwater_image_enhancement_with_vlm.md)
- [\[NeurIPS 2025\] Unifying Vision-Language Latents for Zero-Label Image Caption Enhancement](../../NeurIPS2025/multimodal_vlm/unifying_vision-language_latents_for_zero-label_image_caption_enhancement.md)
- [\[AAAI 2026\] BOFA: Bridge-Layer Orthogonal Low-Rank Fusion for CLIP-Based Class-Incremental Learning](bofa_bridge-layer_orthogonal_low-rank_fusion_for_clip-based_.md)
- [\[ICLR 2026\] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play](../../ICLR2026/multimodal_vlm/vision-zero_scalable_vlm_self-improvement_via_strategic_gamified_self-play.md)

</div>

<!-- RELATED:END -->

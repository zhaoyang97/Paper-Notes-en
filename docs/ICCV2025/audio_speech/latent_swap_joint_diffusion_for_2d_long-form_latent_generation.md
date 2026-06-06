---
title: >-
  [Paper Note] Latent Swap Joint Diffusion for 2D Long-Form Latent Generation
description: >-
  [ICCV 2025][Audio & Speech][Diffusion Models] This paper proposes SaFa (Swap Forward), a modality-agnostic and efficient method that replaces the averaging operation in conventional joint diffusion with two latent swap o…
tags:
  - "ICCV 2025"
  - "Audio & Speech"
  - "Diffusion Models"
  - "Long Audio Generation"
  - "Panorama Generation"
  - "Joint Diffusion"
  - "Latent Swap"
date: 2026-05-08
content_hash: ff3c867ebc48cd59
---

# Latent Swap Joint Diffusion for 2D Long-Form Latent Generation

**Conference**: ICCV 2025
**arXiv**: [2502.05130](https://arxiv.org/abs/2502.05130)  
**Code**: [https://swapforward.github.io](https://swapforward.github.io)  
**Area**: Audio / Image Generation
**Keywords**: Diffusion Models, Long Audio Generation, Panorama Generation, Joint Diffusion, Latent Swap

## TL;DR

This paper proposes SaFa (Swap Forward), a modality-agnostic and efficient method that replaces the averaging operation in conventional joint diffusion with two latent swap operators—Self-Loop Latent Swap and Reference-Guided Latent Swap—to address spectrum aliasing and preserve cross-view consistency, achieving significant improvements over existing methods in both long audio and panoramic image generation.

## Background & Motivation

Diffusion models have demonstrated strong performance in text-to-image and text-to-audio generation, yet they face the challenge of *length extrapolation*—how to generate images of arbitrary shape or audio of arbitrary length using a model trained on fixed-size inputs. Joint Diffusion methods synchronize the denoising processes across multiple sub-views to enable long-form content generation, but suffer from two core problems:

**Spectrum Aliasing**: Applying existing joint diffusion methods (e.g., step-wise averaging in MultiDiffusion) to spectrogram-based audio generation causes severe time-frequency resolution degradation and distortion in overlapping regions—manifesting as white stripes, visual blurring, and audio artifacts. This is particularly pronounced for spectrally rich audio such as soundscapes and concertos.

**Cross-View Inconsistency**: The lack of global consistency guidance between distant sub-views leads to incoherent color, style, or timbre across views.

**Key Challenge**: Through Connectivity Inheritance analysis of VAE latent space and Fourier frequency analysis, the paper identifies that spectrum aliasing originates from the averaging operation excessively suppressing high-frequency components during denoising. Unlike RGB images, VAE latent representations of spectrograms exhibit inherently high-frequency variation, and averaging directly destroys these fine spectral details.

## Method

### Overall Architecture

SaFa achieves long-form content generation via two fundamental latent swap operators: (1) Self-Loop Latent Swap for smooth transitions in overlapping regions between adjacent sub-views, and (2) Reference-Guided Latent Swap for cross-view consistency in non-overlapping regions. The entire process operates in a feed-forward manner, requiring neither additional gradient optimization nor attention window extension.

### Key Designs

1. **Connectivity Inheritance and Spectrum Aliasing Analysis**: The paper demonstrates that a channel-wise linear approximation exists between VAE latent representations and original features: $\text{Downsample}(X) \approx W_c \cdot Z$, where $W_c \in \mathbb{R}^{C_x \times C_z}$ is a learnable constant linear mapping. This implies that the connectivity and structural properties of the original features are inherited in the latent space—including the high-frequency variability, sparsity, and discontinuity of spectral features. 2D Fourier analysis further confirms that in non-overlapping reference regions, the relative amplitude curves of spectrogram latents exhibit dynamic fluctuation with no significant attenuation of high-frequency components; however, in overlapping regions subject to step-wise averaging, high-frequency components are progressively smoothed—especially in late denoising steps—causing spectral detail loss and aliasing.

2. **Self-Loop Latent Swap**: This operator exploits the *step-wise differentiated trajectory* property: overlapping regions of adjacent sub-views diverge after each denoising step due to the influence of their respective non-overlapping regions, yet remain similar due to sharing the initial latent from the previous step. A binary swap operator $W_{swap}$ replaces averaging to perform bidirectional frame-level exchange: $I_{i,i+1}(J_t) = W_{swap} \odot \text{Right}(X_t^i) + (1 - W_{swap}) \odot \text{Left}(X_t^{i+1})$. The swap interval $w$ controls which frequency components are enhanced: $v_m^{(i)} = \frac{1}{2}[1 - (-1)^{\lfloor\frac{i-1}{w}\rfloor}]$, with the optimal setting being $w=1$ (frame-level swap). This hard combination leverages the similarity of differentiated trajectories for stability while adaptively enhancing specific frequency components. Swaps are applied cyclically across all adjacent sub-view pairs (including head and tail), forming a self-loop.

3. **Reference-Guided Latent Swap**: During the first $r_{guide} \times T$ denoising steps, a unidirectional frame-level swap is applied from an independent reference trajectory $X_t^0$ to the non-overlapping regions of each sub-view: $M_i(J_t) = W_{refer} \odot \text{Mid}(X_t^0) + (1 - W_{refer}) \odot \text{Mid}(X_t^i)$. This centralized reference trajectory synchronizes the diffusion processes across sub-views, ensuring global consistency while avoiding repetition (since guidance is not applied in later steps). The parameter $r_{guide}$ (default 0.3) balances similarity and diversity. For image generation, given that 1D token sequences are flattened in row-major order, row-wise swaps are adopted for segment-level mixing to avoid excessive similarity from pixel-wise exchange.

### Loss & Training

SaFa is entirely training-free and operates at inference time without any fine-tuning. The two swap operators are applied directly on top of pretrained text-to-audio/image diffusion models. Experiments use a DDIM sampler (200 steps) with CFG=3.5. Unlike SyncDiffusion, which requires gradient optimization, SaFa operates in a purely feed-forward manner.

## Key Experimental Results

### Main Results

**Long Audio Generation (DiT model, 24-second generation):**

| Method | FD↓ | FAD↓ | KL↓ | CLAP↑ | I-LPIPS↓ | I-CLAP↑ |
|--------|-----|------|-----|-------|----------|---------|
| Reference | 2.92 | 0.22 | 0.74 | 0.54 | 0.39 | 0.86 |
| MAD | 12.77 | 7.56 | 0.86 | 0.51 | 0.32 | 0.93 |
| MD | 11.31 | 6.41 | 0.81 | 0.51 | 0.36 | 0.91 |
| MD* | 9.79 | 5.09 | 0.77 | 0.52 | 0.36 | 0.92 |
| **SaFa** | **6.84** | **4.91** | **0.73** | **0.54** | **0.34** | **0.95** |

**Panoramic Image Generation (SD 3.5 DiT, 512×3200):**

| Method | FID↓ | KID↓ | CLIP↑ | I-StyleL↓ | I-LPIPS↓ | Runtime↓ |
|--------|------|------|-------|-----------|----------|----------|
| MD | 24.50 | 8.12 | 32.37 | 2.58 | 0.59 | 103.85s |
| SyncD | 24.25 | 8.07 | 32.36 | 2.54 | 0.57 | 623.59s |
| MAD | 65.10 | 55.73 | 31.79 | 0.67 | 0.47 | 85.25s |
| **SaFa** | **22.54** | **4.53** | **32.45** | **1.36** | **0.56** | **49.54s** |

SaFa is approximately 12.5× faster than SyncDiffusion while substantially outperforming MAD in quality.

### Ablation Study

| Configuration | Key Metrics | Notes |
|---------------|-------------|-------|
| SaFa* (Self-Loop Swap only) | FD 6.98, I-LPIPS 0.36 | Effectively resolves aliasing; cross-view consistency slightly weaker |
| SaFa (full) | FD 6.84, I-StyleL 1.36 | Reference-Guided Swap further improves global consistency |
| Extension to 72s | FD 6.98, CLAP 0.54 | Performance remains stable |
| SaFa on U-Net vs. DiT | Best performance on both | Architecture-agnostic |
| MAD on DiT | FID 65.10 | Severe degradation due to positional encoding repetition |
| $r_{guide}=0.3$ | Best similarity–diversity trade-off | Default setting |
| $w=1$ (frame-level swap) | Smoothest transition | Optimal swap interval |

### Key Findings

- The averaging operation is the direct cause of spectrum aliasing—Fourier analysis clearly demonstrates its progressive suppression of high-frequency components.
- The latent swap operators adaptively restore high-frequency details by exploiting the divergence of differentiated trajectories, recovering frequency distributions comparable to non-overlapping regions.
- SaFa surpasses even training-based methods in audio generation (vs. AudioGen, Stable Audio) without requiring any additional training.
- MAD degrades severely on DiT architectures due to positional encoding repetition introduced by attention window extension—a problem SaFa entirely avoids.
- Reference-Guided Swap can be interpreted as a frame-level Blended Diffusion, achieving global style synchronization while preserving local coherence.
- SaFa requires an overlap rate of only 0.2 (far below the 0.8 typical for MD-based methods), substantially reducing the number of sub-views and computational overhead.

## Highlights & Insights

- The Connectivity Inheritance analysis of VAE latent space and the identification of the root cause of spectrum aliasing hold independent academic value.
- Replacing averaging with simple binary swapping is intuitively inelegant yet highly effective—exploiting the inherent stability of the diffusion process.
- The triple generality—modality-agnostic (audio + image), architecture-agnostic (U-Net + DiT), and training-free—makes the method exceptionally practical.
- The efficiency advantage is substantial: 2–20× speedup with simultaneously superior quality.

## Limitations & Future Work

- Applicability to 1D wave-based VAE latent representations or discrete token representations remains to be validated.
- Reference-Guided Swap relies on a single reference trajectory, which may constrain content diversity in semantically heterogeneous panoramas.
- The optimal choices of swap interval $w$ and guidance ratio $r_{guide}$ still require task-specific tuning.
- Extension to higher-dimensional long-form generation tasks such as video generation remains unexplored.

## Related Work & Insights

- Relative to MultiDiffusion and SyncDiffusion, SaFa addresses the gap in joint diffusion for spectrogram-based generation.
- The latent swap concept is generalizable to other diffusion-based generation tasks requiring spatial or temporal consistency, such as video and 3D texture synthesis.
- The Connectivity Inheritance finding offers insight into how information preservation properties are encoded by VAEs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ In-depth analysis of spectrum aliasing root causes; the idea of replacing averaging with latent swapping is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Dual modalities (audio + image), dual architectures (U-Net + DiT), multiple lengths, and user studies.
- **Writing Quality**: ⭐⭐⭐⭐ Thorough analysis and rich visualizations, though notation is dense.
- **Value**: ⭐⭐⭐⭐⭐ Training-free and plug-and-play, with both efficiency and quality advantages; extremely high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Space Factorization in LoRA](../../NeurIPS2025/audio_speech/latent_space_factorization_in_lora.md)
- [\[ACL 2026\] Comprehensive Benchmarking of Long-Form Speech Generation in Diverse Scenarios](../../ACL2026/audio_speech/comprehensive_benchmarking_of_long-form_speech_generation_in_diverse_scenarios.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](../../NeurIPS2025/audio_speech/multi-head_temporal_latent_attention.md)
- [\[ACL 2026\] PlanRAG-Audio: Planning and Retrieval Augmented Generation for Long-form Audio Understanding](../../ACL2026/audio_speech/planrag-audio_planning_and_retrieval_augmented_generation_for_long-form_audio_un.md)
- [\[ICLR 2026\] Latent Speech-Text Transformer](../../ICLR2026/audio_speech/latent_speech_text_transformer.md)

</div>

<!-- RELATED:END -->

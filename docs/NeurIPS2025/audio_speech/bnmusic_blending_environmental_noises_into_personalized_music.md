---
title: >-
  [Paper Note] BNMusic: Blending Environmental Noises into Personalized Music
description: >-
  [NeurIPS 2025][Audio & Speech][noise blending] This paper proposes BNMusic, a two-stage framework that blends environmental noises into personalized generated music. Stage 1 generates rhythm-aligned music via mel-spectro…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "noise blending"
  - "auditory masking"
  - "music generation"
  - "spectrogram inpainting"
  - "psychoacoustics"
date: 2026-05-08
content_hash: 07b6facaaacb2401
---

# BNMusic: Blending Environmental Noises into Personalized Music

**Conference**: NeurIPS 2025
**arXiv**: [2506.10754](https://arxiv.org/abs/2506.10754)  
**Code**: [https://d-fas.github.io/BNMusic_page/](https://d-fas.github.io/BNMusic_page/)  
**Area**: Audio Generation
**Keywords**: noise blending, auditory masking, music generation, spectrogram inpainting, psychoacoustics

## TL;DR
This paper proposes BNMusic, a two-stage framework that blends environmental noises into personalized generated music. Stage 1 generates rhythm-aligned music via mel-spectrogram outpainting and inpainting; Stage 2 adaptively amplifies the music signal based on auditory masking theory to reduce noise perception. The approach requires no additional training and significantly outperforms baselines on EPIC-SOUNDS and ESC-50.

## Background & Motivation

**Background**: In public environments such as subways and elevators, people are frequently disturbed by persistent ambient noise. Active Noise Cancellation (ANC) is effective but requires personal devices, making it unsuitable for group settings. Traditional acoustic masking demands high volume and often fails to align the masker with the noise.

**Limitations of Prior Work**: (1) ANC is limited to individual use and cannot serve groups; (2) misalignment between masker and noise (e.g., beat mismatch) in traditional masking necessitates excessively high volume for effectiveness; (3) existing music generation models perform poorly on noisy inputs, as they are trained on clean audio.

**Key Challenge**: Effectively reducing noise perception at a comfortable volume requires the masker (music) to be aligned with the noise in both rhythm and spectral characteristics.

**Goal**: Given environmental noise and a user text prompt, generate music whose rhythm and spectral features are aligned with the noise, so that the noise is perceptually absorbed into the music.

**Key Insight**: The framework draws on psychoacoustic auditory masking theory — when the frequency-temporal characteristics of music are aligned with the noise, effective masking can be achieved at a lower signal-to-noise ratio. Mel-spectrogram-domain image generation techniques (inpainting/outpainting) are leveraged to produce such aligned music.

**Core Idea**: The high-energy regions of the noise spectrogram serve as a "canvas." Spectrogram outpainting extends musical patterns around these regions, inpainting then reconstructs the high-energy regions, and adaptive amplification further enhances the masking effect.

## Method

### Overall Architecture
**Input**: noise audio + text prompt. **Preprocessing**: convert to mel-spectrogram and create a binary mask for high-energy regions. **Stage 1**: dual-step outpainting + inpainting to generate a noise-aligned music mel-spectrogram. **Stage 2**: adaptive amplification guided by masking thresholds. **Output**: music audio suitable for blending with the noise.

### Key Designs

1. **Stage 1: Noise-Aligned Music Synthesis**

    - **Function**: Generate music whose rhythm and spectral content are aligned with the input noise.
    - **Mechanism**: (1) *Preprocessing* — identify high-energy regions in the noise mel-spectrogram using a binary mask. (2) *Outpainting* — preserve high-energy noise regions and generate surrounding musical content, allowing core noise information to diffuse outward into the musical texture. (3) *Inpainting* — invert the mask and reconstruct musical content within the high-energy regions, reintegrating the previously diffused musical information back into the core area. This ensures the generated music fully inherits the rhythmic characteristics of the noise.
    - **Design Motivation**: Directly generating music from noise yields poor results, as models have not been trained on noisy inputs. The two-step approach uses outpainting to first establish a musical framework and then inpainting to fill the core regions — analogous to an "outside-in" compositional strategy.

2. **Stage 2: Adaptive Amplification**

    - **Function**: Maximize masking effectiveness while maintaining a comfortable listening volume.
    - **Mechanism**: A psychoacoustic model computes the masking threshold matrix $\mathbf{T}_\text{Mask}$ (the energy required at each frequency-time bin to mask the noise). Gradient descent is then used to find the optimal amplification factor $\lambda^*$, balancing maximizing masking coverage against minimizing overall volume: $\lambda^* = \arg\min_\lambda \{\text{SUM}(\alpha \cdot \mathbf{S}'_\text{Music}) + \text{SUM}(\max[(\mathbf{T}_\text{Mask} - \mathbf{S}'_\text{Music}) \odot \mathbf{M}, 0])\}$.
    - **Design Motivation**: Because Stage 1 already aligns the spectral features, only moderate amplification is needed for effective masking, avoiding the brute-force volume-raising strategy of traditional masking approaches.

3. **Training-Free Utilization of Existing Models**

    - **Function**: Implemented on top of Riffusion (a music-finetuned variant of Stable Diffusion) without any additional training.
    - **Mechanism**: The noise mel-spectrogram is used as the input for image editing, leveraging Riffusion's inpainting/outpainting capabilities.
    - **Design Motivation**: This avoids the cost of collecting noise–music paired training data and exploits the generalization ability of existing generative models.

### Loss & Training
No training is required. Stage 1 uses Riffusion's LDM inference (DDPM denoising). Stage 2 applies gradient descent to optimize the amplification factor $\lambda$. The global hyperparameter is set to $\alpha = 0.14$. Inference takes approximately 5 seconds per sample on an Nvidia 4090.

## Key Experimental Results

### Main Results

| Method | FAD (EPIC) | KL (EPIC) | FAD (ESC-50) | KL (ESC-50) |
|--------|-----------|---------|------------|-----------|
| Noise Only | 34.17 | - | 27.39 | - |
| Random Music | 14.22 | 2.22 | 8.45 | 2.49 |
| MusicGen | 13.28 | 2.14 | 8.62 | 2.43 |
| Riffusion A2A | 20.06 | 2.90 | 12.62 | 3.26 |
| **BNMusic** | **12.86** | **2.03** | **8.09** | **2.38** |

### Ablation Study

| Component | FAD | KL |
|-----------|-----|----|
| No processing | 34.17 | - |
| Outpainting only | Improved | Improved |
| + Inpainting | Further improved | Further improved |
| + Adaptive amplification | **Best** | **Best** |

### Subjective Evaluation

| Method | OVL (Overall Quality) | PER (Noise Perception) |
|--------|-----------------------|------------------------|
| Random Music | 2.93 | 2.63 |
| MusicGen | 2.97 | 2.68 |
| Riffusion A2A | 2.95 | 3.24 |
| **BNMusic** | **3.67** | **3.84** |

### Key Findings
- **BNMusic consistently outperforms all baselines on both objective and subjective metrics**, with gains of +0.7 in OVL and +0.6 in PER (on a 5-point scale).
- **Riffusion A2A suppresses noise effectively but yields poor musicality** (highest FAD = lowest music quality), as its outputs resemble noise too closely.
- **Each component contributes**: outpainting establishes the musical framework, inpainting ensures coherence, and adaptive amplification strengthens masking.
- **MusicGen, despite its melodic generation capability, achieves limited noise blending** — indicating that effective blending requires not only melodic alignment but also spectral alignment.
- **BNMusic produces the most uniform spectral difference heatmaps**, reflecting the closest alignment with the noise energy distribution.

## Highlights & Insights
- **Reformulating noise masking as an image editing problem** is the core innovation: outpainting and inpainting in the mel-spectrogram domain naturally correspond to the concept of "building music around noise."
- **The two-step design (outpainting → inpainting)** is elegant: noise rhythmic information first diffuses into the surrounding musical texture, and musical content is then filled back into the noise-dominated regions — analogous to a "simple-to-complex" compositional process.
- **Psychoacoustically grounded adaptive amplification**, guided by SMR thresholds, ensures a scientifically principled masking strategy.

## Limitations & Future Work
- **Dependence on Riffusion's generation quality**: Riffusion has limited capability for complex musical genres. Stronger music generation models could be explored in future work.
- **Restricted to repetitive noise**: The rhythm-alignment strategy may fail for impulsive or non-periodic noise.
- **Real-time processing**: The current pipeline requires approximately 5 seconds of processing time, which is insufficient for real-time scenarios such as dynamic noise in moving trains.
- **Mono audio only**: Spatial audio and multi-channel scenarios are not addressed.

## Related Work & Insights
- **vs. ANC**: ANC eliminates noise (for individual users), while BNMusic blends noise (for group settings) — the two approaches are complementary rather than mutually exclusive.
- **vs. Traditional acoustic masking**: Traditional methods use fixed sounds as maskers; BNMusic generates personalized music aligned with the noise, requiring lower playback volume.
- **vs. AudioLDM/MusicGen**: These models generate music from clean inputs; BNMusic innovatively conditions generation on noisy inputs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Novel task definition (noise blending), interdisciplinary approach (psychoacoustics + generative models)
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive objective and subjective evaluation with complete ablation study
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and well-explained psychoacoustic principles
- Value: ⭐⭐⭐⭐ — Pioneering work on a novel task, though practical deployment remains challenging

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Echoes of Humanity: Exploring the Perceived Humanness of AI Music](echoes_of_humanity_exploring_the_perceived_humanness_of_ai_music.md)
- [\[NeurIPS 2025\] Ethics Statements in AI Music Papers: The Effective and the Ineffective](ethics_statements_in_ai_music_papers_the_effective_and_the_ineffective.md)
- [\[NeurIPS 2025\] Unifying Symbolic Music Arrangement: Track-Aware Reconstruction and Structured Tokenization](unifying_symbolic_music_arrangement_track-aware_reconstruction_and_structured_to.md)
- [\[NeurIPS 2025\] LUMIA: A Handheld Vision-to-Music System for Real-Time, Embodied Composition](lumia_a_handheld_vision-to-music_system_for_real-time_embodied_composition.md)
- [\[NeurIPS 2025\] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)

</div>

<!-- RELATED:END -->

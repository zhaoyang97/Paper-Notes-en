---
title: >-
  [Paper Note] Latent Fourier Transform
description: >-
  [ICLR 2026 Oral][diffusion autoencoder] This paper proposes LatentFT, a framework that applies the Discrete Fourier Transform (DFT) to latent time-series representations produced by a diffusion autoencoder…
tags:
  - "ICLR 2026 Oral"
  - "diffusion autoencoder"
  - "Fourier transform"
  - "music generation"
  - "latent frequency"
  - "timescale control"
  - "controllable generation"
date: 2026-05-08
content_hash: 9c317c465be48073
---

# Latent Fourier Transform

**Conference**: ICLR 2026 Oral
**OpenReview**: [ogMxCjdCCq](https://openreview.net/forum?id=ogMxCjdCCq)
**Code**: Available  
**Area**: Others
**Keywords**: diffusion autoencoder, Fourier transform, music generation, latent frequency, timescale control, controllable generation

## TL;DR
This paper proposes LatentFT, a framework that applies the Discrete Fourier Transform (DFT) to latent time-series representations produced by a diffusion autoencoder, decomposing musical patterns by timescale. During training, a correlated log-scale frequency mask is randomly applied so that the decoder learns to reconstruct audio from partial spectra. At inference time, users specify frequency masks to selectively preserve or blend musical elements across different timescales. LatentFT consistently outperforms baselines including ILVR, Guidance, Codec Filtering, and RAVE on conditional generation and music blending tasks, with its superior audio quality and blending capability statistically confirmed by a listening test involving 29 musicians.

## Background & Motivation
**Background**: Music generation models (e.g., diffusion-based autoencoders, language models) can already produce high-quality music, but lack fine-grained control over musical structure at different timescales. Music exhibits multi-scale structure—chord progressions (~0.5 Hz), melodic lines (~2–5 Hz), rhythmic patterns (~8 Hz)—each evolving at a different rate along the time axis.

**Limitations of Prior Work**: Existing control methods (text prompts, style transfer, guidance) operate at a global level and cannot independently manipulate multi-scale aspects such as "preserve the chord progression of this piece while replacing the rhythm." In deep multi-scale representations (e.g., different layers of a UNet), coarse- and fine-grained information are entangled, making independent manipulation difficult.

**Key Challenge**: Music generation requires decoupled control at different timescales, yet no natural, continuous, and interpretable control axis currently exists for this purpose.

**Goal**: Provide a novel control axis grounded in the latent frequency domain, enabling users to manipulate musical structure in a manner analogous to adjusting timbral qualities with an equalizer.

**Key Insight**: Exploit the natural frequency-decomposition property of the Fourier transform—different frequency components are orthogonal and can be modified independently without mutual interference—and apply it to learned latent time series.

**Core Idea**: Apply DFT in the latent space and use frequency masks to control music generation—an "equalizer for musical structure."

## Method

### Overall Architecture
LatentFT consists of three components: (1) an encoder that maps audio to a latent time series $\mathbf{z} \in \mathbb{R}^{C \times T'}$ ($C=80$ channels, $T'=512$ frames, corresponding to 5.9 seconds of audio); (2) a Latent DFT layer that computes the DFT of $\mathbf{z}$ along the time axis and applies a frequency mask $\mathbf{M}$; and (3) a diffusion decoder that reconstructs full audio from the masked latent spectrum $\mathbf{z}_{\text{masked}} = \text{IDFT}(\text{DFT}(\mathbf{z}) \odot \mathbf{M})$. The entire system is trained end-to-end.

### Key Designs
1. **Latent Discrete Fourier Transform (Latent DFT)**:
    - Function: Applies DFT along the time axis of the encoder's output $\mathbf{z}$, yielding the latent spectrum $\mathbf{Z} = \text{DFT}(\mathbf{z})$.
    - Mechanism: The latent frame rate is $f_r \approx 86$ Hz, giving a latent Nyquist frequency of ~43 Hz. Frequency bin $k$ corresponds to latent frequency $f_k = k \cdot f_r / T'$. Zero-padding with factor $L=2$ increases frequency resolution to 1024 bins.
    - Design Motivation: The orthogonality of DFT guarantees that modifying one frequency component does not affect others, making it more stable than alternatives such as bandpass filters (experiments show bandpass training is unstable and requires a reduced learning rate, whereas DFT masking is equivalent to ideal bandpass filtering with periodic padding, avoiding edge artifacts).

2. **Correlated Log-Scale Frequency Mask Training Strategy**:
    - Function: During training, a binary random mask $\mathbf{M}$ is applied to the latent spectrum. The mask generation procedure incorporates correlation and log-scale transformation, causing adjacent frequency bins to be masked or retained together, with higher-frequency bin groups spanning wider ranges.
    - Mechanism: A correlation matrix $\mathbf{K} \in \mathbb{R}^{F \times F}$ is computed across frequency bins (based on log-scale distances); independent random scores are then weighted by $\mathbf{K}$ to produce correlated scores, which are thresholded to obtain the binary mask. The log scale reflects the $1/f$ spectral characteristic of music signals (equal energy per octave).
    - Design Motivation: (a) Correlation causes the mask to form contiguous "groups" rather than scattered points, better aligning with musical semantics (e.g., "preserve the chord frequency band"); (b) the log scale makes higher-frequency groups wider, matching the energy distribution of $1/f$ spectra; (c) exposure to diverse mask combinations during training enables the model to generate from user-specified masks at inference time.

3. **Diffusion Autoencoder Decoder**:
    - Function: A conditional diffusion model that generates complete audio from the masked latent representation $\mathbf{z}_{\text{masked}}$.
    - Mechanism: The decoder is a conditional UNet conditioned on $\mathbf{z}_{\text{masked}}$, generating mel spectrograms (or DAC tokens) via standard DDPM denoising. The loss is the standard diffusion objective $\mathcal{L} = \mathbb{E}_{\epsilon, t}[\|\epsilon - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{z}_{\text{masked}})\|^2]$.
    - Design Motivation: The generative capacity of diffusion models allows the decoder to "imagine" missing musical patterns after masking removes information; stochasticity in each sampling pass yields diverse variants.

### Inference-Time Control
- **Conditional Generation**: Given a reference audio clip, its latent spectrum is extracted; a user-specified mask retains the target frequency range, and the decoder generates a variant that preserves the musical patterns in that range.
- **Music Blending**: Non-overlapping frequency bands are extracted from two separate songs and merged before being passed to the decoder, realizing combinations such as "chord progression from song A + rhythm from song B."

## Key Experimental Results

### Main Results: Conditional Generation

| Method | Loudness ↑ | Rhythm ↑ | Timbre ↓ | Harmony ↓ | FAD ↓ |
|------|-----------|----------|----------|-----------|-------|
| LatentFT-UNet | **0.834** | **0.966** | **0.391** | **0.079** | **0.348** |
| LatentFT-MLP | 0.815 | 0.963 | 0.376 | 0.079 | 0.337 |
| ILVR | 0.540 | 0.881 | 1.183 | 0.116 | 1.478 |
| Guidance | 0.459 | 0.876 | 1.419 | 0.133 | 2.084 |
| Codec Filtering | 0.712 | 0.926 | 0.857 | 0.108 | 2.523 |
| RAVE | -0.016 | 0.718 | 3.836 | 0.180 | 4.695 |

### Ablation Study

| Configuration | FAD (Conditional) | FAD (Blending) | Notes |
|------|----------|----------|------|
| LatentFT-MLP (full) | 0.337 | 1.387 | Baseline |
| w/o frequency mask training | Poor (not reported) | Poor | Decoder cannot reconstruct from masked input |
| w/o log scale | Higher FAD | Higher FAD | High/low-frequency group widths mismatch $1/f$ |
| w/o correlated mask | Higher FAD | Higher FAD | Scattered masks do not align with musical semantics |
| w/ Bandpass alternative | 1.511 | 2.58 | Unstable training; FAD substantially worse |
| LatentFT-DAC (waveform codec) | 0.915 | 1.364 | Generalizes to different frontends |

### Key Findings
- **Different musical attributes correspond to different latent frequency ranges**: chord progressions ~0.5 Hz (low frequency), melody ~2–5 Hz (mid frequency), rhythm/drums ~8 Hz (high frequency)—this mapping is song-dependent and varies slightly across pieces.
- Listening test with 29 musicians using Kruskal-Wallis H and Wilcoxon signed-rank tests: LatentFT is statistically significantly superior to all baselines in audio quality and blending capability.
- Successfully generalizes to 30-second segments (via fine-tuning), capturing section transitions at ~0.05 Hz (20-second periods).
- Generalizes to the Maestro (piano) and GTZAN (10-genre) datasets.

## Highlights & Insights
- **A novel control axis**: Latent frequency is an unprecedented control dimension in music generation—orthogonal, continuous, and interpretable.
- **Practical value of DFT orthogonality**: Modifying one frequency band does not affect others, which is substantially more stable than time-domain methods such as bandpass filters (empirically confirmed by training stability experiments).
- **Elegant structural analogy to the equalizer**: A conventional equalizer manipulates the audio spectrum to alter timbre; LatentFT manipulates the latent spectrum to alter musical structure—a conceptually elegant leap.
- **Transparent failure mode analysis**: The authors demonstrate blending failures when two references use overlapping or adjacent frequency bands, honestly delineating the method's boundaries.

## Limitations & Future Work
- The model is primarily trained on 5.9-second short segments; 30-second segments require fine-tuning, and longer durations (3+ minutes) remain infeasible.
- The mapping between latent frequencies and musical attributes is song-dependent and lacks cross-song consistency.
- The DAC encoder performs slightly worse than the mel encoder, possibly due to smaller training batch sizes.
- No comparison with text-conditioned music generation models (e.g., MusicLM) in combined control scenarios.

## Related Work & Insights
- **vs. ILVR/Guidance**: These methods apply DFT masking to mel spectrograms to guide diffusion denoising, without a dedicated latent space or mask training; they perform significantly worse than LatentFT.
- **vs. RAVE**: RAVE's latent space is not amenable to frequency-domain manipulation—direct masking and decoding yields low-quality audio—demonstrating that not every latent space is suitable for DFT-based control.
- **vs. AudioMAE**: AudioMAE performs patch masking and reconstruction on spectrograms, whereas LatentFT performs frequency masking and reconstruction in the latent spectrum; the directions are analogous but the objectives differ.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Latent Fourier-domain control is an entirely new concept; the "equalizer for musical structure" analogy is highly elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 datasets + musician listening test (statistically significant) + extensive ablations + alternative encoder + failure mode analysis + 30-second generalization.
- Writing Quality: ⭐⭐⭐⭐ Conceptual explanations are intuitively clear, though reviewers noted inconsistent use of certain terminology (timescale vs. latent frequency).
- Value: ⭐⭐⭐⭐⭐ Introduces a novel, interpretable control paradigm for music generation, opening a new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Deep Legendre Transform](../../NeurIPS2025/others/deep_legendre_transform.md)
- [\[ICLR 2026\] FastLSQ: Solving PDEs in One Shot via Fourier Features with Exact Analytical Derivatives](fastlsq_solving_pdes_in_one_shot_via_fourier_features_with_exact_analytical_deri.md)
- [\[ICLR 2026\] Latent Equivariant Operators for Robust Object Recognition: Promises and Challenges](latent_equivariant_operators_for_robust_object_recognition_promises_and_challeng.md)
- [\[ICLR 2026\] Out of the Shadows: Exploring a Latent Space for Neural Network Verification](out_of_the_shadows_exploring_a_latent_space_for_neural_network_verification.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)

</div>

<!-- RELATED:END -->

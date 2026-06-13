---
title: >-
  [Paper Note] MusicDET: Zero-Shot AI-Generated Music Detection
description: >-
  [ICML 2026][Audio & Speech][AI-generated music detection] MusicDET redefines "AI-generated music detection" as a zero-shot problem trained exclusively on real music. By utilizing frequency-wise decomposition…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "AI-generated music detection"
  - "zero-shot detection"
  - "normalizing flows"
  - "frequency-wise decomposition"
  - "likelihood estimation"
date: 2026-05-08
content_hash: 54202a20615004ed
---

# MusicDET: Zero-Shot AI-Generated Music Detection

**Conference**: ICML 2026  
**arXiv**: [2605.18072](https://arxiv.org/abs/2605.18072)  
**Code**: https://github.com/Chaolei98/MusicDET (Available)  
**Area**: AI Security / AI-Generated Content Detection / Audio Forgery  
**Keywords**: AI-generated music detection, zero-shot detection, normalizing flows, frequency-wise decomposition, likelihood estimation

## TL;DR
MusicDET redefines "AI-generated music detection" as a zero-shot problem trained exclusively on real music. By utilizing frequency-wise decomposition, intra-band normalizing flows, and global normalizing flows, it learns the probability distribution of real music power spectrograms. It uses the likelihood value as the "authenticity score," reducing the average EER from ~17% to 4.51% (zero-shot) and 0.89% (with class-conditional priors) under cross-generator evaluation on FakeMusicCaps / SONICS.

## Background & Motivation
**Background**: AI-generated music (AIGM) is rapidly penetrating creation and distribution, but detection (forensics) remains underdeveloped compared to generation. Existing AIGM detectors (SpecTTTra, AASIST, MERT/W2V2-AASIST, WPT, etc.) mostly follow the discriminative paradigm of speech deepfake detection—training a binary classifier on both real and fake samples to capture specific artifacts left by generators.

**Limitations of Prior Work**: This discriminative paradigm achieves high accuracy in closed-set scenarios (same generator for training/testing), but the EER collapses to over 30% when tested on unseen generators. Cross-family transfer (e.g., MusicGen → MusicLDM, Suno V3 → Udio 130) is largely ineffective. Given the continuous emergence of new generators, it is industrially impractical to train a specific detector for each one.

**Key Challenge**: Discriminative detection models "forgery" as "generator-specific artifact distributions," essentially learning a library of generator fingerprints. However, "real music" is a stable and shared target, whereas "forgery" is an open and expanding set. Approximating an open set using the complement of a stable distribution inevitably leads to OOD generalization failure. Furthermore, speech deepfake detectors rely on low-level clues from voice conversion/TTS that do not translate well to music with complex melodies, harmonies, timbres, and rhythms.

**Goal**: Split the task into two sub-problems: ① Perform detection **without any generated samples** in the training set (closer to real-world deployment); ② Provide a unified framework that is generator-agnostic and transfers stably to any unseen generator.

**Key Insight**: Experts identify AI music more easily than average listeners because they have a stronger prior of "what real music should sound like." To formalize this intuition: use Normalizing Flows to build an accurately computable probability density $p_X(x)$ for real music, such that fake samples naturally fall into low-likelihood regions.

**Core Idea**: Use frequency-wise decomposition + intra-band normalizing flows + global normalizing flows to perform one-class density estimation on time-frequency power spectrograms, using the log-likelihood $\log p_X(x)$ as the authenticity score.

## Method

### Overall Architecture
The input is a raw waveform $x_{\text{wav}} \in \mathbb{R}^L$ with 16 kHz sampling and 4s duration. The pipeline consists of five steps:

1.  **Feature Extraction**: STFT produces a power spectrum, followed by convolutional layers to extract energy features $X \in \mathbb{R}^{B \times C \times T \times F}$, preserving time-frequency structures (harmonics, rhythm, timbre texture);
2.  **Frequency-Wise Decomposition**: Slice the features along the frequency axis into multiple sub-bands (default: 2), where low frequencies correspond to rhythm/fundamental frequency and high frequencies to timbre/harmonics/transients;
3.  **Intra-Band Normalizing Flow**: Each sub-band is processed by an independent Glow-style flow $f_\theta^{\text{band}}$ to capture intra-band statistical regularities;
4.  **Global Normalizing Flow**: Latent representations of all sub-bands are concatenated and fed into a global flow $f_\theta^{\text{global}}$ to model cross-band dependencies (e.g., alignment between fundamental and harmonics);
5.  **Likelihood Inference**: The $\log p_X(x)$ of a test sample is used as the score; low likelihood indicates AI-generated music.

Training uses only real music to minimize NLL; inference relies solely on the likelihood under the real music prior. An optional class-conditional extension introduces fake samples during training, but only to modify the prior, preserving the nature of the "detector."

### Key Designs

1.  **Frequency-Wise Decomposition**:
    -   **Function**: Slices the spectrogram along the frequency axis into $K_b$ sub-bands $X = [X^{\text{low}}, X^{\text{high}}, \dots]$, building density models for each.
    -   **Mechanism**: Music is highly non-stationary along the frequency axis—low-frequency statistics (rhythmic pulses, fundamental frequency) differ vastly from high-frequncy statistics (timbre details, transients). Fitting the entire spectrum with a single flow results in high variance and poor discriminative power in $\log p_X$. Decomposition allows each sub-flow to face a more "unimodal" distribution via spatial orthogonalization (note: it does **not** assume frequency independence; cross-band dependencies are handled by the global flow).
    -   **Design Motivation**: Factorizing complex multi-modal modeling into simpler sub-distributions makes likelihood estimation numerically stable and aligns with musical physical priors.

2.  **Intra-band + Global Bi-level Normalizing Flow**:
    -   **Function**: Glow-style sub-flows perform invertible mapping $f_\theta^{\text{band}}: x^{\text{band}} \leftrightarrow h_K^{\text{band}}$, and the global flow $f_\theta^{\text{global}}$ projects concatenated $h_K^{\text{band}}$ to a latent Gaussian prior $p_Z(z) = \mathcal{N}(\mu_{\text{real}}, I)$.
    -   **Mechanism**: Each sub-flow consists of $K$ steps containing ActNorm, invertible $1 \times 1$ conv, and affine coupling layers with analytically computable Jacobians. Being bijective, the exact data likelihood is calculated via change-of-variables: $\log p_X(x) = \log p_Z(f_\theta(x)) + \sum_j \log |\det J_{f_j}|$. Sub-flows capture "intra-band patterns" while the global flow captures "cross-band coupling."
    -   **Design Motivation**: A single flow cannot simultaneously handle intra-band details and global structures. The bi-level structure strikes a compromise between expressiveness (multi-modal) and tractability.

3.  **Class-Conditional MusicDET**:
    -   **Function**: When fake samples are available, the **backbone remains unchanged**, but the prior is switched from a single Gaussian to a class-conditional double Gaussian $p_{Z|Y}(z|y) = \mathcal{N}(\mu_y, I)$, where $\mu_{\text{real}} = 5$ and $\mu_{\text{fake}} = -5$.
    -   **Mechanism**: Flow parameters $\theta$ are shared between classes; class information is injected only through the latent space mean. During inference, **only** $\log p_X(x | y = \text{real})$ is calculated. AI-generated samples are pushed toward $\mu_{\text{fake}}$ in the latent space, naturally falling into low-likelihood regions of the real prior.
    -   **Design Motivation**: Unifying discriminative and generative paradigms into one flow network prevents the model from over-fitting to generator-specific artifacts.

### Loss & Training
Zero-shot setting: minimize NLL of real music, $\min_\theta \mathbb{E}_{x \sim \mathcal{D}_{\text{real}}}[-\log p_X(x)]$. Class-conditional setting: minimize conditional NLL, $\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}_{\text{train}}}[-\log p_{X|Y}(x|y)]$. Training spans 10 epochs using Adam, lr $5 \times 10^{-4}$, batch size 64, $K=2$ flow steps per band, 2 bands, and $\mu_{\text{real}} = 5$. SpecAugment is used for data augmentation.

## Key Experimental Results

### Main Results
Both datasets utilize cross-generator evaluation: training and test subsets come from different generators. Lower average EER is better.

**FakeMusicCaps (Avg. EER across 5 TTM generators)**:

| Method | Zero-Shot | MusicGen | MusicLDM | AudioLDM2 | Stable Audio | Mustango | Avg. EER ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| AASIST | ✗ | 31.13 | 32.91 | 28.04 | 33.64 | 37.93 | 32.73 |
| W2V2-AASIST† (Full FT) | ✗ | 7.78 | 20.87 | 2.87 | 6.66 | 19.13 | 11.46 |
| WPT-W2V2-AASIST | ✗ | 10.84 | 27.31 | 4.62 | 10.44 | 34.84 | 17.61 |
| SpecTTTra-α | ✗ | 11.60 | 31.45 | 7.24 | 10.29 | 27.56 | 17.63 |
| **MusicDET (Zero-Shot)** | ✓ | **5.64** | **6.55** | **2.36** | **3.82** | **4.18** | **4.51** |
| **Class-Conditional MusicDET**| ✗ | 1.67 | 0.15 | 0.22 | 2.40 | 0.04 | **0.89** |

**SONICS (Avg. EER across 5 Suno/Udio subsets)**:

| Method | Zero-Shot | Suno V2 | Suno V3 | Suno V3.5 | Udio 32 | Udio 130 | Avg. EER ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| W2V2-AASIST† | ✗ | 16.20 | 0.37 | 0.47 | 24.97 | 21.70 | 12.74 |
| Spec-ViT | ✗ | 0.43 | 0.50 | 0.44 | 3.80 | 1.00 | 1.23 |
| SpecTTTra-α | ✗ | 0.70 | 1.34 | 0.93 | 7.83 | 2.50 | 2.66 |
| **MusicDET (Zero-Shot)** | ✓ | 2.80 | 3.20 | 2.93 | 2.73 | 2.80 | **2.89** |
| **Class-Conditional MusicDET**| ✗ | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **0.00** |

The class-conditional version achieves perfect EER (0.00) on SONICS. While the zero-shot version is not always the lowest, its variance across subsets is extremely small (2.73–3.20), showing immunity to generator selection.

### Ablation Study
**Efficiency Comparison (Table 3, FakeMusicCaps Training)**:

| Configuration | Inference Speed (M/S) ↑ | FLOPs (G) ↓ | Mem (GB) ↓ | Param (M) ↓ | EER (%) ↓ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MERT-AASIST† | 173 | 73.20 | 3.68 | 315.88 | 15.64 |
| WPT-W2V2-AASIST | 140 | 76.29 | 1.33 | 0.69 | 17.61 |
| SpecTTTra-α | 810 | 2.85 | 0.33 | 16.83 | 17.63 |
| **MusicDET** | 516 | 4.09 | **0.11** | **8.13** | **4.51** |

### Key Findings
- **Optimal Hyperparameters**: Performance peaks at 2 bands and $K=2$ flow steps; further complexity leads to overfitting.
- **Prior Mean Influence**: $\mu_{\text{real}} = 5$ is optimal; smaller values provide insufficient discrimination, while larger values lead to instability.
- **Cross-Generator Confusion**: Discriminative baselines show low diagonal EER but high off-diagonal EER (30–48%). Class-conditional MusicDET keeps the entire matrix near 0.
- **Cross-Task Transfer**: Effective on ASVspoof2019LA and CtrSVDD, showing that "learning real distributions with Normalizing Flows" generalizes to various audio forensics tasks.

## Highlights & Insights
- **Problem Reformulation**: Shifting from "AIGM discrimination" to "Zero-shot AIGM detection" transfers the burden of open-set generalization to the problem definition—learning the real distribution automatically provides immunity to unseen generators.
- **Elegant Factorization**: Factoring multi-modal music distributions into "intra-band × global coupling" maintains high-fidelity density estimation while reducing the instability of a single massive flow.
- **Unified Paradigm**: Using class-conditional priors allows the same backbone to act as a zero-shot detector or a supervised one, maintaining "detector purity" by avoiding discriminative heads.
- **Efficiency**: With 8.13M parameters and 0.11GB VRAM usage, the model is significantly more deployment-friendly than 300M+ parameter schemes.

## Limitations & Future Work
- **Limitations**: Currently verified only on 16kHz, 4s clips; short-duration modeling lacks assessment of long-term musical consistency (e.g., phrase-level coherence). Has not yet been tested against human-imperceptible AI music (e.g., Suno V4 or post-mixed versions).
- **Future Directions**: Replacing the global flow with an autoregressive flow for long-term dependencies; introducing conditional likelihood $p(x | \text{genre}, \text{instrument})$ for fine-grained real priors; and late-fusing "likelihood" with "semantic alignment scores" (e.g., CLAP).

## Related Work & Insights
- **vs. SpecTTTra**: SpecTTTra fits artifacts using spectrograms and Transformers. While it excels within a single generator, its cross-generator EER is 17.63% vs MusicDET's 4.51%.
- **vs. WPT-W2V2-AASIST**: WPT focuses on learning domain-invariant features. MusicDET instead focuses on the distribution of real samples, outperforming WPT on FakeMusicCaps (4.51 vs 17.61).
- **vs. Speech Deepfake Detection**: Unlike speech solutions that localized low-level clues, MusicDET is designed for musical physical characteristics, making it more robust for music-specific tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ (Formally introduces zero-shot AIGM detection; novel bi-level flow combination for audio).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Extensive cross-generator tests and cross-task verification).
- **Writing Quality**: ⭐⭐⭐⭐ (Clear motivation and concise algorithmic description).
- **Value**: ⭐⭐⭐⭐⭐ (Proposes a paradigm shift that industrial deployments can easily adopt).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration](polyphonia_zero-shot_timbre_transfer_in_polyphonic_music_with_acoustic-informed_.md)
- [\[ICML 2026\] VocSim: A Training-Free Benchmark for Zero-Shot Content Identity Recognition of Single-Source Audio](vocsim_a_training-free_benchmark_for_zero-shot_content_identity_in_single-source.md)
- [\[ACL 2026\] ReStyle-TTS: Relative and Continuous Style Control for Zero-Shot Speech Synthesis](../../ACL2026/audio_speech/restyle-tts_relative_and_continuous_style_control_for_zero-shot_speech_synthesis.md)
- [\[ACL 2026\] FC-TTS: Style and Timbre Control in Zero-Shot Text-to-Speech with Disentangled Speech Representations](../../ACL2026/audio_speech/fc-tts_style_and_timbre_control_in_zero-shot_text-to-speech_with_disentangled_sp.md)
- [\[AAAI 2026\] HQ-SVC: Towards High-Quality Zero-Shot Singing Voice Conversion in Low-Resource Scenarios](../../AAAI2026/audio_speech/hq-svc_towards_high-quality_zero-shot_singing_voice_conversion_in_low-resource_s.md)

</div>

<!-- RELATED:END -->

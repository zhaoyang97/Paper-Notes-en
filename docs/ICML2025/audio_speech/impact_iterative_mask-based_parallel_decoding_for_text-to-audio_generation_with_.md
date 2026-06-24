---
title: >-
  [Paper Note] IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling
description: >-
  [ICML2025][Audio & Speech][text-to-audio] This paper proposes the IMPACT framework, which combines iterative mask-based parallel decoding (MGM) with latent diffusion models (LDMs) for text-to-audio generation in a continuous latent space. It replaces heavy attention layers with a lightweight MLP diffusion head and introduces an unconditional pre-training stage, achieving state-of-the-art (SOTA) FD/FAD metrics on AudioCaps while maintaining an inference speed comparable to the…
tags:
  - "ICML2025"
  - "Audio & Speech"
  - "text-to-audio"
  - "diffusion models"
  - "mask-based generative modeling"
  - "iterative parallel decoding"
  - "continuous latent space"
date: 2026-05-08
content_hash: 332d3490fdc5229d
---

# IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling

**Conference**: ICML2025  
**arXiv**: [2506.00736](https://arxiv.org/abs/2506.00736)  
**Code**: [audio-impact.github.io](https://audio-impact.github.io/)  
**Area**: Image Generation  
**Keywords**: text-to-audio, diffusion models, mask-based generative modeling, iterative parallel decoding, continuous latent space

## TL;DR
This paper proposes the IMPACT framework, which combines iterative mask-based parallel decoding (MGM) with latent diffusion models (LDMs) for text-to-audio generation in a continuous latent space. It replaces heavy attention layers with a lightweight MLP diffusion head and introduces an unconditional pre-training stage, achieving state-of-the-art (SOTA) FD/FAD metrics on AudioCaps while maintaining an inference speed comparable to the fastest MAGNET-S model.

## Background & Motivation

### Background
Text-to-audio generation aims to synthesize semantically matching, high-quality audio according to natural language prompts, with applications spanning audio content creation, video games, marketing advertisements, etc. Current SOTA methods are mainly divided into two major categories:

**Diffusion Models**: Represented by Tango (Ghosal et al., 2023; Kong et al., 2024) and AudioLDM (Liu et al., 2023, 2024), these achieve the best performance in audio fidelity and quality. However, they employ heavy network architectures with attention layers as the diffusion backbone, which, combined with the iterative denoising sampling process, leads to extremely high inference latency.

**Masked Generative Models**: MAGNET (Ziv et al., 2024) utilizes iterative mask-based parallel decoding on discrete tokens, achieving much faster inference compared to autoregressive models (such as MusicGen, AudioGen) and diffusion models. However, its audio quality still lags behind state-of-the-art diffusion models.

### Key Challenge & Motivation
There is an obvious quality-speed trade-off: diffusion models provide high quality but are slow, whereas MAGNET is fast but lacks quality. An intuitive direction for improvement is to replace the discrete tokens of MAGNET with continuous representations, as continuous representations have demonstrated superior performance over discrete tokens in tasks such as text-to-image generation (Fan et al., 2024), speech large language models (Yuan et al., 2024), and automatic speech recognition (Xu et al., 2024).

However, the authors' preliminary experiments indicate that simply replacing tokens with continuous representations in MAGNET leads to a significant drop in performance. This suggests that direct replacement is not feasible, and a more reasonable modeling strategy is required to handle continuous representations.

**Key Insight**: Latent diffusion models (LDMs) excel at modeling continuous representations, while the iterative mask-based parallel decoding of MGM can replace the heavy attention backbone in LDMs, executing sampling with a lightweight MLP diffusion head. Combining the two achieves both the high fidelity of continuous representations and the low latency of parallel decoding.

## Method

### Overall Architecture
The IMPACT framework consists of three core components: an audio VAE codec, a Transformer-based Latent Encoder, and a lightweight MLP Diffusion Head. The training comprises two stages, and inference uses iterative mask-based parallel decoding.

### Audio Representation Extraction
Given an audio input, an audio VAE is used to encode it into a continuous latent representation sequence $\mathbf{z} = [z_1, z_2, \cdots, z_N]$, where each $z_i$ is a continuous vector. Unlike MAGNET, which uses a discrete audio codec (such as EnCodec), IMPACT operates directly within the continuous latent space of the VAE, avoiding information loss caused by discretization.

### Stage 1: Unconditional Pre-training
- **Goal**: Learn baseline capabilities for audio generation on large-scale unlabeled audio data.
- **Mechanism**: Perform mask-reconstruction training on the audio latent sequence without using any text conditions.
- **Masking Strategy**: Randomly mask a portion of the latent vectors, and train the model to reconstruct the masked latent vectors from the unmasked contexts.
- **Criticality**: The authors' experiments demonstrate that this stage is indispensable for the final performance, and skipping it directly leads to a significant performance drop.
- **Advantage**: It leverages vast amounts of unpaired text-audio data, significantly expanding the scale of available training data.

### Stage 2: Text-conditional Training with MGM
The training pipeline is as follows:

1. **Text Encoding**: Text prompts are converted into text conditioning vector sequences via a text encoder.
2. **Input Concatenation**: The audio latent sequence $\mathbf{z}$ is concatenated with the text conditioning vector sequence.
3. **Mask Application**: Random masking is applied to partial positions of the audio latent sequence.
4. **Latent Encoding**: The concatenated sequence passes through a Transformer-based Latent Encoder to produce context-aware representations.
5. **Diffusion Head Prediction**: A lightweight MLP Diffusion Head performs diffusion modeling on the latent vectors at masked positions—predicting the noise used to corrupt the masked audio latents.

The execution mechanism of the diffusion head is as follows:
- For a masked latent vector $z_i$, Gaussian noise $\epsilon$ is added to obtain $z_i^t$ in the forward process.
- The MLP diffusion head, conditioned on the output of the latent encoder, learns to predict the noise $\epsilon$.
- Since the MLP structure is much lighter than the U-Net or Transformer backbones traditionally used in LDMs, the computational overhead of each diffusion sampling step is minimal.

### Inference Stage: Iterative Mask-based Parallel Decoding
The inference process starts with an entirely empty sequence (all positions masked) and generates iteratively:

1. **Initialization**: All $N$ positions start in a masked state.
2. **Iterative Generation**: In each decoding iteration:
    - The Latent Encoder encodes the text condition and currently generated context.
    - The MLP diffusion head runs a few diffusion sampling steps on the remaining masked positions to generate candidate latent vectors.
    - A batch of positions is unmasked and transitioned into the generated state based on predictive confidence.
3. **Progressive Unmasking**: Earlier iterations unmask fewer positions (limited context, low confidence), while later iterations unmask more (rich context, high confidence).
4. **Termination**: Once all positions are unmasked, the complete latent sequence is reconstructed back into audio waveforms via the VAE decoder.

**Sources of Speed Advantage**:
- Diffusion sampling only runs on the lightweight MLP head instead of the heavy backbone network.
- Each iteration generates latent vectors for multiple positions simultaneously (parallel decoding).
- The total number of iterations is far lower than the diffusion steps in traditional LDMs.

## Key Experimental Results

### Experimental Setup
- **Dataset**: AudioCaps (text-audio pair dataset)
- **Evaluation Metrics**:
    - Objective Metrics: FD (Fréchet Distance), FAD (Fréchet Audio Distance), KL divergence, IS (Inception Score)
    - Subjective Metrics: REL (text relevance), OVL (overall audio quality)
- **Baseline Methods**: AudioLDM, AudioLDM2, Tango, Tango2, MAGNET-S, MAGNET-M, AudioGen, MusicGen

### Table 1: Main Results Comparison on AudioCaps

| Method | Type | FD ↓ | FAD ↓ | KL ↓ | REL ↑ | OVL ↑ | Inference Speed |
|---|---|---|---|---|---|---|---|
| AudioLDM | Diffusion | 23.31 | 1.96 | 1.26 | — | — | Slow |
| AudioLDM2 | Diffusion | 18.65 | 1.62 | 1.18 | — | — | Slow |
| Tango | Diffusion | 17.52 | 1.59 | 1.15 | 3.65 | 3.72 | Slow |
| Tango2 | Diffusion | 16.83 | 1.36 | 1.12 | 3.78 | 3.81 | Slow |
| AudioGen | Autoregressive | 20.87 | 2.15 | 1.35 | — | — | Medium |
| MAGNET-S | MGM | 22.15 | 2.08 | 1.38 | 3.32 | 3.25 | Fast |
| MAGNET-M | MGM | 19.63 | 1.85 | 1.28 | 3.45 | 3.42 | Medium |
| **IMPACT** | **MGM+LDM** | **15.92** | **1.28** | **1.09** | **3.85** | **3.88** | **Fast** |

IMPACT achieves SOTA on both key metrics FD and FAD, improving upon Tango2 by approximately **5.4%** and **5.9%** respectively, while maintaining an inference speed comparable to MAGNET-S.

### Table 2: Ablation Study

| Configuration | FD ↓ | FAD ↓ | KL ↓ |
|---|---|---|---|
| IMPACT (Full) | **15.92** | **1.28** | **1.09** |
| w/o Unconditional Pre-training | 21.37 | 1.95 | 1.31 |
| Discrete token (MAGNET-style) | 22.15 | 2.08 | 1.38 |
| Continuous representations + direct MSE (w/o diffusion head) | 25.84 | 2.46 | 1.52 |
| Heavy attention diffusion backbone (replacing MLP head) | 16.15 | 1.31 | 1.11 |

**Key Findings**:
- **Unconditional pre-training is crucial**: Removing it deteriorates the FD from 15.92 to 21.37 (+34.2%), indicating that restricted paired text-audio data is insufficient for learning high-quality baseline capabilities for audio generation.
- **Continuous representation + diffusion head is a key combination**: Simply replacing the discrete tokens with continuous representations without diffusion modeling (i.e., direct MSE regression) yields the worst performance, indicating that continuous spaces require diffusion modeling to be handled effectively.
- **MLP Head vs. Heavy Backbone**: Replacing the MLP head with a heavy attention backbone yields only marginal quality improvement (FD 16.15 vs 15.92) while drastically slowing down inference speed, validating the high efficiency of the MLP head.

## Highlights & Insights

- **Precise localization at the intersection of the design space**: Cleverly combines the parallel decoding efficiency of MGM with the continuous representation modeling capability of LDMs, avoiding the respective weaknesses of both paradigms—namely, the discrete token fidelity bottleneck of MGM and the inference latency bottleneck of LDMs.
- **Design philosophy of the lightweight diffusion head**: Since the latent encoder of mask-based parallel decoding already models sequence-level context, the diffusion head only needs to perform denoising at the individual token level. Thus, the heavy backbone can be replaced with an MLP, achieving virtually free diffusion sampling.
- **Empirical contribution of unconditional pre-training**: Demonstrates clearly that in the MGM training paradigm, unconditional pre-training is not merely a cherry on top but is indispensable, providing an important reference for subsequent MGM research.
- **Cross-modal transfer inspired by MAR**: MAR validated the continuous representation + masked parallel decoding + diffusion head paradigm in image generation. IMPACT is the first to successfully transfer this paradigm to the audio generation domain, demonstrating the generality of this scheme.
- **Pareto Optimality**: Reaches a new optimal solution on the quality-speed Pareto frontier, neither sacrificing speed like diffusion models nor sacrificing quality like MAGNET.

## Limitations & Future Work

- **Evaluation Limitations**: Primary evaluation is conducted strictly on the AudioCaps dataset, leaving generalization performance in more diverse audio generation scenarios (e.g., music generation, speech synthesis, and ambient sound effects) unverified.
- **Unconditional Pre-training Reliance on Large-Scale Data**: The unconditional pre-training stage requires substantial volumes of unlabeled audio data, presenting demands on data acquisition and storage.
- **VAE Bottleneck**: The final generation quality of the model is still limited by the reconstruction capability of the audio VAE; if the VAE itself suffers from information loss, IMPACT cannot compensate for it.
- **Fixed Decoding Schedule Strategy**: The unmasking schedule in iterative mask-based parallel decoding (i.e., how many positions to reveal at each step) may need to adaptively adjust according to different audio types; whether the current default scheme is optimal remains unclear.
- **Lack of In-depth Discussion on Text Encoders**: The paper does not thoroughly discuss the impact of different text encoders (such as CLAP, FLAN-T5, or T5) on performance.
- **Long Audio Generation**: The number of iterative decoding steps correlates with sequence length; whether efficiency and quality can be maintained for super-long audio generation (e.g., minutes long) has not been validated.

## Related Work & Insights

### Masked Generative Models (MGM)
- **MaskGIT / MUSE / MAGE**: Validated the effectiveness of masked parallel decoding in the image generation domain, but all operated on discrete tokens.
- **MAGNET**: Applied MGM to audio/music generation, achieving rapid inference but lacking quality.
- **SoundStorm**: Masked parallel decoding applied to speech synthesis.
- **MAR**: First to combine MGM with continuous representations + a diffusion head in image generation, achieving SOTA $\rightarrow$ the direct inspiration for IMPACT.

### Latent Diffusion Models (LDM)
- **AudioLDM / AudioLDM2**: Use VAE to encode audio into latent space, and then employ U-Net/Transformer for diffusion denoising, which yields high quality but is slow.
- **Tango / Tango2**: Similar frameworks, introducing alignment techniques such as DPO to further improve quality.
- **Positioning of IMPACT**: Retains the modeling advantages of LDMs in the continuous latent space but replaces the heavy diffusion backbone with MGM's parallel decoding + a lightweight diffusion head, achieving a win-win in both quality and speed.

### Discrete vs. Continuous Representations
- Multiple studies (Fan et al., 2024; Yuan et al., 2024; Xu et al., 2024) have proven that continuous representations outperform discrete tokens in various generative tasks.
- IMPACT further confirms that, in audio generation, continuous representations likewise outperform discrete tokens, provided that an appropriate modeling method (a diffusion head rather than simple regression) is utilized.

## Rating
- Novelty: ⭐⭐⭐⭐ — First to introduce MAR's continuous masked parallel decoding paradigm into the audio generation domain, with independent contributions from the unconditional pre-training stage.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Includes objective + subjective evaluations and an ablation study that thoroughly validates the contributions of each component; however, validating on only a single dataset is a minor limitation.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, highly structured methodology explanations, and accurate positioning of related work.
- Value: ⭐⭐⭐⭐ — Provides a new quality-speed Pareto optimal solution for text-to-audio generation; the lightweight diffusion head design is generalizable to other modalities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ControlAudio: Tackling Text-Guided, Timing-Indicated and Intelligible Audio Generation via Progressive Diffusion Modeling](../../ACL2026/audio_speech/controlaudio_tackling_text-guided_timing-indicated_and_intelligible_audio_genera.md)
- [\[ICML 2025\] ETTA: Elucidating the Design Space of Text-to-Audio Models](etta_elucidating_the_design_space_of_text-to-audio_models.md)
- [\[ICML 2025\] FLAM: Frame-Wise Language-Audio Modeling](flam_frame-wise_language-audio_modeling.md)
- [\[ICLR 2026\] The Devil behind the Mask: An Emergent Safety Vulnerability of Diffusion LLMs](../../ICLR2026/audio_speech/the_devil_behind_the_mask_an_emergent_safety_vulnerability_of_diffusion_llms.md)
- [\[AAAI 2026\] Diff-V2M: A Hierarchical Conditional Diffusion Model with Explicit Rhythmic Modeling for Video-to-Music Generation](../../AAAI2026/audio_speech/diff-v2m_a_hierarchical_conditional_diffusion_model_with_explicit_rhythmic_model.md)

</div>

<!-- RELATED:END -->

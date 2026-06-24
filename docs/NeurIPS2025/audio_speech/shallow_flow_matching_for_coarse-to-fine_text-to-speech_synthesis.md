---
title: >-
  [Paper Note] Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis
description: >-
  [NeurIPS 2025][Audio & Speech][Flow Matching] This paper proposes Shallow Flow Matching (SFM), which leverages weak generator outputs to construct intermediate states within a flow matching framework for coarse-to-fine TTS. Inference begins from these intermediate states rather than pure noise, simultaneously improving synthesis quality and accelerating inference.
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "Flow Matching"
  - "TTS"
  - "Coarse-to-Fine Generation"
  - "Shallow Inference"
  - "ODE Solving"
date: 2026-05-08
content_hash: e835f8ac7b5105ad
---

# Shallow Flow Matching for Coarse-to-Fine Text-to-Speech Synthesis

**Conference**: NeurIPS 2025
**arXiv**: [2505.12226](https://arxiv.org/abs/2505.12226)  
**Code**: [Available](https://ydqmkkx.github.io/SFMDemo/)  
**Area**: 3D Vision
**Keywords**: Flow Matching, TTS, Coarse-to-Fine Generation, Shallow Inference, ODE Solving

## TL;DR

This paper proposes Shallow Flow Matching (SFM), which leverages weak generator outputs to construct intermediate states within a flow matching framework for coarse-to-fine TTS. Inference begins from these intermediate states rather than pure noise, simultaneously improving synthesis quality and accelerating inference.

## Background & Motivation

Flow Matching (FM)-based TTS models commonly adopt a coarse-to-fine generation paradigm: a weak generator first produces a coarse mel-spectrogram, which is then refined by an FM module into a high-quality output. However, conventional approaches use the coarse representation only as a conditioning input to FM, while generation still starts from pure noise $\boldsymbol{X}_0 \sim \mathcal{N}(0, I)$. Since the coarse representation already encodes substantial semantic and acoustic structure, modeling the early stages from pure noise is redundant and wastes modeling capacity.

Inspired by the "shallow diffusion" concept from DiffSinger—which initiates reverse generation from shallow steps in diffusion models—this work generalizes that idea to the Flow Matching framework, proposing the SFM mechanism. It uses orthogonal projection to map weak generator outputs onto intermediate states along the CondOT path, allowing inference to start directly from these intermediate states, bypassing the early stages.

## Method

### Overall Architecture

The SFM framework consists of three core components: (1) a weak generator $\boldsymbol{g}_\omega$ that produces a coarse mel-spectrogram $\hat{\boldsymbol{X}}_g$; (2) a lightweight SFM head $\boldsymbol{h}_\psi$ that outputs a scaled mel-spectrogram $\hat{\boldsymbol{X}}_h$, time $\hat{t}_h$, and variance $\hat{\sigma}_h^2$; and (3) an FM decoder that generates from the constructed intermediate state. During inference, the ODE solver integrates from $\tilde{t}_h$ to 1 rather than from 0 to 1.

### Key Designs

1. **Orthogonal Projection onto CondOT Paths**: The core idea is to project the SFM head output $\hat{\boldsymbol{X}}_h$ onto the target $\boldsymbol{X}_1$, computing the projection coefficient $t_h$ as the corresponding time step: $t_h = \max(0, \mathbb{E}[\text{sg}[\hat{\boldsymbol{X}}_h] \cdot \boldsymbol{X}_1 / (\boldsymbol{X}_1 \cdot \boldsymbol{X}_1)])$. Theorem 1 is then used to map $\hat{\boldsymbol{X}}_h$ onto the CondOT path via scaling factor $1/\Delta$. The design motivation is to adaptively determine the position of $\hat{\boldsymbol{X}}_h$ on the FM path, avoiding manual specification of shallow steps.

2. **Single-Segment Piecewise Flow**: Based on Theorem 2, the CondOT path is split at intermediate state $\tilde{t}_h$ into two segments; training and inference focus only on the latter segment $t \geq \tilde{t}_h$. The flow and vector field are defined piecewise: $\boldsymbol{X}_t = (1-t_S)\boldsymbol{X}_{\tilde{t}_h} + t_S(\boldsymbol{X}_1 + \sigma_{\min}\boldsymbol{X}_0)$, and training supervises this latter segment using the CFM loss.

3. **SFM Strength $\alpha$**: The adaptively determined $t_h$ during training tends to be small. At inference, a hyperparameter $\alpha \geq 1$ is introduced to scale up $\hat{t}_h$, strengthening the guidance from the coarse representation. The optimal $\alpha$ is found via validation set search to balance quality and determinism.

### Loss & Training

The total loss is the sum of five terms:

$$\mathcal{L}_{\text{SFM}} = \mathcal{L}_{\text{coarse}} + \mathcal{L}_t + \mathcal{L}_\sigma + \mathcal{L}_\mu + \mathcal{L}_{\text{CFM}}$$

- $\mathcal{L}_{\text{coarse}}$: L2 loss on the coarse mel-spectrogram
- $\mathcal{L}_\mu$: guides $\hat{\boldsymbol{X}}_h$ toward $t_h \boldsymbol{X}_1$
- $\mathcal{L}_t, \mathcal{L}_\sigma$: MSE losses for predicted time and variance
- $\mathcal{L}_{\text{CFM}}$: conditional flow matching loss

## Key Experimental Results

### Main Results

Evaluated on LJ Speech, VCTK, and LibriTTS across multiple backbone models including Matcha-TTS (U-Net), StableTTS (DiT), and CosyVoice.

| System | UTMOS↑ | UTMOSv2↑ | Distill-MOS↑ | WER↓ | CMOS↑ |
|------|--------|----------|-------------|------|-------|
| Matcha-TTS Baseline (LJ) | 4.186 | 3.692 | 4.282 | 3.308 | -0.48 |
| Matcha-TTS Ablated (LJ) | 4.217 | 3.763 | 4.311 | 3.355 | -0.27 |
| **Matcha-TTS SFM (LJ)** | **4.257** | **3.848** | **4.386** | 3.413 | **0.00** |
| Ground Truth | 4.380 | 3.964 | 4.241 | 3.566 | +0.22 |

### Ablation Study

| $\alpha$ | $\tilde{t}_g$ | $\tilde{\sigma}_g$ | PMOS↑ | UTMOS↑ | WER↓ |
|----------|---------------|---------------------|-------|--------|------|
| 1.0 | 0.099 | 0.092 | 4.036 | 4.194 | 4.641 |
| 2.0 | 0.198 | 0.183 | 4.158 | 4.305 | 3.496 |
| **2.5** | **0.248** | **0.229** | **4.176** | **4.276** | 3.556 |
| 5.0 | 0.496 | 0.458 | 4.025 | 3.977 | 3.376 |
| 10.0 | 0.520 | 0.480 | 3.987 | 3.955 | 3.315 |

### Key Findings

- SFM yields consistent naturalness improvements across all tested TTS models, as confirmed by both objective and subjective evaluations.
- When using an adaptive step-size ODE solver, SFM substantially accelerates inference with a significant reduction in NFE.
- The optimal $\alpha$ typically falls in the range of 2–4; excessively large values degrade quality.
- SFM-c (using both conditioning and SFM simultaneously) underperforms SFM alone, indicating that the intermediate state already carries sufficient information.

## Highlights & Insights

- **The core contribution lies in generalizing "shallow diffusion" from DDPM to Flow Matching**, providing a rigorous mathematical framework through orthogonal projection and piecewise flows.
- The lightweight SFM head design enables plug-and-play integration into diverse TTS architectures.
- Adaptive determination of the intermediate state position (rather than manual specification) affords greater flexibility than shallow diffusion.
- The SFM strength hyperparameter $\alpha$ offers a flexible quality–speed trade-off at inference time.

## Limitations & Future Work

- The projection assumption $\hat{\boldsymbol{X}}_h \approx t_h \boldsymbol{X}_1$ underlying the SFM head may be inaccurate in certain scenarios.
- Validation is currently limited to TTS; application to other FM domains (image and video generation) remains unexplored.
- $\alpha$ requires validation set search, introducing additional deployment overhead.
- When $\Delta \geq 1$ in the early training phase, the model degenerates into deterministic behavior, which may affect training stability.

## Related Work & Insights

- The shallow diffusion mechanism proposed in DiffSinger serves as the direct inspiration for this work.
- The piecewise reflow idea from PeRFlow provides the theoretical basis for Theorem 2 (piecewise flow).
- The proposed approach generalizes to other coarse-to-fine generation tasks: a simple model first obtains a coarse estimate, which is then refined by a generative model.

## Rating

- Novelty: ⭐⭐⭐⭐ — Generalizes shallow diffusion to FM with a rigorous mathematical framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Multiple models, multiple datasets, subjective and objective evaluation, speed analysis
- Writing Quality: ⭐⭐⭐⭐ — Theoretical derivations are clear and experiments are well organized
- Value: ⭐⭐⭐⭐ — A plug-and-play FM acceleration method with strong practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] BinauralFlow: A Causal and Streamable Approach for High-Quality Binaural Speech Synthesis with Flow Matching Models](../../ICML2025/audio_speech/binauralflow_a_causal_and_streamable_approach_for_high-quality_binaural_speech_s.md)
- [\[NeurIPS 2025\] Adapting Speech Language Model to Singing Voice Synthesis](adapting_speech_language_model_to_singing_voice_synthesis.md)
- [\[ACL 2026\] ZipVoice-Dialog: Non-Autoregressive Spoken Dialogue Generation with Flow Matching](../../ACL2026/audio_speech/zipvoice-dialog_non-autoregressive_spoken_dialogue_generation_with_flow_matching.md)
- [\[ICLR 2026\] AlignSep: Temporally-Aligned Video-Queried Sound Separation with Flow Matching](../../ICLR2026/audio_speech/alignsep_temporally-aligned_video-queried_sound_separation_with_flow_matching.md)
- [\[ICLR 2026\] TangoFlux: Super-Fast and Faithful Text-to-Audio Generation with Flow Matching and CLAP-Ranked Preference Optimization](../../ICLR2026/audio_speech/tangoflux_super_fast_and_faithful_text_to_audio_generation_with_flow_matching_an.md)

</div>

<!-- RELATED:END -->

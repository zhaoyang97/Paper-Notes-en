---
title: >-
  [Paper Note] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space
description: >-
  [NeurIPS 2025][Audio & Speech][speech language model] This paper proposes SLED, which encodes speech waveforms into sequences of continuous latent representations and performs autoregressive modeling in the continuous space via an energy distance objective. This avoids the information loss from discretization and the complex hierarchical architectures required by RVQ, while enabling efficient zero-shot and streaming speech synthesis.
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "speech language model"
  - "continuous latent space"
  - "energy distance"
  - "zero-shot TTS"
  - "streaming synthesis"
date: 2026-05-08
content_hash: c314681a3d6541ea
---

# Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space

**Conference**: NeurIPS 2025
**arXiv**: [2505.13181](https://arxiv.org/abs/2505.13181)  
**Code**: [ictnlp/SLED-TTS](https://github.com/ictnlp/SLED-TTS)  
**Area**: Audio & Speech
**Keywords**: speech language model, continuous latent space, energy distance, zero-shot TTS, streaming synthesis

## TL;DR

This paper proposes SLED, which encodes speech waveforms into sequences of continuous latent representations and performs autoregressive modeling in the continuous space via an energy distance objective. This avoids the information loss from discretization and the complex hierarchical architectures required by RVQ, while enabling efficient zero-shot and streaming speech synthesis.

## Background & Motivation

- The success of text language models (GPT series) has inspired research into modeling speech in a similar autoregressive manner, yet speech is fundamentally a continuous, high-sampling-rate signal that differs intrinsically from discrete text.
- The dominant approach discretizes speech into multi-stream token sequences via residual vector quantization (RVQ), but this introduces two core problems:
    1. **Information bottleneck**: Discretization inevitably discards rich details in the original waveform, degrading reconstruction quality.
    2. **Architectural complexity**: Multi-stream sequences from RVQ require hierarchical autoregressive architectures (e.g., VALL-E's two-stage AR+NAR, or nested Transformers in RQ-Transformer), increasing modeling and engineering difficulty.
- Continuous latent space modeling can circumvent these issues, but the central challenge is: how to construct a **lightweight, expressive, training-stable, and sampling-efficient** step-wise conditional generation module—ideally as simple as softmax in discrete models.

## Core Problem

When performing autoregressive speech modeling in a continuous latent space, how should one design the learning objective and generation module for step-wise distributions so as to jointly achieve modeling capacity, training stability, and inference efficiency?

## Method

### 1. Continuous Latent Space Encoding

Encodec is used to encode raw speech waveforms into continuous vector sequences. Specifically, the token embeddings from Encodec's eight codebooks are summed frame-wise, yielding a continuous representation $\bm{h} \in \mathbb{R}^{Tf_h \times 128}$ at 75 Hz with 128 dimensions, retaining nearly all information.

### 2. Autoregressive Network + Lightweight Generation Module

The overall architecture consists of two components:

- **Autoregressive network $\psi$**: A 12-layer LLaMA-style Transformer (RMSNorm, SwiGLU, RoPE) that captures sequential dependencies and outputs a conditioning vector $\bm{z}_t = \psi(\bm{h}_{<t}; \theta)$.
- **Step-wise generation module $g$**: A lightweight MLP (6 residual blocks + AdaLN) that takes the conditioning vector $\bm{z}_t$ and random noise $\bm{\epsilon}$ as inputs, implicitly defining the continuous distribution $p_g(\bm{h}_t | \bm{z}_t)$:

$$\bm{h}_t = g(\bm{z}_t, \bm{\epsilon}; \phi)$$

The AdaLN module predicts scale and shift parameters from the noise via a linear transformation, applying stochastic modulation to the conditioning vector. At inference, only a single forward pass is required, yielding sampling efficiency comparable to softmax.

### 3. Energy Distance Training Objective

The generalized energy distance (GED), a special case of MMD, is adopted as the training loss. For each time step, the energy distance between the model distribution and the data distribution is minimized:

$$\mathcal{L}_{\text{GED}} = \sum_t \mathbb{E}_{\bm{h}_t, \bm{h}'_t} \left[ 2 \| \bm{h}_t - \bm{h}_t^* \|_2 - \| \bm{h}_t - \bm{h}'_t \|_2 \right]$$

where $\bm{h}_t, \bm{h}'_t$ are two independent samples from $p_g$ and $\bm{h}_t^*$ is the target. Key properties:

- The first term measures distance to the target (analogous to RMSE).
- The second term is a **repulsion term** $\|\bm{h}_t - \bm{h}'_t\|_2$ that prevents the model from degenerating into point regression—removing it is equivalent to an RMSE loss, and ablations show this causes complete model failure (WER rises from 1.59 to 40.60).
- When the distance function is $d(\bm{x}, \bm{y}) = \|\bm{x} - \bm{y}\|_2^\beta$ with $\beta \in (0,2)$, GED constitutes a strictly proper scoring rule, guaranteeing convergence to the true distribution.

### 4. Classifier-Free Guidance (CFG)

At inference, an additional text-masked forward pass is performed at each step to obtain the unconditional output $\bm{z}'_t$, and linear interpolation is used to enhance text alignment:

$$\bm{z}_t^{\text{cfg}} = \bm{z}'_t + \lambda (\bm{z}_t - \bm{z}'_t)$$

The default value is $\lambda = 2.0$, balancing intelligibility and speech quality. During training, text is randomly masked with probability 0.1.

### 5. Streaming Inference

Incremental synthesis is achieved by interleaving text and speech positions: upon receiving $n$ text subwords, $m$ speech vectors are generated (e.g., 5:20 or 5:45). The purely autoregressive architecture requires no post-processing and natively supports streaming. A binary classification head predicts the stopping position.

## Key Experimental Results

Training data: LibriHeavy (~50,000 hours of speech, 6,736 speakers), BF16 precision, batch size 512, trained for 300K steps.

**Zero-shot TTS performance** (LibriSpeech test-clean):

| Setting | WER-C (%) | WER-H (%) | SIM |
|---------|-----------|-----------|-----|
| 3s prefix prompt | 1.59 | 1.99 | 0.515 |
| Reference speech prompt | 1.51 | 1.97 | 0.664 |
| Ground Truth | 1.78 | 2.15 | 0.668/0.778 |

- WER surpasses ground truth (1.59 vs. 1.78), indicating extremely high textual intelligibility.
- Streaming inference DNSMOS (3.59) is close to offline (3.58), with only a modest WER increase (2.18 vs. 1.67).

**Efficiency comparison (10-second audio inference)**:

| Model | Parameters | RTF | FLOPs |
|-------|------------|-----|-------|
| SLED | 0.2B | 0.8 | 280G |
| DiTAR | 0.6B | 0.66 | 2750G |

SLED achieves a comparable real-time factor using approximately 1/10 the FLOPs and 1/3 the parameters of DiTAR.

## Highlights & Insights

- **Minimal architecture**: A single-stage autoregressive network combined with a lightweight MLP generator (~35M parameters), requiring no hierarchical architecture or post-processing—more efficient than VALL-E's NAR module (~159M).
- **Theoretically grounded**: Energy distance as a strictly proper scoring rule provides rigorous mathematical guarantees; the paper further analyzes that the flux loss in MELLE essentially approximates the repulsion term of energy distance.
- **Native streaming support**: The purely autoregressive model produces outputs incrementally without post-processing, making it suitable for real-time voice interaction systems.
- **Valuable empirical findings**: 1,000 hours of data suffices to acquire most generative and in-context learning capabilities.

## Limitations & Future Work

- The current use of Encodec (designed for codec purposes) suggests that training a dedicated encoder for continuous autoregressive modeling could further improve performance.
- Speaker similarity (SIM) still lags behind traditional TTS models (MegaTTS 3: 0.78).
- Validation is limited to speech synthesis; extension to general speech language modeling (speech understanding, dialogue, etc.) has not been explored.
- CFG requires an additional forward pass, approximately doubling inference computation.

## Related Work & Insights

| Method | Latent Space | Per-step Sampling | Post-processing | Streaming |
|--------|-------------|-------------------|-----------------|-----------|
| VALL-E | Discrete (RVQ) | softmax | NAR model | No |
| MELLE | Continuous (mel) | regression + flux loss | NAR refinement | No |
| FELLE | Continuous | ODE multi-step integration | None | No |
| DiTAR | Continuous (patch) | DiT iteration | None | No |
| **SLED** | Continuous (Encodec) | **Single MLP pass** | **None** | **Yes** |

SLED is the only continuous speech language model that simultaneously achieves single-pass sampling, no post-processing, and streaming inference.

**Broader implications**:

- Energy distance as a training objective for implicit generative models is transferable to other continuous sequence modeling domains (video generation, motion generation, etc.).
- The critical role of the repulsion term reveals the fundamental distinction between regression losses and distribution matching losses—a finding with implications for all continuous token prediction tasks.
- The streaming text–speech interleaving scheme is directly applicable as the TTS module in real-time voice interaction systems such as GPT-4o.
- Llasa (8B) demonstrates the scaling potential of discrete methods; scaling SLED in the continuous domain warrants further investigation.

## Rating

- **Novelty**: 8/10 — Applying energy distance to continuous speech LM is a novel and theoretically well-grounded contribution.
- **Experimental Thoroughness**: 8/10 — Comprehensive evaluation covering zero-shot, streaming, ablation, and efficiency analysis, though large-scale experiments are lacking.
- **Writing Quality**: 9/10 — Mathematical derivations are clear, with a complete theoretical chain from MMD to GED.
- **Value**: 8/10 — Significantly simplifies continuous speech LM architecture, laying a foundation for future scaling and generalization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Space Factorization in LoRA](latent_space_factorization_in_lora.md)
- [\[ICLR 2026\] MambaVoiceCloning: Efficient and Expressive Text-to-Speech via State-Space Modeling and Diffusion Control](../../ICLR2026/audio_speech/mambavoicecloning_efficient_and_expressive_text-to-speech_via_state-space_modeli.md)
- [\[NeurIPS 2025\] Multi-head Temporal Latent Attention](multi-head_temporal_latent_attention.md)
- [\[NeurIPS 2025\] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models](e-bats_efficient_backpropagation-free_test-time_adaptation_for_speech_foundation.md)
- [\[ICLR 2026\] Continuous Audio Language Models](../../ICLR2026/audio_speech/continuous_audio_language_models.md)

</div>

<!-- RELATED:END -->

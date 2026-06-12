---
title: >-
  [Paper Note] FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks
description: >-
  [NeurIPS 2025][Image Generation][speech codec] This paper proposes FocalCodec—a low-bitrate speech codec based on Focal Modulation Networks—that compresses speech to 0.16–0.65 kbps using a **single binary codebook**…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "speech codec"
  - "low bitrate"
  - "Focal Modulation"
  - "binary quantization"
  - "single codebook"
  - "speech tokenization"
  - "VQ-VAE"
date: 2026-05-08
content_hash: 594af1a1f72fd8c3
---

# FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks

**Conference**: NeurIPS 2025
**arXiv**: [2502.04465](https://arxiv.org/abs/2502.04465)  
**Code**: [lucadellalib/focalcodec-web](https://lucadellalib.github.io/focalcodec-web/)  
**Area**: image_generation (speech codec / speech tokenization)
**Keywords**: speech codec, low bitrate, Focal Modulation, binary quantization, single codebook, speech tokenization, VQ-VAE

## TL;DR

This paper proposes FocalCodec—a low-bitrate speech codec based on Focal Modulation Networks—that compresses speech to 0.16–0.65 kbps using a **single binary codebook**, achieving performance comparable to or better than multi-codebook state-of-the-art methods on speech resynthesis, voice conversion, and multiple downstream tasks.

## Background & Motivation

The success of large language models has driven a research paradigm of discretizing speech into tokens. Neural audio codecs are central to this pipeline, and their output tokens must simultaneously preserve **semantic information** (for understanding tasks such as ASR) and **acoustic information** (for high-fidelity reconstruction and speaker preservation).

Key limitations of existing approaches:

**Acoustic codecs** (EnCodec, DAC, WavTokenizer, etc.): strong reconstruction quality, but typically rely on multi-codebook RVQ, increasing downstream model complexity while providing insufficient semantic content.

**Semantic codecs** (HuBERT/WavLM + k-means): good semantics but severe acoustic detail loss and low speaker fidelity.

**Hybrid codecs** (SpeechTokenizer, Mimi, Stable Codec, etc.): attempt to balance both objectives but rely on complex multi-codebook designs, distillation losses, or supervised fine-tuning.

**High bitrates**: most methods require ≥0.7 kbps to achieve acceptable performance.

**Core motivation**: Can one design a **purely self-supervised, single-codebook, ultra-low-bitrate** codec that simultaneously retains sufficient semantic and acoustic information?

## Method

### Overall Architecture

FocalCodec is built on the VQ-VAE framework, introducing a **Compressor–Quantizer–Decompressor** architecture between the encoder and decoder:

```
Input waveform → [WavLM Encoder (frozen)] → [Compressor] → [Binary Spherical Quantizer] → [Decompressor] → [Vocos Decoder] → Reconstructed waveform
```

Each module is designed as follows:

**Encoder**: the first 6 layers of WavLM-Large (frozen). Lower layers are chosen because they retain substantial acoustic information while also encoding some semantic content. The encoder accounts for approximately 5× the parameters of the decoder—the authors argue that a strong encoder is more important than a strong decoder.

**Decoder**: the Vocos architecture (rather than the commonly used HiFi-GAN), which processes features through ConvNeXt blocks and projects them into Fourier coefficients, synthesizing waveforms via inverse STFT. This is more efficient and reduces aliasing.

### Key Design: Focal Modulation Compression Block

The core innovation in the compressor is replacing Self-Attention in conventional Transformers with **Focal Modulation**:

$$\mathbf{y}_i = q(\mathbf{x}_i) \odot h\left(\sum_{\ell=1}^{L+1} \mathbf{z}_i^\ell \odot \mathbf{g}_i^\ell \right)$$

Unlike Self-Attention, Focal Modulation **aggregates global context before modulating local interactions**:
- A depthwise convolution stack with progressively increasing kernel sizes captures multi-granularity dependencies from short to long range.
- The final layer uses global average pooling to provide global context.
- Pointwise convolutions compute gating vectors at each level.
- Linear complexity (vs. quadratic complexity of Self-Attention).
- Inductive biases including translation equivariance and explicit input dependence.

The compressor reduces dimensionality via linear projections or 1D convolutions (optionally with temporal downsampling) and uses the Snake activation function to capture periodic patterns. Three variants apply temporal downsampling factors of (1,1,1), (2,1,1), and (2,2,1), corresponding to token rates of 50 Hz, 25 Hz, and 12.5 Hz.

### Key Design: Binary Spherical Quantization (BSQ)

This work presents the first successful application of **Binary Spherical Quantization** (BSQ) in speech, a lookup-free quantization approach:

1. L2-normalize the input vector $\mathbf{v}$ onto the unit hypersphere: $\mathbf{u} = \mathbf{v}/\|\mathbf{v}\|_2$
2. Apply binary quantization independently per dimension: $\hat{\mathbf{u}} = \mathrm{sign}(\mathbf{u}) / \sqrt{L}$
3. The implicit codebook is $\mathcal{C} = \{-1/\sqrt{L}, 1/\sqrt{L}\}^L$ with size $|\mathcal{C}| = 2^L$

With $L=13$, the codebook size is 8192. Advantages include:
- **Parameter-free**: the codebook is implicitly defined, making it lightweight and efficient.
- **High utilization**: the binary bottleneck naturally encourages uniform codebook usage (normalized entropy ≈ 99%).
- **Bounded quantization error**: leads to faster convergence.
- **Suitable for generative models**: codebook size is tied to the latent dimension, avoiding the generation performance degradation caused by large codebooks.

### Loss & Training

A **two-stage decoupled training** strategy is adopted:

**Stage 1** (trains compressor + quantizer + decompressor; encoder frozen):
- Reconstruction loss: $\mathcal{L}_\text{recon} = \|\hat{\mathbf{z}} - \mathbf{z}\|_2^2$ (reconstructs continuous encoder features)
- Entropy loss: encourages confident predictions and uniform codebook usage.
- No commitment loss required (BSQ quantization error is bounded).

**Stage 2** (trains decoder; can run in parallel with Stage 1):
- Adversarial loss (hinge formulation)
- L1 log-mel spectrogram reconstruction loss
- Feature matching loss
- Discriminators: multi-period discriminator + multi-scale discriminator (HiFi-GAN style)

The key benefit of decoupled training: in unconstrained end-to-end training, the reconstruction loss tends to favor acoustic features at the expense of semantic content.

## Key Experimental Results

### Main Results: Speech Resynthesis (Table 2)

Core results on LibriSpeech test-clean:

| Model | Bitrate (kbps) | UTMOS↑ | dWER↓ | Sim↑ | Codebook Util.↑ | RTF↑ |
|-------|---------------|--------|-------|------|----------------|------|
| BigCodec | 1.04 | 4.11 | 2.55 | 98.5 | 100% | 22 |
| Stable Codec | 0.70 | 4.32 | 4.97 | 94.7 | 98.5% | 103 |
| **FocalCodec@50** | **0.65** | 4.05 | **2.18** | 97.4 | **100%** | **185** |
| **FocalCodec@25** | **0.33** | 4.14 | 3.30 | 96.3 | 99.8% | 195 |
| **FocalCodec@12.5** | **0.16** | 4.22 | 7.94 | 93.9 | 98.2% | 208 |

**Key finding**: FocalCodec@50 achieves the lowest dWER (2.18) at 0.65 kbps, outperforming BigCodec at 1.04 kbps, with over 8× faster inference speed.

Multilingual (MLS, 700 utterances): FocalCodec@50 achieves dWER of 12.57, substantially outperforming all other models (trained on English only); the second-best, BigCodec, reaches 15.24 at 60% higher bitrate.

Noisy conditions (VoiceBank): FocalCodec@50 achieves dWER of 8.08, far below the second-best Stable Codec (20.32).

Noisy conditions (Libri1Mix, more challenging): FocalCodec@50 achieves dWER 27.89 and Sim 91.6, substantially outperforming BigCodec (dWER 53.26) and WavTokenizer (dWER 70.10).

**Noteworthy**: UTMOS increases at lower token rates (@12.5: 4.22 > @50: 4.05), attributable to smoothing effects from temporal downsampling. As UTMOS is subject to saturation, dWER and Sim are more reliable evaluation metrics.

### Voice Conversion (Table 3)

One-shot voice conversion (VCTK dataset, 2521 samples). Single-codebook models perform conversion via k-NN token replacement ($k=4$); multi-codebook models concatenate the first codebook (source content) with subsequent codebooks (target speaker):

| Model | Bitrate | UTMOS↑ | dWER↓ | Sim↑ | RTF↑ |
|-------|---------|--------|-------|------|------|
| WavLM6-KM | 0.45 | 2.90 | 26.68 | 92.4 | 57 |
| SpeechTokenizer | 1.00 | 1.49 | 20.32 | 81.2 | 33 |
| Mimi | 0.69 | 2.40 | 110.0 | 89.7 | 71 |
| Stable Codec | 0.70 | 3.76 | 27.63 | 71.1 | 65 |
| **FocalCodec@50** | **0.65** | 3.38 | **21.27** | **92.2** | 116 |
| **FocalCodec@25** | **0.33** | 3.40 | 23.59 | **92.6** | 118 |
| **FocalCodec@12.5** | **0.16** | 3.43 | 29.93 | **92.6** | 117 |

FocalCodec achieves the highest speaker similarity (92.6), surpassing SpeechTokenizer (81.2) and Mimi (89.7), both of which explicitly design multi-codebook disentanglement. Acoustic codecs (EnCodec Sim 72.2, DAC 67.2, BigCodec 68.9) perform poorly due to their inability to separate speaker identity from content. WavLM6-KM ranks second (Sim 92.4), sharing the same encoder with FocalCodec, but exhibits higher dWER.

### Downstream Tasks (Table 4)

**Discriminative tasks** (shallow BiLSTM downstream model, approaching linear probing):

| Model | Bitrate | ASR WER↓ | SI ER↓ | SER ER↓ |
|-------|---------|----------|--------|---------|
| SpeechTokenizer | 1.00 | 14.97 | 2.73 | 41.50 |
| BigCodec | 1.04 | 26.41 | 2.34 | 47.50 |
| WavTokenizer | 0.48 | 35.62 | 2.44 | 49.80 |
| Stable Codec | 0.70 | 16.85 | 16.50 | 46.54 |
| Mimi | 0.69 | 22.98 | 5.43 | 44.70 |
| **FocalCodec@50** | **0.65** | 17.63 | 4.48 | 45.60 |
| **FocalCodec@25** | **0.33** | 21.12 | 6.07 | 46.80 |

- ASR: FocalCodec@50 WER (17.63) is second only to SpeechTokenizer (14.97, 1.5× bitrate + dual codebook) and Stable Codec (16.85, supervised fine-tuning).
- SI: FocalCodec@50 ER (4.48) is slightly higher than purely acoustic codecs (BigCodec 2.34) but far better than Stable Codec (16.50), whose semantic fine-tuning causes acoustic information loss.
- SER: differences across models are small; FocalCodec@50 (45.60) is on par with the best models.
- **WavLM6-KM baseline**: sharing the same encoder but different design choices, it achieves ASR WER 19.04 and SI ER 22.30—demonstrating that FocalCodec's compressor–quantizer design effectively preserves acoustic information.

**Generative tasks** (Transformer downstream model):

| Model | SE dWER↓ | SS dWER↓ | TTS dWER↓ | TTS UTMOS↑ |
|-------|----------|----------|-----------|-----------|
| SpeechTokenizer | 29.82 | 83.99 | 35.46 | 2.69 |
| BigCodec | 26.68 | 89.24 | 54.43 | 3.43 |
| Stable Codec | 35.57 | 103.00 | 49.28 | 3.19 |
| **FocalCodec@50** | **10.93** | **73.87** | 28.10 | **4.11** |
| **FocalCodec@25** | 14.74 | 99.96 | **16.75** | 4.16 |

- Speech Enhancement (SE): FocalCodec@50 dWER (10.93) substantially outperforms all baselines (second-best BigCodec: 26.68).
- Speech Separation (SS): FocalCodec@50 dWER (73.87) is the best, though absolute performance remains far from practical utility—quantization inevitably discards fine time-frequency detail required for separation.
- TTS: FocalCodec@25 achieves the best dWER (16.75) and UTMOS (4.16), as **shorter sequences reduce autoregressive modeling difficulty**—a token rate close to that of text makes next-token prediction more efficient, a finding with important implications for speech LLM design.

### Ablation Study (Table 5)

| Compression Block | Activation | Quantizer | dWER↓ | Sim↑ |
|------------------|-----------|-----------|-------|------|
| **Focal Modulation** | **Snake** | **BSQ** | **2.54** | **95.7** |
| Focal Modulation | Snake | FSQ | 2.61 | 94.8 |
| Focal Modulation | Snake | LFQ | 2.75 | 95.4 |
| Conformer | Snake | LFQ | 3.58 | 94.3 |
| AMP | Snake | LFQ | 4.52 | 94.3 |
| Linear | Snake | LFQ | 9.37 | 82.5 |

Focal Modulation and BSQ are the best-performing compression block and quantizer, respectively; replacing either with Conformer or FSQ leads to significant performance degradation.

## Highlights & Insights

1. **Minimalist design philosophy**: single codebook + binary quantization + purely self-supervised training, requiring no distillation losses or supervised fine-tuning, substantially reducing downstream model design complexity.
2. **Adaptation of Focal Modulation for speech**: the first successful introduction of Focal Modulation from computer vision into speech coding; linear complexity and multi-scale inductive biases naturally align with the properties of speech signals.
3. **First successful application of BSQ in speech**: binary spherical quantization naturally achieves near-perfect codebook utilization (≈100%), resolving the codebook collapse problem endemic to conventional VQ.
4. **Virtue of decoupled training**: the two-stage pipeline can be run in parallel and prevents semantic information from being suppressed by the reconstruction loss during end-to-end training.
5. **Asymmetric encoder–decoder design**: the encoder has approximately 5× the parameters of the decoder, contrary to common practice, prioritizing representation quality over decoding capacity.

## Limitations & Future Work

1. **Low token-rate variants degrade noticeably in multilingual settings**: FocalCodec@12.5 achieves multilingual dWER of 54.15, far higher than @50 (12.57), indicating a non-trivial cost to extreme compression in multilingual generalization.
2. **Encoder dependency on pretrained WavLM**: using the first 6 layers of WavLM-Large as a frozen encoder introduces approximately 127M parameters, limiting on-device deployment feasibility.
3. **Poor performance on speech separation**: absolute performance on the SS task remains far from practical applicability, as quantization discards critical information required for separation.
4. **Training limited to LibriTTS (≈585h of English)**: limited data scale and language diversity may constrain the upper bound for multilingual and multi-style scenarios.
5. **Sampling rate fixed at 16 kHz**: higher sampling rates such as 24 kHz are not supported, restricting applicability to broadband audio domains such as music.

## Related Work & Insights

- **Comparison with BigCodec**: both adopt a single-codebook design, but FocalCodec operates at only 62% of BigCodec's bitrate (0.65 vs. 1.04 kbps), achieves lower dWER (2.18 vs. 2.55), and is 8× faster (RTF 185 vs. 22). The core difference is Focal Modulation + BSQ versus conventional Conformer + VQ. BigCodec is marginally better at SI (ER 2.34 vs. 4.48) but substantially worse at ASR (WER 26.41 vs. 17.63), illustrating that purely reconstruction-based objectives struggle to preserve semantic content.
- **Comparison with Stable Codec**: Stable Codec uses supervised phoneme fine-tuning to enhance semantics at the expense of speaker identification capability (SI ER 16.5%); FocalCodec achieves a better semantic–acoustic balance through purely self-supervised training. Both target low-bitrate single-codebook operation, but Stable Codec requires force-aligned phoneme data.
- **Comparison with SpeechTokenizer / Mimi**: both hybrid codecs explicitly distill semantic information into the first codebook and acoustic information into subsequent ones, yet both underperform FocalCodec in voice conversion (Sim 81.2 / 89.7 vs. 92.6), indicating that FocalCodec's WavLM features combined with k-NN token replacement inherently provide strong speaker–content disentanglement.
- **Comparison with SemantiCodec**: SemantiCodec operates at 0.65 kbps using dual encoders and dual codebooks with 1,033M parameters and RTF of only 0.62 (sub-real-time); FocalCodec uses only 14% of its parameters (142M) and is approximately 300× faster.
- **Broader implications**: (1) the paradigm of binary quantization combined with a strong encoder is transferable to other modalities (visual tokenization, multimodal LLMs); (2) the finding that lower token rates benefit TTS suggests that speech LLMs should pursue more compact representations rather than higher-fidelity reconstruction; (3) the decoupled training strategy (tokenizer first, then decoder) can serve as a general paradigm for codec tasks requiring simultaneous preservation of multiple information types.

## Rating

- **Novelty**: ★★★★☆ — First combination of Focal Modulation and BSQ in speech coding; design is simple yet effective.
- **Technical Quality**: ★★★★★ — Extremely comprehensive experiments (resynthesis × 4 datasets + voice conversion + 3 discriminative tasks + 3 generative tasks + ablations), with both quantitative and subjective evaluation.
- **Practical Value**: ★★★★☆ — Single-codebook, low-bitrate design naturally suits speech LLM pipelines, though WavLM dependency limits on-device deployment.
- **Writing Quality**: ★★★★☆ — Clear structure, thorough baselines, rich figures and tables.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Adapting Speech Language Model to Singing Voice Synthesis](adapting_speech_language_model_to_singing_voice_synthesis.md)
- [\[ICCV 2025\] Bitrate-Controlled Diffusion for Disentangling Motion and Content in Video](../../ICCV2025/image_generation/bitrate-controlled_diffusion_for_disentangling_motion_and_content_in_video.md)
- [\[NeurIPS 2025\] Understanding Representation Dynamics of Diffusion Models via Low-Dimensional Models](understanding_representation_dynamics_of_diffusion_models_via_low-dimensional_mo.md)
- [\[NeurIPS 2025\] StelLA: Subspace Learning in Low-rank Adaptation using Stiefel Manifold](stella_subspace_learning_in_low-rank_adaptation_using_stiefel_manifold.md)
- [\[NeurIPS 2025\] Shallow Diffuse: Robust and Invisible Watermarking through Low-Dimensional Subspaces in Diffusion Models](shallow_diffuse_robust_and_invisible_watermarking_through_low-dimensional_subspa.md)

</div>

<!-- RELATED:END -->

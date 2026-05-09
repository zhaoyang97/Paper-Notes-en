---
title: >-
  [Paper Note] Scaling Speech Tokenizers with Diffusion Autoencoders
description: >-
  [ICLR 2026][Audio & Speech][Speech Tokenizer] This paper proposes SiTok (Speech Diffusion Tokenizer), which employs a diffusion autoencoder to jointly train the encoder–quantizer–decoder in a single stage (rather than two stages), incorporates CTC-based semantic regularization to ensure discrete tokens retain linguistic information, and scales to 1.6B parameters trained on 22 million hours of speech data. SiTok achieves strong performance at an extremely low token rate (12.5 Hz / 200 bps), attaining 3.34% WER (reconstruction) and 4.95 WER (LLM ASR) simultaneously.
tags:
  - ICLR 2026
  - "Audio & Speech"
  - Speech Tokenizer
  - Diffusion Autoencoder
  - Semantic Regularization
  - Low Bitrate
  - CTC Loss
date: 2026-05-08
content_hash: a28a0a94dfa224ef
---

# Scaling Speech Tokenizers with Diffusion Autoencoders

**Conference**: ICLR 2026
**arXiv**: [2602.06602](https://arxiv.org/abs/2602.06602)
**Code**: None (Demo: [https://sitok-demo.github.io/](https://sitok-demo.github.io/))
**Area**: Speech / Tokenization
**Keywords**: Speech Tokenizer, Diffusion Autoencoder, Semantic Regularization, Low Bitrate, CTC Loss

## TL;DR

This paper proposes SiTok (Speech Diffusion Tokenizer), which employs a diffusion autoencoder to jointly train the encoder–quantizer–decoder in a single stage (rather than two stages), incorporates CTC-based semantic regularization to ensure discrete tokens retain linguistic information, and scales to 1.6B parameters trained on 22 million hours of speech data. SiTok achieves strong performance at an extremely low token rate (12.5 Hz / 200 bps), attaining 3.34% WER (reconstruction) and 4.95 WER (LLM ASR) simultaneously.

## Background & Motivation

**State of the Field**: Speech tokenizers serve as the fundamental interface for speech language models, determining how speech is discretized. An ideal speech tokenizer must simultaneously satisfy three objectives: (1) extreme compression to enable efficient language modeling; (2) high-fidelity reconstruction to produce natural speech; and (3) semantically rich representations to support downstream understanding tasks.

**Limitations of Prior Work**: Existing methods address the tension among these three objectives through heuristic compromises rather than principled solutions. (1) Reconstruction quality degrades at low bitrates — many methods increase the number of RVQ (Residual Vector Quantization) codebook layers or raise the frame rate to maintain quality, but this directly inflates token counts (e.g., Mimi at 75 TPS, DualCodec at 75 TPS), undermining the compression objective. (2) Optimizing solely for acoustic fidelity neglects semantics, rendering tokens unsuitable for understanding tasks (e.g., high ASR WER). (3) Two-stage training pipelines — first quantizing speech representations with SSL models, then independently training a diffusion decoder or vocoder — prevent the quantizer from being optimized for reconstruction, forcing the decoder to adapt to suboptimal discrete codes.

**Root Cause**: Under conventional acoustic reconstruction objectives, simply scaling up model size or data yields diminishing returns at low token rates — a structural bottleneck of vector quantization. Deterministic reconstruction losses force the discrete latent space to "collapse uncertainty," prioritizing low-level signal details over semantic structure, so that the more aggressively one compresses, the greater the semantic loss.

**Starting Point**: The uncertainty introduced by low-token-rate quantization calls for a **generative framework** — diffusion models naturally learn to reverse stochastic degradation processes and are thus well-suited to handle the information loss caused by quantization. Moreover, supervising the post-quantization latent space directly with a CTC loss injects semantic information more directly than SSL distillation.

**Core Idea**: A diffusion autoencoder (rather than adversarial training) jointly optimizes quantization and reconstruction, complemented by CTC semantic regularization, achieving simultaneous preservation of both semantics and acoustics at extremely low token rates.

## Method

### Overall Architecture

SiTok takes mel spectrograms as both input and reconstruction target (rather than raw waveforms), avoiding the challenges of processing very long waveform sequences and the instability of adversarial training. The pipeline is as follows: (1) downsampling to 12.5 Hz; (2) a Llama-style causal Transformer encoder (16 layers) extracts latent features $\mathbf{z}$; (3) vector quantization (65,536 entries, 32-dimensional, EMA updates) produces discrete tokens $\mathbf{q}$; (4) a non-causal Llama Transformer diffusion decoder (16 layers) conditions on quantized embeddings $\mathbf{z}_q$ and reconstructs the mel spectrogram via a flow-matching objective; (5) an external Vocos vocoder converts the mel spectrogram to a 24 kHz waveform. An auxiliary CTC decoder (4 layers) operates on the post-quantization latent space to predict text transcriptions.

### Key Designs

1. **Diffusion Autoencoder as a Replacement for Adversarial Training**

    - Function: High-fidelity reconstruction of mel spectrograms conditioned on quantized discrete tokens.
    - Mechanism: The decoder is trained with a flow-matching objective, learning a velocity field $v_\phi(\mathbf{x}_t, t, \mathbf{z}_q)$ for noisy samples $\mathbf{x}_t = t\mathbf{x} + (1-t)\epsilon$ to approximate the true velocity $(\mathbf{x} - \epsilon)$. Advantages over adversarial training: (a) no discriminator or complex loss design is required, yielding more stable training; (b) the diffusion model learns the data distribution and can "hallucinate" missing details from quantized representations; (c) better scalability — waveform-level models require extensive up/downsampling, whereas mel spectrograms are more compact.
    - Design Motivation: Deterministic reconstruction collapses under aggressive compression — forcing all information into 200 bps is fundamentally infeasible. Diffusion models acknowledge that "not all details can be recovered from tokens" and instead learn the conditional distribution $p(\mathbf{x}|\mathbf{z}_q)$, which is the correct modeling paradigm at low token rates.

2. **CTC Semantic Regularization**

    - Function: Ensures that discrete tokens retain semantic and linguistic information.
    - Mechanism: A lightweight CTC decoder $\mathcal{D}_{\phi_{\text{ctc}}}$ (4-layer Transformer) is attached to the quantized embeddings $\mathbf{z}_q$ and directly predicts text transcriptions $\mathbf{y}$. The total loss is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \lambda_{\text{ctc}} \cdot \text{CTC}(\mathcal{D}_{\phi_{\text{ctc}}}(\mathbf{z}_q), \mathbf{y}) + \mathcal{L}_{\text{vq}}$, where $\lambda_{\text{ctc}}$ is a critical hyperparameter. Experiments show $\lambda_{\text{ctc}} = 0.1$ is optimal; too large a value (1.0) degrades reconstruction (WER rising from 4.06 to 10.1).
    - Design Motivation: Unlike prior approaches that align with SSL features via MSE/cosine distillation, CTC directly enforces that tokens can be decoded into text — the most direct supervision signal for semantic preservation. No external SSL model (e.g., HuBERT/WavLM) is required; the approach is fully end-to-end.

3. **Efficient Diffusion Decoding (Shortcut Fine-tuning)**

    - Function: Reduces diffusion inference steps from the standard multi-step setting to 2–4 steps.
    - Mechanism: The encoder and VQ module are frozen, and the decoder is fine-tuned with a shortcut model objective — the network additionally receives the step size $d$ as a condition and is jointly optimized with a flow-matching loss ($d=0$ corresponds to the true velocity) and a self-consistency loss (one large step of $2d$ ≈ two consecutive small steps of $d$), enabling the model to learn to "skip intermediate steps." Practical RTF: 0.041 at 16 steps → 0.013 at 4 steps, a 3.2× speedup.
    - Design Motivation: Multi-step sampling in diffusion decoding is a deployment bottleneck; the shortcut approach lets the model learn its own acceleration strategy, offering greater flexibility than conventional distillation.

### Loss & Training

Total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + 0.1 \cdot \mathcal{L}_{\text{ctc}} + \mathcal{L}_{\text{vq}}$. Training uses AdamW with lr=8e-5, 32K warmup steps, single epoch (~450K steps), on 22 million hours of internal speech data. Optional refinements include: (1) Decoder fine-tuning (encoder and VQ frozen); (2) Token CFG (training an unconditional path by dropping tokens with 10% probability, combining conditional and unconditional predictions at inference).

## Key Experimental Results

### Main Results (Reconstruction Quality Comparison)

| Model | FPS/TPS | # Codebooks | Bitrate | WER↓ | SIM↑ | UTMOS↑ |
|-------|---------|-------------|---------|------|------|--------|
| Ground Truth | - | - | - | 2.14 | 0.730 | 3.53 |
| DualCodec | 12.5/75 | 6 | 0.925 | 2.63 | 0.624 | 3.78 |
| X-codec 2 | 50/50 | 1 | 0.80 | 2.63 | 0.620 | 3.68 |
| Mimi | 12.5/75 | 6 | 0.825 | 4.51 | 0.527 | 3.09 |
| FireRedTTS | 25/25 | 1 | 0.35 | 3.35 | 0.597 | 3.40 |
| CosyVoice | 25/25 | 1 | 0.30 | 5.63 | 0.465 | 3.65 |
| **SiTok (CN=1)** | **12.5/12.5** | **1** | **0.20** | **4.06** | **0.641** | **3.44** |
| + Decoder FT | 12.5/12.5 | 1 | 0.20 | 3.79 | **0.682** | 3.48 |
| + Token CFG | 12.5/12.5 | 1 | 0.20 | **3.34** | 0.635 | **3.60** |

At only 200 bps — the lowest bitrate among all baselines — SiTok achieves competitive WER of 3.34% and SIM of 0.682.

### Ablation Study (Effect of Semantic Regularization)

| CTC Regularization | TPS | Recon. WER↓ | SIM↑ | UTMOS↑ | LLM ASR↓ | ER↑ | SV↓ | KS↑ |
|-------------------|-----|------------|------|--------|----------|-----|-----|-----|
| ✓ (λ=0.1) | 12.5 | 4.06 | 0.641 | 3.44 | 4.95 | 63.5 | 13.8 | 96.9 |
| ✗ | 12.5 | **33.0** | 0.495 | 2.68 | 29.4 | 57.9 | 18.9 | 86.1 |
| ✓ (λ=0.1) | 50 | 2.80 | 0.660 | 3.46 | 4.49 | 64.4 | 8.59 | 97.7 |
| ✗ | 50 | 5.17 | 0.611 | 2.84 | 7.27 | 60.4 | 13.5 | 92.8 |

Without CTC regularization, the 12.5 TPS model's WER surges to 33.0%, demonstrating that semantic regularization is not a supplementary enhancement but an indispensable component.

### Key Findings

- **Non-monotonic effect of model scaling**: As model size increases from 0.63B (S) to 1.61B (XL), reconstruction quality improves consistently (WER 4.18→3.84), but understanding task performance peaks at 1.12B (L), with larger models exhibiting degraded SV scores (13.8→14.7), suggesting that excess capacity may over-encode low-level acoustic details at the expense of semantic abstraction.
- **Complementary nature of Token CFG and Decoder FT**: CFG primarily reduces WER (to 3.34), while FT primarily improves speaker similarity (to 0.682); the two can be combined as needed.
- **Sensitivity of the CTC weight $\lambda_{\text{ctc}}$**: 0.1 is optimal; 0.02 yields good reconstruction but poor understanding, while 0.5–1.0 also degrades reconstruction (over-constraining the latent space).
- **Regression-only (R) training yields poor tokenizers**: WER rises to 4.66 and all understanding metrics deteriorate, confirming that the diffusion loss (D) is the core driver of performance.

## Highlights & Insights

- **A profound insight: uncertainty requires generative modeling.** Low-token-rate quantization inevitably discards information; attempting "perfect recovery" via deterministic reconstruction is fundamentally futile. Diffusion models acknowledge uncertainty and learn conditional distributions — this is the philosophically correct modeling approach, transferable to any high-compression-ratio discretization scenario.
- **Minimalist effectiveness of CTC supervision.** No external SSL model or complex feature alignment is needed; a 4-layer CTC head that directly predicts text suffices. The critical design choice is placing the supervision signal *after* quantization (rather than before), directly shaping the semantic properties of discrete tokens.
- **Pragmatic choice of mel spectrograms as intermediate representation.** This avoids the long sequences and training instability of waveform-level modeling. Although an external vocoder is required, the decoupled design allows the tokenizer and vocoder to be independently optimized and upgraded.

## Limitations & Future Work

- **Dependency on an external vocoder**: The mel-to-waveform conversion relies on Vocos, and overall quality is bounded by the vocoder's performance.
- **Internal training data**: The 22 million hours of speech data are not publicly available, limiting reproducibility.
- **English-dominant training**: Although multilingual coverage is claimed, English constitutes the vast majority, and multilingual generalization has not been thoroughly validated.
- **Diffusion decoding latency**: Even with shortcut fine-tuning, 2–4 iterative steps are still required, which may be insufficient for real-time interactive applications.
- **Understanding performance regression in L and XL models**: Larger models do not consistently improve understanding tasks, suggesting the need for better training strategies or architectural designs to balance acoustic and semantic objectives.

## Related Work & Insights

- **vs. Mimi (Défossez et al., 2024)**: Mimi uses 8-layer RVQ to achieve 1.1 kbps and 23.1 LLM ASR WER; SiTok achieves 4.95 LLM ASR WER with a single codebook at 200 bps — a 5.5× higher compression ratio with substantially better understanding performance.
- **vs. CosyVoice / FireRedTTS**: These two-stage methods first quantize SSL features and then train a diffusion decoder independently; SiTok's end-to-end joint optimization eliminates the objective misalignment between the quantizer and decoder.
- **vs. StableCodec / GLM4-Voice**: Both are designed for low token rates (0.2–0.4 kbps); SiTok's understanding performance (particularly LLM ASR) is significantly superior to these baselines.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of diffusion autoencoder and CTC is innovative, though neither component is entirely new; the core contribution lies in scaled validation and systematic design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers reconstruction, understanding, and generation scenarios; rich ablations over losses, codebook configurations, model scale, and decoding steps; comprehensive comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-argued motivation, and precise mathematical descriptions.
- Value: ⭐⭐⭐⭐⭐ A speech tokenizer that unifies understanding and generation at extremely low bitrates represents an important contribution to the advancement of speech language models.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] The Devil behind the Mask: An Emergent Safety Vulnerability of Diffusion LLMs](the_devil_behind_the_mask_an_emergent_safety_vulnerability_of_diffusion_llms.md)
- [\[NeurIPS 2025\] The Impact of Scaling Training Data on Adversarial Robustness](../../NeurIPS2025/audio_speech/the_impact_of_scaling_training_data_on_adversarial_robustness.md)
- [\[NeurIPS 2025\] Perceptually Aligning Representations of Music via Noise-Augmented Autoencoders](../../NeurIPS2025/audio_speech/perceptually_aligning_representations_of_music_via_noise-augmented_autoencoders.md)
- [\[ACL 2026\] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models](../../ACL2026/audio_speech/breaking_block_boundaries_anchor-based_history-stable_decoding_for_diffusion_lar.md)
- [\[ICLR 2026\] Knowing When to Quit: Probabilistic Early Exits for Speech Separation](knowing_when_to_quit_probabilistic_early_exits_for_speech_separation.md)

<!-- RELATED:END -->

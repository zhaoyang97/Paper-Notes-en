---
title: >-
  [Paper Note] Autoregressive Speech Synthesis without Vector Quantization
description: >-
  [ACL 2025][Audio & Speech][Speech Synthesis] MELLE proposes an autoregressive language model for TTS based on continuous mel-spectrogram frames. By utilizing a regression loss, a variational inference sampling module, and a spectrogram flux loss, it directly predicts continuous spectrogram frames, thereby avoiding the fidelity loss and sampling robustness issues caused by vector quantization. This single-stage model achieves speech synthesis quality comparable to human levels…
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Speech Synthesis"
  - "continuous tokens"
  - "mel-spectrogram"
  - "variational inference"
  - "autoregressive language model"
date: 2026-05-08
content_hash: c20eba74bd6605d2
---

# Autoregressive Speech Synthesis without Vector Quantization

**Conference**: ACL 2025  
**arXiv**: [2407.08551](https://arxiv.org/abs/2407.08551)  
**Code**: [https://aka.ms/melle](https://aka.ms/melle) (Demo)  
**Area**: Speech Synthesis / TTS  
**Keywords**: Speech Synthesis, continuous tokens, mel-spectrogram, variational inference, autoregressive language model

## TL;DR
MELLE proposes an autoregressive language model for TTS based on continuous mel-spectrogram frames. By utilizing a regression loss, a variational inference sampling module, and a spectrogram flux loss, it directly predicts continuous spectrogram frames, thereby avoiding the fidelity loss and sampling robustness issues caused by vector quantization. This single-stage model achieves speech synthesis quality comparable to human levels.

## Background & Motivation

**Background**: Current mainstream zero-shot TTS methods are codec language models (such as VALL-E), which first quantize audio into discrete tokens using neural audio codecs (like EnCodec) and then predict these discrete tokens using an autoregressive language model.

**Limitations of Prior Work**:
   - Vector quantization is inherently designed for audio compression; compared to continuous representations, quantized discrete codes lose fidelity.
   - Highly similar discrete codec codes make random sampling strategies prone to robustness issues such as long silences or persistent noise.
   - Two-stage decoding is required: an AR model generates the coarse first-stage tokens, and then an NAR model iteratively predicts multi-layer codebook codes for refinement, resulting in low inference efficiency.

**Key Challenge**: The discrete token paradigm is inherently unsuited for continuous signals like audio—discretization itself is an information-lossy compression, whereas continuous representations (such as mel-spectrograms) preserve richer acoustic details. However, continuous-value tokens face two major challenges: how to define the training objective (cross-entropy is inapplicable) and how to implement sampling in a continuous space (top-p is inapplicable).

**Goal**: Completely replace discrete quantized tokens with continuous-value mel-spectrogram frames in autoregressive speech synthesis, addressing the two major challenges of training objectives and sampling mechanisms.

**Key Insight**: Speech reconstructed from mel-spectrograms outperforms speech reconstructed from EnCodec in both WER and speaker similarity, demonstrating that continuous representations have higher fidelity. The authors draw inspiration from the variational inference of VAEs to implement sampling in the continuous space.

**Core Idea**: Replace cross-entropy with regression loss and top-p sampling with VAE-style variational inference, realizing a single-stage autoregressive TTS model that predicts continuous mel-spectrograms.

## Method

### Overall Architecture
MELLE is a decoder-only autoregressive language model where the input is a concatenation of BPE text tokens and mel-spectrogram frames, and the output predicts the next continuous mel-spectrogram frame frame-by-frame. The overall pipeline is: Text $\rightarrow$ BPE embedding + mel-spectrogram $\rightarrow$ pre-net projection $\rightarrow$ Transformer decoder $\rightarrow$ Latent Sampling Module $\rightarrow$ continuous mel frames $\rightarrow$ Post-Net refinement $\rightarrow$ vocoder waveform reconstruction. The key difference from VALL-E is that it requires no quantization codec and no NAR second stage; it is completed entirely in a single stage.

### Key Designs

1. **Autoregressive Language Model (Transformer Decoder)**:

    - **Function**: Serves as the core backbone to autoregressively generate continuous acoustic tokens.
    - **Mechanism**: A 12-layer Transformer block (16 heads, 1024 dim). Input text is projected via the embedding layer, and the mel-spectrogram is projected to the model dimension via a 3-layer MLP pre-net. After concatenation, they model the semantic-acoustic dependency. The output $\boldsymbol{e}_t$ at each step is passed to subsequent modules.
    - **Design Motivation**: Utilizes the in-context learning capability of language models to achieve zero-shot TTS, with a highly concise and efficient decoder-only architecture.

2. **Latent Sampling Module (Variational Sampling Module)**:

    - **Function**: Implements a sampling mechanism in the continuous space to enhance output diversity and robustness.
    - **Mechanism**: Based on the reparameterization trick of VAEs. For the LM output $\boldsymbol{e}_t$, a linear layer predicts the mean $\boldsymbol{\mu}_t$ and log-variance $\log \boldsymbol{\sigma}_t^2$ of a Gaussian distribution. Then, a latent variable is sampled using $\boldsymbol{z}_t = \boldsymbol{\mu}_t + \boldsymbol{\sigma}_t \odot \boldsymbol{\epsilon}$ with $\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$, which is then mapped back to the mel-spectrogram space via a 3-layer MLP with residual connections.
    - **Design Motivation**: Discrete models can use top-p sampling to introduce diversity, whereas continuous models cannot. VAE-style sampling automatically learns the distribution corresponding to each input, which is more adaptive than manually designed sampling strategies. Ablation studies demonstrate that this module contributes significantly to preserving speaker similarity (SIM).

3. **Spectrogram Flux Loss (Spectrogram Variation Loss)**:

    - **Function**: Encourages dynamic frame-to-frame variations in generation, preventing repetitions or silence.
    - **Mechanism**: $\mathcal{L}_{\text{flux}} = -\sum_{t=1}^{T-1} \|\boldsymbol{\mu}_t - \boldsymbol{y}_{t-1}\|_1$, which maximizes the L1 difference between the predicted mean and the ground-truth of the previous frame. This is a negative metric that rewards inter-frame changes and penalizes overly-static predictions.
    - **Design Motivation**: Pure regression loss can easily lead to predictions of overly smooth or repetitive frames, making the synthesized speech monotonous. The flux loss directly addresses this issue. Ablation studies show that on cross-sentence tasks, it reduces the WERH from 10.87 to 2.10.

4. **Reduction Factor**:

    - **Function**: Predicts multiple mel-spectrogram frames per step to accelerate inference.
    - **Mechanism**: Groups the sequence by a factor of $r$, predicting $r$ frames instead of 1 frame per step, which increases training and inference speed by approximately $r$ times.
    - **Design Motivation**: A trade-off between the efficiency and robustness of long sequence modeling. When $r=4$, the inference time is only 1.40s (compared to 7.32s for VALL-E), and the performance remains acceptable.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{\text{reg}} + \lambda \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{flux}} + \gamma \mathcal{L}_{\text{stop}}$

- **Regression Loss** $\mathcal{L}_{\text{reg}}$: L1 + L2 loss, applied simultaneously to the intermediate prediction $\boldsymbol{y}'$ and the post-net refined prediction $\boldsymbol{y}''$.
- **KL Divergence Loss** $\mathcal{L}_{\text{KL}}$: Restrains the latent variable distribution $p_\theta(\boldsymbol{z}_t|\boldsymbol{e}_t)$ to approach $\mathcal{N}(\boldsymbol{y}_t, \boldsymbol{I})$ (centered on ground-truth) instead of the standard VAE's $\mathcal{N}(0, I)$, acting as an optimization shortcut.
- **Stop Prediction Loss**: BCE loss + 100x positive sample weight to address the extreme imbalance of positive and negative frames.

Training Data: Libriheavy 50K hours, 6736 speakers; small-scale version uses LibriSpeech 960 hours.

## Key Experimental Results

### Main Results

| System | Training Data | Continuation WERH↓ | Continuation SIM↑ | Cross-Sentence WERH↓ | Cross-Sentence SIM↑ |
|------|---------|-------------------|-------------------|---------------------|---------------------|
| Ground Truth | - | 2.15 | 0.668 | 2.15 | 0.779 |
| VALL-E | 60K h | 3.8 | 0.508 | 5.9 | 0.580 |
| VALL-E 2 | 50K h | 2.32 | 0.504 | 2.44 | 0.643 |
| Voicebox | 60K h | 2.0 | 0.593 | 1.9 | 0.662 |
| **MELLE** | **50K h** | **1.98** | **0.508** | **2.10** | **0.625** |
| MELLE-R4 | 50K h | 2.10 | 0.437 | 2.30 | 0.532 |

Subjective evaluation (Cross-sentence):

| System | MOS↑ | SMOS↑ | CMOS↑ |
|------|------|-------|-------|
| Ground Truth | 4.29 | 3.94 | 0.000 |
| VALL-E | 3.18 | 3.50 | -0.912 |
| VALL-E 2 | 4.08 | 3.88 | -0.085 |
| **MELLE** | **4.20** | **4.40** | **-0.032** |

### Ablation Study

| Configuration | Cont. WERH↓ | Cont. SIM↑ | Cross WERH↓ | Cross SIM↑ |
|------|------------|-----------|------------|-----------|
| w/o LS + w/o SFL | 6.91 | 0.483 | 23.65 | 0.518 |
| w/ LS only | 4.07 | 0.486 | 10.87 | 0.584 |
| w/ SFL only | 2.61 | 0.506 | 5.90 | 0.602 |
| LS training only + SFL | 2.13 | 0.506 | 2.72 | 0.615 |
| **Full (LS + SFL)** | **1.98** | **0.508** | **2.10** | **0.625** |

### Key Findings
- Spectrogram Flux Loss contributes the most to robustness: removing it causes the cross-sentence WERH to surge from 2.10 to 10.87.
- Latent Sampling contributes significantly to speaker similarity: while its WER improvement is less than that of SFL, its SIM improvement is comparable to SFL.
- The reduction factor can reach R4 (active 4x inference acceleration) while still outperforming most baselines; performance begins to drop significantly starting from R5.
- MELLE's SMOS (4.40) even surpasses ground-truth speech (3.94), showing its exceptionally strong ability to capture speaker characteristics.
- CMOS shows no statistically significant difference from ground-truth speech ($p > 0.1$), reaching human-level quality.

## Highlights & Insights
- **Paradigm shift from discrete tokens to continuous tokens**: Against the backdrop of codec LMs dominance, it proves that continuous mel-spectrograms can completely replace discrete codes and yield superior results. This direction is transferable to other audio generation tasks such as music generation and audio editing.
- **Ingenious design of Spectrogram Flux Loss**: A simple negative L1 regularization term addresses the inter-frame repetition problem in continuous frame prediction. The underlying intuition is clear (maximizing inter-frame difference), and the implementation is light but highly effective. Similar ideas can be applied to video generation, motion sequence prediction, and other scenarios requiring sequence diversity.
- **KL divergence centered on ground-truth**: Unlike standard VAEs which use $\mathcal{N}(0, I)$ as the prior, MELLE utilizes $\mathcal{N}(\boldsymbol{y}_t, I)$, which provides regularization while accelerating convergence, serving as an optimization shortcut.
- **Single-stage vs. Two-stage**: By eliminating the complexity of the second-stage NAR, the model is simpler, inference is faster, and storage requirements are lower.

## Limitations & Future Work
- **Vocoder quality limitations**: Uses open-source HiFi-GAN (trained only on LibriTTS 585h); using a more powerful vocoder (trained on large-scale data) could further enhance the outcomes.
- **English-only evaluation**: Experiments were only conducted on LibriSpeech; multilingual capabilities have not been validated.
- **Mel-spectrogram only**: Has not explored other continuous representations such as VAE latent states; potentially better representation spaces may exist.
- **Security risks**: Ethical issues of zero-shot voice cloning—which can be used for voice impersonation, requiring accompanying synthetic detection models and speaker authorization mechanisms.

## Related Work & Insights
- **vs. VALL-E/VALL-E 2**: VALL-E uses discrete codec codes + two stages (AR+NAR), while MELLE uses continuous mel-spectrograms + a single stage. MELLE completely outperforms in robustness (WERH 1.98 vs 2.32) and obtains superior subjective quality (MOS 4.20 vs 4.08).
- **vs. Voicebox**: Voicebox is a non-autoregressive flow-matching method that achieves higher SIM but relies on a private vocoder + phoneme inputs; MELLE is already very close using only BPE + an open-source vocoder.
- **vs. Diffusion/Flow methods (CosyVoice, SEED-TTS)**: These methods still require two stages (AR to generate discrete codes + diffusion to generate continuous features), while MELLE directly generates continuous frames end-to-end, which is simpler.

## Rating
- Novelty: ⭐⭐⭐⭐ Continuous-value autoregressive TTS is an important exploration of paradigm, but the core technologies (VAE, regression loss) are not completely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive subjective and objective evaluations, meticulous ablation, and thorough exploration of the reduction factor.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem motivations, detailed method descriptions, and robust experimental analysis.
- Value: ⭐⭐⭐⭐ Holds significant reference value for the TTS field, though speech synthesis has relatively limited attention within the ACL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Chain-Talker: Chain Understanding and Rendering for Empathetic Conversational Speech Synthesis](chain-talker_chain_understanding_and_rendering_for_empathetic_conversational_spe.md)
- [\[ACL 2025\] Distilling an End-to-End Voice Assistant Without Instruction Training Data](distilling_an_end-to-end_voice_assistant_without_instruction_training_data.md)
- [\[ICLR 2026\] Towards True Speech-to-Speech Models Without Text Guidance](../../ICLR2026/audio_speech/towards_true_speech-to-speech_models_without_text_guidance.md)
- [\[NeurIPS 2025\] E2E-VGuard: Adversarial Prevention for Production LLM-based End-To-End Speech Synthesis](../../NeurIPS2025/audio_speech/e2e-vguard_adversarial_prevention_for_production_llm-based_end-to-end_speech_syn.md)
- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](../../ACL2026/audio_speech/bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)

</div>

<!-- RELATED:END -->

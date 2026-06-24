---
title: >-
  [Paper Note] Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation
description: >-
  [ACL 2025][Audio & Speech][Self-supervised speech representations] This paper proposes Eta-WavLM, which decomposes WavLM self-supervised speech representations into speaker-dependent and speaker-independent components using a simple linear equation. It generates high-quality speaker-disentangled representations without complex training, comprehensively outperforming existing methods in voice conversion tasks.
tags:
  - "ACL 2025"
  - "Audio & Speech"
  - "Self-supervised speech representations"
  - "speaker disentanglement"
  - "linear decomposition"
  - "voice conversion"
  - "WavLM"
date: 2026-05-08
content_hash: b47db21b2e7bdc33
---

# Eta-WavLM: Efficient Speaker Identity Removal in Self-Supervised Speech Representations Using a Simple Linear Equation

**Conference**: ACL 2025  
**arXiv**: [2505.19273](https://arxiv.org/abs/2505.19273)  
**Code**: [https://giuseppe-ruggiero.github.io/eta-wavlm-vc-demo/](https://giuseppe-ruggiero.github.io/eta-wavlm-vc-demo/) (Available, including audio samples)  
**Area**: Speech  
**Keywords**: Self-supervised speech representations, speaker disentanglement, linear decomposition, voice conversion, WavLM

## TL;DR

This paper proposes Eta-WavLM, which decomposes WavLM self-supervised speech representations into speaker-dependent and speaker-independent components using a simple linear equation. It generates high-quality speaker-disentangled representations without complex training, comprehensively outperforming existing methods in voice conversion tasks.

## Background & Motivation

**Background**: Self-supervised learning (SSL) has achieved revolutionary progress in the speech field. Models such as WavLM, HuBERT, and Wav2Vec 2.0 can learn rich, general-purpose representations from raw waveforms. These SSL representations encode various forms of information, such as linguistic content, speaker identity, emotion, and environment, leading to outstanding performance in various downstream tasks.

**Limitations of Prior Work**: (1) Speaker information and linguistic content in SSL representations are highly entangled, causing interference for tasks that require pure content representations (e.g., Voice Conversion (VC), TTS); (2) k-means quantization, while simple, degrades both linguistic content and prosodic information; (3) Existing disentanglement methods either fail to sufficiently remove speaker information or require complex training strategies, loss functions, or fine-tuning processes.

**Key Challenge**: The trade-off between speaker information removal and linguistic content preservation—removing too much degrades content quality, while removing too little hurts downstream task performance.

**Goal**: Efficiently remove speaker identity information from SSL representations while preserving linguistic content without requiring complex training strategies, loss functions, fine-tuning, or even quantization.

**Key Insight**: Based on the assumption that complex non-linear relationships tend to linearize in high-dimensional embedding spaces, SSL representations are modeled as a linear superposition of a speaker component and a speaker-independent component, which can be solved via the pseudo-inverse.

**Core Idea**: SSL representations can be linearly decomposed into $\mathbf{s} = f(\mathbf{d}) + \boldsymbol{\eta}$, where $\mathbf{d}$ is the speaker embedding and $\boldsymbol{\eta}$ is the speaker-independent *eta* representation, which can be solved using the least squares method.

## Method

### Overall Architecture

Eta-WavLM can be viewed as an offline extension module for SSL models, consisting of three components:

1. **SSL Model** (WavLM-Large, frozen): Extracts SSL representations $\mathbf{S}$ from raw waveforms.
2. **Speaker Encoder** (ECAPA-TDNN, frozen): Generates speaker embeddings $\mathbf{d}$ from the same waveform.
3. **Disentanglement Module**: Subtracts the speaker-dependent component from $\mathbf{S}$ using the learned latent basis $\mathbf{A}^*$ and bias $\mathbf{b}^*$ to obtain the eta representation $\boldsymbol{\eta}$.

### Key Designs

#### 1. Linear Decomposition Hypothesis

- **Function**: Assumes SSL representations are an additive combination of a speaker component and a speaker-independent component.
- **Mechanism**: $\mathbf{s} = f(\mathbf{d}) + \boldsymbol{\eta}$, where $f()$ is approximated by a linear model: $\mathbf{S} = \mathbf{D}^T \mathbf{A} + \mathbf{1}_N \mathbf{b}^T$.
- **Design Motivation**: Based on the finding of Ethayarajh et al. (2018) that large embedding spaces tend to linearize complex non-linear relationships. The linear assumption enables extremely efficient solving (via pseudo-inverse) without gradient-based training.

#### 2. Calculation of Latent Basis and Bias

- **Function**: Learns the linear mapping parameters from speakers to SSL representations from a multi-speaker dataset.
- **Mechanism**:
    - Extract WavLM representations $\mathbf{S} \in \mathbb{R}^{N \times Q}$ ($Q = 1024$, using output from layer 15) from each speech segment of the full LibriSpeech training set.
    - Extract ECAPA-TDNN speaker embeddings, and apply PCA to reduce the dimension to $P = 128$ to obtain $\mathbf{D} \in \mathbb{R}^{P \times N}$.
    - Solve via weight pseudo-inverse: $\tilde{\mathbf{A}}^* = (\tilde{\mathbf{D}}^T \tilde{\mathbf{D}})^{-1} \tilde{\mathbf{D}}^T \mathbf{S}$.
- **Design Motivation**: The entire "training" process requires only a single matrix operation, rather than iterative optimization. Reducing dimensionality via PCA ($V = 192 \to P = 128$) removes redundant information and is shown to positively impact performance.

#### 3. Generation of the Eta Representation

- **Function**: Generates the speaker-independent eta representation from audio during inference.
- **Mechanism**: Given audio $\mathbf{u}'$, extract SSL representation $\mathbf{S}$ and speaker embedding $\mathbf{d}$, then:
  $$\boldsymbol{\eta} = \mathbf{S} - \mathbf{1}_K (\mathbf{d}^T \mathbf{A}^* + \mathbf{b}^*)$$
- **Design Motivation**: Extremely simple operation—requiring only matrix multiplication and subtraction, which can be viewed as subtracting the speaker-related "offset" from the original representation.

### Loss & Training

- This method **does not involve traditional training**: it only requires a single least-squares optimization (pseudo-inverse computation).
- Parameters are computed using the full LibriSpeech training set (approx. 1000 hours of English speech).
- Fixed-length representations are constructed by randomly sampling $L = 100$ frames from each speech segment.
- The acoustic model in downstream VC tasks follows standard training pipelines.

## Key Experimental Results

### Main Results

**Speaker Classification Experiment** (10 speakers, 5-fold cross-validation, lower accuracy is better):

| Representation | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Mean ± Std |
|---------|-------|-------|-------|-------|-------|------------|
| WavLM | 83.46 | 82.33 | 80.85 | 83.30 | 81.55 | 82.30 ± 0.01 |
| **Eta-WavLM** | 53.82 | 55.14 | 58.77 | 53.94 | 56.96 | **55.73 ± 0.01** |

Eta-WavLM's speaker classification accuracy drops from 82.30% to 55.73% (a decrease of nearly 27%), proving that speaker information is effectively removed. p-value = 5.12×10⁻⁵, showing a significant difference.

**Voice Conversion Experiment** (two target speakers):

| Method | LJSpeech WER↓ | LJ T-SSIM↑ | LJ MOS↑ | Elliot WER↓ | Elliot T-SSIM↑ | Elliot MOS↑ |
|------|---------------|------------|---------|-------------|----------------|-------------|
| Perturbation | 6.29 | 91.69 | 3.45 | 10.76 | 87.41 | 3.13 |
| Normalization | 4.13 | 90.34 | 3.80 | 5.16 | 85.91 | 3.41 |
| Soft units | 4.82 | 91.81 | 3.84 | 5.50 | 86.69 | 3.32 |
| VQ | 4.79 | 90.05 | 3.90 | 7.72 | 86.30 | 3.50 |
| Original WavLM | 4.56 | 89.52 | 3.84 | 5.14 | 86.18 | 3.66 |
| **Eta-WavLM** | **3.81** | **92.46** | **4.00** | **4.64** | **89.32** | **3.79** |

Eta-WavLM achieves the best performance across all metrics: WER of 3.81% on LJSpeech (close to ground-truth speech of 3.22%), T-SSIM of 92.46%, and MOS of 4.00.

### Ablation Study

**Ablation on Speaker Encoder and PCA**:

| Configuration | WER↓ | T-SSIM↑ | SPK ACC↓ |
|------|------|---------|----------|
| Resemblyzer w/o PCA | 4.94 | 89.02 | 74.01 |
| WavLM-SV w PCA-128 | 3.91 | 89.76 | 65.83 |
| ECAPA-TDNN w/o PCA | 4.18 | 89.90 | 60.87 |
| ECAPA-TDNN w PCA-64 | 3.95 | 90.91 | 58.14 |
| **ECAPA-TDNN w PCA-128** | **3.81** | **92.46** | **55.73** |

ECAPA-TDNN + PCA-128 is the optimal choice across all metrics. PCA dimension reduction positively impacts performance, but excessive reduction (PCA-64) leads to performance degradation.

### Key Findings

1. **The Linear Decomposition Hypothesis Holds**: A simple linear equation can effectively separate speaker and content information, which is a powerful empirical finding.
2. **Speaker Encoder Selection is Critical**: ECAPA-TDNN outperforms Resemblyzer and WavLM-SV by a wide margin, showing that the encoder quality directly impacts disentanglement performance.
3. **PCA Dimension Reduction Aids Disentanglement**: Removing redundant information in speech embeddings enhances decomposition quality.
4. **Balancing the Voice Conversion Trilemma**: Eta-WavLM is the only method achieving optimal performance across intelligibility, speaker similarity, and audio quality.

## Highlights & Insights

1. **Extreme Simplicity**: The entire "training" only requires a single matrix pseudo-inverse operation without GPU-based iterative training, presenting an elegant solution.
2. **Clear Theoretical Intuition**: The subtraction operation $\boldsymbol{\eta} = \mathbf{s} - f(\mathbf{d})$ is intuitive and easy to understand—it simply subtracts the speaker's "offset."
3. **Excellent Versatility**: Acting as a post-processing module for SSL models, this method does not modify the original model parameters and is plug-and-play.
4. **Compelling UMAP/PaCMAP Visualizations**: Moving from clusters in WavLM to a uniform distribution in Eta-WavLM is visually convincing.

## Limitations & Future Work

1. Speaker information is not completely removed—the 55.73% classification accuracy is still higher than the random chance level (10%), leaving some residual speaker information.
2. Validated only on an English dataset (LibriSpeech); its multilingual generalization capability remains unverified.
3. The linear assumption might be imprecise in certain scenarios, and non-linear decomposition methods might offer further improvements.
4. Only WavLM was tested among SSL models; applicability to other SSL models such as HuBERT and Wav2Vec 2.0 has yet to be verified.

## Related Work & Insights

- **ContentVec (Qian et al., 2022)**: Performs speaker disentanglement via training; high complexity but shares a similar idea.
- **k-means Quantization (Hsu et al., 2021)**: The simplest disentanglement approach, but loses both content and prosody.
- **van Niekerk et al. (2022)**: The soft speech units method performs well in VC but does not achieve complete disentanglement.
- **RepCodec (Huang et al., 2024)**: Vector quantization approach; achieves decent MOS but is inferior to Eta-WavLM in intelligibility.
- **Insight**: In high-dimensional spaces, linear methods might be far more powerful than expected; disentanglement tasks may not necessarily require complex adversarial training or specialized loss functions.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Highly creative in solving complex speaker disentanglement problems using a single linear equation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Thoroughly validated with both speaker classification and VC tasks, accompanied by comprehensive ablation studies, though lacking multilingual experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear mathematical derivations and detailed experimental descriptions.
- **Value**: ⭐⭐⭐⭐⭐ — Simple, efficient, and highly versatile, with a broad potential impact on the speech processing community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] An Exploration of Mamba for Speech Self-Supervised Models](../../ACL2026/audio_speech/an_exploration_of_mamba_for_speech_self-supervised_models.md)
- [\[ACL 2026\] \[b\] = \[d\] − \[t\] + \[p\]: Self-supervised Speech Models Discover Phonological Vector Arithmetic](../../ACL2026/audio_speech/bd-tp_self-supervised_speech_models_discover_phonological_vector_arithmetic.md)
- [\[ICML 2025\] Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech](../../ICML2025/audio_speech/do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech.md)
- [\[ACL 2025\] Different Speech Translation Models Encode and Translate Speaker Gender Differently](different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)
- [\[ACL 2025\] MMS-LLaMA: Efficient LLM-based Audio-Visual Speech Recognition with Minimal Multimodal Speech Tokens](mms-llama_efficient_llm-based_audio-visual_speech_recognition_with_minimal_multi.md)

</div>

<!-- RELATED:END -->

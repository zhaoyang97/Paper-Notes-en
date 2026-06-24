---
title: >-
  [Paper Note] FLAM: Frame-Wise Language-Audio Modeling
description: >-
  [ICML2025][Audio & Speech][Open-Vocabulary Sound Event Detection] Proposes FLAM, a frame-level audio-language contrastive model that achieves precise temporal localization of open-vocabulary sound events through text-dependent logit bias correction and a million-scale synthetic SED dataset, while maintaining outstanding performance in global retrieval and zero-shot classification.
tags:
  - "ICML2025"
  - "Audio & Speech"
  - "Open-Vocabulary Sound Event Detection"
  - "Frame-Level Contrastive Learning"
  - "Logit Adjustment"
  - "Audio-Language Alignment"
  - "Data Augmentation"
date: 2026-05-08
content_hash: 129f464346420821
---

# FLAM: Frame-Wise Language-Audio Modeling

**Conference**: ICML2025  
**arXiv**: [2505.05335](https://arxiv.org/abs/2505.05335)  
**Code**: [flam-model.github.io](https://flam-model.github.io/)  
**Area**: Audio & Speech  
**Keywords**: Open-Vocabulary Sound Event Detection, Frame-Level Contrastive Learning, Logit Adjustment, Audio-Language Alignment, Data Augmentation

## TL;DR

Proposes FLAM, a frame-level audio-language contrastive model that achieves precise temporal localization of open-vocabulary sound events through text-dependent logit bias correction and a million-scale synthetic SED dataset, while maintaining outstanding performance in global retrieval and zero-shot classification.

## Background & Motivation

- **Limitations of Prior Work (ALMs)**: Audio-language models like CLAP learn instance-level global embeddings and excel at text-audio retrieval, but fail to precisely localize the temporal boundaries of sound events.
- **Limitations of Traditional SED**: Traditional sound event detection (SED) models can localize events precisely but are confined to predefined categories, failing to handle out-of-distribution events in an open-vocabulary manner.
- **Scarcity of Annotated Data**: Unlike the image domain, audio frame-level annotations are extremely scarce, and manual labeling is highly costly. Existing SED datasets are small in scale and limited in categories.
- **Insufficiency of Self-Supervised Methods**: Previous self-supervised local alignment methods (e.g., MGA-CLAP) improve frame-level capabilities to some extent, but lack fine-grained annotations, resulting in limited localization performance.

**Core Problem**: How can ALMs be endowed with open-vocabulary, frame-level sound event localization capabilities while preserving their global retrieval capacity?

## Method

### Overall Architecture

FLAM is based on the LAION-CLAP architecture (HTSAT audio encoder + RoBERTa text encoder), extended to simultaneously output both global embeddings and frame-level embedding sequences. A 10-second audio input processed by HTSAT yields a frame-level embedding sequence $\mathbf{e}^{a,loc}(x) \in \mathbb{R}^{L \times d}$ of length $L=32$, and the global representation is obtained by averaging these frame embeddings.

### Frame-Level Contrastive Loss

Open-vocabulary SED is modeled as a frame-level binary classification task, which determines whether an event is active in each frame for every (audio frame, text event description) pair:

$$\mathcal{L}_{\text{SED}} = -\frac{1}{BKL}\sum_{i=1}^{B}\sum_{k=1}^{K}\sum_{l=1}^{L}\log\sigma\big(z_{i,k,l}\cdot h(X_i, l, \mathcal{Y}_k)\big)$$

where the logit function incorporates **text-dependent scaling and bias**:

$$h(x, l, y) = \alpha^t(y)\;\mathbf{e}^{a,loc}(x)_l \cdot \mathbf{e}^t(y) + \beta^t(y)$$

- $\alpha^t(y) > 0$: Text-dependent logit scaling, produced by $\text{MLP}^\alpha(E^t(y))$
- $\beta^t(y)$: Text-dependent logit bias, produced by $\text{MLP}^p(E^t(y))$

### Logit Adjustment: Handling Event-Dependent Class Imbalance

Frame-level labels suffer from severe imbalance (most frame-text pairs are negative samples), and the degree of imbalance varies across events (e.g., "thunder" is rare and short-lived, while "rain" is frequent and long-lasting).

**Bayesian Optimal Classifier**: During inference, an unbiased classifier is used to normalize raw predictions into a localization score independent of the event prior:

$$s(x, l, y) = \frac{p(z=1|x,l,y)}{p(z=1|x,l,y) + p(z=1|y)} \approx \sigma\!\left(\log\frac{p(y|x,l)}{p(y)}\right)$$

**Bias Training**: $\beta^t$ is trained independently to approximate the optimal bias $\beta^*(y) = \log\frac{p(z=1|y)}{p(z=-1|y)}$ using an auxiliary loss $\mathcal{L}_p$, while blocking gradient propagation from $\mathcal{L}_{\text{SED}}$ to it:

$$\mathcal{L}_p = -\frac{1}{K}\sum_{k=1}^{K}\big[\bar{z}_k\log\sigma(\beta^t(\mathcal{Y}_k)) + (1-\bar{z}_k)\log\sigma(-\beta^t(\mathcal{Y}_k))\big]$$

### Joint Training Objective

$$\mathcal{L} = \gamma^{\text{CLIP}}\mathcal{L}_{\text{CLIP}} + \gamma^{\text{SED}}\mathcal{L}_{\text{SED}} + \gamma^p\mathcal{L}_p$$

where $\gamma^{\text{CLIP}}=1$, $\gamma^{\text{SED}}=200$, and $\gamma^p=1$.

### Memory-Efficient Training

A SigLIP-style block-wise ring communication strategy is adopted: each GPU processes a local subset of the frame-text pair loss, and text embeddings are passed around the GPUs in a ring. This prevents the gathering of all embeddings on a single GPU, enabling large-batch training.

### Data Augmentation Pipeline

Synthesizes **1 million 10-second mixed samples** from 1.1M audio source clips (licensed sound effects library + CC-licensed general-purpose datasets):

1. Randomly select a background audio ($\ge$10s, containing the "ambiance" keyword).
2. Sample $N \sim \mathcal{U}(1, 10)$ foreground events (80% from sound effects, 20% from general datasets).
3. Randomly place events, with a maximum of 3 overlapping concurrently.
4. Split 10% of the events into 2-3 fragments, and repeat another 10% 2-3 times to simulate realistic scenarios.
5. Apply a random loudness shift of $\mathcal{U}(6,30)$ dB and a 10ms fade-in/fade-out.
6. Perform boundary calibration based on A-weighted RMS loudness (regions below -70 dB are marked inactive).

Using Mixtral to generate 2-13 word captions for the sound effects.

## Key Experimental Results

### Sound Event Detection (Table 1)

| Model | Held-out AUROC | ASFX-SED AUROC | DESED AUROC | MAESTRO MPAUC | AudioSet-S AUROC | UrbanSED AUROC |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| FLAM-Global | 67.76 | 65.14 | 85.52 | 51.13 | 82.54 | 67.39 |
| **FLAM** | **91.00** | **81.23** | **91.66** | **56.97** | **94.76** | **93.62** |
| MGA-CLAP* | 74.17 | 69.56 | 89.28 | 52.50 | 79.12 | 78.22 |

FLAM significantly outperforms baselines across almost all metrics, improving the open-vocabulary SED (Held-out) AUROC from 67.76 to 91.00.

### Zero-Shot Classification (Table 3)

| Model | ESC-50 | US8K | VGGSound |
|------|:---:|:---:|:---:|
| FLAM-Global | 81.6 | 65.4 | 38.9 |
| **FLAM** | **86.9** | **75.6** | **39.3** |
| MGA-CLAP* | 72.6 | 69.9 | 38.6 |

Frame-level supervision does not compromise global representations; on the contrary, it enhances zero-shot classification accuracy.

### Retrieval Performance (Table 2)

FLAM achieves retrieval performance close to that of FLAM-Global (AudioCaps T2A R@1: 32.1 vs 36.0), demonstrating that frame-level training has minimal negative impact on global retrieval.

## Highlights & Insights

1. **Clear Modeling of Open-Vocabulary SED**: Formulates frame-level event localization as frame-text pair binary classification. By inheriting the contrastive learning framework, it only needs to encode new text queries once audio embeddings are precomputed during inference, achieving high efficiency.
2. **Text-Dependent Logit Adjustment**: Introduces event-specific scaling and bias to handle event-level class imbalance, transforming predictions from raw cosine similarity into calibrated probabilities with mathematically rigorous derivations.
3. **Large-Scale Synthetic Data Pipeline**: Overcomes the scarcity of frame-level annotations by synthesizing 1 million mixed samples with precise boundary labels, successfully simulating realistic scenarios using splitting and repetition strategies.
4. **Frame-Level Supervision Benefits Global Representations**: The improvement in zero-shot classification performance (ESC-50 81.6 $\to$ 86.9) demonstrates that fine-grained alignment also boosts overall discriminative ability.
5. **Memory-Efficient Training**: Implementing ring transmission avoids centralized gathering, making large-scale frame-level contrastive training highly feasible.

## Limitations & Future Work

1. **Fixed Input Length**: Limits support to 10-second audio with a coarse frame resolution (32 frames/10s), making it challenging to handle long-form audio or scenarios requiring finer temporal granularity.
2. **Synthetic vs. Real Data Gap**: Since training data consists of synthesized mixtures, real-world scenes feature more complex reverberations, masking, and co-occurrence patterns.
3. **Lightweight Model Scale**: The HTSAT + RoBERTa architecture is relatively lightweight; adopting larger encoders or more expressive backbones may yield further improvements.
4. **Suboptimal PSDS on DESED**: Having only 692 real annotated samples leads to high variance, hinting at potential limitations in scenarios with small-scale real annotations.
5. **Minor Drop in Retrieval Performance**: AudioCaps T2A R@1 drops from 36.0 to 32.1, indicating a slight trade-off between frame-level training and global retrieval targets.

## Related Work & Insights

- **CLAP / LAION-CLAP**: Serves as the backbone architecture for FLAM, extending its capability from instance-level to frame-level alignment.
- **SigLIP**: The conceptual ideas of sigmoid contrastive loss and logit bias directly inspired the design of FLAM's frame-level objective.
- **MGA-CLAP**: A representative method for self-supervised local alignment that serves as a primary baseline, though it lacks explicit frame-level supervision.
- **Scaper / Synthetic SED Data**: Traditional synthesizer frameworks for SED data, which FLAM extends to open-vocabulary contexts.
- **GLIP / PACL (Computer Vision)**: Open-vocabulary detection/segmentation works in the visual domain that provide analogous inspiration for the audio domain.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Successfully combines frame-level contrastive learning with text-dependent logit adjustment for open-vocabulary SED with a clear and highly original approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Thoroughly covers open/closed SED, retrieval, and zero-shot classification, supported by comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematically rigorous derivations, consistent notations, and highly intuitive diagrams.
- **Value**: ⭐⭐⭐⭐ — Open-vocabulary frame-level audio localization is a practical and vital direction; the proposed method is generalizable to broader audio understanding tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] IMPACT: Iterative Mask-based Parallel Decoding for Text-to-Audio Generation with Diffusion Modeling](impact_iterative_mask-based_parallel_decoding_for_text-to-audio_generation_with_.md)
- [\[ICML 2025\] NTPP: Generative Speech Language Modeling for Dual-Channel Spoken Dialogue via Next-Token-Pair Prediction](ntpp_generative_speech_language_modeling_for_dual-channel_spoken_dialogue_via_ne.md)
- [\[ICLR 2026\] FlexiCodec: A Dynamic Neural Audio Codec for Low Frame Rates](../../ICLR2026/audio_speech/flexicodec_a_dynamic_neural_audio_codec_for_low_frame_rates.md)
- [\[ACL 2025\] Improving Language and Modality Transfer in Translation by Character-level Modeling](../../ACL2025/audio_speech/improving_language_and_modality_transfer_in.md)
- [\[NeurIPS 2025\] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space](../../NeurIPS2025/audio_speech/efficient_speech_language_modeling_via_energy_distance_in_continuous_latent_spac.md)

</div>

<!-- RELATED:END -->

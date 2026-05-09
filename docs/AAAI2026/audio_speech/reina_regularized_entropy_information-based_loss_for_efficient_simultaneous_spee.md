---
title: >-
  [Paper Note] REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation
description: >-
  [AAAI 2026][Audio & Speech][Simultaneous speech translation] This paper proposes REINA (Regularized Entropy INformation Adaptation), a loss function grounded in mutual information theory that efficiently converts a non-streaming speech translation model into a streaming simultaneous speech translation model. REINA achieves state-of-the-art streaming translation performance across multiple language directions and introduces a new streaming efficiency metric, NoSE.
tags:
  - AAAI 2026
  - "Audio & Speech"
  - Simultaneous speech translation
  - streaming translation
  - information theory
  - adaptive policy
  - speech translation
date: 2026-05-08
content_hash: e98a86e5bac8959d
---

# REINA: Regularized Entropy Information-Based Loss for Efficient Simultaneous Speech Translation

**Conference**: AAAI 2026
**arXiv**: [2508.04946](https://arxiv.org/abs/2508.04946)
**Code**: None
**Area**: Audio & Speech
**Keywords**: Simultaneous speech translation, streaming translation, information theory, adaptive policy, speech translation

## TL;DR

This paper proposes REINA (Regularized Entropy INformation Adaptation), a loss function grounded in mutual information theory that efficiently converts a non-streaming speech translation model into a streaming simultaneous speech translation model. REINA achieves state-of-the-art streaming translation performance across multiple language directions and introduces a new streaming efficiency metric, NoSE.

## Background & Motivation

### Core Problem

Simultaneous speech translation (SimulST) requires producing translated text while continuously receiving a speech stream. The central challenge lies in the **trade-off between translation quality and latency**. The system must follow a READ/WRITE policy to decide when to wait for more input and when to generate output.

### Limitations of Prior Work

**Fixed policies (e.g., wait-k)**: Simple but suboptimal, as the audio frame sampling rate does not match the output token rate, and different language pairs exhibit different word-order divergences.

**Monotonic attention methods (e.g., EMMA/MMA)**: Embed the policy into the model architecture, offering strong expressiveness but at the cost of extremely expensive and unstable training. EMMA requires computing matrices of size $[\text{batch} \times \text{heads} \times \text{tokens} \times \text{audio\_len} \times \text{audio\_len}]$ at every cross-attention layer, limiting batch size to 1 on an A100-80G, and the cumulative product operation is numerically unstable.

**Reinforcement learning methods**: Directly optimize the quality–latency trade-off but suffer from training instability and lack convergence guarantees.

**DiG-SST** (most closely related): Guides policy training via output distribution divergence from a non-streaming model, which is efficient but **does not leverage ground-truth label information** when computing divergence scores.

### Core Idea

**The system should READ only when waiting yields an information gain.** This intuition can be rigorously formalized via mutual information theory: the additional information that the full audio provides about the next translation token, compared to partial audio, determines whether the system should continue waiting.

## Method

### Overall Architecture

REINAStream is trained in three stages:
1. Non-streaming S2TT model training (multi-task learning: ASR + NMT + S2TT)
2. Truncated audio adaptation training
3. REINA policy network training

### Key Designs

#### 1. **REINA Policy Loss: Formalizing Information Gain**

**Core derivation**:

The information gain of waiting for the remaining audio with respect to the next token $s_{n+1}$ is defined as:

$$\mathcal{F}(a, S, n, t) := I(s_{n+1}; a_T, S_n) - I(s_{n+1}; a_t, S_n)$$

Expanding via mutual information yields a difference of conditional entropies:

$$\mathcal{F} = H(s_{n+1}|a_t, S_n) - H(s_{n+1}|a_T, S_n) = \mathbb{E}[\log p(s_{n+1}|a_T, S_n) - \log p(s_{n+1}|a_t, S_n)]$$

**Key approximation**: The log-probabilities of the non-streaming S2TT model are used to estimate the true conditional probabilities — specifically, the difference between the cross-entropy loss under full audio and that under partial audio.

**Policy network training**: Since the complete target text is unavailable at inference time, a lightweight policy network $q_\theta$ is trained to estimate the information gain by maximizing the covariance between $q_\theta$ and $\hat{\mathcal{F}}$:

$$\mathcal{L}_p = \frac{1}{N} \sum_{n=0}^{N-1} q_\theta^n \cdot \text{BN}[\log \hat{p_t}^{s_{n+1}} - \log \hat{p_T}^{s_{n+1}}]$$

where BN denotes batch normalization (zero-centering the information gain estimates), which eliminates the constant term in the covariance formula.

**Design Motivation**: The key distinction from DiG-SST is that REINA leverages ground-truth labels $s_{n+1}$ to compute information gain (by evaluating the model's log-probability on the correct token), whereas DiG-SST only compares distributional divergences, ignoring which tokens are actually correct.

#### 2. **Regularization Terms**

**Monotonicity constraint**: At inference time, once the policy decides to READ, no further tokens are generated; accordingly, the training objective encourages $q_\theta$ to be approximately non-decreasing along the token sequence:

$$\mathcal{L}_m = \frac{1}{N} \sum_{n=1}^{N} \max(\max_{m<n}\{q_\theta^m\} - q_\theta^n - \epsilon, 0)$$

This encourages committed behavior — once the information gain exceeds a threshold, the policy consistently stops generating.

**L2 regularization**: $\mathcal{L}_r = \frac{1}{N}\sum (q_\theta^n)^2$ prevents $q_\theta$ values from exploding.

**Full REINA loss**: $\mathcal{L}_{\text{REINA}} = \mathcal{L}_p + \mathcal{L}_m + \lambda \mathcal{L}_r$, with $\lambda = 0.05$.

#### 3. **Model Architecture and Training**

**Non-streaming model** (408M parameters at inference): Whisper Medium encoder (307M) + 16-layer Transformer decoder (101M) + trainable T5 text encoder (38M, used only during training for the NMT task).

**Policy network** (6M parameters): 2-layer Transformer encoder applied to the final decoder hidden states, followed by a linear layer and sigmoid for binary READ/WRITE decisions.

**Three-stage training**:
- Stage 1: Non-streaming S2TT, multi-task (ASR + NMT + S2TT), 24× A100, 5 days, 130k hours of data
- Stage 2: Truncated audio adaptation, 80% randomly truncated + 20% full audio, 2 days
- Stage 3: REINA policy training, all other parameters frozen, only the 6M policy network trained, completed in <12 hours over 20 epochs

#### 4. **NoSE Evaluation Metric**

**Motivation**: Existing AL vs. BLEU curve comparisons are unfair — a model with a higher non-streaming BLEU may appear to have better streaming performance for that reason alone.

**Definition**: NoSE = area under the AL/BLEU curve / area under the non-streaming BLEU line (within bounds $[x, y]$). After normalization, this enables fair comparison of streaming conversion efficiency across models with different non-streaming baselines.

### Loss & Training

- Stages 1–2: $\mathcal{L} = \mathcal{L}_{\text{asr}} + \mathcal{L}_{\text{nmt}} + \mathcal{L}_{\text{s2tt}}$
- Stage 3: $\mathcal{L}_{\text{REINA}} = \mathcal{L}_p + \mathcal{L}_m + 0.05 \cdot \mathcal{L}_r$, policy network only
- Data: MLS (Multilingual LibriSpeech), MUST-C, CVSS-C, MOSEL + CCMatrix text pairs

## Key Experimental Results

### Main Results

NoSE scores (↑ higher is better) — MUST-C dataset:

| Model | en→de | en→fr | en→es |
|-------|-------|-------|-------|
| DiG-SST (Original) | 0.888 | 0.903 | 0.879 |
| DiSeg | 0.838 | - | 0.774 |
| EDAtt | 0.704 | - | 0.740 |
| DiG-SST (Authors' reproduction) | 0.665 | 0.774 | 0.607 |
| **REINA (MUST-C only)** | **0.940** | **0.953** | **0.960** |
| REINA (Full data) | 0.925 | 0.944 | 0.952 |

NoSE scores — CVSS-C dataset:

| Model | de→en | fr→en | es→en |
|-------|-------|-------|-------|
| StreamSpeech* | 0.842 | 0.886 | 0.837 |
| REINA | **0.974** | **0.983** | **0.981** |

Selected operating point comparison (MUST-C, low latency):

| Model | en→de AL↓ | BLEU↑ | en→es AL↓ | BLEU↑ | en→fr AL↓ | BLEU↑ |
|-------|----------|-------|----------|-------|----------|-------|
| REINA | **1.01** | **21.44** | **0.86** | **26.92** | **0.77** | **33.13** |
| DiG-SST | 1.08 | 21.13 | 0.90 | 23.92 | 1.11 | 30.51 |

### Ablation Study

| Configuration | en→de NoSE | en→fr NoSE | en→es NoSE |
|---------------|-----------|-----------|-----------|
| REINA (Full) | 0.925 | 0.944 | 0.952 |
| REINA w/o monotonicity | 0.899 | 0.920 | 0.909 |
| REINA (MUST-C only) | 0.940 | 0.953 | 0.960 |
| REINA w/o truncation training | 0.840 | 0.839 | 0.895 |
| DiG-SST (Authors' reproduction) | 0.665 | 0.774 | 0.607 |

### Key Findings

1. **REINA substantially outperforms DiG-SST**: The MUST-C-only variant achieves a NoSE improvement of 3.0% over the original DiG-SST and 8.9% over DiSeg; the gains are not solely attributable to additional training data.
2. **Clear advantage at low latency**: At ~35 BLEU, the monotonicity constraint reduces AL from 1.95 to 1.57 (a 19% reduction).
3. **Truncation training is critical**: Skipping Stage 2 leads to a substantial drop in NoSE (e.g., en→de falls from 0.932 to 0.840), demonstrating that REINA relies on accurate log-probability estimates for partial audio.
4. **Monotonicity is most effective in low-latency regimes**: It helps the policy make clear READ/WRITE boundary decisions when information gain fluctuates.
5. **SOTA achieved with open-source data only**: 130k hours of open-source and synthetic data, a scale far smaller than Seamless's 600k hours.

## Highlights & Insights

1. **Solid information-theoretic foundation**: The intuition ("only wait if waiting is informative") is rigorously derived as a mutual information difference and then approximated into a tractable loss function, with a clear derivation chain.
2. **Highly efficient training**: The policy network has only 6M parameters and training completes in under 12 hours — a qualitative leap compared to the enormous computational cost of EMMA.
3. **Introduction of the NoSE metric**: Addresses a long-standing fairness issue in evaluation, enabling equitable comparison of streaming conversion efficiency across models of different scales.
4. **In-depth analysis of EMMA** (appendix): The authors thoroughly document their failed attempts to reproduce EMMA (memory explosion, numerical instability, ambiguous layer/head selection choices), providing valuable lessons for the community.
5. **Open-source-first research philosophy**: The work emphasizes exclusive use of open-source training data, aiming to bridge the gap between industrial large-data systems and academic small-scale settings.

## Limitations & Future Work

1. The policy threshold $\alpha$ must be determined by trial and error, with no automatic selection mechanism.
2. Only en↔{de, fr, es} language pairs are covered; low-resource languages remain untested.
3. The model scale (408M) sits between small and industrial-grade; behavior at much larger scales may differ.
4. The NoSE metric is sensitive to the choice of boundary $[x, y]$.
5. The work does not extend to speech-to-speech translation (SimulS2ST), which the authors identify as future work.
6. Although categorized under "human body understanding," the work properly belongs to the NLP/speech domain.

## Related Work & Insights

- **DiG-SST**: The most direct point of comparison; REINA builds upon it by incorporating ground-truth label information, with the essential distinction being the use of mutual information rather than KL divergence.
- **SeamlessM4T/EMMA**: Industrial-scale systems that demonstrate the capability ceiling of monotonic attention methods while also exposing their training difficulties.
- **Transducer architectures**: Natively support streaming but face similar convergence challenges during training.
- **wait-k policy**: As the simplest baseline, it remains competitive in works such as SimulS2S-LLM.

## Rating

- Novelty: ⭐⭐⭐⭐ (The mutual information perspective is elegant, though the difference from DiG-SST is incremental)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple languages, multiple datasets, complete ablations, fair comparisons, detailed appendix)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear derivations; the appendix is exceptionally informative)
- Value: ⭐⭐⭐⭐ (Efficient streaming conversion approach; the NoSE metric has broad applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SimulMEGA: MoE Routers are Advanced Policy Makers for Simultaneous Speech Translation](../../NeurIPS2025/audio_speech/simulmega_moe_routers_are_advanced_policy_makers_for_simultaneous_speech_transla.md)
- [\[AAAI 2026\] Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning](modelling_the_effects_of_hearing_loss_on_neural_coding_in_the_auditory_midbrain_.md)
- [\[ICLR 2026\] Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](../../ICLR2026/audio_speech/scalable_multilingual_multimodal_machine_translation_with_speech-text_fusion.md)
- [\[AAAI 2026\] DualSpeechLM: Towards Unified Speech Understanding and Generation via Dual Speech Token Modeling](dualspeechlm_towards_unified_speech_understanding_and_generation_via_dual_speech.md)
- [\[ICCV 2025\] Lyra: An Efficient and Speech-Centric Framework for Omni-Cognition](../../ICCV2025/audio_speech/lyra_an_efficient_and_speech-centric_framework_for_omni-cognition.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems
description: >-
  [ICML 2025][Audio & Speech][Speaker Diarization] This work proposes Sortformer—an encoder-based speaker diarization model that resolves the permutation problem by sorting speakers according to their arrival times using Sort Loss, thereby replacing or supplementing the traditional Permutation Invariant Loss (PIL). It designs a sinusoidal kernel function to inject speaker labels into the ASR encoder, enabling multi-speaker ASR training with standard cross-entropy loss and achie…
tags:
  - "ICML 2025"
  - "Audio & Speech"
  - "Speaker Diarization"
  - "Multi-Speaker ASR"
  - "Permutation Invariant Training"
  - "Arrival Time Sorting"
  - "Sort Loss"
  - "End-to-End Diarization"
date: 2026-05-08
content_hash: 086aaa96094dc8ea
---

# Sortformer: A Novel Approach for Permutation-Resolved Speaker Supervision in Speech-to-Text Systems

**Conference**: ICML 2025  
**arXiv**: [2409.06656](https://arxiv.org/abs/2409.06656)  
**Code**: [NVIDIA NeMo Framework](https://huggingface.co/nvidia/diar_sortformer_4spk-v1)  
**Area**: Speech/Speaker Diarization  
**Keywords**: Speaker Diarization, Multi-Speaker ASR, Permutation Invariant Training, Arrival Time Sorting, Sort Loss, End-to-End Diarization

## TL;DR

This work proposes Sortformer—an encoder-based speaker diarization model that resolves the permutation problem by sorting speakers according to their arrival times using Sort Loss, thereby replacing or supplementing the traditional Permutation Invariant Loss (PIL). It designs a sinusoidal kernel function to inject speaker labels into the ASR encoder, enabling multi-speaker ASR training with standard cross-entropy loss and achieving relative error reductions of 30% and 25% on 2-mix and 3-mix LibriSpeechMix, respectively.

## Background & Motivation

**Core Challenge of Multi-Speaker ASR**: With the widespread deployment of ASR in industrial scenarios (e.g., meeting transcription, conversation analysis), an increasing number of applications require transcription results to be accompanied by speaker labels. However, integrating speaker diarization with ASR models has long faced key challenges such as data scarcity, domain adaptation difficulties, and permutation matching.

**Limitations of Permutation Invariant Loss (PIL)**: Existing end-to-end diarization systems (such as EEND-SA, EEND-EDA) rely on PIL to solve the speaker label permutation problem. PIL requires iterating through all permutations to find the minimum loss, which introduces two issues: (1) computational complexity grows factorially with the number of speakers; (2) specialized loss functions are required at the model output layer, limiting integration with multi-task ASR systems.

**Gap between End-to-End and Cascaded Systems**: Although end-to-end multi-speaker ASR systems (such as SOT) feature elegant architectures, their performance still lags behind cascaded systems due to a lack of sufficient speaker-attributed training data. While cascaded systems are powerful, they require separate tuning of individual modules for specific domains, leading to high deployment costs.

**Core Motivation of This Work**: Can a permutation-resolving method be designed such that multi-speaker ASR training becomes completely consistent with single-speaker ASR training at the loss function level (i.e., using standard cross-entropy), thereby achieving "plug-and-play" speaker attribution capabilities?

## Method

### Overall Architecture

The Sortformer framework comprises two core components:

1. **Sortformer Diarization Model**: An end-to-end speaker diarization model based on the Transformer encoder. It learns to generate speaker labels ordered by arrival time via Sort Loss, outputting a frame-level speaker existence probability matrix $\mathbf{P} \in \mathbb{R}^{K \times T}$.
2. **Multi-Speaker ASR System (MS-ASR)**: Injects the outputs of Sortformer into the encoder states of a pre-trained ASR model through a sinusoidal kernel function, enabling differentiable transmission of speaker information.

The overall data flow is: Audio $\rightarrow$ NEST encoder extracting frame-level features $\rightarrow$ Sortformer generating permutation-resolved speaker probabilities $\rightarrow$ Sinusoidal kernel encoding speaker information into ASR encoder states $\rightarrow$ ASR decoder generating speaker-tagged transcriptions.

### Key Designs

#### 1. Multi-Label Binary Classification Modeling

Sortformer models speaker diarization as a frame-level multi-label binary classification problem. Given a sequence of $T$ frames of $D$-dimensional embeddings $\{\mathbf{x}_t\}_{t=1}^T$, the model outputs the existence probability of $K$ speakers at each frame. The key assumption is that individual speakers are conditionally independent given the features, thus using Sigmoid instead of Softmax as the output activation function:

$$P(\xi_1, \ldots, \xi_T \mid \mathbf{x}_1, \ldots, \mathbf{x}_T) = \prod_{k=1}^{K} \prod_{t=1}^{T} P(y_{k,t} \mid \mathbf{x}_1, \ldots, \mathbf{x}_T)$$

This contrasts with the design of the EEND series which utilizes Softmax, thereby allowing the handling of overlapped speech.

#### 2. Sort Loss — Loss Function Sorted by Arrival Time

The core innovation of Sort Loss is the introduction of an arrival-time sorting function $\eta$. For each speaker $k$, their arrival time is defined as the starting frame of their first speech segment: $\Psi(\mathbf{y}_k) = \min\{t' \mid y_{k,t'} \neq 0\}$. The ground-truth label matrix is then sorted by arrival time, so that the speaker who speaks first corresponds to the first output row, the second speaker to the second output row, and so on:

$$\mathcal{L}_{\text{Sort}}(\mathbf{Y}, \mathbf{P}) = \frac{1}{K} \sum_{k=1}^{K} \mathcal{L}_{\text{BCE}}(\mathbf{y}_{\eta(k)}, \mathbf{q}_k)$$

Key difference from PIL: PIL requires searching through all $K!$ permutations to find the optimal match ($O(K!)$ complexity), whereas Sort Loss directly establishes the label-output correspondence via deterministic sorting ($O(K \log K)$ complexity).

#### 3. Hybrid Loss

When Sort Loss is used alone, arrival time estimation can be inaccurate (especially when there are many speakers). Therefore, a hybrid loss is proposed:

$$\mathcal{L}_{\text{hybrid}} = \alpha \cdot \mathcal{L}_{\text{Sort}} + (1 - \alpha) \cdot \mathcal{L}_{\text{PIL}}$$

where $\alpha = 0.5$ is an empirical weight. PIL provides permutation robustness while Sort Loss provides sorting constraints, making them complementary.

#### 4. Position Embeddings

Unlike EEND-SA and EEND-EDA, Sortformer must use positional encodings. The reason is that the multi-head self-attention mechanism of the Transformer is permutation equivariant without positional encodings—permuting the input order does not change the output. The core objective of Sortformer is to learn the arrival time order, which requires the model to perceive positional information in the sequence.

#### 5. Sinusoidal Kernel Speaker Encoding

To bridge diarization results to ASR, a sinusoidal kernel function index is designed:

$$\tilde{\mathbf{A}} = \frac{\mathbf{A}}{\|\mathbf{A}\|_2} + \mathbf{\Gamma}^T \cdot \mathbf{P}$$

where $\mathbf{A}$ is the ASR encoder state, $\mathbf{\Gamma}$ is the sinusoidal kernel matrix for $K$ speakers ($\kappa_{k,z} = \sin(2\pi kz / M)$), and $\mathbf{P}$ is the speaker probability matrix output by Sortformer. Different speakers correspond to sinusoidal functions of different frequencies, encoding speaker identities into the ASR features via additive injection.

### Loss & Training

**Two-Stage Training**:
- **Pre-training Stage**: Trained using 2,030 hours of real data + 5,150 hours of simulated mixed data, with a simulated overlap ratio of 0.12 and an average silence ratio of 0.1.
- **Fine-tuning Stage**: Fine-tuned using only real data.

**Architectural Parameters**: Based on an L-size NEST encoder (115M parameters) + 18-layer Transformer encoder (hidden size 192) + 2-layer Feed-Forward Network + 4 Sigmoid outputs, totaling 123M parameters.

**MS-ASR Training**: Based on the Canary ASR model at two scales: 170M and 1B parameters. The 170M version undergoes full-parameter fine-tuning for 50K steps (batch size 64); the 1B version uses an Adapter (learning only Adapter parameters while freezing the rest), trained for 75K steps. The optimizer is AdamW with a learning rate of $3 \times 10^{-4}$, weight decay of $10^{-3}$, inverse square root annealing, and warmup of 2,500 steps.

**Permutation-Resolved Transcription Format**: Sorted Serialized Transcript (SST) is proposed, which inserts speaker tags (e.g., `<spk0>`, `<spk1>`) into the transcription text ordered by arrival time. It supports word-level and segment-level granularities (word-level is better). Standard cross-entropy loss is used directly during training without extra permutation processing.

## Key Experimental Results

### Main Results: Speaker Diarization Performance (DER%)

| System | Parameters | DH3 (≤4spk, 0s collar) | CALLHOME-2spk (0.25s) | CALLHOME-3spk (0.25s) | CALLHOME-4spk (0.25s) | CH109 (2spk, 0.25s) |
|:---|:---|:---|:---|:---|:---|:---|
| EEND-EDA | 6.4M | 15.55 | 7.83 | 12.29 | 17.59 | - |
| AED-EEND | 11.6M | - | 6.18 | 11.51 | 18.44 | - |
| WavLM-L+EEND-VC† | 317M | - | 6.46 | 10.69 | 11.84 | - |
| Sortformer-PIL | 123M | 17.04 | 6.94 | 10.30 | 17.52 | 6.89 |
| Sortformer-Sort-Loss| 123M | 17.10 | 6.52 | 10.36 | 17.40 | 10.85 |
| **Sortformer-Hybrid** | **123M** | **14.76** | **5.87** | **8.46** | **12.59** | **6.86** |

> Hybrid Loss achieves the best performance across almost all datasets. On DH3, DER falls from 17.04 (PIL) to 14.76 (a relative reduction of 13.4%), and on CALLHOME-4spk from 17.52 to 12.59 (a relative reduction of 28.1%).

### Ablation Study: Contribution of Multi-Speaker ASR Components (AMI-test & CH109)

| System | Speaker Supervision | Train/Inference Speaker | Adapter | AMI-test WER | AMI-test cpWER | CH109 WER | CH109 cpWER |
|:---|:---|:---|:---|:---|:---|:---|:---|
| Baseline (Canary-170M) | None | - | - | 26.93% | - | 21.81% | - |
| Sys1 (Unsupervised) | None | - | - | 19.67% | 32.94% | 18.57% | 24.80% |
| Sys2 (Sortformer Frozen) | Sortformer | Frozen | - | 20.08% | 28.17% | 18.65% | 22.22% |
| Sys4 (GT Training) | GT $\rightarrow$ Sortformer Inference | - | - | 19.48% | 26.83% | 18.74% | 24.39% |
| **Sys6 (1B+Adapter+Word-level)** | **Sortformer** | **Frozen** | **256** | **18.04%** | **26.71%** | **16.46%** | **21.45%** |

> Sys6, which combines Sortformer supervision, Adapter, and the word-level target, performs best across all metrics. The cpWER decreases from 32.94% in the unsupervised baseline to 26.71%.

### LibriSpeechMix Comparative Results

| System | Parameters | Speaker Supervision | 1-mix WER | 2-mix WER | 3-mix WER |
|:---|:---|:---|:---|:---|:---|
| SOT-ASR | 135.6M | ✗ | 4.6 | 11.2 | 24.0 |
| DOM-SOT | 33M | ✗ | 5.17 | 5.56 | 9.96 |
| MT-LLM | 8.4B | ✓ | 2.3 | 5.2 | 10.2 |
| MS-Canary (No Sortformer) | 170M | ✗ | 2.74 | 6.55 | 12.14 |
| **Sortformer-MS-Canary** | **293M** | **✓** | **2.26** | **4.61** | **9.05** |

> With only 293M parameters, Sortformer-MS-Canary outperforms the 8.4B-parameter MT-LLM on both 2-mix and 3-mix, while the integration of Sortformer increases inference time overhead by only 0.78%.

## Highlights & Insights

1. **Elegant Solution to the Permutation Problem**: Sort Loss simplifies the $O(K!)$ permutation search to an $O(K\log K)$ deterministic sorting scheme, while maintaining comparable or even superior performance to PIL. This represents a fundamental breakthrough over the PIL-dependent paradigm that has dominated the end-to-end diarization field for a decade.

2. **Unified Training Paradigm**: Through the Sorted Serialized Transcript + sinusoidal kernel injection, multi-speaker ASR training can entirely utilize standard cross-entropy loss, aligning perfectly with the single-speaker ASR training pipeline. This significantly lowers the barrier to integrating multi-speaker capabilities.

3. **Complementary Effects of Hybrid Loss**: Sort Loss offers global sorting constraints but is sensitive to arrival time estimation, while PIL provides permutation robustness but lacks sorting awareness. Combining them significantly outperforms using either loss in isolation.

4. **Extreme Parameter Efficiency**: At 293M parameters, Sortformer-MS-Canary outperforms the 8.4B MT-LLM on LibriSpeechMix while utilizing only 3.5% of its parameter size, with almost zero increase in inference time (+0.78%).

5. **Theoretical Insight on Position Embeddings**: The authors point out that Transformers without positional encodings exhibit permutation equivariance, which explains why the EEND series does not require positional encodings (permutation-invariant) whereas Sortformer does (permutation-dependent).

## Limitations & Future Work

1. **Upper Limit on Speaker Numbers**: The current model assumes a fixed maximum of 4 speakers and has not yet addressed open-set speaker scenarios. Although the number of output heads can be scaled up, the stability of Sort Loss with more speakers remains unvalidated.

2. **Vulnerability of Arrival-Time Sorting**: When multiple speakers start talking almost simultaneously, the determinism of arrival-time sorting can be compromised. Performance degrades when using Sort Loss alone, necessitating the support of PIL.

3. **Streamed Inference Unverified**: The current model requires a complete 90-second audio segment to operate, and has not yet been adapted to streaming/online diarization scenarios.

4. **Scalability to Long Audio**: Training samples are limited to 90 seconds. Multi-hour meeting recordings require chunking and subsequential merging, and the issue of speaker consistency across chunk boundaries is not discussed.

5. **Lack of Theoretical Support for Sinusoidal Kernel**: Selecting a sinusoidal function as the speaker kernel is guided more by engineering intuition (resembling positional encoding), lacking a theoretical analysis explaining why it outperforms other encoding schemes (such as learnable embeddings).

## Related Work

- **End-to-End Diarization**: EEND-SA (Fujita et al., 2019) first introduced PIL to frame-level diarization; EEND-EDA (Horiguchi et al., 2020) used an encoder-decoder to generate attractors to handle variable speaker numbers; AED-EEND (Chen et al., 2024) introduced attention enhancement. The proposed Sortformer operates in the same end-to-end framework while introducing Sort Loss as an alternative/supplement to PIL.
- **Multi-Speaker ASR**: SOT (Kanda et al., 2020b) resolves the permutation problem via serialized output, t-SOT (Kanda et al., 2022a) extends this to streaming scenarios, and DOM-SOT (Shi et al., 2024) utilizes dominance-based sorting. The proposed SST (Sorted Serialized Transcript) is conceptually similar to SOT but employs sorted speaker indices instead of speaker change tags.
- **Cascaded Systems**: TS-VAD (Medennikov et al., 2020a) and the CHiME challenge-winning system (Cornell et al., 2023) demonstrate the strong performance of cascaded systems, which nevertheless remain difficult to deploy. This work aims to bridge the performance gap between end-to-end and cascaded systems while retaining deployment convenience.
- **LLM Integration**: MT-LLM (Meng et al., 2025) employs an 8.4B-parameter multi-task LLM to address multi-speaker scenarios. This work demonstrates that a much lighter scheme can achieve comparable or even superior performance.

## Rating

| Dimension | Rating (1-5) | Description |
|:---|:---|:---|
| Novelty | ⭐⭐⭐⭐ | Sort Loss introduces a novel solution to the permutation problem, shifting from the search paradigm of PIL to a deterministic sorting paradigm. |
| Theoretical Depth | ⭐⭐⭐⭐ | The analysis of permutation equivariance and the design of the hybrid loss are backed by solid mathematical proof. |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Validated on 3 diarization datasets and 2 ASR datasets, featuring exhaustive ablations and a broad range of baseline systems. |
| Engineering Value | ⭐⭐⭐⭐⭐ | Code is open-sourced (NeMo) and pre-trained models are publicly available, with minimal training overhead (+0.22%~2.26%). |
| Writing Quality | ⭐⭐⭐⭐ | Clear mathematical derivations and rich figures, though some notations are somewhat heavy. |
| Overall | ⭐⭐⭐⭐ | A significant contribution to the speaker diarization field; the Sort Loss concept is simple and effective, and holds the potential to become a new baseline in this direction. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Do Not Mimic My Voice: Speaker Identity Unlearning for Zero-Shot Text-to-Speech](do_not_mimic_my_voice_speaker_identity_unlearning_for_zero-shot_text-to-speech.md)
- [\[ACL 2025\] It's Not a Walk in the Park! Challenges of Idiom Translation in Speech-to-text Systems](../../ACL2025/audio_speech/its_not_a_walk_in_the_park_challenges_of_idiom_translation_in_speech-to-text_sys.md)
- [\[ICML 2025\] BinauralFlow: A Causal and Streamable Approach for High-Quality Binaural Speech Synthesis with Flow Matching Models](binauralflow_a_causal_and_streamable_approach_for_high-quality_binaural_speech_s.md)
- [\[ICLR 2026\] TTSDS2: Resources and Benchmark for Evaluating Human-Quality Text to Speech Systems](../../ICLR2026/audio_speech/ttsds2_resources_and_benchmark_for_evaluating_human-quality_text_to_speech_syste.md)
- [\[ACL 2025\] Different Speech Translation Models Encode and Translate Speaker Gender Differently](../../ACL2025/audio_speech/different_speech_translation_models_encode_and_translate_speaker_gender_differen.md)

</div>

<!-- RELATED:END -->

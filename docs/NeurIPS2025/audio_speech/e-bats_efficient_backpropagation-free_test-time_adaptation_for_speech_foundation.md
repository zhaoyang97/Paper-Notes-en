---
title: >-
  [Paper Note] E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models
description: >-
  [NeurIPS 2025][Audio & Speech][test-time adaptation] This paper proposes E-BATS, the first backpropagation-free test-time adaptation framework for speech foundation models. Through lightweight prompt adaptation…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "test-time adaptation"
  - "speech foundation model"
  - "backpropagation-free"
  - "prompt tuning"
  - "CMA-ES"
date: 2026-05-08
content_hash: e3ce91eedcfa69d4
---

# E-BATS: Efficient Backpropagation-Free Test-Time Adaptation for Speech Foundation Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.07078](https://arxiv.org/abs/2506.07078)  
**Code**: [JiahengDong/E-BATS](https://github.com/JiahengDong/E-BATS)  
**Area**: Audio & Speech
**Keywords**: test-time adaptation, speech foundation model, backpropagation-free, prompt tuning, CMA-ES

## TL;DR

This paper proposes E-BATS, the first backpropagation-free test-time adaptation framework for speech foundation models. Through lightweight prompt adaptation, multi-scale loss functions, and a test-time EMA mechanism, E-BATS achieves 2.0×–6.4× GPU memory savings while maintaining competitive accuracy.

## Background & Motivation

Speech Foundation Models (SFMs) such as Wav2Vec2 and HuBERT perform well on clean data but suffer significant performance degradation in real-world deployment due to acoustic domain shifts caused by background noise, speaker accents, and microphone characteristics.

Test-Time Adaptation (TTA) adapts models to new domains using unlabeled test data during inference, without access to source data or labels. Existing TTA methods fall into two categories:

- **BP-based**: Methods such as TENT, SUTA, and DSUTA update parameters via gradient-based entropy minimization or pseudo-labels. These achieve good accuracy but incur large memory overhead, as storing intermediate gradients is required even when updating only BatchNorm layers.
- **BP-free**: Methods such as LAME, T3A, and FOA update models using only forward passes, offering high memory efficiency but inferior accuracy. Moreover, these methods are primarily designed for vision tasks.

The root cause of the problem is that **existing BP-free methods are tailored for vision tasks**, whereas speech tasks differ substantially in model architecture (CNN + Transformer hybrid), task formulation (sequence-to-sequence), noise characteristics (temporally dynamic), and batching requirements (single-utterance processing). Direct transfer of vision-oriented methods yields poor results.

## Core Problem

1. **Architectural mismatch**: SFMs use LayerNorm rather than BatchNorm and employ a hybrid CNN feature encoder + Transformer encoder architecture, making existing BN-statistics-based BP-free methods inapplicable.
2. **Task mismatch**: Automatic speech recognition involves sequence-to-sequence mapping where noise varies dynamically along the time dimension, requiring multi-scale adaptation rather than single image-level adaptation.
3. **Batch size constraint**: Speech TTA must process one utterance at a time (batch size = 1), precluding statistical estimation over large batches.
4. **Memory bottleneck**: Memory consumption of BP-based methods grows sharply with utterance length, limiting deployment in resource-constrained settings.

## Method

E-BATS consists of three core modules:

### 1. Lightweight Prompt Adaptation (LPA)

**Core observation**: Across different acoustic conditions, the mean shift between source- and target-domain hidden-space embeddings is up to **7.8× larger** than the covariance shift, indicating that domain shift primarily manifests as geometric translation in the hidden space.

**Design rationale**: Rather than prepending prompt tokens at the Transformer input as in conventional prompt tuning, E-BATS directly adds a learnable prompt vector $\mathbf{s}_t$ to the hidden features $\mathbf{Z}_t$ at the output of the CNN encoder:

$$\hat{\mathbf{Z}}_t = \mathbf{Z}_t + \mathbf{s}_t \cdot \mathbf{1}_N^\top$$

Prompts are injected at the CNN layer rather than the Transformer layer because the CNN captures local spectral features (pitch, formants) that are more sensitive to acoustic domain shifts, while the Transformer focuses on global contextual dependencies and is less suited for modeling fine-grained acoustic variations. Ablation experiments confirm that CNN-layer injection substantially outperforms Transformer-layer injection (WER 24.0 vs. 34.2).

### 2. Multi-Scale Loss Function

The total loss is a weighted combination of three terms: $L_{adapt} = \alpha L_{ent} + \beta L_{utt} + c \cdot L_{token}$

**(a) Blank-excluded entropy minimization $L_{ent}$**:
In CTC decoding, a large proportion of frames are predicted as the blank class, causing class imbalance. Shannon entropy is computed only over non-blank prediction frames. Using entropy minimization alone leads to degenerate solutions where all frames predict blank.

**(b) Utterance-level hidden embedding alignment $L_{utt}$**:
At each Transformer layer, the squared Euclidean distance between the source- and target-domain utterance-level embedding centroids is computed. Utterance-level embeddings are obtained by averaging frame-level embeddings. This term effectively prevents the degenerate solutions of entropy minimization, with a storage cost of only $L \times d$.

**(c) Adaptive-confidence token-level alignment $L_{token}$**:
Frames are grouped by pseudo-label into token categories, and the means and standard deviations of source- and target-domain embeddings within each token category are aligned. An adaptive confidence coefficient $c$ is introduced: when domain shift is large or entropy is high, $c$ is reduced to mitigate the influence of unreliable pseudo-labels; when shift is small, $c$ is increased to strengthen alignment.

### 3. Prompt Optimization via CMA-ES

The prompt vector is optimized using the gradient-free Covariance Matrix Adaptation Evolution Strategy (CMA-ES). At each iteration, $J=50$ candidate prompts are sampled; they are ranked by $L_{adapt}$, and the search distribution parameters (mean $\mathbf{m}$, covariance $\mathbf{C}$, step size $\sigma$) are updated accordingly. The process iterates until convergence, and the best prompt is selected.

### 4. Test-Time EMA (T-EMA)

A stabilization mechanism for adaptation across an utterance stream: after processing each utterance, the CMA-ES search distribution parameters are updated via EMA:

$$\mathbf{m}_{ema} = \gamma \mathbf{m}_{ema} + (1-\gamma)\mathbf{m}_t^{(K)}$$

The covariance and step size are updated analogously. This balances retention of historical knowledge with adaptation to new utterances, avoiding both forgetting and overfitting.

## Key Experimental Results

**Datasets**: 4 noisy speech datasets, 16 acoustic conditions
- LibriSpeech + Gaussian noise ($\sigma$ = 0.0–0.02)
- CHiME-3 single-domain / mixed-domain
- CommonVoice (accent diversity)
- TEDLIUM-v2 (speaking style diversity)

### Main Results (Wav2Vec2-Base)

| Metric | E-BATS Performance |
|--------|--------------------|
| Improvement over BP-free baselines | WER reduction of 4.1%–13.5% (absolute) |
| Comparison with best BP-based method | Lowest WER on 3/5 datasets; max relative gain 30.7% |
| Memory savings vs. BP-based | 2.0×–6.4× (3.3× vs. DSUTA) |
| High-noise scenario ($\sigma$=0.02) | WER 25.3, vs. 45.3 for best BP-free baseline FOA (−20.0) |

**HuBERT-Large results**: WER reduced by 1.8%–17.1% over BP-free baselines; memory savings of 2.4×–6.8×.

**Memory vs. utterance length** (TED dataset, HuBERT-Large): BP-based methods reach 6–12 GB at 30 seconds; E-BATS uses only ~1.9 GB with near-linear growth.

### Ablation Study

- CNN-layer prompt injection >> Transformer-layer injection (WER 24.0 vs. 34.2)
- All three loss terms are necessary: $L_{ent}$ alone causes degeneracy (WER 49.6); adding $L_{utt}$ yields a large correction (25.5); adding $L_{token}$ further improves performance (25.4)
- T-EMA > continuous adaptation without reset > reset each utterance (WER 24.3 vs. 25.4 vs. 26.5)

## Highlights & Insights

1. **First BP-free TTA method for SFMs**, filling a gap in the field while matching or surpassing BP-based methods in accuracy.
2. **Compelling analysis of hidden-space shift**—the empirical observation that mean shift dominates covariance shift provides a solid foundation for the "translation-as-alignment" design choice.
3. **Elegant multi-scale loss design**—hierarchical alignment from utterance level to token level, combined with adaptive confidence control, addresses the challenge of unreliable pseudo-labels.
4. **Memory advantage grows with model scale**—6.8× savings on HuBERT-Large, offering substantial practical value for deployment.

## Limitations & Future Work

1. **Inference latency**: The iterative optimization of CMA-ES introduces additional latency; the current implementation does not fully exploit GPU parallelism, making it unsuitable for real-time scenarios.
2. **Evaluated on ASR only**: The framework has not been extended to other speech tasks such as speaker recognition or emotion detection.
3. **Dependence on source-domain statistics**: Pre-collected embedding statistics from source-domain layers are required, which may not be available in all settings.
4. **CMA-ES population size**: The efficiency of $J=50$ samples in high-dimensional search spaces is uncertain; more efficient gradient-free optimizers could be explored.

## Related Work & Insights

| Method | Category | Key Characteristics | Disadvantages vs. E-BATS |
|--------|----------|---------------------|--------------------------|
| SUTA/CEA/SGEM | BP-based speech TTA | Per-utterance reset; entropy minimization + speech-specific losses | Cannot accumulate cross-utterance knowledge; high memory overhead |
| DSUTA | BP-based speech TTA | Continuous adaptation; dual slow-fast models | Frequent parameter updates cause catastrophic forgetting; WER 5.5 higher than E-BATS on CommonVoice |
| FOA | BP-free general TTA | CMA-ES + prompt tuning | Transformer-layer prompt injection unsuitable for acoustic domain shift |
| T3A/LAME | BP-free general TTA | Classifier adjustment only | Insufficient adaptation capacity; sometimes worse than source model |

**Broader implications**:
- The **"mean-shift dominance" observation** may generalize to other modalities (e.g., video, multimodal models) for understanding domain shift.
- The **choice of prompt injection layer** (CNN vs. Transformer) provides important design guidance for adapting hybrid-architecture models.
- The adaptive confidence weighting of token-level loss is transferable to other semi-supervised or adaptive methods that rely on pseudo-labels.
- The use of gradient-free optimizers in TTA warrants further attention; beyond CMA-ES, Natural Evolution Strategies and OpenAI-ES are promising alternatives.

## Rating
- Novelty: ⭐⭐⭐⭐ — First BP-free speech TTA; both problem formulation and method design are innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 4 datasets, 16 conditions, 2 backbones, 13 baselines, detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — Clear logic, rich figures and tables, data-supported motivation.
- Value: ⭐⭐⭐⭐ — Practical significance for deploying speech systems in resource-constrained settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [\[ICLR 2026\] When and Where to Reset Matters for Long-Term Test-Time Adaptation](../../ICLR2026/audio_speech/when_and_where_to_reset_matters_for_long-term_test-time_adaptation.md)
- [\[NeurIPS 2025\] Instance-Specific Test-Time Training for Speech Editing in the Wild](instance-specific_test-time_training_for_speech_editing_in_the_wild.md)
- [\[NeurIPS 2025\] Data-Juicer 2.0: Cloud-Scale Adaptive Data Processing for and with Foundation Models](data-juicer_20_cloud-scale_adaptive_data_processing_for_and_with_foundation_mode.md)
- [\[NeurIPS 2025\] Efficient Speech Language Modeling via Energy Distance in Continuous Latent Space](efficient_speech_language_modeling_via_energy_distance_in_continuous_latent_spac.md)

</div>

<!-- RELATED:END -->

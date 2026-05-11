---
title: >-
  [Paper Note] TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning
description: >-
  [NeurIPS 2025][Video Understanding][Time series forecasting] This paper proposes TiRex, a pretrained time series forecasting model based on xLSTM. By introducing a Contiguous Patch Masking (CPM) strategy and data augment…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Time series forecasting"
  - "zero-shot forecasting"
  - "xLSTM"
  - "in-context learning"
  - "data augmentation"
  - "pretrained models"
date: 2026-05-08
content_hash: bf7d2ac50b0dbeb4
---

# TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning

**Conference**: NeurIPS 2025
**arXiv**: [2505.23719](https://arxiv.org/abs/2505.23719)
**Authors**: Andreas Auer (NXAI/JKU Linz), Patrick Podest (JKU Linz), Daniel Klotz (ITUA Linz), Sebastian Böck (NXAI), Günter Klambauer (NXAI/JKU), Sepp Hochreiter (NXAI/JKU)
**Code**: Not released
**Area**: Video Understanding
**Keywords**: Time series forecasting, zero-shot forecasting, xLSTM, in-context learning, data augmentation, pretrained models

## TL;DR

This paper proposes TiRex, a pretrained time series forecasting model based on xLSTM. By introducing a Contiguous Patch Masking (CPM) strategy and data augmentation techniques, TiRex with only 35M parameters comprehensively outperforms larger models such as Chronos Bolt (200M) and TimesFM (500M) on the GiftEval and Chronos-ZS benchmarks, achieving state-of-the-art performance in both short- and long-horizon zero-shot forecasting.

## Background & Motivation

### State of the Field
Time series forecasting is a core task in domains such as energy, retail, and healthcare. Inspired by large language models, pretrained time series models have recently enabled zero-shot forecasting via in-context learning, making advanced forecasting tools accessible to non-expert users and significantly improving performance in data-scarce scenarios.

### Limitations of Prior Work
- **Transformer dominance with limited gains**: Mainstream zero-shot forecasting models (Chronos, TimesFM, Moirai, etc.) are all Transformer-based, yet Transformers frequently underperform expectations on time series tasks and can even be surpassed by the simple linear model DLinear.
- **LSTM lacks in-context learning capability**: Traditional LSTMs possess strong state-tracking ability well suited to time series modeling, but lack the in-context generalization required for zero-shot forecasting.
- **Error accumulation in long-horizon forecasting**: Existing methods generate multi-step forecasts autoregressively, using point estimates from previous steps as inputs to the next, which disrupts uncertainty propagation and leads to rapid quality degradation over long horizons.
- **Underutilization of data augmentation**: While synthetic data has been used in pretraining, systematic data augmentation strategies—successful in computer vision—have rarely been explored for time series pretraining.

### Root Cause
By leveraging xLSTM (an enhanced LSTM), the paper aims to combine the state-tracking strengths of LSTM with Transformer-level in-context learning capability, constructing a pretrained model that excels at both short- and long-horizon zero-shot forecasting.

## Method

### Overall Architecture
TiRex adopts a decoder-only design, stacking multiple xLSTM blocks (using sLSTM modules) with lightweight residual blocks at the input and output layers.

**Input Layer**:
1. Each time series undergoes z-score instance normalization to eliminate cross-domain scale discrepancies.
2. The series is segmented into non-overlapping patches of size $m_{\text{in}}=32$.
3. Each patch is concatenated with a binary mask (indicating missing values) and projected to the xLSTM hidden dimension $d$ via a two-layer residual block.

**xLSTM Blocks**:
- sLSTM (rather than mLSTM) is used as the sequence mixing component to preserve **true recurrence** for state tracking.
- Each block contains an sLSTM module and a feed-forward network, both with RMSNorm and residual skip connections.
- An optimized kernel architecture enables efficient training and inference.

**Output Layer**:
- Nine equally spaced quantiles $Q=\{0.1, 0.2, \ldots, 0.9\}$ are predicted to enable probabilistic forecasting.
- Optimization is performed using quantile loss.

### Contiguous Patch Masking (CPM)
CPM is one of TiRex's core innovations, addressing error accumulation in multi-step forecasting:

1. For each training sample, the number of contiguous masked segments is sampled as $c_{\text{mask}} \sim U(1, c_{\text{mask}}^{\text{max}})$.
2. A masking probability is sampled as $p_{\text{mask}} \sim U(0, p_{\text{mask}}^{\text{max}})$.
3. A binary mask is generated and repeated in units of $c_{\text{mask}} \cdot m_{\text{out}}$, masking entire contiguous patch segments.
4. Masked patches are represented as missing values in the input.

**Core Idea**: By randomly masking contiguous segments during training, the model learns to propagate predictive information and uncertainty through its internal memory states at inference time, rather than relying on autoregressive point estimates. The key distinction from autoregressive methods (e.g., TimesFM) is that TiRex treats future inputs as missing values during multi-step forecasting, allowing xLSTM's internal state to naturally propagate uncertainty across patches.

### Multi-Step Forecasting at Inference
When the forecast horizon $h$ exceeds a single output patch length:
- **Existing methods**: Use the median/mean of the previous step as input to the next (autoregressive), resetting the probability distribution at each step.
- **TiRex**: Marks future inputs as missing values, allowing internal memory to propagate predictive states and uncertainty across patches.

### Data Augmentation Strategies
Three augmentation techniques designed specifically for time series pretraining:

1. **Amplitude Modulation**: $y'_t = y_t \cdot a_t$, introducing trends and change points; applied with probability 0.5.
2. **Censor Augmentation**: $y'_t = \max/\min(y_t, c)$, truncating at a random threshold; applied with probability 0.5.
3. **Spike Injection**: $y'_t = y_t + s_t$, adding short-duration spike signals (tophat/RBF/linear kernels); applied with probability 0.05.

### Training Details
- Training data: Chronos training set + KernelSynth synthetic data + GiftEval pretraining data, totaling 47.5 million time series samples.
- Context length of 2048, patch size of 32.
- Model parameters: 35M (substantially fewer than competitors).

## Key Experimental Results

### Experiment 1: GiftEval-ZS Benchmark (Comprehensive Short- and Long-Horizon Evaluation)

After excluding 16 evaluation settings that overlap with the training set, 81 zero-shot evaluation settings remain:

| Model | Parameters | CRPS (Overall) | CRPS (Short) | CRPS (Long) | Avg Rank |
|-------|-----------|---------------|-------------|------------|----------|
| **TiRex** | **35M** | **0.411±0.002** | **0.455±0.001** | **0.325±0.003** | **Best** |
| TimesFM-2.0 | 500M | 0.459 | — | — | Second tier |
| TabPFN-TS | — | 0.463 | — | — | Second tier |
| Chronos-Bolt-Base | 200M | 0.481 | — | — | Second tier |

Key findings:
- TiRex with 35M parameters comprehensively outperforms TimesFM-2.0 (500M) and Chronos-Bolt (200M).
- Competing models each hold advantages in either short- or long-horizon forecasting; TiRex is the only model leading in both simultaneously.
- In long-horizon forecasting, TiRex becomes the first zero-shot model to surpass task-specific models PatchTST and TFT.

### Experiment 2: Ablation Study

| Variant | Gift-ZS Overall | Gift-ZS Long | Gift-ZS Short | Chronos-ZS WQL |
|---------|----------------|-------------|-------------|---------------|
| **TiRex (Full)** | **0.411** | **0.325** | **0.455** | **0.592** |
| Naive multi-step training | 0.424 ↓ | 0.335 ↓ | 0.475 ↓ | 0.650 ↓ |
| No multi-step (autoregressive) | 0.445 ↓ | 0.370 ↓ | 0.471 ↓ | 0.589 |
| No augmentation | 0.430 ↓ | 0.339 ↓ | 0.473 ↓ | 0.623 ↓ |
| Transformer backbone | 0.422 ↓ | 0.342 ↓ | 0.461 ↓ | 0.597 |
| mLSTM backbone | 0.457 ↓ | 0.430 ↓ | 0.456 | 0.588 |
| Chronos Bolt Base | 0.454 ↓ | 0.418 ↓ | 0.458 | 0.627 ↓ |

Key conclusions:
- CPM is critical for long-horizon forecasting; removing it degrades long-horizon CRPS from 0.325 to 0.370.
- The sLSTM backbone significantly outperforms mLSTM (long-horizon: 0.325 vs. 0.430) and Transformer (0.325 vs. 0.342).
- Data augmentation provides a clear overall benefit; removing it raises overall CRPS from 0.411 to 0.430.
- Inference speed: TiRex is 11× faster than TimesFM-2.0, 4× faster than Chronos-Bolt Base, and 2176× faster than TabPFN-TS.

## Highlights & Insights

- **Small model, strong performance**: With only 35M parameters, TiRex comprehensively surpasses competitors at the 200M–500M scale, achieving state-of-the-art results in both short- and long-horizon forecasting on two major benchmarks.
- **Elegant and effective CPM strategy**: Contiguous Patch Masking elegantly resolves the uncertainty propagation discontinuity inherent in autoregressive multi-step forecasting; the random masking pattern during training naturally aligns with missing-value inputs at inference time.
- **Theoretically motivated architecture selection**: Choosing sLSTM over mLSTM or Transformer leverages its unique state-tracking capability to achieve consistent uncertainty propagation across long horizons.
- **First systematic exploration of data augmentation for time series pretraining**: The three augmentation strategies are each purposefully designed; spike injection in particular enables the model to capture rare, sharp events.
- **High inference efficiency**: 4× to 2176× faster than major competitors, with substantially reduced GPU memory consumption.

## Limitations & Future Work

- **Univariate only**: Like most pretrained forecasting models, TiRex handles only univariate time series and does not model inter-variable dependencies.
- **Limited hyperparameter tuning**: Computational constraints precluded extensive hyperparameter search; only sensitivity analyses for key parameters were conducted.
- **Downstream tasks unexplored**: The transferability of the learned representations to other time series tasks such as classification and anomaly detection remains unknown.
- **Area classification questionable**: This paper belongs to the time series forecasting domain; the current categorization under 3d_vision may be inaccurate.
- **Data leakage risk in evaluation benchmarks**: Although overlapping datasets were excluded by the authors, fair comparison with competing models (e.g., Moirai has 82% overlap with Chronos-ZS) still warrants caution.

## Related Work & Insights

- **Chronos/Chronos-Bolt (Amazon)**: Based on an encoder-decoder Transformer; Chronos-Bolt-Base has 200M parameters and is competitive in short-horizon forecasting but notably weaker than TiRex in long-horizon settings.
- **TimesFM (Google)**: A 500M-parameter decoder-only Transformer; v2.0 achieves a CRPS of 0.459 on GiftEval-ZS, substantially surpassed by TiRex at 35M parameters.
- **Moirai (Salesforce)**: Encoder-only design with masked modeling; performs well on Chronos-ZS but has 82% overlap with the test set, raising questions about genuine zero-shot capability.
- **TabPFN-TS (Prior Labs)**: An improved Transformer encoder pretrained solely on synthetic data; slightly outperforms TiRex on Chronos-ZS MASE but is 2176× slower at inference.
- **xLSTM (Beck et al., 2024)**: The backbone architecture of TiRex; the original paper demonstrated in-context learning capability comparable to Transformer LLMs, yet its potential for general-purpose time series pretraining had not been fully explored.
- **DLinear (Zeng et al., 2023)**: A simple linear model that demonstrated Transformers do not necessarily dominate in time series tasks, indirectly supporting TiRex's choice of a non-Transformer architecture.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The CPM strategy and the application of xLSTM to zero-shot forecasting are novel, though the overall approach applies existing architectures to a new task.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Two major benchmarks, 6-seed repetitions, comprehensive ablations, inference efficiency analysis, and qualitative analysis constitute a very rigorous experimental design.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, well-articulated motivation, and well-organized experiments.
- **Value**: ⭐⭐⭐⭐⭐ — Achieving state-of-the-art in both short- and long-horizon zero-shot forecasting with an extremely small parameter count has significant practical implications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Frequency-Semantic Enhanced Variational Autoencoder for Zero-Shot Skeleton-based Action Recognition](../../ICCV2025/video_understanding/frequency-semantic_enhanced_variational_autoencoder_for_zero-shot_skeleton-based.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)
- [\[ICCV 2025\] RainbowPrompt: Diversity-Enhanced Prompt-Evolving for Continual Learning](../../ICCV2025/video_understanding/rainbowprompt_diversity-enhanced_prompt-evolving_for_continual_learning.md)
- [\[NeurIPS 2025\] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning](adversarial_locomotion_and_motion_imitation_for_humanoid_policy_learning.md)
- [\[NeurIPS 2025\] VideoLucy: Deep Memory Backtracking for Long Video Understanding](videolucy_deep_memory_backtracking_for_long_video_understanding.md)

</div>

<!-- RELATED:END -->

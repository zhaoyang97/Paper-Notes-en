---
title: >-
  [Paper Note] MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory
description: >-
  [ECCV 2024][Test-Time Adaptation] This paper proposes MemBN (Memory-based Batch Normalization). By maintaining a statistics memory queue within each BN layer and designing dedicated memory management and aggregation algorithms, TTA methods can robustly estimate test domain statistics across various batch sizes, significantly improving accuracy and robustness in small-batch scenarios.
tags:
  - "ECCV 2024"
  - "Test-Time Adaptation"
  - "Batch Normalization"
  - "Statistics Memory"
  - "Small-Batch Robustness"
  - "Distribution Shift"
date: 2026-05-08
content_hash: 02d9211b7096f11d
---

# MemBN: Robust Test-Time Adaptation via Batch Norm with Statistics Memory

**Conference**: ECCV 2024  
**Authors**: Juwon Kang, Nayeong Kim, Jungseul Ok, Suha Kwak  
**Code**: None  
**Area**: Model Robustness / Test-Time Adaptation  
**Keywords**: Test-Time Adaptation, Batch Normalization, Statistics Memory, Small-Batch Robustness, Distribution Shift

## TL;DR

This paper proposes MemBN (Memory-based Batch Normalization). By maintaining a statistics memory queue within each BN layer and designing dedicated memory management and aggregation algorithms, TTA methods can robustly estimate test domain statistics across various batch sizes, significantly improving accuracy and robustness in small-batch scenarios.

## Background & Motivation

**Background**: Test-Time Adaptation (TTA) is an effective paradigm to handle train-test distribution shifts. Current mainstream TTA methods typically utilize the batch statistics of test data to replace the BN statistics learned during training, thereby rapidly adapting to new domains.

**Limitations of Prior Work**: Existing TTA methods heavily rely on large batches of test data to estimate reliable BN statistics. When the test batch size is small (e.g., online scenarios, edge devices, real-time inference), the estimation of batch statistics becomes highly unstable, leading to a drastic performance drop in TTA, sometimes even underperforming the non-adapted baseline. This issue is highly prevalent in real-world deployment but has rarely been systematically studied.

**Key Challenge**: TTA requires replacing training statistics with test batch statistics to adapt to new distributions, but the statistics of a single batch under small-batch settings are highly noisy and cannot reliably represent the test domain distribution. This creates a contradiction between "the need for test statistics" and "the unreliability of test statistics."

**Goal**: (1) How to obtain reliable BN statistics in extreme scenarios of small batch sizes, even when batch size = 1? (2) How to design a plug-and-play BN module that is robust to batch sizes? (3) How to prevent the historical statistics preserved in memory from being contaminated by outdated information?

**Key Insight**: The authors observe that while statistics from a single small batch are unreliable, aggregating the statistics of multiple historical batches can yield a more robust estimation. The key lies in how to manage this historical information, maintaining a sufficient history to reduce variance while avoiding the accumulation of outdated information.

**Core Idea**: Maintain a fixed-length statistics memory queue inside each BN layer, and estimate reliable statistics of the current test domain through intelligent enqueue/dequeue management and weighted aggregation.

## Method

### Overall Architecture

The core design of MemBN is to introduce a Statistics Memory Queue within each Batch Normalization layer of a pre-trained model. Upon receiving a test batch, the mean and variance of the current batch are calculated and stored in the memory queue. Subsequently, an aggregation algorithm is applied to fuse multiple historical statistics in the queue via weighted combination, yielding the final statistics for normalization. The entire pipeline does not modify the model architecture and can replace the standard BN layer of any model in a plug-and-play manner.

### Key Designs

1. **Statistics Memory Queue**:
    - **Function**: Maintain a FIFO queue in each BN layer to store the mean $\mu$ and variance $\sigma^2$ of the last $K$ test batches.
    - **Mechanism**: When a new batch arrives, its batch statistics are calculated and enqueued; if the queue is full, the oldest statistics are dequeued. Consequently, the queue consistently retains information from the $K$ most recent batches, providing sufficient history to minimize estimation variance while preserving adaptability to distribution changes by discarding obsolete information.
    - **Design Motivation**: Compared to directly using Exponential Moving Average (EMA), the memory queue offers more flexible control over the historical window size and avoids EMA's sensitivity to hyperparameters (momentum coefficient).

2. **Adaptive Memory Management**:
    - **Function**: Dynamically determine whether to insert the new statistics into memory based on the consistency between the current batch statistics and the historical statistics in the queue.
    - **Mechanism**: Compute the distance between the current batch statistics and the aggregated statistics of the queue. If the distance is excessively large (indicating the current batch might be a noisy outlier or a sudden shift in distribution has occurred), a conservative strategy is adopted; if the distance is moderate, the statistics are enqueued normally. This mechanism prevents highly noisy outlier batches from contaminating the memory.
    - **Design Motivation**: Under small-batch scenarios, statistics of individual batches can deviate severely from the true distribution due to sampling noise. Direct insertion would skew the aggregated estimation; hence, a filtering mechanism is necessary to guarantee memory quality.

3. **Weighted Aggregation**:
    - **Function**: Fuse multiple historical statistics in the memory queue into final normalization parameters.
    - **Mechanism**: Rather than a simple average, different weights are assigned based on the reliability of each historical batch (i.e., its consistency with the global distribution estimate). Batches closer to the global distribution estimation receive higher weights, while those with larger deviations receive lower weights. The final mean and variance are obtained via weighted summation: $\mu_{agg} = \sum_{i} w_i \mu_i$ and $\sigma^2_{agg} = \sum_{i} w_i \sigma^2_i$.
    - **Design Motivation**: A simple average assigns identical weights to all historical batches, whereas in reality, the quality of statistics varies significantly across different batches. Weighted aggregation leverages high-quality historical information more effectively.

### Loss & Training

MemBN itself does not require extra training losses. It is a module integrated at inference time that can be combined with any TTA loss function (such as entropy minimization, pseudo-labeling, contrastive learning, etc.). The core contribution of MemBN is to provide more reliable normalization statistics, rather than modifying the optimization objective.

## Key Experimental Results

### Main Results

| Dataset/Setting | Metric | MemBN (Ours) | Standard BN TTA | Gain |
|-------------|------|-------------|------------|------|
| CIFAR-10-C (BS=1) | Error Rate (%) | ~12.5 | ~25.0 | ↓12.5 |
| CIFAR-100-C (BS=1) | Error Rate (%) | ~38.0 | ~52.0 | ↓14.0 |
| ImageNet-C (BS=4) | Error Rate (%) | ~48.0 | ~58.0 | ↓10.0 |
| ImageNet-C (BS=64) | Error Rate (%) | ~42.0 | ~44.0 | ↓2.0 |

### Ablation Study

| Configuration | Key Metric (Error %) | Description |
|------|-------------------|------|
| Full MemBN | 12.5 | Full Model |
| w/o Adaptive Management | 15.2 | Outlier batches contaminate memory after eliminating enqueue filtering |
| w/o Weighted Aggregation | 14.0 | Simple average performs worse than weighted average, as low-quality batches degrade performance |
| w/o Memory Queue (current batch only) | 25.0 | Degrades to standard BN TTA |
| Queue Length K=5 | 13.8 | History is too short, leaving variance still large |
| Queue Length K=50 | 13.0 | Performance stabilized with longer queue |
| Queue Length K=100 | 12.5 | Optimal range |

### Key Findings
- The memory queue contributes most under small-batch scenarios: the improvement is most prominent at BS=1, while relatively minor at BS=64, indicating that MemBN primarily addresses the unreliability of batch statistics.
- The adaptive management mechanism is particularly crucial in continual TTA scenarios, preventing historical statistics of previous distributions from contaminating estimates of the new distribution.
- The queue length K does not require meticulous tuning, showing stable performance within the range of 50–200.
- As a plug-and-play module, MemBN can boost the performance of diverse TTA methods (e.g., TENT, CoTTA).

## Highlights & Insights
- **Plug-and-Play Design**: MemBN does not modify the model architecture or training pipeline, only replacing how BN statistics are calculated. It can be integrated with any existing TTA method. This design philosophy renders it highly practical.
- **Robustness to Small Batches**: It remains effective even in extreme scenarios such as BS=1, which is of great value for edge devices and real-time inference scenarios.
- **The Concept of Memory Queue + Weighted Aggregation** can be applied to other scenarios requiring online estimation of statistics, such as BN statistics aggregation in federated learning and feature normalization in online learning.

## Limitations & Future Work
- In scenarios where distributions shift rapidly (e.g., each batch originates from a different domain), stale statistics in the queue might introduce interference, necessitating a more aggressive forgetting mechanism.
- The paper assumes test data comes from a single target domain; its efficacy might be limited in mixed scenarios where multiple domains appear concurrently (e.g., alternative occurrences of different corruption types).
- Although the queue length K is not highly sensitive, it still requires a certain prior in extreme scenarios. Adaptive adjustment of K represents a promising direction for future improvement.
- The integration of MemBN with other normalization techniques (such as Layer Normalization and Group Normalization) remains unexplored.

## Related Work & Insights
- **vs TENT**: TENT updates the affine parameters of BN via entropy minimization, while the operational statistics are still calculated using the current batch. MemBN is orthogonal to TENT, and both can be used and combined.
- **vs CoTTA**: CoTTA introduces a teacher-student framework and augmentation averaging to enhance TTA robustness, but the BN statistics issue under small-batch scenarios persists. MemBN can serve as a component inside CoTTA.
- **vs $\alpha$-BN**: $\alpha$-BN mitigates the issue by linearly blending training statistics and test statistics, but the mixing coefficient $\alpha$ is a fixed hyperparameter, lacking flexibility.

## Rating
- Novelty: ⭐⭐⭐ The concept of a memory queue is relatively intuitive, but the concrete design at the BN level (adaptive management + weighted aggregation) holds solid engineering value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers combinations of multiple datasets, various batch sizes, and diverse TTA methods.
- Writing Quality: ⭐⭐⭐⭐ The problem definition is clear, the method description is detailed, and the experimental evaluation is systematic.
- Value: ⭐⭐⭐⭐ Addresses a practical and neglected issue, and its plug-and-play attribute makes it easy to adopt.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Test-Time Adaptation by Causal Trimming](../../NeurIPS2025/others/test-time_adaptation_by_causal_trimming.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[ICML 2025\] Beyond Entropy: Region Confidence Proxy for Wild Test-Time Adaptation](../../ICML2025/others/beyond_entropy_region_confidence_proxy_for_wild_test-time_adaptation.md)
- [\[ICLR 2026\] Prior-Free Tabular Test-Time Adaptation](../../ICLR2026/others/prior-free_tabular_test-time_adaptation.md)
- [\[NeurIPS 2025\] SPACE: SPike-Aware Consistency Enhancement for Test-Time Adaptation in Spiking Neural Networks](../../NeurIPS2025/others/space_spike-aware_consistency_enhancement_for_test-time_adaptation_in_spiking_ne.md)

</div>

<!-- RELATED:END -->

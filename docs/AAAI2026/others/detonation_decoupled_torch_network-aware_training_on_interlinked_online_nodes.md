---
title: >-
  [Paper Note] DeToNATION: Decoupled Torch Network-Aware Training on Interlinked Online Nodes
description: >-
  [AAAI 2026][distributed training] This paper proposes FlexDeMo — a hybrid sharding training strategy that integrates Fully Sharded Data Parallelism (FSDP) with decoupled momentum optimization. It applies FSDP sharding wi…
tags:
  - "AAAI 2026"
  - "distributed training"
  - "decoupled momentum"
  - "FSDP"
  - "gradient compression"
  - "large language models"
date: 2026-05-08
content_hash: 46307d67da360615
---

# DeToNATION: Decoupled Torch Network-Aware Training on Interlinked Online Nodes

**Conference**: AAAI 2026  
**arXiv**: [2502.06728](https://arxiv.org/abs/2502.06728)  
**Code**: [github.com/schneiderkamplab/DeToNATION](https://github.com/schneiderkamplab/DeToNATION)  
**Area**: Other (Distributed Training / Systems Optimization)  
**Keywords**: distributed training, decoupled momentum, FSDP, gradient compression, large language models

## TL;DR

This paper proposes FlexDeMo — a hybrid sharding training strategy that integrates Fully Sharded Data Parallelism (FSDP) with decoupled momentum optimization. It applies FSDP sharding within nodes and synchronizes only the fast-moving momentum components across nodes, achieving loss convergence comparable to full-synchronization AdamW while substantially accelerating training.

## Background & Motivation

### Distributed Training Bottleneck

Training large deep neural networks requires transmitting gradients across accelerators, and network bandwidth is increasingly becoming a bottleneck. As the number of participating nodes grows and network congestion intensifies, the overhead of gradient synchronization escalates sharply.

### Limitations of Prior Work

The Decoupled Momentum (DeMo) optimizer reduces communication by exchanging only the fast-moving components of gradients, but it has three critical limitations:

**Models must fit on a single accelerator**: DeMo is built on Distributed Data Parallelism (DDP), requiring the model and optimizer states to reside entirely within each accelerator's memory, making it unsuitable for LLMs.

**all_gather bandwidth grows linearly**: Communication cost scales linearly with the number of accelerators (rather than nodes), leading to poor scalability.

**Unclear hyperparameter selection**: Hyperparameters such as chunk size, TopK, and the sign function lack systematic investigation.

### Paper Goals

This work combines the memory efficiency of FSDP with the communication compression of DeMo to overcome the "large model + low bandwidth" training bottleneck. It also introduces new replication schemes and systematically analyzes key hyperparameters.

## Method

### Overall Architecture

FlexDeMo adopts a **hybrid sharding strategy**:
- **Intra-node**: FSDP is used to shard model and optimizer states across multiple accelerators.
- **Inter-node**: Only selected gradient components (rather than full gradients) are synchronized using a DeMo-style replication scheme.

Core Idea: The typically higher intra-node bandwidth supports full communication, while the lower inter-node bandwidth transmits only compressed, critical information.

### Key Designs

#### 1. **FlexDeMo Optimizer**: Fusion of FSDP and Decoupled Momentum

Algorithm flow (Algorithm 1):
1. **Gradient Reduce-Scatter**: Perform reduce-scatter within the intra-node sharding group $S$ to obtain gradients for local parameter shards.
2. **Local SGD**: Compute local gradient $\Delta_t$.
3. **Momentum Accumulation**: $m_t \leftarrow \beta m_t + \Delta_t$
4. **Extract Fast Component**: Extract the fast-moving component $q_t$ of momentum $m_t$ via DCT-II (or alternative schemes).
5. **Momentum Update**: $m_{t+1} \leftarrow m_t - q_t$ (remove the already-synchronized portion from momentum).
6. **Inter-node Synchronization**: Synchronize $q_t$ within the replication group $R$.
7. **Parameter Update**: $\theta_{t+1} \leftarrow \theta_t - \eta Q_t$

**Key implementation details**:
- The `no_sync` context manager is used to disable automatic gradient synchronization.
- Accelerator 0 of node 0 replicates only with accelerator 0 of node 1, substantially reducing cross-node communication.
- Degenerate behavior: $|R|=1$ reduces to FSDP; $|S|=1$ reduces to DDP+DeMo.

#### 2. **Replication Schemes**: Challenging DeMo's Design Choices

Four replication schemes are introduced and compared:

| Replication Scheme | Selection Strategy | Index Transmission Required | Characteristics |
|---------|---------|-------------|------|
| **DeMo** | Extract fast-moving momentum components via DCT-II | Yes | Original scheme, strong theoretical foundation |
| **Random** | Randomly select $n$ indices | No (shared seed) | Halves bandwidth, independent of frequency-domain transforms |
| **Striding** | Select indices at uniform intervals of $n$ | No (shared seed) | Structured sampling |
| **DiLoCo** | Full synchronization every $n$ steps | No | Federated learning style |

**Advantage of Random**: Because no index transmission is required, Random achieves half the bandwidth of DeMo at the same compression ratio.

#### 3. **Sign Function and Hyperparameter Analysis**

Systematic experiments confirm the following key design choices:
- **Sign before synchronization**: Quantizes gradient values into a ternary system (−1, 0, 1), substantially reducing transmitted data. Experiments demonstrate that directional information is more important than magnitude.
- **Communication precision**: fp32 outperforms fp16; full precision has a significant impact on both DeMo and Random schemes.
- **chunk size = 32**: Validated as the default choice through experimentation.
- **TopK = 4**: Achieves the best performance in T5 experiments.

#### 4. **Decoupled AdamW**: A Decoupled Variant of AdamW

A variant of AdamW is implemented in which the first- and second-order moments (EMA and moving average of squared gradients) are not synchronized. However, experiments show that DeMo-SGD outperforms Decoupled AdamW in most settings.

### Loss & Training

- Standard task losses are used (translation: cross-entropy; classification: cross-entropy; language modeling: causal LM loss).
- Optimizer: DeMo-SGD (SGD + momentum accumulation + decoupled replication).
- No additional loss terms are introduced; the core contribution lies in the communication strategy rather than the objective function.

## Key Experimental Results

### Main Results

**T5-base Translation Task** (Opus Books En-Fr):

| Replication Scheme | Compression Ratio | Validation Loss Rank | Notes |
|---------|--------|------------|------|
| Random 1/2, 1/4 | 50%, 25% | Best | Fast convergence |
| DeMo 1/8, 1/4 | 12.5%, 25% | Second | Higher compression but slightly inferior |
| DiLoCo | Various | Slower | Converges more slowly than DeMo/Random |
| Striding | Various | Slowest | Not competitive |

**OLMo2-1B Causal Language Modeling** (Dolma v1.6, 2 nodes × 4 GPUs, 10K steps):

| Method | Training Loss | Wall-clock Time | Speedup vs. Full Sync |
|------|---------|---------|-------------|
| DeMo 1/32 | **Best** | ~2.6× faster | Significant improvement |
| DeMo 1/16 | Near best | ~2.6× faster | Significant improvement |
| Random 1/4 | Good | ~2.6× faster | Significant improvement |
| Hybrid-FSDP + AdamW | Baseline | Baseline | — |

### Ablation Study

**Bandwidth-Constrained Experiments** (ViT-B, 2 nodes, varying bandwidth):

| Bandwidth (Mbps) | Random SGD 1/32 | DeMo SGD 1/32 | Decoupled AdamW Full Sync |
|------------|----------------|---------------|---------------------|
| 10 | ~3.33× faster than DeMo | Baseline | ~18× slower than Random |
| 100 | Noticeably faster | Moderate | Significantly slower |
| 1000 | Gap narrows | Moderate | Gap narrows |
| 10000 | Nearly identical | Nearly identical | Nearly identical |

**Bandwidth Usage Measurement** (T5-small, compression ratio 1/16):

| Method | Average Bandwidth (Mbps) | Relative Ratio |
|------|----------------|---------|
| Full Sync | 1070 | 7× |
| DeMo | 291 | 2× |
| Random | **152** | 1× |

### Key Findings

1. **Random uses half the bandwidth of DeMo**: No index transmission for selected gradients is required.
2. **Sign is foundational**: Applying the sign function before synchronization yields significant positive effects across all replication schemes.
3. **Optimal replication scheme is task/architecture-dependent**: DeMo performs best on ViT and decoder architectures; Random performs best on encoder-decoder architectures.
4. **64-node scaling experiments**: DeMo does not scale well due to all_gather; Random is 64% faster than full synchronization.
5. **DeMo-SGD > Decoupled AdamW**: SGD is more suitable for decoupled training in the vast majority of settings.

## Highlights & Insights

- **Engineering meets theory**: The work addresses DeMo's practical incompatibility with FSDP while introducing new replication schemes that challenge existing design choices.
- **Elegance of the Random scheme**: Requiring no frequency-domain transforms, no index transmission, and minimal implementation complexity, it achieves competitive performance in many settings.
- **Significance of Sign**: The finding that gradient direction is more important than magnitude carries profound implications for optimization theory.
- **Comprehensive hyperparameter analysis**: Provides practitioners with clear guidance for tuning.

## Limitations & Future Work

1. **Asynchronous communication not yet exploited**: The potential for overlapping communication and computation via CUDA streams remains untapped.
2. **FSDP2/SimpleFSDP not adopted**: Newer FSDP variants may further accelerate base communication.
3. **Cross-node sharding not implemented**: The current approach assumes the model fits within the combined memory of all accelerators on a single node.
4. **No final model quality evaluation**: Only training/validation losses are reported; downstream task performance is not assessed.
5. **Decoupled AdamW underperforms**: Dedicated moment synchronization strategies may be required.

## Related Work & Insights

- **ZeRO / DeepSpeed**: Pioneered staged parameter sharding.
- **DiLoCo**: Local optimization with periodic global averaging in a federated learning style.
- **SignSGD**: Theoretical foundation for gradient sign compression.
- **GradZip / PowerSGD**: Low-rank gradient compression approaches.
- **Insight**: The hybrid sharding + decoupled optimization paradigm is extensible to heterogeneous clusters (e.g., CPU+GPU, cloud+edge).

## Rating

- Novelty: ⭐⭐⭐⭐ (Combining FSDP and DeMo is natural yet non-trivial)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (T5/ViT/OLMo2 across three domains + bandwidth/scalability/hyperparameter analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich figures and tables)
- Value: ⭐⭐⭐⭐⭐ (Directly lowers the bandwidth barrier for large model training; highly practical)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ParaRevSNN: A Parallel Reversible Spiking Neural Network for Efficient Training and Inference](pararevsnn_a_parallel_reversible_spiking_neural_network_for_efficient_training_a.md)
- [\[AAAI 2026\] A Switching Framework for Online Interval Scheduling with Predictions](a_switching_framework_for_online_interval_scheduling_with_pr.md)
- [\[AAAI 2026\] Learning Network Dismantling Without Handcrafted Inputs](learning_network_dismantling_without_handcrafted_inputs.md)
- [\[CVPR 2026\] Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](../../CVPR2026/others/rethinking_snn_online_training_and_deployment_grad.md)
- [\[CVPR 2026\] ZO-SAM: Zero-Order Sharpness-Aware Minimization for Efficient Sparse Training](../../CVPR2026/others/zo-sam_zero-order_sharpness-aware_minimization_for_efficient_sparse_training.md)

</div>

<!-- RELATED:END -->

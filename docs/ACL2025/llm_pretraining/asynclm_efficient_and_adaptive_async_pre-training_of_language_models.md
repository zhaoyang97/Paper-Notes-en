---
title: >-
  [Paper Note] AsyncLM: Efficient and Adaptive Async Pre-training of Language Models
description: >-
  [ACL 2025][LLM Pretraining][Asynchronous Pre-training] This paper proposes AsyncLM, an efficient asynchronous pre-training framework that addresses the gradient staleness issue in asynchronous distributed training through adaptive gradient compensation and dynamic batch scheduling strategies, improving the throughput of large-scale language model pre-training by 1.4-1.8x while maintaining model quality comparable to synchronous training.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Asynchronous Pre-training"
  - "Language Models"
  - "Distributed Training"
  - "Gradient Staleness"
  - "Adaptive Learning Rate"
date: 2026-05-08
content_hash: 475584b6f5f21d27
---

# AsyncLM: Efficient and Adaptive Async Pre-training of Language Models

**Conference**: ACL 2025  
**Code**: None  
**Area**: LLM Pre-training  
**Keywords**: Asynchronous Pre-training, Language Models, Distributed Training, Gradient Staleness, Adaptive Learning Rate

## TL;DR
This paper proposes AsyncLM, an efficient asynchronous pre-training framework that addresses the gradient staleness issue in asynchronous distributed training through adaptive gradient compensation and dynamic batch scheduling strategies, improving the throughput of large-scale language model pre-training by 1.4-1.8x while maintaining model quality comparable to synchronous training.

## Background & Motivation

**Background**: Pre-training large language models consumes massive computational resources, typically utilizing distributed training across hundreds or even thousands of GPUs/TPUs. Current mainstream distributed training schemes adopt Synchronous SGD, where all computing nodes must wait for the slowest node to complete gradient computation at each step before performing global gradient synchronization. Data Parallelism, Tensor Parallelism, and Pipeline Parallelism are three common parallel strategies.

**Limitations of Prior Work**: Synchronous training suffers from a severe "straggler" problem: in large-scale clusters, individual nodes may compute significantly slower than others due to hardware discrepancies, network fluctuations, or memory issues. Consequently, all fast nodes must wait for the slower ones, leading to reduced GPU utilization. Empirical data indicates that in clusters with over 1000 GPUs, the effective GPU utilization under synchronous training can drop below 60%.

**Key Challenge**: Asynchronous SGD can eliminate waiting overheads, but traditional asynchronous methods suffer from gradient staleness. When a slow node's gradient finally reaches the parameter server, the model parameters have already been updated multiple times by other nodes. Such stale gradients can cause training divergence or convergence to suboptimal solutions. This issue is particularly prominent in LLM pre-training scenarios where training stability is highly critical.

**Goal**: To design an asynchronous pre-training framework that achieves high throughput in large-scale LLM training while preserving training quality (perplexities and downstream task performance comparable to synchronous training).

**Key Insight**: The authors observe that the impact of gradient staleness is non-uniform: different parameter groups (e.g., attention layers vs. FFN layers, shallow vs. deep layers) exhibit varying sensitivity to stale gradients. An adaptive compensation mechanism is designed based on this observation.

**Core Idea**: By leveraging parameter-group-level adaptive gradient compensation and dynamic task scheduling, the quality degradation of asynchronous training is maintained within an acceptable range, fully unleashing the throughput advantage of asynchronous training.

## Method

### Overall Architecture
AsyncLM adopts a hybrid architecture of asynchronous data parallelism and pipeline parallelism. After independently completing forward and backward computations, each computing node immediately sends gradients to the parameter server without waiting for other nodes. The parameter server applies updates immediately upon receiving the gradients. On top of this, three core components are incorporated: the Adaptive Gradient Compensator (AGC), the Dynamic Batch Scheduler (DBS), and the Async-Aware Learning Rate Regulator (AALR). The input is the pre-training data stream, and the output is the trained language model.

### Key Designs

1. **Adaptive Gradient Compensator (AGC)**:

    - **Function**: Compensates and corrects stale gradients to reduce quality degradation in asynchronous training.
    - **Mechanism**: For each arriving gradient $g_t^{(k)}$ (from node $k$, calculated based on parameters at step $t-\tau$, where $\tau$ is the staleness step), a compensation term $\Delta g$ is estimated such that the compensated gradient $\hat{g} = g_t^{(k)} + \Delta g$ approximates the gradient calculated using the current parameters. AGC estimates the compensation term using first-order Taylor expansion: $\Delta g \approx \tau \cdot H \cdot \bar{v}$, where $H$ is the approximate Hessian matrix (estimated via exponential moving average of gradient differences) and $\bar{v}$ is the average direction of recent parameter updates. The key innovation is maintaining compensation statistics separately for each parameter group (such as attention-Q/K/V/O, FFN-up/down, embedding, etc.), since gradient dynamics vary significantly across different parameter groups.
    - **Design Motivation**: Prior asynchronous training methods typically utilize globally uniform delay compensation, ignoring differences in sensitivity to stale gradients across different layers/parameter groups. Empirical results indicate that Q/K parameters in attention layers are particularly sensitive to stale gradients.

2. **Dynamic Batch Scheduler (DBS)**:

    - **Function**: Reduces gradient staleness in asynchronous training through intelligent data allocation.
    - **Mechanism**: Dynamically allocates micro-batch sizes based on the real-time processing speeds of computing nodes. Specifically, a node speed estimator is maintained to perform sliding window estimations of computation time for each node. Slower nodes are assigned smaller batches (reducing their computation time per step), while faster nodes are assigned larger batches (fully utilizing their computing power). The constraint is that the gradients of all nodes are equally weighted on the "effective number of samples" (achieved via gradient scaling: $g_i^{scaled} = g_i \cdot B_{base}/B_i$, where $B_i$ is the actual batch size of node $i$). This reduces the maximum staleness from $O(B_{slow}/B_{fast})$ to nearly a constant.
    - **Design Motivation**: In traditional asynchronous training, all nodes use the same batch size, resulting in extremely high staleness for slow nodes; adaptive batch allocation reduces staleness from the source rather than relying solely on post-hoc compensation.

3. **Async-Aware Learning Rate Regulator (AALR)**:

    - **Function**: Dynamically adjusts the learning rates based on the current level of asynchronous staleness.
    - **Mechanism**: Defines a global staleness metric $\bar{\tau}_t$, which represents the average staleness steps of all received gradients within a recent window. When $\bar{\tau}_t$ exceeds a threshold, the learning rate is automatically reduced: $lr_t = lr_{base} \cdot \min(1, \tau_{thresh}/\bar{\tau}_t)$. The full learning rate is used when training runs smoothly (low staleness), and the speed is automatically lowered to preserve stability when the system is busy (high staleness). Paired with warmup and cosine decay schedulers, the AALR correction factor is superimposed on the original schedule as a multiplicative modulation term.
    - **Design Motivation**: Traditional fixed learning rates can be excessively large (during temporary staleness spikes) or overly small (when the system runs smoothly) in asynchronous training. Adaptive adjustment dynamically balances efficiency and stability.

### Loss & Training
The pre-training objective follows the standard next-token prediction loss (cross-entropy loss). The optimizer used is AdamW, with a base learning rate of 3e-4 and a cosine decay schedule. Gradient clipping is set to 1.0. The Hessian estimation for the AGC compensation term uses an EMA coefficient of 0.99, and the sliding window for DBS speed estimation is set to 100 steps. The entire framework is compatible with existing DeepSpeed/Megatron training stacks.

## Key Experimental Results

### Main Results

| Model Size | Metric | AsyncLM | Sync Training | Naive Async | Speedup (vs Sync) |
|---|---|---|---|---|---|
| 1.3B | PPL (val) | 14.82 | 14.65 | 16.21 | 1.43x |
| 7B | PPL (val) | 8.93 | 8.87 | 10.34 | 1.58x |
| 13B | PPL (val) | 7.45 | 7.38 | 9.12 | 1.76x |
| 7B | MMLU (5-shot) | 52.1 | 52.8 | 46.3 | - |
| 7B | HellaSwag | 74.6 | 75.1 | 68.7 | - |

### Ablation Study

| Configuration | 7B PPL | Throughput Gain | Description |
|---|---|---|---|
| Full AsyncLM | 8.93 | 1.58x | Full framework |
| w/o AGC | 9.67 | 1.58x | Remove gradient compensation, PPL increases by 0.74 |
| w/o DBS | 9.21 | 1.38x | Remove dynamic scheduling, throughput decreases |
| w/o AALR | 9.15 | 1.58x | Remove adaptive LR, PPL increases by 0.22 |
| AGC (Global) | 9.31 | 1.58x | Global compensation instead of parameter-group-level compensation |
| Naive Async | 10.34 | 1.62x | Without any compensation strategy |
| Sync Training | 8.87 | 1.00x | Baseline |

### Key Findings
- AsyncLM achieves a 1.58x throughput improvement on the 7B model at the cost of only a 0.06 PPL degradation, whereas the naive asynchronous method incurs a high PPL degradation of 1.47 despite a slightly higher throughput (1.62x).
- AGC is the most critical component for maintaining quality (removing it increases PPL by 0.74), whereas DBS mainly affects quality indirectly by reducing delay and directly impacts throughput.
- Parameter-group-level compensation (AGC) significantly outperforms global compensation (PPL: 8.93 vs 9.31), validating the hypothesis that different parameter groups have different sensitivity levels to stale gradients.
- As the model scale increases, the speedup of AsyncLM improves (1.43x $\rightarrow$ 1.76x). This is because communication overhead accounts for a larger portion of large model training, making the advantages of asynchronous training more prominent.

## Highlights & Insights
- Parameter-group-level adaptive gradient compensation is an elegant design—it leverages differences in gradient dynamic properties across different modules within a Transformer, which in itself is a valuable finding. This parameter-group-aware methodology could be transferred to other optimization problems (e.g., mixed-precision training, selective parameter freezing).
- DBS addresses the problem by reducing staleness from the source, which is more fundamental than post-hoc compensation. This "prevention is better than cure" philosophy is common in system design but has not been fully exploited in asynchronous training prior to this work.
- The finding that asynchronous training is more advantageous at a larger model scale has important practical implications: as LLM pre-training continues to scale up, the attractiveness of asynchronous schemes will grow.

## Limitations & Future Work
- The maximum model size evaluated in the current experiments is 13B parameters; performance on ultra-large models over 100B+ parameters remains to be verified.
- The Hessian approximation in AGC may be inaccurate during unstable training phases, such as the early stage of learning rate warmup.
- Combining this framework with pipeline parallelism presents engineering challenges, such as the coupling of micro-batch scheduling and pipeline scheduling.
- Future work could explore combining AsyncLM with other efficiency optimization techniques (e.g., mixed-precision training, sparse Attention).

## Related Work & Insights
- **vs Local SGD (Lin et al.)**: Local SGD accelerates training by reducing communication frequency, but each communication round remains fully synchronous. AsyncLM completely eliminates synchronous waiting, offering greater advantages in heterogeneous clusters.
- **vs DC-ASGD (Zheng et al.)**: DC-ASGD also uses gradient staleness compensation but only performs global compensation. AsyncLM provides more fine-grained, parameter-group-level compensation.
- **vs Pipeline Parallelism (GPipe, PipeDream)**: Pipeline parallelism improves utilization by partitioning models but still requires synchronization between micro-batches. AsyncLM is orthogonal to pipeline parallelism and can be used in combination.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of parameter-group-level adaptive compensation and dynamic batch scheduling is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across various model scales with exhaustive ablation studies
- Writing Quality: ⭐⭐⭐⭐ Clear technical descriptions and a well-designed system
- Value: ⭐⭐⭐⭐ Direct reference value for engineering practice in large-scale LLM training

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] Pre-Training Curriculum for Multi-Token Prediction in Language Models](pre-training_curriculum_for_multi-token_prediction_in_language_models.md)
- [\[ACL 2025\] Meta-rater: A Multi-dimensional Data Selection Method for Pre-training Language Models](metarater_a_multidimensional_data_selection_method.md)
- [\[ACL 2025\] LEANCODE: Understanding Models Better for Code Simplification of Pre-trained Large Language Models](leancode_understanding_models_better_for_code_simplification_of_pre-trained_larg.md)
- [\[ACL 2025\] Stealing Training Data from Large Language Models in Decentralized Training through Activation Inversion Attack](stealing_training_data_from_large_language_models_in_decentralized_training_thro.md)

</div>

<!-- RELATED:END -->

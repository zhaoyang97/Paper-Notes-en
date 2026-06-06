---
title: >-
  [Paper Note] TimeBill: Time-Budgeted Inference for Large Language Models
description: >-
  [AAAI 2026][Autonomous Driving][Time-budgeted inference] This paper proposes TimeBill, a framework that adaptively adjusts the KV cache eviction ratio under a given time budget via a fine-grained Response Length Predicto…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Time-budgeted inference"
  - "KV cache eviction"
  - "response length prediction"
  - "execution time estimation"
  - "real-time systems"
date: 2026-05-08
content_hash: d915dfa4d364d4c1
---

# TimeBill: Time-Budgeted Inference for Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2512.21859](https://arxiv.org/abs/2512.21859)  
**Code**: None  
**Area**: Autonomous Driving / LLM Inference Optimization
**Keywords**: Time-budgeted inference, KV cache eviction, response length prediction, execution time estimation, real-time systems

## TL;DR

This paper proposes TimeBill, a framework that adaptively adjusts the KV cache eviction ratio under a given time budget via a fine-grained Response Length Predictor (RLP) and a workload-guided Execution Time Estimator (ETE), simultaneously maximizing LLM response quality while guaranteeing inference completion rate.

## Background & Motivation

### State of the Field

Large language models (LLMs) are increasingly deployed in time-critical systems such as robotics, autonomous driving, embodied intelligence, and industrial automation. In these scenarios, LLMs must generate accurate responses within **hard real-time deadlines**; failure to do so is treated as a system fault. Representative examples include:
- Autoware.Flex, which leverages LLMs to translate natural language instructions into formats interpretable by autonomous driving systems
- DriveGPT4, which uses LLMs to perceive the driving environment and produce driving decisions

### Core Challenges

**Execution time uncertainty**: Unlike CNNs, the autoregressive generation process of LLMs renders end-to-end execution time highly uncertain, as it depends on response length.

**Coarse-grained response length prediction**: Existing predictors (e.g., 5-class classification in ProxyModel, 10-class in S3) operate at insufficient granularity, and BERT-based architectures struggle to handle long inputs.

**Inflexibility of fixed KV cache eviction ratios**: Different tasks carry different time budgets; a fixed eviction ratio either causes timeout (ratio too low) or severely degrades response quality (ratio too high).

### Limitations of Prior Work

- **Offline methods** (quantization, pruning): Compress the model before deployment and cannot adapt to time budgets at runtime.
- **Online methods** (KV cache eviction/quantization): Methods such as StreamingLLM and SnapKV employ fixed eviction ratios and ignore time budget constraints.
- **Existing predictors**: BERT-based predictors are constrained by context length and cannot handle long inputs; coarse-grained classification fails to provide sufficiently precise response time estimates.

## Method

### Overall Architecture

The TimeBill framework comprises three core components:

1. **Fine-grained Response Length Predictor (RLP)**: Based on a small language model (SLM), it predicts the response length of the target LLM.
2. **Workload-guided Execution Time Estimator (ETE)**: Combines FLOPs analysis with performance profiling to estimate end-to-end execution time.
3. **Time-budget-efficient inference mechanism**: Adaptively adjusts the KV cache eviction ratio $\alpha$ based on predicted execution time and the time budget.

### Key Designs

#### 1. Problem Formulation

Time-budgeted LLM inference is formulated as a constrained optimization problem:

$$\max_{\theta} \mathcal{M}(\hat{\mathbf{y}}(\theta), \mathbf{y})$$
$$\text{s.t.} \quad t_{\text{e2e}}(x, \theta) \leq T, \quad N \leq N_{\max}$$

where $\mathcal{M}(\cdot)$ denotes the response quality metric, $T$ is the time budget, and $N_{\max}$ is the maximum generation length. The objective is to maximize response quality subject to the time constraint.

#### 2. Fine-Grained Response Length Predictor (RLP)

**Core Idea**: Response length prediction is formulated as a fine-grained classification task. An SLM (Qwen2.5-0.5B-Instruct) replaces BERT to support long-input processing.

- **Architecture**: Embedding layer + $L$ decoder layers (RMSNorm–CausalAttention–RMSNorm–FFN/SwiGLU) + classification head
- **Bucket design**: Response lengths are partitioned into buckets of fixed size $B$, with 512 buckets by default ($B=16$)
- **Knowledge distillation alignment**: Actual response lengths $N_j$ from the target LLM are collected to construct a training dataset $(x_j, \lceil N_j/B \rceil)$, aligning the RLP with the target LLM

Post-processing caps the maximum predicted length:

$$\hat{N} = \min(N_{\max}, \text{Predict}(x) \cdot B)$$

**Design Motivation**: Compared to BERT, the SLM provides a longer context window and handles long inputs more effectively. Fine-grained classification (512 classes) yields more precise predictions than coarse-grained schemes (5–10 classes), and knowledge distillation ensures predictor–target alignment.

#### 3. Workload-Guided Execution Time Estimator (ETE)

**Core Idea**: Execution time is accurately estimated by combining theoretical FLOPs modeling with performance profiling curve fitting.

**FLOPs-based analysis**:
- Prefill stage: execution time is quadratic in input length $N_x$ (due to the $QK^T$ computation in CausalAttention)
- Decoding step: execution time is linear in KV cache length $N_{kv}$

$$\hat{t}_{\text{prefill}}(x) = aN_x^2 + bN_x + c$$
$$\hat{t}_{\text{decoding}}^i(N_{kv}^i) = pN_{kv}^i + q$$

**Performance profiling**: Execution times are measured under varying $N_x$ and $N_{kv}$ configurations, and coefficients $a, b, c, p, q$ are fitted via least squares.

**Effect of KV cache eviction on execution time**: Under eviction ratio $\alpha$, the KV cache length at decoding step $i$ is:

$$N_{kv}^i(x, \alpha) = (1-\alpha)N_x + i - 1$$

A **pessimism factor** $k$ ($k \geq 1$) is introduced to estimate the worst-case execution time (WCET), ensuring hard real-time constraints are satisfied.

#### 4. Time-Budget-Efficient Inference Mechanism

**Core Idea**: The original optimization problem is reformulated as minimizing the KV cache eviction ratio $\alpha$, since a higher eviction ratio degrades response quality.

The closed-form optimal eviction ratio is:

$$\alpha^* = \min\left(\alpha_{\max}, 1 - \frac{T - \hat{t}_{\text{prefill}}(x) - t_{\text{Predict}}(x)}{pN_x(\hat{N}_W - 1)} + \frac{\hat{N}_W - 2}{2pN_x} + \frac{q}{pN_x}\right)$$

**System deployment**: RLP prediction can be executed in parallel with the LLM prefill stage (on a CPU or a separate GPU); if the predictor's execution time is less than the prefill time, the prediction overhead is effectively zero.

### Loss & Training

- The RLP is trained with cross-entropy loss for the classification task.
- The Arena-Human-Preference-100k dataset is used to construct training data, avoiding contamination of test sets.
- The ETE is fitted via performance profiling data and least squares, requiring no neural network training.
- KV cache eviction is implemented using SnapKV, with $\alpha_{\max}$ set to 95%.

## Key Experimental Results

### Main Results

Experiments are conducted on Qwen2.5-7B-Instruct with the LongBench dataset on an NVIDIA A40 GPU.

| Method | Time Budget | Avg. Score (Kill) | Completion Rate (Kill) | Notes |
|--------|-------------|-------------------|------------------------|-------|
| Vanilla | 5–10s | Lowest | Lowest | Frequently times out |
| $\alpha$=25% | 5–10s | Low–medium | Medium | Insufficient eviction |
| $\alpha$=50% | 5–10s | Medium | Medium–high | Rises then falls |
| $\alpha$=95% | 5–10s | Low–medium | Among highest | Excessive eviction, poor quality |
| AWQ | 5–10s | Marginally above Vanilla | Marginally above Vanilla | Orthogonally composable with TimeBill |
| **TimeBill** | **5–10s** | **Highest** | **Comparable to $\alpha$=95%** | Adaptive balance |

### Ablation Study

**Response length predictor comparison**:

| Method | # Buckets | MAE↓ | RMSE↓ | R²↑ |
|--------|-----------|------|-------|-----|
| Ours (regression) | — | 64.21 | 103.30 | 0.516 |
| Ours (128 buckets) | 128 | 48.95 | 87.57 | 0.652 |
| Ours (256 buckets) | 256 | 44.15 | 78.63 | 0.719 |
| **Ours (512 buckets)** | **512** | **42.71** | **78.13** | **0.723** |
| ProxyModel | 5 | 105.72 | 136.79 | 0.152 |
| S3 | 10 | 108.96 | 148.91 | −0.004 |

**Execution time estimation accuracy**:
- Prefill stage MAPE: 1.22%
- Decoding step MAPE: 1.69%

**Effect of pessimism factor $k$** (T=5s, Kill strategy):
- $k=1$–$5$: Increasing $k$ improves both completion rate and average score.
- $k=6$–$8$: Excessively large $k$ leads to overly aggressive eviction ($\alpha$ too high), severely degrading response quality and reducing average score.

### Key Findings

1. Fine-grained classification (512 buckets) achieves more than 2.5× higher prediction accuracy than coarse-grained schemes (5/10 buckets).
2. The SLM-based predictor reduces MAE by 60% compared to BERT-based predictors.
3. TimeBill achieves the highest average response score across all tested time budgets (5–10s).
4. A pessimism factor of $k=5$ is identified as optimal, consistent with common practice in hard real-time systems.

## Highlights & Insights

1. **Novel problem formulation**: This work is the first to formally cast LLM inference as a time-budget-constrained optimization problem, providing a theoretical framework for the field.
2. **Elegant closed-form solution**: By combining FLOPs modeling with performance profiling, the optimal KV cache eviction ratio is derived analytically, eliminating the need for online search.
3. **Efficient system design**: Parallel execution of the RLP alongside the prefill stage eliminates any additional prediction overhead.
4. **Strong practicality**: The framework accommodates heterogeneous time budgets across different inference tasks and is orthogonally composable with offline methods such as quantization.

## Limitations & Future Work

1. Validation is limited to single-GPU, single-request scenarios; batched inference and multi-request scheduling are not considered.
2. The RLP must be retrained for each target LLM, limiting transferability.
3. The pessimism factor $k$ requires manual selection; an adaptive adjustment mechanism is absent.
4. The KV cache eviction strategy is fixed to SnapKV; integration with alternative eviction strategies is unexplored.
5. End-to-end validation on real autonomous driving systems has not been conducted.

## Related Work & Insights

- TimeBill is complementary to KV cache eviction methods such as SnapKV, providing a mechanism for dynamically adjusting the eviction ratio.
- The work offers both a theoretical and practical framework for deploying LLMs in real-time systems.
- The proposed approach may inspire analogous time-budget allocation strategies in multi-model collaborative inference scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ — Novel problem formulation, though the methodology primarily combines existing components.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multiple baselines, eviction strategies, and time budgets are evaluated comprehensively.
- Writing Quality: ⭐⭐⭐⭐⭐ — Mathematical derivations are clear and system design diagrams are thorough.
- Value: ⭐⭐⭐⭐ — Strong reference value for real-time LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TawPipe: Topology-Aware Weight Pipeline Parallelism for Accelerating Long-Context Large Models Training](tawpipe_topology-aware_weight_pipeline_parallelism_for_accelerating_long-context.md)
- [\[CVPR 2026\] Traffic Scene Generation from Natural Language Description for Autonomous Vehicles with Large Language Model](../../CVPR2026/autonomous_driving/traffic_scene_generation_from_natural_language_description_for_autonomous_vehicl.md)
- [\[CVPR 2026\] Learning Vision-Language-Action World Models for Autonomous Driving](../../CVPR2026/autonomous_driving/vla_world_learning_vision_language_action_world_models_for_autonomous_driving.md)
- [\[ICLR 2026\] ST4VLA: Spatially Guided Training for Vision-Language-Action Models](../../ICLR2026/autonomous_driving/st4vla_spatially_guided_training_for_vision-language-action_models.md)
- [\[ICML 2026\] Threshold-Based Exclusive Batching for LLM Inference](../../ICML2026/autonomous_driving/threshold-based_exclusive_batching_for_llm_inference.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Understanding Silent Data Corruption in LLM Training
description: >-
  [ACL 2025][LLM (Other)][Silent Data Corruption] This paper presents the first systematic study on the impact of real-world Silent Data Corruption (SDC) on LLM training. By pairing unhealthy nodes with healthy ones and introducing synchronization mechanisms, the authors reveal SDC characteristics and impact patterns across three levels: submodule computation, single-step gradients, and cumulative training.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Silent Data Corruption"
  - "LLM Training"
  - "Hardware Failures"
  - "Training Stability"
  - "Gradient Noise"
date: 2026-05-08
content_hash: 88551ec27b8080c1
---

# Understanding Silent Data Corruption in LLM Training

**Conference**: ACL 2025  
**arXiv**: [2502.12340](https://arxiv.org/abs/2502.12340)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: Silent Data Corruption, LLM Training, Hardware Failures, Training Stability, Gradient Noise

## TL;DR

This paper presents the first systematic study on the impact of real-world Silent Data Corruption (SDC) on LLM training. By pairing unhealthy nodes with healthy ones and introducing synchronization mechanisms, the authors reveal SDC characteristics and impact patterns across three levels: submodule computation, single-step gradients, and cumulative training.

## Background & Motivation

As the scale of LLM training continues to grow (e.g., Llama 3 405B utilizes 16K H100 GPUs), the probability of hardware failures increases accordingly. Silent Data Corruption (SDC) is an emerging hardware error where hardware silently outputs incorrect computational results without triggering any error signals. Meta reported that 6 unplanned outages during a 54-day pre-training run were attributed to SDC, while Google estimates that SDC events occur every 1-2 weeks during Gemini training.

Despite the growing prevalence of SDC in large-scale training, no prior public work has systematically characterized the impact of real-world SDC on LLM training. The limitations of prior work include: (1) relying mostly on fault-injection simulations rather than real SDC; and (2) focusing primarily on the inference phase, with scarce attention paid to training dynamics. This paper represents the first study to leverage real unhealthy nodes screened from production-level hardware management pipelines.

## Method

### Overall Architecture

In collaboration with a cloud computing platform, the authors obtained 15 unhealthy nodes that failed production-level stress tests due to SDC, alongside 15 healthy nodes that passed the tests. Through the XLA compiler, deterministic execution was ensured to eliminate non-deterministic sources non-related to SDC, enabling precise comparisons of computational results between unhealthy and healthy nodes.

The study is conducted across three levels:
- **RQ1**: The impact of SDC on Transformer submodule computation outputs.
- **RQ2**: The impact of SDC on single-step optimizer gradients.
- **RQ3**: The cumulative impact of SDC over multi-step training.

### Key Designs

1. **Computation Synchronization Mechanism**:

    - Used to isolate the submodule-level SDC impact in RQ1.
    - A "Lock-step Parallelism" communication grid was designed: during the forward pass, after each submodule completes computation and before the reduce-scatter operation, the outputs of the unhealthy and healthy nodes are compared, and the unhealthy node's value is overwritten by the healthy node's value to prevent SDC errors from accumulating into the next submodule.
    - The same logic applies to the backward pass, preventing error accumulation during backpropagation.
    - Key assumption: The communication process itself is unaffected by SDC (guaranteed by checksums).

2. **Parameter Synchronization Mechanism**:

    - Used to isolate the single-step training SDC impact in RQ2.
    - After each optimizer update step, the parameters from the healthy node are broadcasted to the unhealthy node to overwrite them.
    - This ensures that every optimizer step starts with identical parameters.

3. **Metrics Design**:

    - **Mismatch Frequency**: The ratio of mismatched elements to total elements, measuring the prevalence of SDC.
    - **Mismatch Severity**: The mean of non-zero relative differences, measuring the magnitude of deviation caused by SDC.
    - **Worst-Case Noise-to-Signal Ratio (WCNTS)**: The maximum ratio of the $L_2$ norm of the gradient difference to the $L_2$ norm of the true gradient.

### Loss & Training

The experiments utilize a decoder-only Transformer with a Llama3-8B-like configuration (16 layers, hidden dimension of 4096), employing tensor parallelism (TP) to fit the model within a single node. The fine-tuning experiments perform instruction tuning on Mistral-7B-v0.3 across 6 multiple-choice question answering tasks.

## Key Experimental Results

### Main Results

**RQ1: Submodule Computation Impact**

| Node | fwd/attn frequency | fwd/ffn frequency | bwd/attn frequency | bwd/ffn frequency |
|------|------|------|------|------|
| Node 1 | 1.55e-5 | 5.06e-7 | 1.56e-4 | 2.81e-6 |
| Node 10 | 4.78e-3 | 1.03e-3 | 1.92e-3 | 7.98e-5 |
| Node 11 | 2.89e-2 | 2.25e-3 | 6.71e-3 | 1.08e-4 |

The impact of SDC varied drastically across different nodes: Nodes 10 and 11 exhibited high mismatch frequencies, whereas no SDC occurred on Nodes 2 and 3 under this setup.

**RQ2: Gradient Impact**

| Node | WCNTS Ratio |
|------|---------|
| Node 11 (Worst) | 0.051 |
| Node 1 | 0.037 |
| Node 4 | 0.019 |

In the worst-case scenario (Node 11), the gradient difference was only 5.1% of the true gradient norm, suggesting that the direct noise introduced by single-step SDC is relatively small.

**RQ3: Cumulative Impact - Fine-Tuning Results**

| Configuration | CosmosQA Accuracy (Divergence Rate) | MathQA Accuracy (Divergence Rate) |
|------|---------|------|
| Healthy Node | 90.79 (-) | 37.22 (-) |
| Healthy Node (Reseeded) | 89.50 (6.70%) | 38.83 (56.75%) |
| Unhealthy Node 1 | 90.53 (5.15%) | 36.78 (42.24%) |
| Unhealthy Node 6 | 0.00 (100%) | 36.92 (36.82%) |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| SDC Temporal Distribution | Non-uniform, containing spikes | SDC occurs non-uniformly, which may be caused by system-level factors. |
| Mismatch Severity | Up to $10^{12}$ | Individual elements deviate drastically, despite a low overall frequency. |
| Parameter Drift | Loss curves are nearly identical but parameters continuously drift | The model converges to different local optima. |
| Node 13 Latency Behavior | No SDC before step 450, followed by rapid drift | Confirms the temporal non-uniformity of SDC during training. |

### Key Findings

1. **SDC impact varies by node**: Different unhealthy nodes demonstrate distinct SDC patterns, ranging from no impact at all to high-frequency mismatches.
2. **SDC occurs non-uniformly in time**: Spikes often appear at certain steps, likely due to system loads.
3. **Single-step gradient noise is relatively small**: In the worst case, it is only 5.1%, indicating that the immediate impact of single-step SDC is limited.
4. **Cumulative effects lead to parameter drift**: Although training losses remain almost identical, model parameters on unhealthy nodes gradually deviate from those on healthy nodes, with the model converging to different local optima.
5. **Fine-tuning performance is maintained on most nodes**: On most unhealthy nodes, the fine-tuned model performance is comparable to that of healthy nodes, though individual nodes (e.g., Node 6 on CosmosQA) can suffer from complete collapse.
6. **SDC acts as a trigger rather than the primary cause**: The rate of parameter drift is similar across most nodes, suggesting that SDC acts more like a trigger pushing the optimization trajectory onto a different path, rather than continuously exerting massive deviations.

## Highlights & Insights

1. **First Real-World SDC Study**: Unlike previous works that rely on fault-injection simulations, this study utilizes real unhealthy hardware screened from production environments, making the results highly practical and representative.
2. **Ingenious Experimental Design**: The computation synchronization and parameter synchronization mechanisms elegantly decouple the impact of SDC into three independent levels for analysis, providing a valuable experimental methodology.
3. **Discovery of "Different Local Optima"**: It reveals the nature of parameter drift caused by SDC—not a slow degradation of quality, but rather the model being pushed toward alternative, potentially equally good solutions.
4. **Practical Guidance for Large-scale Training**: It highlights the necessity and prioritization of SDC detection and mitigation, specifically pointing out that attention should be directed toward cumulative effects rather than single-step perturbations.

## Limitations & Future Work

1. The experiments leverage an 8B-like configuration model and have not been validated on truly large models (100B+).
2. Only tensor parallelism is employed, without accounting for the interactive effects of SDC in pipeline parallelism and data parallelism.
3. The paper lacks concrete implementations of SDC detection and mitigation strategies.
4. Unhealthy nodes' SDC patterns may be influenced by specific hardware batches, leaving their generalizability unknown.
5. The sensitivity differences to SDC across various training stages (pre-training vs. fine-tuning vs. alignment) are not analyzed.

## Related Work & Insights

This work is related to large-scale training system reliability and hardware fault tolerance. For practical large-scale training, this study suggests: (1) a continuous hardware health monitoring mechanism is needed rather than only scanning during idle times; (2) training checkpoint strategies should account for the cumulative effects of SDC; (3) normal-looking loss curves do not guarantee healthy model weights; periodic parameter consistency verification is necessary.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Represents the first systematic study of the impact of real SDC on LLM training, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ The three-level analysis is comprehensive, though the coverage of model scales and parallel strategies is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ The structure is clear, problem definitions are precise, and the experimental design and analysis are highly logical.
- Value: ⭐⭐⭐⭐⭐ Provides highly valuable references for the practice and theory of large-scale LLM training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Big5-Chat: Shaping LLM Personalities Through Training on Human-Grounded Data](big5-chat_shaping_llm_personalities_through_training_on_human-grounded_data.md)
- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [\[ACL 2025\] Training-free LLM Merging for Multi-task Learning](training-free_llm_merging_for_multi-task_learning.md)
- [\[ACL 2025\] Cross-Modal Alignment for LLM-Enhanced Spoken Language Understanding](cross-modal_alignment_for_llm-enhanced_spoken_language_understanding.md)
- [\[ACL 2025\] AAD-LLM: Neural Attention-Driven Auditory Scene Understanding](aad-llm_neural_attention-driven_auditory_scene_understanding.md)

</div>

<!-- RELATED:END -->

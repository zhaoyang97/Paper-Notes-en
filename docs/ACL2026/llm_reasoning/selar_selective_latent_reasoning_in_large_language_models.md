---
title: >-
  [Paper Note] SeLaR: Selective Latent Reasoning in Large Language Models
description: >-
  [ACL 2026][LLM Reasoning][Latent Reasoning] This paper proposes SeLaR, a lightweight training-free framework that activates soft-embedding latent reasoning through an entropy gating mechanism only during uncertain "explo…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Latent Reasoning"
  - "Entropy Gating"
  - "Soft Embedding"
  - "Contrastive Regularization"
  - "Training-free Inference Enhancement"
date: 2026-05-08
content_hash: b7bd355cf6f67c43
---

# SeLaR: Selective Latent Reasoning in Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.08299](https://arxiv.org/abs/2604.08299)  
**Code**: [GitHub](https://github.com/Parker-rfu/SeLaReasoning)  
**Area**: Model Compression  
**Keywords**: Latent Reasoning, Entropy Gating, Soft Embedding, Contrastive Regularization, Training-free Inference Enhancement

## TL;DR

This paper proposes SeLaR, a lightweight training-free framework that activates soft-embedding latent reasoning through an entropy gating mechanism only during uncertain "exploration steps" while maintaining discrete decoding for high-confidence "certain steps." By introducing entropy-aware contrastive regularization to prevent soft embeddings from collapsing toward dominant tokens, SeLaR consistently outperforms standard CoT and SOTA training-free methods across five reasoning benchmarks.

## Background & Motivation

**Background**: Chain-of-Thought (CoT) has become the dominant paradigm for multi-step reasoning in LLMs, improving performance on complex tasks by generating explicit intermediate reasoning steps. Recent latent reasoning methods attempt to replace discrete token sampling with soft embeddings or hidden states to implicitly explore multiple reasoning paths within a single forward pass.

**Limitations of Prior Work**: (1) Standard CoT must commit to a single discrete token at each step, discarding distributional information about alternative reasoning paths; (2) Training-based latent reasoning methods (e.g., Coconut) suffer from catastrophic forgetting due to the domain gap between hidden states and embedding spaces; (3) Training-free methods (e.g., Soft Thinking) activate soft embeddings globally, introducing unnecessary perturbations in high-confidence steps and undermining reasoning stability.

**Key Challenge**: The entropy distribution during CoT decoding exhibits a clear long-tail structure—most steps are low-entropy "certain steps," while only a few are high-entropy "exploration steps." Global activation ignores this structure, introducing noise in certain steps while failing to maintain multi-path exploration in exploration steps due to soft embeddings collapsing toward dominant tokens.

**Goal**: To address two questions—when to activate latent reasoning (selective activation) and how to maintain effective exploration (preventing collapse).

**Key Insight**: Utilize the entropy of token-level prediction distributions as a confidence signal to categorize decoding steps into certain steps and exploration steps, enabling latent reasoning only during critical exploration steps.

**Core Idea**: Entropy gating for selective activation + entropy-aware contrastive regularization. The former determines "when" to use latent reasoning, while the latter solves "how" to maintain multi-path exploration after activation.

## Method

### Overall Architecture

At each step of decoding, SeLaR: (1) calculates the normalized entropy $\bar{H}_t$ of the top-k tokens; (2) if $\bar{H}_t \leq \tau$ (certain step), standard discrete decoding is used; (3) if $\bar{H}_t > \tau$ (exploration step), a probability-weighted soft embedding of top-k candidates is constructed and contrastive regularization is applied. The regularized soft embedding serves as the input for the next step. The entire process is training-free and plug-and-play.

### Key Designs

1.  **Entropy Gating Selective Activation Mechanism**:
    - **Function**: Enables latent reasoning only when the model is uncertain, maintaining standard decoding for certain steps.
    - **Mechanism**: Computes the truncated entropy $H_t = -\sum_{v \in \mathcal{V}_k} \hat{p}_t(v) \log \hat{p}_t(v)$ for the top-k tokens, normalized as $\bar{H}_t = H_t / \log k$. If $\bar{H}_t \leq \tau$, a sampled/greedy discrete token embedding is used; otherwise, a probability-weighted soft embedding $e_t = \sum_{v \in \mathcal{V}_k} \hat{p}_t(v) \cdot e_v$ is employed. The threshold $\tau$ is stable within the $[0.3, 0.7]$ range, located in the low-density transition zone of the entropy distribution.
    - **Design Motivation**: Experiments show that only a small fraction of steps are high-entropy exploration steps. Introducing soft embeddings in certain steps has a net negative impact—removing selective activation leads to a 5.19% drop in average accuracy.

2.  **Entropy-Aware Contrastive Regularization**:
    - **Function**: Prevents soft embeddings from collapsing toward the dominant token during latent reasoning.
    - **Mechanism**: Calculates the difference vector $\Delta_t = e_t - e_{v_t^*}$ between the soft embedding and the dominant token embedding. After normalization, it uses entropy as a weight to push the embedding away from the dominant direction: $\tilde{e}_t = e_t + \bar{H}_t \cdot \hat{\Delta}_t \cdot \|\Delta_t\|$. Higher entropy results in a stronger push, which naturally diminishes as the model becomes more confident.
    - **Design Motivation**: Previous work found that soft embeddings are quickly dominated by the highest-probability token, degenerating into greedy decoding. Logit lens analysis verifies that without regularization, top-1 overlap dominates in deep layers; with it, top-1 and top-2 overlaps remain comparable, indicating the coexistence of multiple reasoning trajectories.

3.  **Top-k Truncated Entropy Estimation**:
    - **Function**: Efficiently and accurately estimates the model's decision uncertainty.
    - **Mechanism**: Computes entropy only on top-k tokens (rather than the full vocabulary) by re-normalizing their probabilities. This captures uncertainty among the most likely candidates while avoiding interference from low-probability tokens.
    - **Design Motivation**: Full-vocabulary entropy computation is computationally expensive and sensitive to long-tail noise. Top-k truncation is both efficient and focused on decision-relevant uncertainty.

### Loss & Training

SeLaR is entirely training-free. Evaluations were conducted using Qwen3-1.7B/8B/32B and DeepSeek-R1-Distill-Llama-8B. Decoding settings: temperature=0.6, top-p=0.95, top-k=20, min-p=0.0.

## Key Experimental Results

### Main Results

**Accuracy Comparison on Five Reasoning Benchmarks (Qwen3-8B)**

| Method | GSM8K | MATH500 | GPQA | AIME24 | AIME25 | Avg |
|------|-------|---------|------|--------|--------|-----|
| CoT (Sampling) | 95.45 | 98.00 | 61.62 | 76.67 | 66.67 | 79.68 |
| Soft Thinking | 94.92 | 95.80 | 57.58 | 70.00 | 66.67 | 76.99 |
| SwiR | 95.68 | 97.00 | 62.63 | 60.00 | 66.67 | 76.40 |
| **SeLaR** | **95.83** | **97.00** | **61.62** | **83.33** | **80.00** | **83.56** |

### Ablation Study

**Component Ablation (Qwen3-8B)**

| Configuration | Avg | Description |
|------|-----|------|
| Full SeLaR | 83.56 | Complete model |
| w/o Selective Activation | 78.37 | Global activation drops performance by 5.19% |
| w/o Contrastive Regularization | 75.74 | Lack of collapse prevention drops performance by 7.82% |

### Key Findings

- SeLaR consistently outperforms CoT across all model scales, with an average gain of +3.88% on Qwen3-8B, and is the only method to consistently exceed the baseline across all models.
- The most significant improvements are observed on the most difficult AIME benchmarks: +6.66% on AIME24 and +13.33% on AIME25 (Qwen3-8B).
- Contrastive regularization contributes the most (a 7.82% drop if removed), especially on AIME24/25 where performance falls from 83.33/80.00 to 70.00/60.00.
- Computational Efficiency: On AIME24, SeLaR reduces TPCA by 19.2% compared to CoT, whereas SwiR increases it by 33.2%.
- Logit lens analysis confirms: Contrastive regularization keeps top-1 and top-2 overlaps comparable, maintaining genuine multi-path exploration.

## Highlights & Insights

- The observation of the long-tail entropy distribution is the cornerstone of the paper—models are already certain for most steps, and latent reasoning is only valuable at a few critical junctures.
- The design of contrastive regularization is elegant: using entropy itself as the weight for the push-away intensity allows for strong repulsion during exploration steps and natural fading as the model approaches certainty.
- Logit lens analysis provides mechanistic evidence rather than relying solely on ablation studies—it directly visualizes whether multiple trajectories coexist.

## Limitations & Future Work

- Although the threshold $\tau$ is stable within $[0.3, 0.7]$, it remains a dataset-specific hyperparameter and is not yet fully adaptive.
- Effectiveness is limited on knowledge-intensive tasks (GPQA), where domain knowledge recall is more critical than multi-step reasoning.
- Evaluation was limited to reasoning LLMs, and its effects on general instruction-following or code generation tasks have not been verified.
- The direction chosen for contrastive regularization (pushing away only from top-1) might be insufficient—top-2 or top-3 might also be collapse directions that require repulsion.

## Related Work & Insights

- **vs Soft Thinking (Zhang et al., 2025)**: The latter activates soft embeddings globally, while SeLaR uses selective activation—verifying the harm of global activation (a 5.19% drop without selective activation).
- **vs SwiR (Shi et al., 2025)**: The latter triggers switching based on entropy changes between adjacent steps, which is prone to false triggers and requires window smoothing; SeLaR uses absolute entropy thresholds directly, making it simpler and more stable.
- **vs Coconut (Hao et al., 2025)**: The latter requires fine-tuning to propagate hidden states, which leads to catastrophic forgetting; SeLaR is entirely training-free.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of selective activation and contrastive regularization is novel with clear motivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 benchmarks × 4 models × detailed ablations + logit lens mechanistic analysis.
- Writing Quality: ⭐⭐⭐⭐ Complete logical chain from observation to methodology to analysis.
- Value: ⭐⭐⭐⭐ High practical value as a training-free, plug-and-play solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)
- [\[ACL 2026\] TInR: Exploring Tool-Internalized Reasoning in Large Language Models](tinr_exploring_tool-internalized_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Parallel Test-Time Scaling for Latent Reasoning Models](parallel_test-time_scaling_for_latent_reasoning_models.md)
- [\[ICML 2026\] Reasoning Structure of Large Language Models](../../ICML2026/llm_reasoning/reasoning_structure_of_large_language_models.md)
- [\[ACL 2026\] Dissecting Failure Dynamics in Large Language Model Reasoning](dissecting_failure_dynamics_in_large_language_model_reasoning.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] SeLaR: Selective Latent Reasoning in Large Language Models
description: >-
  [ACL 2026][Model Compression][Latent reasoning] This paper proposes SeLaR, a lightweight training-free framework that activates soft-embedding latent reasoning exclusively at high-entropy "exploration steps" via an entropy gating mechanism, while retaining discrete decoding at high-confidence "certainty steps." An entropy-aware contrastive regularization is introduced to prevent soft embeddings from collapsing toward the dominant token. SeLaR consistently outperforms standard CoT and state-of-the-art training-free methods across five reasoning benchmarks.
tags:
  - ACL 2026
  - Model Compression
  - Latent reasoning
  - entropy gating
  - soft embeddings
  - contrastive regularization
  - training-free reasoning enhancement
date: 2026-05-08
content_hash: 625cd3c8aede3a3f
---

# SeLaR: Selective Latent Reasoning in Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.08299](https://arxiv.org/abs/2604.08299)
**Code**: [GitHub](https://github.com/Parker-rfu/SeLaReasoning)
**Area**: Model Compression
**Keywords**: Latent reasoning, entropy gating, soft embeddings, contrastive regularization, training-free reasoning enhancement

## TL;DR

This paper proposes SeLaR, a lightweight training-free framework that activates soft-embedding latent reasoning exclusively at high-entropy "exploration steps" via an entropy gating mechanism, while retaining discrete decoding at high-confidence "certainty steps." An entropy-aware contrastive regularization is introduced to prevent soft embeddings from collapsing toward the dominant token. SeLaR consistently outperforms standard CoT and state-of-the-art training-free methods across five reasoning benchmarks.

## Background & Motivation

**Background**: Chain-of-thought (CoT) has become the dominant paradigm for multi-step reasoning in LLMs, improving performance on complex tasks by explicitly generating intermediate reasoning steps. Recent latent reasoning methods attempt to replace discrete token sampling with soft embeddings or hidden states, enabling implicit exploration of multiple reasoning paths within a single forward pass.

**Limitations of Prior Work**: (1) Standard CoT must commit to a single discrete token at each step, discarding distributional information about alternative reasoning paths. (2) Training-based latent reasoning methods (e.g., Coconut) suffer from catastrophic forgetting due to the domain gap between hidden states and the embedding space. (3) Training-free methods (e.g., Soft Thinking) globally activate soft embeddings, introducing unnecessary perturbations at steps where the model is already confident, thereby destabilizing the reasoning process.

**Key Challenge**: The entropy distribution across CoT decoding steps exhibits a clear heavy-tail structure—most steps are low-entropy certainty steps, while only a small fraction are high-entropy exploration steps. Global activation ignores this structure, introducing perturbations at certainty steps while losing multi-path exploration at exploration steps due to soft embedding collapse toward the dominant token.

**Goal**: Address two problems—when to activate latent reasoning (selective activation) and how to maintain effective exploration (preventing collapse).

**Key Insight**: The entropy of the token-level predictive distribution serves as a confidence signal to partition decoding steps into certainty steps and exploration steps, activating latent reasoning only at critical exploration steps.

**Core Idea**: Entropy-gated selective activation combined with entropy-aware contrastive regularization—the former determines *when* to apply latent reasoning, while the latter ensures *how* to maintain multi-path exploration once activated.

## Method

### Overall Architecture

At each decoding step, SeLaR: (1) computes the normalized entropy $\bar{H}_t$ over the top-$k$ tokens; (2) if $\bar{H}_t \leq \tau$ (certainty step), applies standard discrete decoding; (3) if $\bar{H}_t > \tau$ (exploration step), constructs a probability-weighted soft embedding over the top-$k$ candidates, applies contrastive regularization, and feeds the regularized soft embedding as input to the next step. The entire process requires no training and is plug-and-play.

### Key Designs

1. **Entropy-Gated Selective Activation**:

    - **Function**: Enables latent reasoning only when the model is uncertain; certainty steps retain standard decoding.
    - **Mechanism**: The truncated entropy over the top-$k$ tokens is computed as $H_t = -\sum_{v \in \mathcal{V}_k} \hat{p}_t(v) \log \hat{p}_t(v)$ and normalized to $\bar{H}_t = H_t / \log k$. If $\bar{H}_t \leq \tau$, a discrete token embedding from sampling/greedy decoding is used; otherwise, the probability-weighted soft embedding $e_t = \sum_{v \in \mathcal{V}_k} \hat{p}_t(v) \cdot e_v$ is used. The threshold $\tau$ lies in the low-density transition band of the entropy distribution and remains stable in the range $[0.3, 0.7]$.
    - **Design Motivation**: Experiments show that only a small fraction of steps are high-entropy exploration steps. Applying soft embeddings at certainty steps has a net negative effect—removing selective activation degrades average accuracy by 5.19%.

2. **Entropy-Aware Contrastive Regularization**:

    - **Function**: Prevents soft embeddings from collapsing toward the dominant token during latent reasoning.
    - **Mechanism**: The difference vector between the soft embedding and the dominant token embedding is computed as $\Delta_t = e_t - e_{v_t^*}$, normalized, and then used to push the soft embedding away from the dominant direction with entropy-weighted magnitude: $\tilde{e}_t = e_t + \bar{H}_t \cdot \hat{\Delta}_t \cdot \|\Delta_t\|$. The pushing force increases with entropy and naturally diminishes as the model becomes more confident.
    - **Design Motivation**: Prior work has observed that soft embeddings are rapidly dominated by the highest-probability token, degenerating to greedy decoding. Logit lens analysis validates the regularization: without it, top-1 overlap dominates in deeper layers; with it, top-1 and top-2 overlaps remain comparable, indicating the coexistence of multiple reasoning trajectories.

3. **Top-$k$ Truncated Entropy Estimation**:

    - **Function**: Efficiently and accurately estimates the model's decision-level uncertainty.
    - **Mechanism**: Entropy is computed only over the top-$k$ tokens (rather than the full vocabulary), with probabilities renormalized prior to computation. This captures uncertainty among the most likely candidates while avoiding noise from low-probability tokens.
    - **Design Motivation**: Full-vocabulary entropy computation is costly and susceptible to long-tail noise; top-$k$ truncation is both efficient and focused on decision-relevant uncertainty.

### Loss & Training

SeLaR is entirely training-free. Evaluations use Qwen3-1.7B/8B/32B and DeepSeek-R1-Distill-Llama-8B. Decoding settings: temperature=0.6, top-p=0.95, top-k=20, min-p=0.0.

## Key Experimental Results

### Main Results

**Accuracy comparison across five reasoning benchmarks (Qwen3-8B)**

| Method | GSM8K | MATH500 | GPQA | AIME24 | AIME25 | Avg |
|--------|-------|---------|------|--------|--------|-----|
| CoT (Sampling) | 95.45 | 98.00 | 61.62 | 76.67 | 66.67 | 79.68 |
| Soft Thinking | 94.92 | 95.80 | 57.58 | 70.00 | 66.67 | 76.99 |
| SwiR | 95.68 | 97.00 | 62.63 | 60.00 | 66.67 | 76.40 |
| **SeLaR** | **95.83** | **97.00** | **61.62** | **83.33** | **80.00** | **83.56** |

### Ablation Study

**Component ablation (Qwen3-8B)**

| Configuration | Avg | Note |
|---------------|-----|------|
| Full SeLaR | 83.56 | Complete model |
| w/o Selective Activation | 78.37 | Global activation: −5.19% |
| w/o Contrastive Regularization | 75.74 | No collapse prevention: −7.82% |

### Key Findings

- SeLaR consistently outperforms CoT across all model scales, achieving an average gain of +3.88% on Qwen3-8B and being the only method to consistently surpass CoT across all models.
- The largest gains are observed on the most challenging AIME benchmarks: AIME24 +6.66%, AIME25 +13.33% (Qwen3-8B).
- Contrastive regularization contributes the most (removal causes −7.82%), with AIME24/25 dropping from 83.33/80.00 to 70.00/60.00 without it.
- Computational efficiency: SeLaR reduces TPCA by 19.2% relative to CoT on AIME24, whereas SwiR increases it by 33.2%.
- Logit lens analysis confirms that contrastive regularization maintains comparable top-1 and top-2 overlaps, preserving genuine multi-path exploration.

## Highlights & Insights

- The observation of the heavy-tail entropy distribution is the cornerstone of the paper—most steps are already high-confidence, and latent reasoning is valuable only at a small number of critical steps.
- The contrastive regularization design is elegant: entropy itself serves as the weight for the push-away magnitude, applying strong pressure at exploration steps and naturally fading as the model approaches certainty.
- Logit lens analysis provides mechanistic evidence beyond ablation studies, directly visualizing the presence or absence of multi-trajectory coexistence.

## Limitations & Future Work

- Although $\tau$ is stable in the range $[0.3, 0.7]$, it remains a dataset-specific hyperparameter without fully adaptive selection.
- Performance gains are limited on knowledge-intensive tasks (GPQA), where domain knowledge retrieval is more critical than multi-step reasoning.
- Evaluation is restricted to reasoning-oriented LLMs; effectiveness on general instruction following or code generation tasks has not been verified.
- The direction of contrastive regularization (pushing away from top-1 only) may be insufficient—top-2 and top-3 tokens could also represent collapse directions worth repelling.

## Related Work & Insights

- **vs. Soft Thinking (Zhang et al., 2025)**: Soft Thinking applies global soft embedding activation; SeLaR applies selective activation—removing selectivity causes a 5.19% drop, validating the harm of global activation.
- **vs. SwiR (Shi et al., 2025)**: SwiR triggers switching based on entropy changes between adjacent steps, which is susceptible to false triggers and requires window smoothing; SeLaR uses an absolute entropy threshold directly, yielding a simpler and more stable approach.
- **vs. Coconut (Hao et al., 2025)**: Coconut requires fine-tuning to propagate hidden states and suffers from catastrophic forgetting; SeLaR is entirely training-free.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of selective activation and contrastive regularization is novel with well-motivated design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five benchmarks × four models × detailed ablations + logit lens mechanistic analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The logical chain from observation to method to analysis is complete and coherent.
- **Value**: ⭐⭐⭐⭐ Training-free and plug-and-play, offering high practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] JudgeMeNot: Personalizing Large Language Models to Emulate Judicial Reasoning in Hebrew](judgemenot_personalizing_large_language_models_to_emulate_judicial_reasoning_in_.md)
- [\[AAAI 2026\] Efficient Reasoning for Large Reasoning Language Models via Certainty-Guided Reflection Suppression](../../AAAI2026/model_compression/efficient_reasoning_for_large_reasoning_language_models_via_certainty-guided_ref.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)

<!-- RELATED:END -->

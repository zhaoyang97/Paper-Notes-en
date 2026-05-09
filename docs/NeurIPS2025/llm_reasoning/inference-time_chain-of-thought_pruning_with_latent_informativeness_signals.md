---
title: >-
  [Paper Note] Inference-Time Chain-of-Thought Pruning with Latent Informativeness Signals
description: >-
  [NeurIPS 2025][LLM Reasoning][inference-time scaling] This paper proposes KAPPA (KL-Adjusted Pruned Path Algorithm), which progressively prunes reasoning branches in Best-of-N sampling using three training-free signals — KL divergence, confidence, and entropy — achieving up to 60% peak memory reduction and 90% token generation reduction while maintaining accuracy.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - inference-time scaling
  - chain-of-thought pruning
  - KL divergence
  - Best-of-N
  - reasoning efficiency
date: 2026-05-08
content_hash: f8b73dc841f9a5fa
---

# Inference-Time Chain-of-Thought Pruning with Latent Informativeness Signals

**Conference**: NeurIPS 2025
**arXiv**: [2511.00699](https://arxiv.org/abs/2511.00699)
**Code**: Not open-sourced
**Area**: LLM Reasoning
**Keywords**: inference-time scaling, chain-of-thought pruning, KL divergence, Best-of-N, reasoning efficiency

## TL;DR
This paper proposes KAPPA (KL-Adjusted Pruned Path Algorithm), which progressively prunes reasoning branches in Best-of-N sampling using three training-free signals — KL divergence, confidence, and entropy — achieving up to 60% peak memory reduction and 90% token generation reduction while maintaining accuracy.

## Background & Motivation
**Background**: LLMs improve reasoning accuracy via Chain-of-Thought (CoT) and Best-of-N (BoN) sampling — generating N reasoning paths and selecting the best. This is a core paradigm for inference-time scaling.

**Limitations of Prior Work**: Standard BoN requires fully generating all N paths, with compute and memory costs scaling linearly with N. Existing methods such as ST-BoN truncate poor branches via consistency heuristics, but consistency criteria do not directly assess branch quality. DeepConf uses confidence-weighted voting but still requires completing multiple paths.

**Key Challenge**: The effectiveness of inference-time scaling relies on sampling more paths (larger N is better), yet the cost of fully generating all paths limits the practical range of N — the central challenge is how to substantially reduce redundant computation without sacrificing accuracy.

**Goal**: Design a training-free, information-theoretic branch pruning algorithm that progressively eliminates low-quality reasoning branches during inference.

**Key Insight**: KL divergence is used as a self-supervised signal for branch "informativeness" — branches that deviate more from the unconditional distribution carry more information — combined with confidence and entropy for a composite score.

**Core Idea**: Without any external reward model, the logit distribution characteristics of the model itself suffice to determine which reasoning paths are worth continuing.

## Method

### Overall Architecture
KAPPA consists of three phases: Draft (exploration) → Scoring & Gating (scoring and gate-based pruning) → Continuation (exploitation).

### Key Designs
1. **Draft Phase**

    - **Function**: Generates the prefixes of N reasoning branches in parallel until the truncation point $c$ (the earliest step at which all branches are pairwise inconsistent).
    - **Mechanism**: Sufficiently explores the space to ensure adequate diversity among branches before evaluation begins.
    - **Design Motivation**: Pruning too early risks eliminating potentially promising branches; branches must first "unfold."

2. **Scoring & Gating Phase**

    - **Function**: Within the window $[c, c+\tau)$, computes a composite score for each surviving branch at every step and progressively prunes them.
    - **Mechanism**: Three-signal fusion scoring:
        - **KL Divergence**: $D_{KL}(p_t^i \| q)$, measuring the KL divergence between the current branch logits and the unconditional (BOS token) logits, quantifying information gain.
        - **Confidence**: $C_t^i = \max_v p_t^i(v)$, the top-1 token probability, reflecting the model's certainty about the current prediction.
        - **Entropy**: $H_t^i$, the uncertainty of the logit distribution.
    - **Score Computation**: $s_t^i = w_{KL} \cdot \hat{EMA}_t^i + w_C \cdot \hat{C}_t^i + w_H \cdot \hat{H}_t^i$, with weights $(0.7, 0.2, 0.1)$.
    - **Stabilization**: Median-of-Means (MoM, 4 buckets) combined with exponential moving average (EMA, $\alpha=0.5$) smooths the signals; scores are z-score normalized and clipped to $[-3, 3]$.
    - **Pruning Schedule**: A linear schedule eliminates the lowest-scoring branch at each step, leaving exactly 1 branch after $\tau$ steps.
    - **Design Motivation**: KL divergence serves as a training-free branch quality signal, avoiding the overhead of an external reward model.

3. **Continuation Phase**

    - **Function**: Continues autoregressive decoding on the sole surviving branch until EOS.
    - **Mechanism**: All remaining computation is concentrated on the highest-quality branch.

## Key Experimental Results

### Main Results (Accuracy vs. Efficiency)
Evaluated on GSM8K and MATH500 with N = 5/10/20:

| Model | Dataset | Method | N=20 Accuracy | N=20 Peak Memory | Token Reduction |
|-------|---------|--------|--------------|-----------------|----------------|
| DeepSeek-R1-1.5B | MATH500 | BoN | ~70% | 16240 MB | - |
| DeepSeek-R1-1.5B | MATH500 | KAPPA | **72.2%** (+1–2%) | **6495 MB** | **~90%** |
| Qwen2.5-7B | GSM8K | BoN | baseline | baseline | - |
| Qwen2.5-7B | GSM8K | KAPPA | ≈baseline | ~40%↓ | ~65% |

### Core Efficiency Metrics
- **Peak memory reduction**: 4%–60% (depending on model and N).
- **Token generation reduction**: 65%–90% relative to BoN.
- **Largest observed difference**: DeepSeek-R1-1.5B on MATH500, N=20 → KAPPA uses only 2,113 tokens vs. BoN's 20,053 tokens (89.5% reduction).

### Ablation Study / Key Findings
- **Smaller models benefit more**: KAPPA consistently improves accuracy by 1–2% on DeepSeek-R1-1.5B, as smaller models produce more low-quality branches, making pruning more effective.
- **Over-pruning risk for larger models**: Accuracy gains are inconsistent on Qwen2.5-7B, because the overall branch quality of larger models is higher and linear pruning may eliminate promising branches prematurely.
- **Hyperparameters**: KL weight of 0.7 is the most critical; EMA rate 0.5; MoM window 16; 4 buckets.

## Highlights & Insights
- **Training-free branch quality signals**: KL divergence, confidence, and entropy are computed directly from model logits without any external reward model or additional training, yielding a lightweight and general-purpose solution.
- **Progressive pruning outperforms one-shot truncation**: Finer-grained than ST-BoN's one-shot truncation; gradually eliminating low-scoring branches allows scoring signals more time to accumulate.
- **Information-theoretic perspective on path evaluation**: Using KL divergence to measure how far a branch deviates from the unconditional distribution as a proxy for informativeness is intuitively compelling — conditional reasoning should deviate more from the prior to carry more information.
- **Practical solution under the "more thinking is not always better" paradigm**: Directly addresses the N-scaling bottleneck of BoN.

## Limitations & Future Work
- **Only two models and two datasets tested**: Generalizability remains to be verified across tasks such as code reasoning and commonsense reasoning.
- **max_new_tokens capped at 1024**: DeepSeek-R1 on MATH500 frequently requires more than 1,024 tokens, and this truncation affects results.
- **Linear pruning schedule may be suboptimal**: Overly aggressive for larger models; a cosine schedule or adaptive schedule may be preferable.
- **Accuracy may degrade as N increases**: More branches amplify scoring signal noise and exacerbate over-pruning.
- **Choice of reference distribution for KL divergence**: Whether the unconditional logits of the BOS token constitute the optimal reference is an open question; alternative reference distributions warrant exploration.

## Related Work & Insights
- **ST-BoN** (Wang et al., 2025): The direct predecessor of this work, using consistency-based one-shot truncation → KAPPA replaces this with information-theoretic progressive pruning.
- **DeepConf** (Fu et al., 2025): Confidence-weighted voting still requires complete generation of multiple paths → KAPPA prunes early.
- **INFORM** (Zhou et al., 2024): Adaptively determines the number of sampled paths without intervening within individual paths → KAPPA applies fine-grained control inside each path.
- **ThinkPrune** (Hou et al., 2025): Uses RL to train models to produce shorter reasoning chains → KAPPA provides a training-free alternative.
- Insight: KL divergence signals may also be applicable to draft quality assessment in speculative decoding.

## Rating
- Novelty: ⭐⭐⭐ The idea of combining KL divergence, confidence, and entropy for branch pruning is original, though each individual component is not novel in isolation.
- Experimental Thoroughness: ⭐⭐⭐ Only two models and two datasets, with token length constraints and insufficiently comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ The algorithm is described clearly; the three-phase structure is easy to follow; figures and tables are presented intuitively.
- Value: ⭐⭐⭐⭐ Inference-time efficiency optimization is a high-demand research direction, and the training-free nature of the approach offers strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Latent Chain-of-Thought for Visual Reasoning](latent_chain-of-thought_for_visual_reasoning.md)
- [\[ICLR 2026\] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](../../ICLR2026/llm_reasoning/fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)
- [\[ACL 2026\] CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning](../../ACL2026/llm_reasoning/crisp_compressing_redundancy_in_chain-of-thought_via_intrinsic_saliency_pruning.md)
- [\[NeurIPS 2025\] Re-FORC: Adaptive Reward Prediction for Efficient Chain-of-Thought Reasoning](re-forc_adaptive_reward_prediction_for_efficient_chain-of-thought_reasoning.md)
- [\[NeurIPS 2025\] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)

</div>

<!-- RELATED:END -->

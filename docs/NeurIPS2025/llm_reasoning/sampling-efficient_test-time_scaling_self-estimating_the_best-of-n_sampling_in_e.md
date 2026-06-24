---
title: >-
  [Paper Note] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding
description: >-
  [NeurIPS 2025 Spotlight][Reasoning][Best-of-N] This paper proposes Self-Truncation Best-of-N (ST-BoN), a decoding method that leverages a theoretical guarantee showing early hidden-state consistency predicts final consistency, enabling identification and truncation of suboptimal samples at early decoding steps. ST-BoN reduces memory usage by over 80% and latency by ~50% while preserving standard BoN performance.
tags:
  - "NeurIPS 2025 Spotlight"
  - "Reasoning"
  - "Best-of-N"
  - "test-time scaling"
  - "early truncation"
  - "latent consistency"
  - "token efficiency"
date: 2026-05-08
content_hash: c60ec61113fe97c9
---

# Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2503.01422](https://arxiv.org/abs/2503.01422)  
**Code**: [https://github.com/Alsace08/ST-BoN](https://github.com/Alsace08/ST-BoN)  
**Area**: LLM Reasoning / Test-Time Computation
**Keywords**: Best-of-N, test-time scaling, early truncation, latent consistency, token efficiency

## TL;DR
This paper proposes Self-Truncation Best-of-N (ST-BoN), a decoding method that leverages a theoretical guarantee showing early hidden-state consistency predicts final consistency, enabling identification and truncation of suboptimal samples at early decoding steps. ST-BoN reduces memory usage by over 80% and latency by ~50% while preserving standard BoN performance.

## Background & Motivation

**Background**: Best-of-N (BoN) sampling is a widely adopted test-time scaling strategy—N candidate responses are generated and the best is selected via a reward model or self-consistency. BoN effectively exploits high-quality solutions in the model's distribution.

**Limitations of Prior Work**: (a) Generating N complete samples requires substantial GPU memory due to linear KV cache growth, constraining the feasible value of N; (b) reward models (RMs) consume additional memory and inference time, and training strong RMs is costly with limited generalizability. Existing improvements (e.g., FastRM, TreeBoN) address only one of these challenges.

**Key Challenge**: The efficiency bottleneck of BoN stems from two implicit assumptions—complete generation of all N samples and external RM scoring—raising the question of whether the most promising sample can be identified much earlier.

**Goal**: To autonomously identify the most promising sample at early decoding steps and truncate the rest, without any external reward model.

**Key Insight**: The core insight of self-consistency is that "if multiple reasoning paths converge to the same answer, that answer is more reliable." If early hidden-state consistency already predicts final consistency, truncation can be applied at early steps.

**Core Idea**: Measure inter-sample consistency using Chain-of-Embedding (CoE) features derived from the LLM's own hidden states, predict the final best sample at the first divergence point, and truncate all others.

## Method

### Overall Architecture
ST-BoN operates in three stages: (1) generate N samples in parallel until the earliest divergence point $c$ (where all sample sequences differ pairwise); (2) continue generation for $\tau$ additional steps, using hidden-state consistency at each step to self-evaluate sample quality and vote for the most promising candidate; (3) truncate the remaining $N-1$ samples and complete only the selected best sample.

### Key Designs

1. **Theoretical Foundation: Early Consistency Predicts Final Consistency (Theorem 1)**:

    - Function: Proves that samples with smaller pairwise distances at early steps are more likely to remain close at generation end.
    - Mechanism: Let $S_t = \sum_i d_t^i$ denote the total pairwise distance at step $t$. Under local Lipschitz continuity and bounded increment assumptions, $\mathbb{E}[S_{t+1}] \leq \Gamma \cdot S_t$ where $\Gamma = 1+LM$. Markov's inequality yields $\Pr[S_T \leq \epsilon | S_t] \geq 1 - \frac{\Gamma^{T-t}}{\epsilon} S_t$.
    - Design Motivation: Provides a principled theoretical guarantee for early truncation—probabilistically controlled rather than purely heuristic.

2. **Chain-of-Embedding (CoE) Hidden-State Consistency Measure**:

    - Function: Measures inter-sample divergence via the curvature of "latent reasoning trajectories" represented by the LLM's internal hidden states.
    - Mechanism: For each sample, sentence embeddings $\mathbf{h}_l^T$ are extracted from each layer; the cross-layer difference of normalized Manhattan distance and angular distance, $\mathcal{F}(\mathbf{H})$, is computed. Inter-sample distance is defined as $\mathcal{D}(Y^i, Y^j) = (\mathcal{F}(\mathbf{H}^i) - \mathcal{F}(\mathbf{H}^j))^2$. The sample minimizing $\mathcal{D}$ (most consistent) is selected.
    - Design Motivation: At early decoding steps, token-level differences are minimal and difficult to distinguish; hidden states carry richer semantic information and capture divergences in reasoning trajectories not yet visible at the token level.

3. **Buffer Window for Robust Estimation**:

    - Function: Aggregates votes over a buffer window $[c, c+\tau]$ to reduce variance from single-step estimation.
    - Mechanism: $\tau = m \cdot c$ (default $m=1$); the best sample is independently selected at each step, with the final decision determined by majority vote.
    - Design Motivation: Divergences at point $c$ may be too subtle to reliably detect; the buffer window allows differences to accumulate before truncation.

### Loss & Training
ST-BoN is a training-free inference method requiring no additional training and applied directly during decoding.

## Key Experimental Results

### Main Results
Evaluated on four objective tasks (MATH, TheoremQA, GPQA, MMLU) and two subjective tasks (CNNDM, AlpacaFarm) across three models:

| Method | Memory Reduction | Latency Reduction | Performance |
|--------|-----------------|-------------------|-------------|
| Full-BoN w/o RM (self-consistency) | Baseline | Baseline | Baseline |
| Full-BoN w/ RM (PRM) | Higher (RM loaded) | Higher | Typically best |
| ST-BoN | **>80%** | **~50%** | ≈ Full-BoN w/o RM |

Computational savings to match Full-BoN performance:

| Model | Task | Full-BoN N | ST-BoN Equivalent | Savings |
|-------|------|-----------|-------------------|---------|
| Llama3-8B | MATH | N=16 | N=64 ST-BoN ≈ N=16 Full | 70–80% |
| Qwen2.5-7B | MATH | N=16 | Similar | 70–80% |

Performance gains under equal compute budget:

| Model | Task | Full-BoN Acc | ST-BoN Acc | Gain |
|-------|------|-------------|-----------|------|
| Llama3-8B | MATH | 50.2% (N=8) | **53.4%** | +3.2 |
| Qwen2.5-7B | TheoremQA | 46.8% (N=8) | **50.1%** | +3.3 |

### Ablation Study

| Configuration | Effect | Notes |
|---------------|--------|-------|
| $\tau=0$ (no buffer) | 2–3% performance drop | Buffer window needed for stable estimation |
| $m=0.5$ vs $m=1$ vs $m=2$ | $m=1$ optimal | Too short: unstable; too long: wasteful |
| CoE vs output token distance | CoE significantly better | Hidden states more discriminative than output tokens |
| 72B model | Remains effective | Generalizes across model scales |

### Key Findings
- The divergence point $c$ typically occurs at 5–10% of total generation length $T$, meaning the truncation window closes very early.
- GPU memory savings increase with N and exceed 80% for $N \geq 5$.
- ST-BoN remains effective on subjective tasks (summarization, instruction-following), demonstrating the generality of CoE consistency.
- Compared to PRM-based BoN, ST-BoN achieves competitive performance without requiring any external model.

## Highlights & Insights
- **Elegant theory-to-practice loop**: The paper first proves a probabilistic guarantee from early to final consistency, then designs a practical CoE measure, and finally mitigates estimation variance via a buffer window—each step with clear motivation.
- **Complete elimination of RM dependency**: Hidden states from the model itself serve as a self-verifier, achieving true plug-and-play deployment without any external reward model.
- **Practical deployment value**: Over 80% memory reduction and ~50% latency reduction are highly significant for LLM inference serving—the same GPU budget can support more requests or larger N.

## Limitations & Future Work
- CoE feature computation requires extracting hidden states from all layers, which may be inconvenient in certain inference frameworks (e.g., vLLM).
- While effective for reasoning tasks, early truncation may sacrifice valuable diversity in tasks where variety is important, such as creative generation.
- The Lipschitz continuity and bounded increment assumptions underlying the theoretical analysis require further validation in practical LLMs.
- Comparisons are limited to self-consistency and PRM; no evaluation against other efficiency-oriented methods such as speculative decoding.

## Related Work & Insights
- **vs. Full-BoN**: ST-BoN is an efficient approximation of Full-BoN, replacing complete generation and global scoring with early consistency estimation.
- **vs. VG-Search**: VG-Search optimizes verification granularity (frequency of verifier calls), while ST-BoN optimizes sample count (early truncation of unnecessary samples); the two perspectives are orthogonal.
- **vs. Self-Consistency**: ST-BoN inherits the core intuition of self-consistency but measures consistency in latent space rather than output space, and advances consistency detection to early decoding steps.

## Rating
- Novelty: ⭐⭐⭐⭐ — The idea of using early hidden-state consistency to predict final performance is novel and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Six datasets, four models (including 72B), and multi-dimensional ablations provide comprehensive validation.
- Writing Quality: ⭐⭐⭐⭐ — The theory–method–experiment logical chain is clear, though the presentation of the main method could be more concise.
- Value: ⭐⭐⭐⭐⭐ — Highly practical: no model training required, plug-and-play, with substantial inference resource savings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)
- [\[NeurIPS 2025\] Scalable Best-of-N Selection for Large Language Models via Self-Certainty](scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)
- [\[ICLR 2026\] CaTS: Calibrated Test-Time Scaling for Efficient LLM Reasoning](../../ICLR2026/llm_reasoning/cats_calibrated_test-time_scaling_for_efficient_llm_reasoning.md)

</div>

<!-- RELATED:END -->

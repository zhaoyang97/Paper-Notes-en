---
title: >-
  [Paper Note] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling
description: >-
  [NeurIPS 2025][LLM Reasoning][test-time scaling] This paper proposes Variable Granularity Search (VG-Search), which unifies Beam Search and Best-of-N under a tunable verification granularity parameter $g$. It demonstrates that conventional per-step verification is suboptimal, and that adaptively adjusting $g$ can improve accuracy by 3%+ while reducing computation by 52%+.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - test-time scaling
  - verification granularity
  - beam search
  - process reward model
  - compute efficiency
date: 2026-05-08
content_hash: a6dadfbc77a45e02
---

# Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling

**Conference**: NeurIPS 2025
**arXiv**: [2505.11730](https://arxiv.org/abs/2505.11730)
**Code**: [https://github.com/hmarkc/VG-Search](https://github.com/hmarkc/VG-Search)
**Area**: LLM Reasoning / Test-Time Compute
**Keywords**: test-time scaling, verification granularity, beam search, process reward model, compute efficiency

## TL;DR
This paper proposes Variable Granularity Search (VG-Search), which unifies Beam Search and Best-of-N under a tunable verification granularity parameter $g$. It demonstrates that conventional per-step verification is suboptimal, and that adaptively adjusting $g$ can improve accuracy by 3%+ while reducing computation by 52%+.

## Background & Motivation

**Background**: Test-time scaling (TTS) improves LLM performance by allocating additional computation at inference time. Sampling-based methods (Beam Search, Best-of-N) employ verifiers (PRMs) to guide search. The current standard practice is to invoke the verifier at every generation step, typically delimited by newlines.

**Limitations of Prior Work**: (a) PRM score differences between consecutive steps are negligible in more than 50% of cases (score gap < 1%), indicating substantial redundancy in per-step verification; (b) verifier call latency constitutes an increasingly large fraction of total inference time, becoming a computational bottleneck.

**Key Challenge**: High verification frequency yields precise search but is computationally expensive; low frequency is cheaper but misses opportunities for error correction. The prevailing fixed setting of $g=1$ is heuristic and not necessarily optimal.

**Goal**: (1) Systematically study how verification granularity affects the accuracy–compute trade-off; (2) design an adaptive strategy for dynamically selecting the optimal $g$.

**Key Insight**: Drawing an analogy to the infinite monkey theorem — the verification frequency $g$ controls the timing of error pruning. Too frequent wastes computation; too sparse misses correction opportunities. An optimal equilibrium point exists that depends on generator capability and task difficulty.

**Core Idea**: The verifier need not be invoked at every step. Strong generators on easy tasks benefit from sparse verification, whereas weak generators on hard tasks require frequent verification.

## Method

### Overall Architecture
In each search cycle, VG-Search: (1) scores and selects among $B_1 \times B_2$ candidates via the verifier → (2) retains the top $B_1$ beams → (3) extends each beam by $g-1$ steps without verification (Extend) → (4) branches each beam into $B_2$ single-step continuations (Branch) → returns to (1). Setting $g=1$ recovers standard Beam Search; setting $g=L$ recovers Best-of-N.

### Key Designs

1. **Unified Search Framework — VG-Search**:

    - **Function**: Unifies Beam Search and Best-of-N into a continuous spectrum via the granularity parameter $g$.
    - **Mechanism**: The key innovation lies in the Extend step — selected beams freely generate $g-1$ steps without invoking the verifier before branching and verification. Since only $B_1$ beams participate in Extend (rather than $B_1 \times B_2$), generation FLOPs are substantially reduced.
    - **Design Motivation**: The Extend step occurs prior to verification, realizing "early pruning" — verify and select first, then expand, rather than the reverse.

2. **Computational Cost Model**:

    - **Function**: Precisely quantifies the computational overhead at different values of $g$ using FLOPs.
    - **Mechanism**: $C_{total} = 2B_1 \frac{L}{g}[T(g-1+B_2)P_g + \alpha B_2 P_v]$. Increasing $g$ reduces the number of verification calls $L/g$, linearly decreasing verification FLOPs; generation FLOPs also decrease due to the reduced branching overhead of $(g-1+B_2)/g$.
    - **Design Motivation**: Provides a theoretical framework for rigorously analyzing the cost of different $g$ values, guiding the adaptive strategy.

3. **Adaptive VG-Search Strategy**:

    - **Function**: Dynamically selects the optimal $g$ based on the computational budget and task properties.
    - **Mechanism**: Performance–compute curves are profiled over a validation set (MATH-250) for different $(g, n)$ combinations to identify Pareto-optimal configurations. Two strategies are proposed: (a) Budget-Adaptive — selects the optimal $g$ given a FLOPs budget; (b) Task-Adaptive — dynamically adjusts $g$ based on problem difficulty.
    - **Design Motivation**: Experiments reveal that the optimal $g$ varies jointly with generator capability, verifier quality, and computational budget, such that no single fixed value is universally appropriate.

## Key Experimental Results

### Main Results
Evaluated on MATH-500 and AIME:

| Configuration | $g=1$ (Beam) | $g=3$ (Optimal) | Gain |
|---|---|---|---|
| Strong G + Small V (MATH-500) | 87.5% | **~88%** | +0.5%, 75% FLOPs reduction |
| Strong G + Small V (AIME) | ~50% | **~54%** | +4%, $g=4$ optimal |
| Medium G + Large V (MATH-500) | ~83% | **~84%** ($g=2$) | +1% |
| Weak G + Small V (MATH-500) | **Best** | Decreases | Weak generators require $g=1$ |

Final results of Adaptive VG-Search:

| Method | MATH-500 Acc | AIME Acc | FLOPs Saved |
|---|---|---|---|
| Beam Search ($g=1$) | 87.4% | 50.0% | Baseline |
| Best-of-N | 86.0% | 46.7% | — |
| Adaptive VG-Search | **90.5%** | **53.6%** | **>52%** |

### Ablation Study

| Ablation | Finding |
|---|---|
| Random PRM (uninformative verifier) | Optimal $g$ increases substantially (sparser verification preferred) |
| Different model scales | The dynamic optimality of $g$ holds consistently across scales |
| Different difficulty levels | Easy problems require small $g$; hard problems benefit from larger $g$ |
| DVTS vs. VG-Search | VG-Search achieves higher compute efficiency while maintaining diversity |

### Key Findings
- **Strong generators prefer sparse verification**: Qwen2.5-Math-7B peaks at $g=3$, as it reliably generates extended correct reasoning segments; frequent verification introduces noise by truncating at immature points.
- **Weak generators require frequent verification**: Llama-3B is optimal at $g=1$ on MATH-500, but also shifts toward sparser verification on the harder AIME benchmark.
- **Optimal $g$ varies with compute budget**: At low budgets, $g=1$ (aggressive pruning) is preferred; at high budgets, $g>1$ (broader exploration) becomes superior.
- **PRM score stability**: More than 50% of consecutive step score differences are below 1%, directly validating the redundancy of current per-step verification.

## Highlights & Insights
- **Redefining the "reasoning step"**: The newline-delimited "step" currently in use is too fine-grained and does not correspond to a meaningful reasoning unit. Larger $g$ defines semantically coherent reasoning blocks, and delayed verification avoids injecting noise on incomplete reasoning fragments.
- **Deeper understanding of generator–verifier co-adaptation**: Verification frequency should be dynamically adjusted according to the ratio of the two components' capabilities. This is not merely an efficiency concern — it also affects search quality, as excessively frequent verification suppresses the generator's autonomous reasoning.
- **Practically valuable compute savings**: Reducing FLOPs by 52%+ without sacrificing — and even improving — performance is highly valuable for LLM inference deployment.

## Limitations & Future Work
- Validation is limited to mathematical reasoning; the behavior of optimal $g$ on code generation, logical reasoning, and other tasks remains unexplored.
- The adaptive strategy requires prior profiling on a validation set, introducing setup overhead.
- Only PRM-based verifiers are considered; outcome reward models (ORM) and LLM-as-verifier settings are not covered.
- $g$ is fixed throughout the search process — ideally, $g$ should be adjusted dynamically as reasoning progresses (e.g., small $g$ early on, larger $g$ later).

## Related Work & Insights
- **vs. Beam Search**: VG-Search with $g=1$ reduces to Beam Search, but $g>1$ achieves significant speedup by reducing verification frequency.
- **vs. Best-of-N**: Best-of-N is the extreme case of $g=L$ and lacks intermediate guidance; VG-Search provides a continuous interpolation.
- **vs. DVTS**: DVTS targets search diversity while VG-Search targets verification efficiency; the two are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐ The dimension of verification granularity has been previously overlooked and warrants systematic study. The VG-Search unified framework is concise and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The combinatorial experiments across multiple models, datasets, and granularity levels are highly comprehensive, with rigorous ablation controls.
- Writing Quality: ⭐⭐⭐⭐⭐ The paper is clearly structured, the research-question-driven narrative is effective, and the figures and tables are well designed.
- Value: ⭐⭐⭐⭐⭐ Highly practical insight — tuning a single hyperparameter yields substantial savings in inference cost.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[NeurIPS 2025\] LIMOPro: Reasoning Refinement for Efficient and Effective Test-time Scaling](limopro_reasoning_refinement_for_efficient_and_effective_test-time_scaling.md)
- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)
- [\[NeurIPS 2025\] Atom of Thoughts for Markov LLM Test-Time Scaling](atom_of_thoughts_for_markov_llm_testtime_scaling.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)

<!-- RELATED:END -->

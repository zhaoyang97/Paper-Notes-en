---
title: >-
  [Paper Note] DISC: Dynamic Decomposition Improves LLM Inference Scaling
description: >-
  [NeurIPS 2025][LLM Efficiency][Test-time compute] DISC proposes a dynamic decomposition algorithm that automatically and recursively adjusts the granularity of reasoning steps at inference time based on the z-score (normalized maximum of sampled rewards) at each step — decomposing difficult steps more finely while taking larger strides over easy ones. It can be plugged into greedy search, Beam Search, and MCTS, achieving higher pass@k with fewer token budgets on APPS, MATH…
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "Test-time compute"
  - "dynamic decomposition"
  - "search algorithms"
  - "inference-time scaling"
  - "reasoning efficiency"
date: 2026-05-08
content_hash: 39ce218bde36138e
---

# DISC: Dynamic Decomposition Improves LLM Inference Scaling

**Conference**: NeurIPS 2025
**arXiv**: [2502.16706](https://arxiv.org/abs/2502.16706)  
**Code**: [disc-search.github.io](https://disc-search.github.io/)  
**Area**: LLM Efficiency
**Keywords**: Test-time compute, dynamic decomposition, search algorithms, inference-time scaling, reasoning efficiency

## TL;DR
DISC proposes a dynamic decomposition algorithm that automatically and recursively adjusts the granularity of reasoning steps at inference time based on the z-score (normalized maximum of sampled rewards) at each step — decomposing difficult steps more finely while taking larger strides over easy ones. It can be plugged into greedy search, Beam Search, and MCTS, achieving higher pass@k with fewer token budgets on APPS, MATH, and LiveCodeBench.

## Background & Motivation

**Background**: Test-time compute has become the dominant paradigm for enhancing LLM reasoning. Existing methods typically decompose a problem into steps and then sample and select at each step (e.g., Best-of-N, Tree of Thoughts, MCTS).

**Limitations of Prior Work**: The granularity of problem decomposition is usually **statically predefined** — either token-level (too fine, search is too slow), line/sentence-level (too coarse, may skip critical decision points), or full-sequence generation (unable to leverage intermediate step information). These static strategies cannot adaptively allocate compute: they waste computation on steps that are easy for the LLM while under-sampling at difficult steps.

**Key Challenge**: Coarse-grained step decomposition is fast but may miss the optimal solution; fine-grained decomposition is precise but extremely slow to search. Manually designed decomposition rules lack generality, and the critical decision points for autoregressive LLMs (e.g., pivot words like "which" or "therefore") are often not intuitively predictable by humans.

**Goal**: How to automatically and dynamically determine step granularity at inference time, efficiently allocating reasoning compute without requiring manual design, domain knowledge, or process reward models?

**Key Insight**: Exploit the LLM's own sampling statistics — if the z-score of the reward distribution when continuing from a given prefix is low (i.e., there is a higher probability of sampling a better result), the prefix is promising and should be accepted for continuation; otherwise, reduce the step granularity (shrink to a shorter prefix) for finer-grained search.

**Core Idea**: Use z-scores to compare sampled reward distributions across different prefixes, dynamically deciding whether to accept (take a large step forward) or reject (shrink to a smaller step), thereby achieving adaptive decomposition of reasoning steps.

## Method

### Overall Architecture
Given problem $x$, DISC maintains a "base prefix" $p_b$ (initialized to $x$). At each iteration: (1) sample multiple complete solutions from $p_b$, compute rewards and z-score; (2) take the first $\alpha$ fraction of the best suffix as candidate prefix $p_c = p_b \cdot \text{split}(s^*, \alpha)$; (3) sample from $p_c$ and compute its z-score; (4) if $z_c < z_b$ (the candidate is more promising than the base), accept the candidate as the new base prefix and reset $\alpha$; otherwise shrink $\alpha \leftarrow \alpha \cdot \alpha_0$ (finer steps). This repeats until the budget is exhausted or a correct solution is found.

### Key Designs

1. **Z-score-Driven Prefix Accept/Reject**:

    - **Function**: Determines whether the candidate prefix should be committed as the new base prefix.
    - **Mechanism**: Assuming the reward distribution belongs to a location-scale family (e.g., Gaussian), a lower z-score $z = \frac{r_{\max} - \mu}{\sigma}$ implies a higher tail probability (i.e., a greater chance of sampling a better result). The candidate is accepted when $z_c < z_b$.
    - **Design Motivation**: The z-score jointly accounts for mean and variance, making it more robust than mean-only comparison (DISC-Q). Ablation studies confirm that z-score yields the best performance.

2. **Recursive Shrinkage Mechanism**:

    - **Function**: Reduces step size and retries when a candidate is rejected.
    - **Mechanism**: Upon each rejection, $\alpha \leftarrow \alpha \cdot \alpha_0$ (default $\alpha_0 = 0.15$), making the candidate prefix progressively shorter. This implements a binary-search-style recursive refinement — automatically pausing at difficult decision points for finer sampling.
    - **Design Motivation**: Prevents blind progression through difficult steps (too coarse) and avoids over-segmentation of easy steps (too fine).

3. **Negative Binomial Sampling Strategy**:

    - **Function**: Controls how many samples are drawn from each prefix.
    - **Mechanism**: Sampling continues until the cumulative reward reaches threshold $\sigma$: $M = \min\{m : \sum_{i=1}^m R(y_i) \geq \sigma\}$. This results in fewer samples for easy prefixes (where correct solutions are found quickly) and more for difficult ones.
    - **Design Motivation**: Adaptively balances sample quantity and quality, yielding higher efficiency than fixed sample counts.

### Loss & Training
DISC is a purely inference-time method with **no training involved**. It only requires a scalar reward model (which can be a ground-truth verifier, an ORM, or self-generated test cases) to guide the search, and is plug-and-play with existing LLMs.

## Key Experimental Results

### Main Results
Experiments use gpt-4o-mini as the primary LLM, with default $\alpha_0 = 0.15$, $\sigma = 1.0$, and temperature $\tau = 0.2$.

| Benchmark | Metric | DISC | Best Baseline | Relative Error Reduction |
|-----------|--------|------|---------------|--------------------------|
| APPS (competition-level) | pass@10 error | 0.475 | 0.50 (BoN) | **5.0%** |
| MATH500 | pass@10 error | 0.14 | 0.15 (LineSplit) | **6.7%** |
| LiveCodeBench | pass@10 error | 0.51 | 0.57 (BoN) | **10.5%** |

On the DeepSeek-R1 reasoning model: pass@10 improves 85% relative to the base model, and still improves 33% under the same token budget as single-pass sampling.

### Ablation Study

| Configuration | APPS pass@10 | Notes |
|---------------|-------------|-------|
| DISC-Z (z-score) | **Best** | Normalized reward maximum |
| DISC-Q (mean comparison) | Second best | Ignores variance |
| DISC-R (random accept) | Moderate | No informative guidance |
| DISC-negZ (inverted z-score) | Worst | Selects worst prefix |
| $\alpha_0 = 0.15$ | **Best range** | Conservative step size |
| $\alpha_0 = 0.50$ | Worse | Step size too large |

### Key Findings
- **DISC prefers low temperature**: Unlike BoN, which is optimal at $\tau = 0.6$–$0.8$, DISC performs best at $\tau = 0.2$. Lower temperature reduces sampling variance, enabling more accurate z-score estimation.
- DISC typically commits only 1–5 steps before finding a good solution, whereas LineSplit/TokenSplit require more steps — DISC identifies critical prefixes more efficiently.
- As decomposition depth increases (more committed steps), average reward grows linearly while reward variance decreases linearly.
- Computational overhead is negligible: >90% of wall-clock time remains LLM token generation; DISC's auxiliary computation is comparable to BoN.
- DISC is also effective on open-source models including LLaMA, Mistral, and Qwen: LLaMA pass@10 improves from 0.01 to 0.04 (300% relative improvement).

## Highlights & Insights
- **Pivot words are critical decision points for LLMs**: DISC frequently allocates substantial computation at tokens such as "which" and "therefore" that appear trivial to humans. This reveals that critical decision points in autoregressive models differ fundamentally from human intuition, and automatically discovering these points is more effective than manually designed rules.
- **The recursive shrinkage mechanism as binary search is elegantly designed**: Each rejection triggers a finer step size until an appropriate granularity is found. This simple mechanism theoretically guarantees eventual convergence to an optimal solution.
- **The counter-intuitive finding that low temperature is optimal is highly valuable**: It exposes a fundamental difference in optimal temperature between decomposition-based search methods (which require accurate estimation) and pure sampling methods (which benefit from diversity).

## Limitations & Future Work
- Relies on a scalar reward model: for open-ended tasks without explicit verifiers (e.g., creative writing), an additional reward signal must be constructed.
- In the greedy search variant, accepted prefixes are irrevocable — early incorrect decisions cannot be corrected (the MCTS variant partially mitigates this).
- Currently supports only single-turn generation; multi-turn interactive reasoning is not supported.
- For simple or non-compositionally structured tasks, the gains from decomposition are limited.

## Related Work & Insights
- **vs. Best-of-N (BoN)**: BoN treats the entire generated sequence as a single step, keeping the success probability constant across samples. DISC commits prefixes to monotonically increase the success probability (Lemma 1), theoretically converging faster.
- **vs. s1 (Simple Test-Time Scaling)**: s1 also addresses test-time compute but focuses on controlling the length of the thinking budget, whereas DISC focuses on dynamically adjusting step granularity; the two approaches are complementary.
- **vs. Tree of Thoughts / MCTS**: ToT/MCTS also perform step decomposition and search, but step granularity is typically fixed (by line breaks or manual design). DISC's core contribution lies in automating granularity selection, and it can serve as a plug-and-play module within these frameworks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of dynamic granularity decomposition is novel; the z-score-driven accept/reject mechanism is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks × multiple models × multiple search methods, with extensive ablations, theoretical analysis, and computational overhead analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear figures, well-motivated intuitions, and a seamless interplay between theory and experiments.
- **Value**: ⭐⭐⭐⭐⭐ A plug-and-play general inference-time method requiring no training, directly improving the reasoning performance of existing LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Yggdrasil: Bridging Dynamic Speculation and Static Runtime for Latency-Optimal Tree-Based LLM Decoding](yggdrasil_bridging_dynamic_speculation_and_static_runtime_for_latency-optimal_tr.md)
- [\[ICLR 2026\] Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models](../../ICLR2026/llm_efficiency/inference-cost-aware_dynamic_tree_construction_for_efficient_inference_in_large_.md)
- [\[ICLR 2026\] When Does Divide and Conquer Work for Long Context LLM? A Noise Decomposition Framework](../../ICLR2026/llm_efficiency/when_does_divide_and_conquer_work_for_long_context_llm_a_noise_decomposition_fra.md)
- [\[ICLR 2026\] Dynamic-dLLM: Dynamic Cache-Budget and Adaptive Parallel Decoding for Training-Free Acceleration of Diffusion LLM](../../ICLR2026/llm_efficiency/dynamic-dllm_dynamic_cache-budget_and_adaptive_parallel_decoding_for_training-fr.md)
- [\[NeurIPS 2025\] Long-Context Modeling with Dynamic Hierarchical Sparse Attention for On-Device LLMs](long-context_modeling_with_dynamic_hierarchical_sparse_attention_for_on-device_l.md)

</div>

<!-- RELATED:END -->

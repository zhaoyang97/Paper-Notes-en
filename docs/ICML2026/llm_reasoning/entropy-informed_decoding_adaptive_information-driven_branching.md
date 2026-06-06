---
title: >-
  [Paper Note] Entropy-informed Decoding: Adaptive Information-Driven Branching
description: >-
  [ICML 2026][LLM Reasoning][Entropy Adaptation] EDEN (Entropy-informed DEcodiNg) sets the beam width $B_t$ at each step to be monotonically proportional to the normalized entropy $\bar H_t$—high entropy forks more branche…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "Entropy Adaptation"
  - "Branching Factor"
  - "beam search"
  - "Inference-time Compute Allocation"
  - "regret bound"
date: 2026-05-08
content_hash: 9996b2e00f3616a1
---

# Entropy-informed Decoding: Adaptive Information-Driven Branching

**Conference**: ICML 2026  
**arXiv**: [2605.09745](https://arxiv.org/abs/2605.09745)  
**Code**: None  
**Area**: LLM Decoding / Adaptive Inference / Information Theory  
**Keywords**: Entropy Adaptation, Branching Factor, beam search, Inference-time Compute Allocation, regret bound

## TL;DR
EDEN (Entropy-informed DEcodiNg) sets the beam width $B_t$ at each step to be monotonically proportional to the normalized entropy $\bar H_t$—high entropy forks more branches, low entropy steps approach greedy decoding—thus approximating a wider beam search with fewer total expansions. Theoretically, it is proven that entropy-monotonic branching factors are strictly superior to any fixed beam width in terms of expected cumulative regret, with an explicit regret rate $\mathbb{E}[R_T] \leq G P_\max \sum_t \exp(-c m_t \Delta_\min^2)$.

## Background & Motivation

**Background**: LLM decoding at inference time generally follows two lines: (1) Sampling-based methods (top-$k$, nucleus $p$, min-$p$, top-$H$) trade randomness for diversity but typically follow a single path; (2) Search-based methods (beam search, best-of-$n$, majority voting) explicitly expand multiple candidates and select the best, but their compute cost is unrelated to task difficulty—even simple tasks run all $n$ branches.

**Limitations of Prior Work**: Sampling-based methods **explore too narrowly**, committing to a single path and easily getting stuck on early low-probability tokens at reasoning forks; search-based methods **explore uniformly**, allocating the same compute to both easy and hard tokens, leading to significant waste. Existing entropy-related works (Simonds 2025, Entropix, Top-$H$, HARP, etc.) either use entropy as a binary trigger (branch or not) or for model switching/sampling truncation, but none **map entropy directly and continuously to beam width** with theoretical guarantees.

**Key Challenge**: Effective decoding should "be greedy when possible, branch when necessary," but current methods fix the branching factor as a hyperparameter, unable to adaptively allocate compute based on token difficulty during generation.

**Goal**: (1) Design a pluggable, model-agnostic search strategy that allocates compute per step according to model uncertainty; (2) Provide a strict regret bound using the noisy-maximization framework; (3) Ensure applicability to closed-source models (entropy can be estimated via API access).

**Key Insight**: Treat next-token selection as a noisy maximization problem under sub-Gaussian noise—if the estimation budget $m_t$ matches the "step difficulty," the single-step error probability drops exponentially; Shannon entropy $H_t$ characterizes both the number of candidates (perplexity $\text{PP}_t = e^{H_t}$) and the top-2 gap $\gamma = \log(p_1 / p_2)$.

**Core Idea**: Set the beam width at each step as $B_t = \max(1, \lfloor B_\max \cdot \bar H_t \rfloor)$, where $\bar H_t = H_t / \log |\mathcal{V}|$ is the normalized entropy; high-entropy forks automatically branch more, low-entropy steps degenerate to greedy decoding.

## Method

### Overall Architecture
EDEN is a variant of beam search: it maintains an active candidate set $\mathcal{B}$, estimates the normalized entropy $\bar H_t$ for each candidate's next-token distribution $P(y_{t+1} | x, y_{1:t})$ at each step, and determines the number of top-$B_t$ tokens to expand via $B_t = \max(1, \lfloor B_\max \cdot \bar H_t \rfloor)$. Each new candidate is ranked by cumulative log-prob $s(y_{1:t+1})$ and length-normalized score $\text{Score}_\alpha = s / t^\alpha$, with pruning via upper and lower bounds (retain only if $\bar S \geq S^*$); EOS candidates are scored directly, others use "all-1 future probability" as upper bound and "all-uniform" as lower bound. The sequence with the highest normalized score is returned.

### Key Designs

1. **Monotonic Mapping from Entropy to Branching Factor**:

    - **Function**: Directly converts quantified uncertainty into search budget ("how many candidates needed").
    - **Mechanism**: Uses a simple piecewise linear mapping $f(H, B_\max) = \max(1, \lfloor B_\max \cdot \bar H \rfloor)$, where $0 \leq \bar H \leq 1$ is entropy normalized by $\log |\mathcal{V}|$. As $\bar H \to 0$ (high confidence), the branch count degenerates to 1 (greedy); as $\bar H \to 1$ (max uncertainty), expansion approaches $B_\max$.
    - **Design Motivation**: The authors prove two lemmas: (a) Lemma 3.1: High entropy implies the $\varepsilon$-typical set size is at least $(1-\varepsilon)\text{PP}^{1/\varepsilon}$, so more candidates are worth exploring; (b) Lemma 3.2: The top-2 log gap $\gamma \geq \log(e^{-H}/(1-e^{-H}))$, so low entropy means large, easy-to-select gaps, while high entropy means small, hard-to-select gaps. Together, these show "higher entropy requires more estimation budget."

2. **Upper/Lower Bound Pruning and EOS Handling**:

    - **Function**: Prunes hopeless branches early while maintaining theoretical admissibility.
    - **Mechanism**: For each non-EOS candidate, provide upper and lower bounds $\bar S, \underline S$ for future scores—upper bound assumes all remaining token probabilities are 1 (optimistic), lower bound assumes all are $1/|\mathcal{V}|$ (pessimistic). If $\bar S < S^*$ (current best), prune the branch; otherwise, update $S^*$ with $\underline S$. For EOS candidates, both bounds equal the actual score.
    - **Design Motivation**: Beam search wastes compute mainly on "impossible to win" candidates; admissible upper bounds enable effective pruning without harming the optimal solution, essentially bringing A* ideas into LLM decoding.

3. **Entropy Estimation for Closed-Source Models**:

    - **Function**: Enables EDEN in API-only scenarios.
    - **Mechanism**: For closed-source APIs where only samples (not logits) are available, the authors provide a sub-linear sample complexity upper bound—estimating entropy to $\epsilon$ accuracy requires only $\tilde O(1/\epsilon^2)$ samples, much less than the vocabulary size. With bucketed estimation, entropy can be estimated in tens of samples per step.
    - **Design Motivation**: EDEN is not just a "needs logits" local inference optimization, but can also be deployed on closed APIs like ChatGPT or Claude that only expose sampling interfaces.

### Loss & Training
This work does not train any parameters; it is a pure inference-time algorithm. The theoretical part formulates next-token selection as noisy max: $V_t(i) = \log P(i | x_t) + \text{OPT}_{t+1}(i)$, with estimator variance $\sigma_t^2 = \delta^2 / m_t$. Proposition 3.3, under Lipschitz continuity, yields "required budget $m_t \gtrsim \frac{1}{(\Delta_t^\text{eff})^2} \log(\text{PP}_t / \delta_t)$, monotonically increasing with $H_t$"; Theorem 3.4 further proves that entropy-monotonic branching factors are strictly better in expected regret than any fixed beam width, provided entropy varies during decoding (empirically shown by Wang 2026, Cao 2026).

## Key Experimental Results

### Main Results

| Method | GSM8K ↑ | MATH500 ↑ | HumanEval ↑ | SciBench ↑ | Friedman Rank ↓ | Posterior Probability EDEN > Method |
|--------|---------|-----------|-------------|------------|-----------------|-------------------------------------|
| Greedy | 73.5% | 27.4% | 27.0% | 4.9% | 6.12 | 0.99 |
| Top-$k$ | 70.7% | 23.0% | 27.6% | 4.9% | 7.00 | 1.00 |
| Top-$p$ | 73.5% | 27.4% | 27.0% | 4.8% | 6.62 | 0.99 |
| Top-$H$ | 69.7% | 26.0% | 27.0% | 4.5% | 8.12 | 1.00 |
| Min-$p$ | 72.3% | 28.0% | 25.8% | 4.3% | 8.00 | 1.00 |
| Best-of-5 | 78.2% | 28.2% | 27.0% | 5.2% | 4.62 | 0.96 |
| Beam search (width 3) | — | — | — | — | — | 0.77 |
| **EDEN ($B_\max = 5$)** | **best avg** | — | — | — | **best** | — |

(Friedman test $p = 0.012$, differences are statistically significant; Bayesian hierarchical analysis gives EDEN a 75% posterior probability of being "overall best.")

### Ablation Study

| Configuration | Key Findings | Notes |
|---------------|-------------|-------|
| EDEN (full) | Highest accuracy, fewest expansions | Entropy-adaptive branching |
| Fixed beam width = 3/5/7 | Accuracy similar or slightly lower, expansions scale linearly | Validates "fixed beam wastes compute" |
| Entropy threshold binary trigger | Accuracy between greedy and EDEN | Validates continuous mapping outperforms binary trigger |
| API-only closed-source (sampling-based entropy) | Slight performance drop but still better than greedy | Validates feasibility of sub-linear entropy estimation |

### Key Findings
- **EDEN achieves the best Friedman rank across 4 benchmarks**: Math, code, and science tasks are all covered, demonstrating robust benefits across task types, not just a single domain.
- **Matches or exceeds beam search accuracy with fewer expansions**: The total expansions in Table 1 are fewer than width=3 beam search, confirming the "approximate wider beam" promise.
- **Robust across model families**: Improvements observed on Llama-3.2-3B, Gemma3, IBM Granite, Mistral (see Appendix B), indicating entropy is a reliable signal across distributions.
- **Pareto dominance**: Bayesian analysis shows EDEN's pairwise dominance probability ≥ 96% over other methods, and 77% over beam search.
- **Greater variance, greater gain**: Theorem 3.4's geometric intuition is "the more entropy varies, the more adaptive allocation helps"—empirically, reasoning/code tasks have large entropy fluctuations, so EDEN's gains are most pronounced there.

## Highlights & Insights
- **Dual theoretical and empirical contributions**: Combines noisy max + sub-Gaussian tools for regret rates with Bayesian posterior analysis across 4 tasks × 4 models, yielding a highly complete argument.
- **Branching factor as a first-class compute variable**: The decoding community has long treated beam width as a hyperparameter; this work explicitly upgrades it to a step-wise state variable, pointing to a parameterized approach for inference-time optimizations like speculative decoding and early exit.
- **API friendliness**: Sub-linear entropy estimation enables EDEN on closed-source models, making it a rare "logit-free" search-based decoding scheme with broad deployment potential.

## Limitations & Future Work
- Evaluated only on 3B-scale models and standard benchmarks; whether branching gains persist for 70B+ models with flatter entropy distributions is unknown.
- $f(H, B_\max)$ is piecewise linear with default parameters $a=1, b=0$; the paper does not systematically tune nonlinear mapping forms, so better function families may exist.
- The constants $c$ in the regret bound and Lipschitz $\Lambda$ are hard to measure in real systems; theoretical guarantees serve more as "qualitative guidance" than "quantitative budgeting."
- Experiments use a maximum generation length $T = 400$; cumulative gains and compute curves for long-form/agent chains (thousands of tokens) are not covered.
- Orthogonal integration with RL/process reward model-based adaptive inference methods remains unexplored.

## Related Work & Insights
- **vs Entropix / Simonds**: They use entropy for model switching or CoT token insertion; EDEN maps entropy continuously to beam width, enabling finer-grained compute allocation.
- **vs Top-$H$ decoding (Potraghloo et al. 2026)**: Top-$H$ truncates the sampling distribution by entropy; EDEN uses entropy to decide "how many branches to expand," an orthogonal and potentially complementary approach.
- **vs HARP**: HARP uses entropy to trigger extra transformer computation; EDEN adapts at the higher-level search algorithm, and can be deployed alongside such low-level modifications.
- **vs Best-of-$n$ / Majority voting**: These methods allocate budget uniformly to independent full rollouts; EDEN allocates at the token level as needed, using compute more efficiently at "critical forks" in reasoning chains.

## Rating
- Novelty: ⭐⭐⭐⭐ Continuously mapping entropy to beam width with regret rates is a clear advance for this research line.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 tasks × multiple model families + Bayesian analysis; lacks large model and long-generation scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear lemma-proposition-theorem structure, well-bridged intuition and proof.
- Value: ⭐⭐⭐⭐ Inference-time compute optimization is core to LLM deployment; EDEN offers a cheap and interpretable solution with strong practical potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Chain-of-Thought Works? Tracing Information Flow from Decoding, Projection, and Activation](../../ACL2026/llm_reasoning/how_chain-of-thought_works_tracing_information_flow_from_decoding_projection_and.md)
- [\[ICLR 2026\] AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](../../ICLR2026/llm_reasoning/aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)
- [\[AAAI 2026\] LLMs for Game Theory: Entropy-Guided In-Context Learning and Adaptive CoT Reasoning](../../AAAI2026/llm_reasoning/llms_for_game_theory_entropy-guided_in-context_learning_and_adaptive_cot_reasoni.md)
- [\[ACL 2026\] AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](../../ACL2026/llm_reasoning/aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)
- [\[ICML 2026\] Game of Thought: Robust Information Seeking with Large Language Models Using Game Theory](game_of_thought_robust_information_seeking_with_large_language_models_using_game.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Entropy-informed Decoding: Adaptive Information-Driven Branching
description: >-
  [ICML 2026][Code Intelligence][Entropy-adaptive] EDEN (Entropy-informed DEcodiNg) sets the beam width $B_t$ at each step monotonically proportional to the normalized entropy $\bar H_t$—branching more at high-entropy step…
tags:
  - "ICML 2026"
  - "Code Intelligence"
  - "Entropy-adaptive"
  - "Branching Factor"
  - "Beam Search"
  - "Inference-time Compute Allocation"
  - "Regret Bound"
date: 2026-05-08
content_hash: 9acad51b8f2e725f
---

# Entropy-informed Decoding: Adaptive Information-Driven Branching

**Conference**: ICML 2026  
**arXiv**: [2605.09745](https://arxiv.org/abs/2605.09745)  
**Code**: None  
**Area**: LLM Decoding / Adaptive Inference / Information Theory  
**Keywords**: Entropy-adaptive, Branching Factor, Beam Search, Inference-time Compute Allocation, Regret Bound

## TL;DR
EDEN (Entropy-informed DEcodiNg) sets the beam width $B_t$ at each step monotonically proportional to the normalized entropy $\bar H_t$—branching more at high-entropy steps and behaving nearly greedily at low-entropy steps. This approximates a wider beam search with fewer total expansions. The authors theoretically prove that entropy-monotonic branching factors strictly outperform any fixed beam width in terms of expected cumulative regret, providing an explicit regret rate of $\mathbb{E}[R_T] \leq G P_\max \sum_t \exp(-c m_t \Delta_\min^2)$.

## Background & Motivation

**Background**: LLM inference-time decoding branches into two main lines: (1) Sampling-based methods (top-$k$, nucleus $p$, min-$p$, top-$H$) exchange randomness for diversity but typically follow a single path; (2) Search-based methods (beam search, best-of-$n$, majority voting) explicitly expand multiple candidates and then select the best, but their computational cost is independent of task difficulty—simple problems consume as many branches as difficult ones.

**Limitations of Prior Work**: Sampling-based methods suffer from **narrow exploration**, committing to a single path which can lead to being locked into early low-probability tokens during reasoning forks. Search-based methods have **uniform exploration**, allocating equal compute to both simple and difficult tokens, resulting in significant waste. While existing entropy-related works (Simonds 2025, Entropix, Top-$H$, HARP, etc.) use entropy as a trigger (to branch or not) or for model switching/sampling truncation, none have **directly mapped entropy continuously to beam width** with theoretical guarantees.

**Key Challenge**: Optimal decoding should be "greedy when possible and exploratory when necessary," yet current methods treat the branching factor as a fixed hyperparameter, failing to adaptively allocate compute based on token-level difficulty during generation.

**Goal**: (1) Design a plug-and-play, model-agnostic search strategy where compute scales with the model's own uncertainty; (2) Provide a rigorous regret bound using a noisy-maximization framework; (3) Ensure compatibility with closed-source models (estimating entropy via API access only).

**Key Insight**: Treat next-token selection as a noisy maximization problem under sub-Gaussian noise. If the estimation budget $m_t$ matches the "step difficulty," the single-step error probability decreases exponentially. Shannon entropy $H_t$ characterizes both the number of candidates (perplexity $\text{PP}_t = e^{H_t}$) and the top-2 gap $\gamma = \log(p_1 / p_2)$.

**Core Idea**: Set the beam width at each step as $B_t = \max(1, \lfloor B_\max \cdot \bar H_t \rfloor)$, where $\bar H_t = H_t / \log |\mathcal{V}|$ is the normalized entropy. This automatically forks more branches at high entropy and degrades to greedy decoding at low entropy.

## Method

### Overall Architecture
EDEN is a variant of beam search that maintains an active candidate set $\mathcal{B}$. At each step, it estimates the normalized entropy $\bar H_t$ based on the next-token distribution $P(y_{t+1} | x, y_{1:t})$ for each candidate, then uses $B_t = \max(1, \lfloor B_\max \cdot \bar H_t \rfloor)$ to decide how many top-$B_t$ tokens to expand. Each new candidate is ranked using cumulative log-probability $s(y_{1:t+1})$ and length normalization $\text{Score}_\alpha = s / t^\alpha$, supported by upper and lower bound pruning (retained only if $\bar S \geq S^*$). EOS candidates are scored directly, while others use "future probability of 1" as an upper bound and "future uniform distribution" as a lower bound. The sequence with the highest normalized score is finally returned.

### Key Designs

1.  **Monotonic Mapping from Entropy to Branching Factor**:
    - **Function**: Directly converts uncertainty quantification into a search budget for "how many candidates are needed."
    - **Mechanism**: Employs a simple piecewise linear mapping $f(H, B_\max) = \max(1, \lfloor B_\max \cdot \bar H \rfloor)$, where $0 \leq \bar H \leq 1$ is the entropy normalized by $\log |\mathcal{V}|$. When $\bar H \to 0$ (high model confidence), the branch count degrades to 1 (greedy); when $\bar H \to 1$ (high ambiguity), expansion approaches $B_\max$.
    - **Design Motivation**: The authors use two lemmas to prove that: (a) Lemma 3.1: High entropy implies the $\varepsilon$-typical set size is at least $(1-\varepsilon)\text{PP}^{1/\varepsilon}$, making exploration worthwhile; (b) Lemma 3.2: The top-2 log-gap $\gamma \geq \log(e^{-H}/(1-e^{-H}))$, meaning low entropy makes selection easy while high entropy makes it difficult. Combined, these indicate that "higher entropy requires a larger estimation budget."

2.  **Pruning and EOS Handling Based on Bounds**:
    - **Function**: Prunes hopeless branches early while maintaining theoretical admissibility.
    - **Mechanism**: Provides upper and lower bounds $\bar S, \underline S$ for future scores of non-EOS candidates. The upper bound assumes all remaining token probabilities are 1 (optimistic), while the lower bound assumes a uniform distribution $1/|\mathcal{V}|$ (pessimistic). If $\bar S < S^*$ (current best), the branch is pruned; otherwise, $S^*$ is updated with $\underline S$. For EOS candidates, both bounds equal the actual score.
    - **Design Motivation**: Beam search compute is largely wasted on candidates that "cannot win." Admissible upper bounds allow for effective pruning without discarding the optimal solution, effectively bringing A* search principles into LLM decoding.

3.  **API-Friendly Entropy Estimation for Closed-Source Models**:
    - **Function**: Enables the use of EDEN in API-only scenarios.
    - **Mechanism**: For closed-source APIs providing only samples (no logits), the authors provide a sub-linear sample complexity upper bound. Estimating entropy to $\epsilon$ precision requires only $\tilde O(1/\epsilon^2)$ samples, significantly less than the vocabulary size. Bucket-based estimation can complete entropy estimation within a few dozen samples per step.
    - **Design Motivation**: Ensures EDEN is not just a local inference optimization requiring logits, but a strategy that can empower closed models like ChatGPT or Claude APIs that only expose sampling interfaces.

### Loss & Training
This work does not train any parameters and is a pure inference-time algorithm. The theoretical section formulates next-token selection as noisy maximization: $V_t(i) = \log P(i | x_t) + \text{OPT}_{t+1}(i)$, with the estimator $\hat V_t(i)$ having a variance proxy $\sigma_t^2 = \delta^2 / m_t$. Proposition 3.3 derives that under Lipschitz continuity, the required budget $m_t \gtrsim \frac{1}{(\Delta_t^\text{eff})^2} \log(\text{PP}_t / \delta_t)$ increases monotonically with $H_t$. Theorem 3.4 further proves that entropy-monotonic branching factors are strictly superior in expected regret to any fixed beam width, provided that entropy varies during the decoding process (an empirical observation supported by Wang 2026 and Cao 2026).

## Key Experimental Results

### Main Results

| Method | GSM8K ↑ | MATH500 ↑ | HumanEval ↑ | SciBench ↑ | Friedman Rank ↓ | Posterior Prob (EDEN > Method) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Greedy | 73.5% | 27.4% | 27.0% | 4.9% | 6.12 | 0.99 |
| Top-$k$ | 70.7% | 23.0% | 27.6% | 4.9% | 7.00 | 1.00 |
| Top-$p$ | 73.5% | 27.4% | 27.0% | 4.8% | 6.62 | 0.99 |
| Top-$H$ | 69.7% | 26.0% | 27.0% | 4.5% | 8.12 | 1.00 |
| Min-$p$ | 72.3% | 28.0% | 25.8% | 4.3% | 8.00 | 1.00 |
| Best-of-5 | 78.2% | 28.2% | 27.0% | 5.2% | 4.62 | 0.96 |
| Beam search (width 3) | — | — | — | — | — | 0.77 |
| **EDEN ($B_\max = 5$)** | **best avg** | — | — | — | **best** | — |

(Friedman test $p = 0.012$, indicating statistically significant differences; Bayesian hierarchical analysis gives EDEN a 75% posterior probability of being "overall best.")

### Ablation Study

| Configuration | Key Finding | Description |
| :--- | :--- | :--- |
| EDEN (full) | Highest accuracy, fewest expansions | Entropy-adaptive branching |
| Fixed beam width = 3/5/7 | Accuracy flat or lower, expansions linear | Validates that fixed width wastes compute |
| Binary entropy threshold | Accuracy between greedy and EDEN | Confirms continuous mapping is superior to binary triggers |
| Closed-source API only | Performance slight drop but > greedy | Validates feasibility of sub-linear entropy estimation |

### Key Findings
- **EDEN achieved the best Friedman rank across 4 benchmarks**: Covering Math, Code, and Science, proving consistent benefits across diverse task types.
- **Wider-beam accuracy with fewer total expansions**: The total number of expansions (shown in Table 1 parentheses) is lower than width=3 beam search, fulfilling the promise of approximating wider beams more efficiently.
- **Robust across model families**: Improvements were observed on Llama-3.2-3B, Gemma3, IBM Granite, and Mistral (Appendix B), indicating that entropy is a reliable signal across different distribution families.
- **Pareto advantage**: Bayesian analysis shows a pairwise dominance probability of $\geq 96\%$ against most methods and $77\%$ against standard beam search.
- **Gains scale with variance**: The geometric intuition of Theorem 3.4 is that the more entropy fluctuates, the more advantageous adaptive allocation becomes. Empirically, reasoning and coding tasks show high entropy volatility, where EDEN's improvements are most significant.

## Highlights & Insights
- **Dual Track of Theory + Empirics**: The paper uses standard tools like noisy max and sub-Gaussianity to derive regret rates while providing a complete Bayesian posterior analysis across 4 tasks and 4 models.
- **Branching Factor as a First-order Compute Variable**: Rather than treating beam width as a hyperparameter, it is upgraded to a step-wise state variable, suggesting a parameterized path for inference-time optimizations like speculative decoding or early exiting.
- **API Friendliness**: Sub-linear entropy estimation allows EDEN to remain viable for closed-source models, which is a rare "non-logit dependent" search-based decoding solution in the community.

## Limitations & Future Work
- Evaluated only on 3B-scale models and standard benchmarks; it remains unknown if branching benefits remain significant on 70B+ models where entropy distributions may be flatter.
- $f(H, B_\max)$ uses piecewise linear defaults ($a=1, b=0$); non-linear mapping forms were not systematically optimized and superior function families may exist.
- Regret bound constants $c$ and Lipschitz $\Lambda$ are difficult to measure in real systems; the theoretical guarantees serve more as "qualitative guidelines" than "quantitative budgets."
- Generation length was capped at $T=400$; cumulative gains and compute curves for long-form sequences or agentic chains (thousands of tokens) were not covered.
- Orthogonal integration with RL or process reward model-based adaptive inference methods has not been explored.

## Related Work & Insights
- **vs Entropix / Simonds**: They use entropy for model switching or CoT token insertion; EDEN maps entropy continuously to beam width for finer-grained compute allocation.
- **vs Top-$H$ Decoding (Potraghloo et al. 2026)**: Top-$H$ uses entropy to truncate the sampling distribution; EDEN applies it to the dimension of "how many branches to expand," offering orthogonal and stackable concepts.
- **vs HARP**: HARP triggers additional Transformer computation via entropy; EDEN adapts at the higher search algorithm level, allowing simultaneous deployment with low-level modifications like HARP.
- **vs Best-of-$n$ / Majority Voting**: These methods allocate budget uniformly to independent full rollouts; EDEN allocates at the token level as needed, improving compute efficiency at "critical forks" in the reasoning chain.

## Rating
- Novelty: ⭐⭐⭐⭐ Continuous mapping of entropy to beam width with regret rates is a clear advancement in this research line.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 tasks across multiple model families plus Bayesian analysis; lacks large-scale models and long-generation scenarios.
- Writing Quality: ⭐⭐⭐⭐⭐ The lemma-proposition-theorem chain is clear, bridging intuition and proof effectively.
- Value: ⭐⭐⭐⭐ Inference-time compute optimization is a core LLM deployment issue; EDEN provides a cheap, interpretable, and highly deployable solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Probability-Entropy Calibration: An Elastic Indicator for Adaptive Fine-tuning](probability-entropy_calibration_an_elastic_indicator_for_adaptive_fine-tuning.md)
- [\[ICML 2026\] Locally Coherent Parallel Decoding in Diffusion Language Models](locally_coherent_parallel_decoding_in_diffusion_language_models.md)
- [\[ICML 2026\] HE-SNR: Uncovering Latent Logic via Entropy for Guiding Mid-Training on SWE-bench](he-snr_uncovering_latent_logic_via_entropy_for_guiding_mid-training_on_swe-bench.md)
- [\[ACL 2026\] Learning Adaptive Parallel Execution for Efficient Code Localization](../../ACL2026/code_intelligence/learning_adaptive_parallel_execution_for_efficient_code_localization.md)
- [\[ACL 2026\] Static Program Slicing Using Language Models With Dataflow-Aware Pretraining and Constrained Decoding](../../ACL2026/code_intelligence/static_program_slicing_using_language_models_with_dataflow-aware_pretraining_and.md)

</div>

<!-- RELATED:END -->

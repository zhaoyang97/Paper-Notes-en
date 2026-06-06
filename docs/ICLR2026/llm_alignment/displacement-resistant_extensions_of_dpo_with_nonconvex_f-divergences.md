---
title: >-
  [Paper Note] Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences
description: >-
  [ICLR 2026][LLM Alignment][DPO] This paper establishes that the solvability of f-DPO does not require convexity of $f$ — only $\lim_{t\to 0^+} f'(t) = -\infty$ is needed — and further proves that $\arg\min f(t) \geq 1$ i…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "DPO"
  - "f-divergence"
  - "likelihood displacement"
  - "preference optimization"
  - "SquaredPO"
date: 2026-05-08
content_hash: 4447c59e6bfa83f3
---

# Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences

**Conference**: ICLR 2026
**arXiv**: [2602.06788](https://arxiv.org/abs/2602.06788)  
**Code**: None  
**Area**: LLM Alignment / Preference Optimization
**Keywords**: DPO, f-divergence, likelihood displacement, preference optimization, SquaredPO

## TL;DR
This paper establishes that the solvability of f-DPO does not require convexity of $f$ — only $\lim_{t\to 0^+} f'(t) = -\infty$ is needed — and further proves that $\arg\min f(t) \geq 1$ is a necessary condition for displacement resistance. Based on these findings, the paper proposes SquaredPO ($f(t) = \frac{1}{2}(\log t)^2$, nonconvex), which significantly alleviates the winner probability degradation problem while maintaining competitive performance.

## Background & Motivation

**Background**: DPO and its variants are the dominant methods for LLM alignment, essentially constraining policy deviation from a reference model via KL divergence within the RLHF objective. Wang et al. (2024) generalize the KL divergence to $f$-divergences, but restrict $f$ to be convex.

**Limitations of Prior Work**: DPO suffers from *probability displacement* — during training, the probabilities of both winner and loser responses tend toward zero. This causes severe performance degradation under overtraining and is widely regarded as the most critical practical drawback of DPO.

**Key Challenge**: The KL divergence corresponds to $f_{KL}(t) = t\log t$, whose $\arg\min = e^{-1} < 1$, which theoretically implies that DPO must cause the winner probability to decrease by at least a factor of $e^{-1}$. Within the class of convex $f$-divergences, it is difficult to find an $f$ that simultaneously satisfies solvability and displacement resistance.

**Goal**: (1) What are the precise solvability conditions for f-DPO? (2) Which choices of $f$ can theoretically prevent probability displacement? (3) Can a loss be designed that is simultaneously solvable and displacement-resistant?

**Key Insight**: Relax the convexity requirement and search for $f$ satisfying both conditions within a broader function class.

**Core Idea**: Replace $f(t) = t\log t$ (convex, displacement-prone) with $f(t) = \frac{1}{2}(\log t)^2$ (nonconvex, displacement-resistant) to derive the theoretically superior SquaredPO loss.

## Method

### Overall Architecture
Starting from the general RLHF objective: $\max_{\pi_\theta} \mathbb{E}[r(x,y)] - \beta D_f[\pi_\theta \| \pi_{ref}]$. The f-DPO loss takes the form $-\log\sigma(\beta f'(\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}) - \beta f'(\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))$. The paper proceeds in two steps: (1) characterize solvability conditions on $f$, and (2) characterize displacement-resistance conditions on $f$.

### Key Designs

1. **DPO-Inducing Condition (Solvability)**:

    - Function: Precisely characterizes which $f$ keep the RLHF problem tractable.
    - Core Result (Corollary 1): $f$ is DPO-inducing if and only if $\lim_{t\to 0^+} f'(t) = -\infty$.
    - Significance: Convexity is not a necessary condition. As long as the derivative of $f$ tends to negative infinity near zero (ensuring the optimal policy assigns positive probability to all responses), $f$ is admissible. This substantially expands the usable function class.

2. **Displacement-Resistant Condition**:

    - Function: Characterizes which $f$ can prevent winner probability degradation.
    - Core Result (Lemma 2): If $\arg\min_{t \geq 0} f(t) < 1$, the optimal policy must assign lower probability to in-sample responses than $c \cdot \pi_{ref}$. Therefore, a necessary condition for displacement resistance is $\arg\min f(t) \geq 1$.
    - The DPO Problem: $f_{KL}(t) = t\log t$ achieves its minimum at $t = e^{-1} < 1$, so DPO is theoretically guaranteed to cause displacement.
    - Key Insight (Lemma 1): f-DPO simultaneously solves the full RLHF problem (5) and a degenerate problem (7) whose regularization covers only in-sample responses. This means f-DPO imposes no constraint on out-of-sample behavior, which is the fundamental cause of displacement.

3. **SquaredPO Loss**:

    - Function: A concrete loss satisfying both conditions.
    - $f(t) = \frac{1}{2}(\log t)^2$ is a nonconvex function with $\lim_{t\to 0^+} f'(t) = -\infty$ (DPO-inducing) and $\arg\min f(t) = 1$ (displacement-resistant).
    - Loss Form: Equivalent to "DPO with adaptive $\beta$", where $\beta_\theta(y,x) = \beta / \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}$. When the winner probability decreases, the effective $\beta$ automatically increases, strengthening regularization and suppressing further degradation.
    - Distinction from SimPO/$\beta$-DPO: SimPO's adaptive $\beta$ depends only on length and is fixed during training; $\beta$-DPO introduces additional hyperparameters. SquaredPO's adaptive $\beta$ is derived naturally from theory, requiring no additional hyperparameters.

## Key Experimental Results

### Displacement Mitigation

| Metric | SquaredPO | DPO |
|--------|-----------|-----|
| Median chosen log-ratio at Epoch 1 | Higher (less displacement) | Lower (more displacement) |
| Fraction of winners with monotonically decreasing probability (4 epochs) | **4.21%** | **99.63%** |

Key finding: In DPO, 99.63% of winner probabilities that decrease in the first epoch continue to decrease monotonically in every subsequent epoch. SquaredPO reduces this fraction to 4.21%.

### Overtraining Robustness (TL;DR Win Rate vs. Base Model)

| Epochs | SquaredPO | χPO | DPO |
|--------|-----------|-----|-----|
| 1 | 50.8% | 51.2% | 51.8% |
| 2 | 50.6% | 48.9% | 45.0% |
| 4 | **51.0%** | 48.3% | **34.7%** |

DPO's win rate drops to 34.7% after 4 epochs (severe overtraining), while SquaredPO maintains 51.0%.

### Standard Benchmarks (1 Epoch)

| Method | AlpacaEval LC↑ | AlpacaEval WR↑ | MT-Bench↑ |
|--------|---------------|----------------|-----------|
| SquaredPO | 29.2 | 24.5 | 7.924 |
| DPO | 29.6 | 24.8 | 7.925 |

Performance is essentially on par, with SquaredPO using DPO's default hyperparameters without task-specific tuning.

## Highlights & Insights
- **Theory-Derived Adaptive $\beta$**: The core intuition behind SquaredPO is elegantly simple — automatically increase regularization when winner probability decreases. Crucially, this is not a heuristic design but a natural consequence derived from $f$-divergence theory.
- **The Depth of Lemma 1**: The fact that f-DPO simultaneously solves both the full problem and a degenerate problem reveals a structural deficiency shared by all f-DPO variants: the lack of constraint over out-of-sample behavior. Displacement is not a bug but a mathematical inevitability.
- **99.63% Monotonic Decrease**: This paper is the first to report the monotonic decline of winner probabilities in DPO, a finding that is more precise and striking than previously reported results on average probability decrease.

## Limitations & Future Work
- Experiments are conducted on only a single dataset (TL;DR) and a single model (Llama-3-8B) using LoRA fine-tuning.
- The displacement-resistant condition is proven to be necessary but not sufficient — satisfying the condition does not guarantee complete elimination of displacement.
- SquaredPO slightly underperforms DPO at epoch 1, and the hyperparameter $\beta$ is not tuned specifically for SquaredPO.
- Only a single concrete $f$ ($(\log t)^2/2$) is explored; many other functions satisfying both conditions remain to be investigated.

## Related Work & Insights
- **vs. DPO (Rafailov et al., 2023)**: DPO is a special case with $f_{KL}(t) = t\log t$, where $\arg\min = e^{-1}$, making displacement theoretically unavoidable. SquaredPO addresses this at the root by using $f(t) = \frac{1}{2}(\log t)^2$, which guarantees $\arg\min = 1$.
- **vs. χPO (Huang et al., 2025)**: χPO also operates within the f-DPO framework ($\chi^2$ divergence) and exhibits some overtraining robustness, but less than SquaredPO. The theoretical framework in this paper is more general (covering all $f$), whereas χPO analyzes only a single instance.
- **vs. SimPO/$\beta$-DPO**: These methods employ heuristic adaptive $\beta$ schedules; SquaredPO's adaptive $\beta$ is theory-derived and requires no additional hyperparameters.
- **vs. RCPO (Beyond Pairwise)**: RCPO focuses on the format of preference data (pairwise → ranked), while SquaredPO addresses the mathematical properties of regularization. The two approaches are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Complete characterization of DPO-inducing conditions + first formulation of displacement-resistant conditions; deep theoretical contribution
- Experimental Thoroughness: ⭐⭐⭐ Limited to a single dataset/model, though the displacement analysis is thorough
- Writing Quality: ⭐⭐⭐⭐⭐ Theoretical structure is clear; the logical chain from definitions to lemmas to theorems is rigorous, with intuitive Venn diagram illustrations
- Value: ⭐⭐⭐⭐ Provides principled design criteria (two conditions) for DPO-class methods, offering guidance for future preference optimization research

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Uni-DPO: A Unified Paradigm for Dynamic Preference Optimization of LLMs](uni-dpo_a_unified_paradigm_for_dynamic_preference_optimization_of_llms.md)
- [\[ICLR 2026\] Why DPO is a Misspecified Estimator and How to Fix It](why_dpo_is_misspecified_estimator.md)
- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](token-importance_guided_direct_preference_optimization.md)
- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](../../NeurIPS2025/llm_alignment/g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[ICML 2026\] TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](../../ICML2026/llm_alignment/tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)

</div>

<!-- RELATED:END -->

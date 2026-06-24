---
title: >-
  [Paper Note] Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences
description: >-
  [ICLR 2026][LLM Alignment][DPO] It is discovered that the solvability of f-DPO does not require $f$ to be convex (only $\lim_{t\to 0^+} f'(t) = -\infty$). Furthermore, it is proven that $\arg\min f(t) \geq 1$ is a necessary condition to resist probability displacement. Based on this, SquaredPO ($f(t) = \frac{1}{2}(\log t)^2$, non-convex) is proposed, which significantly alleviates the decline in winner probability while maintaining performance.
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "DPO"
  - "f-divergence"
  - "likelihood displacement"
  - "preference optimization"
  - "SquaredPO"
date: 2026-05-08
content_hash: a0561e03d163ecd5
---

# Displacement-Resistant Extensions of DPO with Nonconvex $f$-Divergences

**Conference**: ICLR 2026  
**arXiv**: [2602.06788](https://arxiv.org/abs/2602.06788)  
**Code**: None  
**Area**: LLM Alignment / Preference Optimization  
**Keywords**: DPO, f-divergence, likelihood displacement, preference optimization, SquaredPO

## TL;DR
It is discovered that the solvability of f-DPO does not require $f$ to be convex (only $\lim_{t\to 0^+} f'(t) = -\infty$). Furthermore, it is proven that $\arg\min f(t) \geq 1$ is a necessary condition to resist probability displacement. Based on this, SquaredPO ($f(t) = \frac{1}{2}(\log t)^2$, non-convex) is proposed, which significantly alleviates the decline in winner probability while maintaining performance.

## Background & Motivation
**Background**: DPO and its variants are the mainstream methods for LLM alignment, essentially using KL divergence to constrain the policy from deviating from a reference model in the RLHF objective. Wang et al. (2024) generalized KL to f-divergence, but restricted it to convex $f$.

**Limitations of Prior Work**: DPO exhibits a "probability displacement" phenomenon—where the probabilities of both the winner and loser approach zero during training. This leads to a sharp performance drop during overtraining, which is a widely criticized practical issue of DPO.

**Key Challenge**: The $f$-function for KL divergence is $f_{KL}(t) = t\log t$, which has an $\arg\min = e^{-1} < 1$. This theoretically dictates that DPO inevitably leads to at least an $e^{-1}$ factor decrease in winner probability. It is difficult to find an $f$ within the class of convex f-divergences that satisfies both solvability and displacement resistance.

**Goal**: (1) What are the actual solvability conditions for f-DPO? (2) Which $f$ functions can theoretically prevent probability displacement? (3) Can a loss be designed that is both solvable and displacement-resistant?

**Key Insight**: Abandon the convexity requirement and search for $f$ within a broader class of functions that satisfy both conditions.

**Core Idea**: Replace $f(t) = t\log t$ (convex, prone to displacement) with $f(t) = \frac{1}{2}(\log t)^2$ (non-convex, displacement-resistant) to obtain the theoretically superior SquaredPO loss.

## Method

### Overall Architecture
The objective of this paper is not to reinvent an alignment algorithm but to answer a more fundamental question: what properties of $f$ determine the behavior of the f-DPO loss family. The starting point is the generalized RLHF objective $\max_{\pi_\theta} \mathbb{E}[r(x,y)] - \beta D_f[\pi_\theta \| \pi_{ref}]$. By solving this in closed form under f-divergence constraints and substituting it into the Bradley-Terry preference model, the f-DPO loss $-\log\sigma(\beta f'(\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}) - \beta f'(\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))$ is derived. Notably, the derivative $f'$—and not $f$ itself—appears in the loss. All subsequent conclusions depend on the properties of $f'$ and the location of the minimum of $f$. Following this logic, the paper proceeds in three steps: defining solvability conditions, identifying displacement-resistant conditions, and providing a specific $f$ that satisfies both.

### Key Designs

**1. DPO-Inducing Condition: Solvability Does Not Require Convexity**

To transform the RLHF objective into a closed-form preference loss like DPO, the optimal policy under the f-divergence constraint must exist and be well-defined. Previous f-DPO work (Wang et al., 2024) assumed $f$ must be convex. Corollary 1 replaces this threshold precisely: $f$ is DPO-inducing if and only if $\lim_{t\to 0^+} f'(t) = -\infty$. Intuitively, $f'$ approaching negative infinity near 0 is equivalent to requiring the optimal policy to assign a strictly positive probability to any response (otherwise, the gradient would push some responses toward zero probability, breaking solvability). Convexity is only one sufficient condition for this, not a necessity—this step expands the available $f$ functions from the convex class to all functions satisfying this limit condition, opening the door for non-convex $f$.

**2. Displacement-Resistant Condition: $\arg\min f \geq 1$ is a Necessary Threshold**

Probability displacement refers to the winner probability decreasing instead of increasing during training. Lemma 2 characterizes its cause: if $\arg\min_{t \geq 0} f(t) < 1$, the optimal policy's probability for in-sample responses will inevitably be suppressed below $c \cdot \pi_{ref}$, meaning displacement is certain. Conversely, a necessary condition for displacement resistance is $\arg\min f(t) \geq 1$. Applying this to DPO makes it clear: it corresponds to $f_{KL}(t) = t\log t$, which has a minimum at $t = e^{-1} < 1$, so DPO is theoretically destined to suppress winner probability.

A deeper reason comes from Lemma 1—f-DPO nominally solves the complete RLHF problem (5), but it also solves a degraded problem (7). The regularization in the latter only covers in-sample responses appearing in the training data, with no constraints on out-of-sample behavior. Consequently, the model can shift probability mass to unseen responses at no cost, causing winner probability to drop. Displacement is therefore not an implementation bug but a mathematical necessity of the loss structure. The reason $\arg\min f \geq 1$ prevents this is that it requires $f$ to pay a higher cost when $t < 1$ (i.e., when probability is pushed below the reference model), thus blocking this probability leakage channel.

**3. SquaredPO: Hitting Both Conditions with $f(t)=\frac{1}{2}(\log t)^2$**

Taking the first two conditions as design constraints, the problem becomes finding an $f$ that satisfies both $\lim_{t\to 0^+} f'(t) = -\infty$ and $\arg\min f(t) \geq 1$. The paper proposes $f(t) = \frac{1}{2}(\log t)^2$: it is a non-convex function, yet $\lim_{t\to 0^+} f'(t) = -\infty$ (solvable) and $\arg\min f(t) = 1$ (displacement-resistant). When substituted into the f-DPO loss, it is equivalent to a "DPO with adaptive $\beta$," where the effective coefficient is

$$\beta_\theta(y,x) = \beta \Big/ \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)}.$$

The meaning is straightforward: when the winner probability $\pi_\theta/\pi_{ref}$ drops, the denominator decreases, causing $\beta_\theta$ to automatically increase. This strengthens regularization and pushes back against further decreases—this is the specific manifestation of the displacement-resistant condition in training dynamics. Compared to methods like SimPO or $\beta$-DPO that also modify $\beta$, the difference is that their adaptations are heuristic (SimPO's $\beta$ only varies with length and is fixed during training; $\beta$-DPO introduces extra hyperparameters), while SquaredPO's adaptive $\beta$ is derived naturally from f-divergence theory without new hyperparameters.

## Key Experimental Results

### Probability Displacement Mitigation

| Metric | SquaredPO | DPO |
|--------|-----------|-----|
| Median chosen log-ratio (Epoch 1) | Higher (Lower displacement) | Lower (Severe displacement) |
| Proportion of winners with monotonic decrease (4 epochs) | **4.21%** | **99.63%** |

Key finding: In DPO, 99.63% of winners whose probability decreases in the 1st epoch continue to decrease in every subsequent epoch (monotonic decrease). SquaredPO reduces this proportion to 4.21%.

### Overtraining Robustness (TL;DR Win Rate vs Base Model)

| Epochs | SquaredPO | $\chi$PO | DPO |
|--------|-----------|-----|-----|
| 1 | 50.8% | 51.2% | 51.8% |
| 2 | 50.6% | 48.9% | 45.0% |
| 4 | **51.0%** | 48.3% | **34.7%** |

DPO's win rate drops to 34.7% after 4 epochs (severe overtraining), while SquaredPO maintains 51.0%.

### Main Results (1 epoch)

| Method | AlpacaEval LC↑ | AlpacaEval WR↑ | MT-Bench↑ |
|--------|---------------|----------------|-----------|
| SquaredPO | 29.2 | 24.5 | 7.924 |
| DPO | 29.6 | 24.8 | 7.925 |

Performance is essentially on par, but SquaredPO was not hyperparameter-tuned (using DPO defaults).

## Highlights & Insights
- **Theoretically Derived "Adaptive $\beta$"**: The core intuition of SquaredPO is extremely simple—automatically increase regularization when the winner probability drops. However, this is not a heuristic design but a natural derivation from f-divergence theory.
- **Profound Revelation of Lemma 1**: The fact that f-DPO solves both the complete and degraded problems implies that all f-DPO variants have a structural flaw regarding lack of constraints on out-of-sample behavior. Displacement is a mathematical necessity rather than a bug.
- **99.63% Monotonic Decrease**: The first report of the monotonic decrease of winner probability in DPO, which is more precise and shocking than previous reports of "average probability decrease."

## Limitations & Future Work
- Validated only on a single dataset (TL;DR) and a single model (Llama-3-8B) using LoRA.
- The displacement-resistant condition is proven to be a necessary condition, but not a sufficient one—satisfying it does not guarantee complete elimination of displacement.
- SquaredPO is slightly inferior to DPO in the 1st epoch; hyperparameters ($\beta$) were not optimized for SquaredPO.
- Only one specific $f$ ($(\log t)^2/2$) was explored; many other functions satisfying both conditions warrant investigation.

## Related Work & Insights
- **vs DPO (Rafailov et al., 2023)**: DPO is a special case with $f_{KL}(t) = t\log t$, where $\arg\min = e^{-1}$, necessitating displacement theoretically. SquaredPO uses $f(t) = \frac{1}{2}(\log t)^2$ to ensure $\arg\min = 1$, addressing the root cause.
- **vs $\chi$PO (Huang et al., 2025)**: $\chi$PO also uses the f-DPO framework ($\chi^2$ divergence) and shows overtraining robustness but is inferior to SquaredPO. This paper's theory is more general (covering all $f$), whereas $\chi$PO only analyzes a specific case.
- **vs SimPO/$\beta$-DPO**: These methods utilize heuristic adaptive $\beta$, whereas SquaredPO's adaptive $\beta$ is theoretically derived without extra hyperparameters.
- **vs RCPO (Beyond Pairwise)**: RCPO focuses on preference data formats (pairwise → ranked), while SquaredPO focuses on the mathematical properties of regularization. The two are orthogonal and can be combined.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Fully characterizes the DPO-inducing condition and proposes the displacement-resistant condition for the first time; profound theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐ Only one dataset/model, but the displacement analysis is very detailed.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical structure; the logical chain from definitions to lemmas to theorems is perfect, and Venn diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Provides design principles (two conditions) for DPO-like methods, offering significant guidance for future preference optimization research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Alignment-Weighted DPO: A Principled Reasoning Approach to Improve Safety Alignment](alignment-weighted_dpo_a_principled_reasoning_approach_to_improve_safety_alignme.md)
- [\[ICLR 2026\] Why DPO is a Misspecified Estimator and How to Fix It](why_dpo_is_misspecified_estimator.md)
- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](token-importance_guided_direct_preference_optimization.md)
- [\[ICML 2025\] DPO Meets PPO: Reinforced Token Optimization for RLHF](../../ICML2025/llm_alignment/dpo_meets_ppo_reinforced_token_optimization_for_rlhf.md)
- [\[ACL 2026\] S2H-DPO: Hardness-Aware Preference Optimization for Vision-Language Models](../../ACL2026/llm_alignment/s2h-dpo_hardness-aware_preference_optimization_for_vision-language_models.md)

</div>

<!-- RELATED:END -->

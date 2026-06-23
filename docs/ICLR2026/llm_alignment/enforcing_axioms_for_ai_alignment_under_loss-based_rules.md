---
title: >-
  [Paper Note] Enforcing Axioms for AI Alignment under Loss-Based Rules
description: >-
  [ICLR 2026][Alignment & RLHF][RLHF] Under a linear social choice framework, loss-based reward models (including polynomial rewards) fail to guarantee Pareto Optimality (PO), but PO can be recovered in the limit when training data uniformly covers the embedding space—offering a provable data design for constitutional-style alignment.
tags:
  - ICLR 2026
  - Alignment & RLHF
  - RLHF
  - Constitutional AI
date: 2026-05-08
content_hash: 1cb43562f9ee49f9
---
# Enforcing Axioms for AI Alignment under Loss-Based Rules

**Conference**: ICLR 2026  
**Code**: None  
**Area**: LLM Alignment / Social Choice Theory  
**Keywords**: RLHF, Pareto Optimality, Social Choice, Constitutional AI, Reward Model Axioms, Data Design

## TL;DR

Under a linear social choice framework, loss-based reward models (including polynomial rewards) fail to guarantee Pareto Optimality (PO), but PO can be recovered in the limit when training data uniformly covers the embedding space—offering a provable data design for constitutional-style alignment.

## Background & Motivation

**Background**: RLHF and its variants (Constitutional AI, NLHF) are the mainstream paradigms for aligning Large Language Models. The core step involves minimizing loss on binary preference data to train a reward model, which then guides policy optimization. Constitutional AI further introduces a small set of "principles" (e.g., HHH: helpfulness/honesty/harmlessness) as guides for comparative judgments, where principles act as "voters."

**Limitations of Prior Work**: Ge et al. (2024) proved a surprising negative result in linear social choice models—the optimal linear reward model may violate Pareto Optimality (PO): even if all principles prefer response A over B, the trained reward function might still assign a higher score to B. This directly contradicts alignment goals, but whether it can be fixed via stronger reward function classes or more rational data distributions remains unclear.

**Key Challenge**: Existing analyses are fixed on a worst-case perspective over finite candidate sets, while actual training relies on the model's generalization to new data and the choice of data distribution—neither of which is discussed in classical social choice frameworks.

**Goal**: This paper explores the robustness and restoration paths of PO violations from three directions: (1) expanding the reward function class (polynomial rewards); (2) generalizing axioms to the entire embedding space (generalization perspective); and (3) restoring axiomatic guarantees through data design.

**Key Insight**: The fundamental cause of PO violation is the implicit "norm constraint" in loss optimization—comparisons in certain directions contribute more to the loss, and optimization tends to satisfy these directions first. When data uniformly covers the embedding sphere $S^{d-1}$, this bias disappears, and PO is guaranteed in the limit.

## Method

### Overall Architecture

The paper builds its theoretical analysis on the linear social choice model of Ge et al. (2024). $n$ "principles" act as voters, each holding a linear utility direction $v_i \in \mathbb{R}^d$, generating preference data via pairwise binary comparisons over embedding vectors of candidate responses. The loss-based voting rule outputs a reward function $r_\theta(x) = \langle \theta, x \rangle$ that minimizes the total loss $L(\theta)$. The research progresses along three lines: proving polynomial rewards still violate PO (negative result), showing uniform data restores PO (positive result), and analyzing why PMC is harder to guarantee.

### Key Designs

**1. Intuition for PO Violation—Implicit Norm Constraints**

Ge et al.'s counterexample is complex (6 candidates); this paper provides a minimal intuition: a single voter $v = (\varepsilon, 1)$ and three candidates $a=(1,0), b=(0,0), c=(-\delta,\delta)$ ($\delta \ll 1$). Under the unit norm constraint $\|\theta\|=1$, the Bradley-Terry loss $\ell_{BT}(x) = \log(1 + e^x)$ is dominated by comparisons of $(a,b)$ and $(a,c)$ because these pairs have "longer" directional vectors. This forces the optimal $\theta$ toward the $x$-axis, leading to $\langle\theta, b\rangle > \langle\theta, c\rangle$, while the voter prefers $b \succ c$. Core Mechanism: "Different directions contribute differently to the loss; directions with length or quantity advantages hijack the limited norm 'budget'."

**2. Polynomial Rewards Still Violate PO—Theorem 4.1**

A natural conjecture is that richer reward classes (bounded-degree polynomials) could bypass linear limitations. This paper proves otherwise. Construction: $m+1 = d(d+1)+2$ candidate points, two (weighted) voters $v_1=(1,0)$ and $v_2=(0,1)$, agreed only on $c_0 \succ c_1$ (the PO requirement). By distributing candidates on $d$ lines $L_j$ with slope $-2$, the unique optimal polynomial for a degenerate instance ($c_0=c_1$) is exactly the PO-violating $p^*(x,y) = -x-y$. Using **Berge’s Maximum Theorem** (upper hemicontinuity), the paper proves that for a sufficiently small $\delta > 0$, the optimal polynomial for the non-degenerate instance still satisfies $p(c_1) > p(c_0)$, contradicting the PO requirement of all voters.

**3. Restoring PO with Uniform Data—Theorem 5.1**

The analysis is extended from finite candidate sets to a continuous embedding space. An "idealized" uniform data setting is defined where loss is the integral over the hypersphere $S^{d-1}$:
$$L(\theta) := \sum_{i=1}^n \int_{x \in S^{d-1},\, \langle v_i, x\rangle \geq 0} \ell(-\langle\theta, x\rangle)\, dx$$
meaning each voter provides comparisons for all unit vectors in their preferred half-space. In this setting, for any direction $x$ required by PO ($\langle v_i, x\rangle > 0, \forall i$), if $\theta$ satisfies $\langle\theta, x\rangle \leq 0$, it is shown that $\theta$ cannot be optimal. Conclusion: **With at least two distinct voters, any optimal $\theta^*$ satisfies PO.** Data uniformity eliminates directional bias, restoring social choice axioms.

**4. PMC Still Fails Under Uniform Data—Theorem 5.2**

Pairwise Majority Consistency (PMC) requires that if a strict majority of voters prefer direction $x$, the reward must satisfy it. Under uniform data, when two voters $v_1, v_2$ have proportions $p > 1/2$ and $1-p < 1/2$ respectively, PMC requires outputting $v_1$; however, loss minimization tends to interpolate between $v_1$ and $v_2$. This continuity bias is fundamentally incompatible with the discrete jump required by PMC.

## Key Experimental Results

This is a purely theoretical work; the core contributions are theorems and proofs.

### Main Results

| Theorem | Conclusion | Condition |
|------|------|------|
| Theorem 4.1 | Polynomial rewards still violate PO and PMC | Strictly convex loss, $\ell'(0)>0$, finite candidate set |
| Theorem 5.1 | PO is restored under uniform data | Convex non-decreasing loss, $\ell'(0)>0$, $\geq 2$ voters |
| Theorem 5.2 | PMC still fails under uniform data | Strictly convex loss, $\ell'(0)>0$, $\geq 2$ voters |

### Key Findings

- PO violation does not disappear as the reward function class grows—the root cause lies in the data distribution, not the expressive power of the function class.
- Uniform data is a sufficient condition for restoring PO; in practice, the degree of uniformity can be evaluated by analyzing the directional distribution of embedding differences via PCA.
- PMC currently cannot be guaranteed by loss rules in either finite or infinite data settings; its necessity for practical alignment deserves re-examination.
- The optimal solution for Bradley-Terry loss is equivalent to Borda ranking, which is consistent with this framework.

## Highlights & Insights

- The root cause of RLHF reward models violating alignment axioms is precisely pinpointed as "unbalanced directional distribution of data" rather than "weakness of the linear class."
- The use of Berge’s Maximum Theorem elegantly extends the uniqueness conclusion of degenerate instances to perturbed instances, serving as the core technique for the negative theorem.
- A practical recipe with provable guarantees is provided: restore PO through dataset design (balanced coverage of comparison directions) rather than changing the training pipeline.
- Social choice axioms are generalized from "fixed candidate sets" to the "entire embedding space," establishing a new foundation for the theoretical analysis of reward generalization.

## Limitations & Future Work

- The Linear Representation Hypothesis is a simplifying assumption; whether actual LLM feature spaces satisfy it remains to be verified.
- The uniform data setting is an idealized continuous limit; explicit sample complexity bounds for convergence under finite samples are not yet established.
- Only frozen embeddings are considered; end-to-end fine-tuning that changes embedding distribution is not covered.
- Whether there are relaxed versions of PMC in this framework is worth further study; based on reweighted data-query strategies, whether PO can be restored more effectively remains an open question.

## Related Work & Insights

- **vs Ge et al. (2024)**: The latter found linear rewards violate PO and used combinatorial rules to fix it; this paper proves extensions to polynomial rewards do not help, but restoration is achieved via data design.
- **vs Christiano et al. (2017) / RLHF**: BT optimal solutions correspond to Borda ranking; this paper reveals the axiomatic flaws of Borda rules in linear social choice.
- **vs Constitutional AI (Bai et al., 2022b)**: Provides theoretical grounding—principles act as voters, and uniform data coverage is a provable data scheme for constitutional-style alignment.
- **vs Nash Learning from Human Feedback (Munos et al., 2024)**: NLHF is equivalent to von Neumann winner/Fishburn lotteries; this framework can further explore if such game-theoretic alignment satisfy PO.

## Rating

- Novelty: ⭐⭐⭐⭐ The angle is novel—attributing PO violation to data distribution rather than function class and providing an actionable data design.
- Experimental Thoroughness: ⭐⭐⭐ Purely theoretical work; theorems are rigorous, but empirical validation on real LLM embeddings is lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear structure; the narrative progression from negative to positive results is smooth.
- Value: ⭐⭐⭐⭐ Provides a theoretical basis for the data design of Constitutional AI, with high practical reference value for the industry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Humanline: Online Alignment as Perceptual Loss](humanline_online_alignment_as_perceptual_loss.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)
- [\[ICLR 2026\] Align Once, Benefit Multilingually: Enforcing Multilingual Consistency for LLM Safety Alignment](align_once_benefit_multilingually_enforcing_multilingual_consistency_for_llm_saf.md)
- [\[ICML 2025\] Challenges and Future Directions of Data-Centric AI Alignment](../../ICML2025/llm_alignment/challenges_and_future_directions_of_data-centric_ai_alignment.md)
- [\[ICLR 2026\] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)

</div>

<!-- RELATED:END -->

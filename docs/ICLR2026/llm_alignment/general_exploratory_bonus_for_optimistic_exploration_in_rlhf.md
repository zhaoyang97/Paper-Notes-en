---
title: >-
  [Paper Note] General Exploratory Bonus for Optimistic Exploration in RLHF
description: >-
  [ICLR 2026][LLM Alignment][exploratory bonus] It is theoretically proven that existing RLHF exploratory bonuses under KL and α-divergence regularization actually guide the policy toward high-probability regions of the reference model (contradicting the principle of optimism). This paper proposes the General Exploratory Bonus (GEB) framework, which counteracts the conservative bias of divergence regularization through reference-model-dependent reward adjustment and is provably…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "exploratory bonus"
  - "optimistic exploration"
  - "RLHF"
  - "α-divergence"
  - "sample efficiency"
date: 2026-05-08
content_hash: 5346277e29550237
---

# General Exploratory Bonus for Optimistic Exploration in RLHF

**Conference**: ICLR 2026  
**arXiv**: [2510.03269](https://arxiv.org/abs/2510.03269)  
**Code**: Available (see paper link)  
**Area**: RLHF Alignment  
**Keywords**: exploratory bonus, optimistic exploration, RLHF, α-divergence, sample efficiency

## TL;DR
It is theoretically proven that existing RLHF exploratory bonuses under KL and α-divergence regularization actually guide the policy toward high-probability regions of the reference model (contradicting the principle of optimism). This paper proposes the General Exploratory Bonus (GEB) framework, which counteracts the conservative bias of divergence regularization through reference-model-dependent reward adjustment and is provably optimistic.

## Background & Motivation

**Background**: Iterative online RLHF is the core paradigm for LLM alignment (used by Claude and LLaMA series). Standard methods rely on the policy's own stochasticity for "passive exploration." However, when the optimal behavior lies in a low-probability region, passive exploration may never discover it, causing the policy to remain stuck in local optima.

**Limitations of Prior Work**: To improve sample efficiency, recent works (Zhang et al. 2024, Xie et al. 2024, Cen et al. 2025) introduce an exploratory bonus $\mathcal{L}_{bonus} = \max_\pi \mathcal{J}_{\beta,KL}(\pi, r)$ to incentivize exploration. However, these methods have fundamental theoretical flaws.

**Key Challenge**: Divergence regularization (KL/α-divergence) aims to prevent the policy from deviating too far from the reference model, but this directly conflicts with the goal of "exploring unknown regions." The divergence term in existing bonus formulations inadvertently guides exploration back toward high-probability regions of $\pi_{ref}$ — reinforcing conservative behavior rather than promoting discovery.

**Goal**: (a) Rigorously prove why existing exploratory bonuses fail; (b) design a new framework that provably satisfies the principle of optimism.

**Key Insight**: By using the reward reparameterization trick $r(x,y) = \beta \log \frac{\pi(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$, the bonus can be converted into an expression involving the policy. Analyzing the gradient relationship between this expression, $\pi$, and $\pi_{ref}$ allows one to determine if the optimism condition is met.

**Core Idea**: Introduce a reference-model-dependent adjustment term in the reward to offset the conservative bias introduced by divergence regularization, ensuring the exploratory bonus truly incentivizes exploration in low-probability (unexplored) regions.

## Method

### Overall Architecture
This paper addresses a counter-intuitive question: why do exploratory bonuses specifically designed to "encourage exploration" in recent RLHF literature instead push the policy back into the safety zone of the reference model under divergence regularization. GEB does not modify the outer loop of iterative RLHF (the "reward modeling → policy optimization → sampling with new policy → retraining reward" cycle); it only modifies the bonus term added during reward modeling. The methodology follows a theoretical chain of derivation: first, establishing a **verifiable formal condition** for what constitutes true exploration (determined by the sign of the second-order cross-derivative); next, using this metric to **prove the failure of existing bonuses** (showing they are either redundant under KL or inversely incentivize conservative regions under general divergence); and finally **back-solving a family of new bonuses** from this condition, showing that previous heuristic approaches are merely special cases. In terms of the **Mechanism**, whereas old bonuses directly use the policy ratio $\pi/\pi_{ref}$ (focusing incentives on high $\pi_{ref}$ areas), GEB employs an atomic function $u$ negatively correlated with $\pi$ to construct the bonus. This ensures that areas with lower policy probability receive higher bonuses, pulling the policy toward low $\pi_{ref}$ (unexplored) regions. While these bonuses formally depend on the relationship between $\pi$ and $\pi_{ref}$, they can be instantiated using only the current policy $\pi$ via reward reparameterization, thus adding no sampling overhead.

### Key Designs

**1. Formal Definition of the Optimism Condition (Definition 3.1): Determining optimism via gradient relationships**

To correct the problem where an "exploration bonus inhibits exploration," a calculable standard is needed to judge if a bonus is truly optimistic. Since direct uncertainty quantification (e.g., Bayesian or Ensemble methods) is computationally infeasible at LLM scale, this paper instead focuses on the gradient relationships between policy distributions. The intuition for optimism is that "regions sampled more frequently should receive less exploration incentive." The paper formalizes this as:

$$\frac{\partial}{\partial \pi_s(y|x)} \left(\frac{\partial \mathcal{L}_{bonus}}{\partial \pi(y|x)}\right) < 0$$

That is, the marginal contribution of the bonus to the policy $\pi$ should decrease as the sampling policy $\pi_s$ increases. This condition requires no additional sampling or explicit uncertainty estimation, as it only looks at the sign of the second-order cross-derivative of the bonus with respect to the two distributions, making it naturally scalable for LLMs and providing a unified metric for all subsequent proofs.

**2. Theorem Proof of the Failure of Existing Methods (Lemma 3.1 / 3.2, Theorem 3.3): Formalizing the "anti-optimism" flaw**

Using the optimism condition as a benchmark, the paper quantifies the issues with existing bonuses. Lemma 3.1 addresses the common KL setting: under KL regularization, the sets of policies obtained with or without a bonus are identical, meaning the bonus is redundant. Lemma 3.2 examines more general α-divergences, where the cross-gradient is $\frac{\partial^2 \mathcal{L}_{bonus}}{\partial \pi_{ref} \partial \pi} \geq 0$, which is exactly opposite to the optimism condition. This implies the bonus provides more incentive to high $\pi_{ref}$ regions, which constitutes anti-optimism and encourages conservative behavior. Theorem 3.3 extends this to the general f-divergence family (e.g., JS divergence, Pearson $\chi^2$), proving that as long as the generator function satisfies $xf''(x)$ monotonicity, this failure is universal, indicating a structural flaw rather than an issue with specific divergences.

**3. GEB Framework (Eq. 8-11): Back-deriving a family of bonuses that satisfy optimism**

Since the problem stems from bonuses directly using the policy ratio $\pi/\pi_{ref}$, GEB introduces an atomic function $u(x,y)$ as an intermediary to construct the bonus:

$$\mathcal{L}_{bonus} = \beta \, \mathbb{E}_{x,y \sim \pi_{ref}}\big[u \cdot f'(u) - f(u)\big]$$

The key is to make $u$ negatively correlated with $\pi$ (e.g., $u = 1/\pi$ or $u = 1+\alpha - \pi$). This ensures that lower policy probability $\pi$ leads to higher $u$ and higher bonuses, naturally focusing incentives on unexplored regions. Theorem 4.2 proves that when $u$ satisfies corresponding conditions, the bonus strictly satisfies the optimism condition $\frac{\partial^2 \mathcal{L}_{bonus}}{\partial \pi \partial \pi_{ref}} \leq 0$. Unlike prior work that designs bonuses heuristically, GEB is back-derived from the optimism condition, correcting the mechanism itself.

**4. Unifying Prior Methods (Table 2): Heuristic bonuses as special cases, all practically computable**

GEB is not just another bonus but a framework accommodating multiple instances. Depending on the divergence (reverse KL, forward KL, Hellinger) and the choice of $u$, it instantiates various specific bonuses, proving that heuristic bonuses from prior work (Zhang/Xie/Cen) are just special cases. Crucially, through reward reparameterization, these instantiated bonuses eventually depend only on the current policy $\pi$ and do not require explicit computation of $\pi_{ref}$ during training. Thus, they can be integrated into standard iterative RLHF cycles with zero additional sampling cost.

### Loss & Training
- Reward Modeling: $r_t = \arg\min_r [\mathcal{L}_{BT}(\mathcal{D}_t, r) - \kappa \mathcal{L}_{bonus}(r)]$
- Policy Optimization: $\pi_t = \arg\max_\pi \mathcal{J}_{\beta,f}(\pi, r_t)$
- GEB can be seamlessly integrated into standard iterative RLHF loops without additional sampling costs.

## Key Experimental Results

### Main Results

Alignment tasks across multiple divergence settings and LLM backbones:

| Method | Description | vs Iterative f-DPO |
|------|------|-------------------|
| Passive Exploration | Standard passive exploration | baseline |
| Prior Bonus (Zhang/Xie/Cen) | Existing bonuses | Inconsistent improvement |
| **GEB (reverse KL)** | Ours | **Consistently superior** |
| **GEB (forward KL)** | Ours | **Consistently superior** |
| **GEB (Hellinger)** | Ours | **Consistently superior** |

The three GEB variants consistently outperform iterative f-DPO and existing bonus methods under different divergence regularizations.

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| $u = 1/\pi$ vs $u = 1+\alpha-\pi$ | Different choices of $u$ perform differently depending on the divergence. |
| Sampling Distribution Analysis | GEB indeed increases the sampling probability in low $\pi_{ref}$ regions. |
| Different Backbones | Consistently effective across multiple LLM backbones. |

### Key Findings
- **Existing bonuses indeed fail**: Sampling distribution analysis confirms prior methods concentrate on high $\pi_{ref}$ regions.
- **GEB achieves optimistic exploration**: The sampling distribution noticeably shifts toward low $\pi_{ref}$ regions.
- **Consistent and significant performance Gain**: Effective across divergence types and model scales.

## Highlights & Insights
- **Counter-intuitive discovery that "exploration rewards inhibit exploration"**: This is the most profound contribution—bonuses that seemingly encourage exploration actually reinforce conservative behavior under divergence regularization. This challenges the common understanding of exploratory bonuses in the RLHF community.
- **Elegance of the unified framework**: GEB not only fixes the problem but unifies prior heuristic methods as special cases and naturally extends to the entire α-divergence family.
- **Seamless transition from theory to practice**: The optimism of GEB is proven, and all instantiated bonuses depend only on $\pi$ (not requiring $\pi_{ref}$ computation), keeping computational costs identical to standard RLHF.

## Limitations & Future Work
- The theoretical analysis relies on the policy-reparameterized reward hypothesis, which may have biases in actual training (non-exact optimization).
- Experimental scale did not cover the largest models (70B+); effectiveness at ultra-large scales remains to be verified.
- The optimal choice of the atomic function $u$ depends on the divergence type; an automatic selection mechanism is currently lacking.
- Further comparison with other RL exploration strategies (e.g., intrinsic reward, count-based methods) is needed.

## Related Work & Insights
- **vs Zhang et al. 2024 / Xie et al. 2024 / Cen et al. 2025**: The bonuses in these prior works are theoretically shown to fail the optimism principle; GEB fixes this fundamental flaw.
- **vs Uncertainty Quantification (Bayesian, Ensemble)**: These methods are computationally infeasible for LLMs; GEB avoids direct uncertainty quantification through formula design.
- **vs DPO / f-DPO**: GEB serves as an enhancement to iterative DPO/f-DPO and can be directly applied on top of them.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reveals the fundamental failure of existing exploratory bonuses and provides a provably correct fix—high theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified across multiple divergences and backbones, though model scale and benchmark coverage could be broader.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation with a smooth narrative flow from failure analysis to correction.
- Value: ⭐⭐⭐⭐⭐ Fundamental contribution to RLHF exploration theory, directly guiding bonus design in practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RE-PO: Robust Enhanced Policy Optimization as a General Framework for LLM Alignment](re-po_robust_enhanced_policy_optimization_as_a_general_framework_for_llm_alignme.md)
- [\[ICLR 2026\] COMAL: A Convergent Meta-Algorithm for Aligning LLMs with General Preferences](comal_a_convergent_meta-algorithm_for_aligning_llms_with_general_preferences.md)
- [\[ICLR 2026\] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](unifying_stable_optimization_and_reference_regularization_in_rlhf.md)
- [\[ICLR 2026\] Learning to Summarize User Information for Personalized RLHF (PLUS)](learning_to_summarize_user_information_for_personalized_reinforcement_learning_f.md)
- [\[ICLR 2026\] Beyond RLHF and NLHF: Population-Proportional Alignment under an Axiomatic Framework](beyond_rlhf_and_nlhf_population-proportional_alignment_under_an_axiomatic_framew.md)

</div>

<!-- RELATED:END -->

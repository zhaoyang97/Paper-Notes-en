---
title: >-
  [Paper Note] Learning in Structured Stackelberg Games
description: >-
  [ICML 2026 Spotlight][Reinforcement Learning][Stackelberg game] This paper introduces a structural assumption to "contextual Stackelberg games" (where the mapping context $\to$ follower type originates from a hypothesis class $\mathcal{H}$) and constructs two new types of learning-theoretic dimensions: the Stackelberg-Littlestone dimension (SLdim), which characterizes online regret bounds, and the $\gamma$-SG / $\gamma$-SN dimensions, which characterize lower and upper bounds…
tags:
  - "ICML 2026 Spotlight"
  - "Reinforcement Learning"
  - "Stackelberg game"
  - "online learning"
  - "Littlestone dimension"
  - "PAC learning"
  - "AI safety"
date: 2026-05-08
content_hash: b75486c88f751823
---

# Learning in Structured Stackelberg Games

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2504.09006](https://arxiv.org/abs/2504.09006)  
**Code**: Not released  
**Area**: Multi-agent / Game Learning / Learning Theory  
**Keywords**: Stackelberg game, online learning, Littlestone dimension, PAC learning, AI safety  

## TL;DR
This paper introduces a structural assumption to "contextual Stackelberg games" (where the mapping context $\to$ follower type originates from a hypothesis class $\mathcal{H}$) and constructs two new types of learning-theoretic dimensions: the Stackelberg-Littlestone dimension (SLdim), which characterizes online regret bounds, and the $\gamma$-SG / $\gamma$-SN dimensions, which characterize lower and upper bounds for PAC sample complexity. The authors prove these dimensions strictly outperform various Littlestone / Natarajan dimensions and provide instance-optimal online algorithms (SSOA) and batch algorithms ($\mathfrak L^*$).

## Background & Motivation

**Background**: Stackelberg games provide a classic framework for studying "commitment-response" strategic interactions, where a leader commits to a strategy first, and the follower best-responds after observing it. This is widely applied in security patrolling, congestion pricing, and AI red-teaming. Harris et al. (2024) extended this to a *contextual* form, where the utilities of both the leader and follower are influenced by additional side information $\mathbf z$.

**Limitations of Prior Work**: Work by Harris et al. provided a pessimistic conclusion: when context sequences and follower types are both chosen adversarially, the worst-case regret grows linearly with time $T$. This stems from the fact that the problem can be reduced to online classification, where an adversary can encode any difficult-to-learn "context $\to$ type" mapping. In other words, "online no-regret learning" is impossible in the most general contextual Stackelberg models.

**Key Challenge**: In reality, the context $\to$ follower type mapping almost always possesses structure—site cameras can predict poaching types, and AI deployment environments can predict attack types. However, tools to formalize this structure and "apply" it to learnability are missing. Furthermore, forcing the problem into a multi-class classification framework (using Littlestone dimension to measure difficulty) ignores the utility space structure of Stackelberg games: often, the leader's strategy can remain optimal even if the classifier consistently predicts the wrong type.

**Goal**: (1) Formalize "structured Stackelberg games" where context $\to$ type stems from a known hypothesis class $\mathcal H$; (2) Identify new dimensions capable of simultaneously characterizing *utility space* and *hypothesis class complexity*; (3) Provide instance-optimal algorithms and matching sample/regret bounds for both online and batch learning paradigms.

**Key Insight**: Retain the utility structure of the game (the leader’s piecewise-linear payoff surface) and utilize classical tools like "shattered trees / shattered sets." However, replace the *regressor* in node weights with the *Stackelberg regret* itself, allowing the dimension to naturally distinguish between "wrong classification but correct strategy" and "wrong classification and wrong strategy."

**Core Idea**: Online learning is feasible if and only if the *Stackelberg-Littlestone dimension* is finite; batch learning is feasible if and only if the *$\gamma$-SN dimension* is finite. Both can be strictly smaller than their classical counterparts.

## Method

### Overall Architecture
The authors adopt the notation of contextual Stackelberg games: $\mathbf z \in \mathcal Z$ is the context, the leader commits to a mixed strategy $\mathbf x$ over $\Delta(\mathcal A)$, and the follower best-responds $b_f(\mathbf z, \mathbf x)$ based on one of $K$ types $\{f^{(1)}, \dots, f^{(K)}\}$. The novel structural assumption is that there exists an unknown true mapping $h^* \in \mathcal H \subseteq [K]^{\mathcal Z}$ such that $h^*(\mathbf z_t) = f_t$ for all $t$ (realizable setting). The leader's instantaneous loss is defined as the *Stackelberg regret* $r(\mathbf z, \hat{\mathbf x}, f^{(h^*(\mathbf z))}) = \sup_{\mathbf x} u(\mathbf z, \mathbf x, b(\mathbf z, \mathbf x)) - u(\mathbf z, \hat{\mathbf x}, b(\mathbf z, \hat{\mathbf x}))$. The work focuses on finding matching complexity measures for this regret.

### Key Designs

**1. Stackelberg-Littlestone (SL) Dimension: Embedding Utility Structure into Shattered Tree Node Weights**

The classic multi-class Littlestone dimension is blind to utility space—it only cares about the ability to distinguish classes. In Stackelberg games, the true cost is the leader's utility deficit, not the number of classification errors. The SL dimension modifies the shattered tree tool: each internal node contains a context $\mathbf z_s$, and each edge represents a type label $j \in [K]$. The recursive node weight is defined as $\rho_s = \inf_{\mathbf x \in \Delta(\mathcal A)} \max_{j: sj \in S_d} \bigl( r(\mathbf z_s, \mathbf x, f^{(j)}) + \rho_{sj} \bigr)$ (with leaf nodes $\rho_s = 0$). A tree is shattered by $\mathcal H$ if for every root-to-leaf path, there exists $h \in \mathcal H$ matching the edge labels. The SL dimension is the supremum of root weights across all shattered trees. The key difference is embedding the Stackelberg regret $r(\cdot)$—disagreements in labels where the leader's optimal strategy remains the same will result in weights collapsing to 0.

**2. SSOA: Adapting the Standard Optimal Algorithm for Stackelberg Regret**

The SSOA (Stackelberg Standard Optimal Algorithm) maintains a version space $V_t \subseteq \mathcal H$ consistent with history. After observing $\mathbf z_t$, for each possible type $j \in V_t(\mathbf z_t)$, it calculates the leader's optimal utility $u_*^{(j)} = \sup_{\mathbf x} u(\mathbf z_t, \mathbf x, b_{f^{(j)}}(\mathbf z_t, \mathbf x))$ and selects:

$$\mathbf x_t \in \arg \inf_{\mathbf x} \max_{j \in V_t(\mathbf z_t)} \bigl( u_*^{(j)} - u(\mathbf z_t, \mathbf x, b_{f^{(j)}}(\mathbf z_t, \mathbf x)) + \mathrm{SLdim}_{\mathcal G}(V_t^{(\mathbf z_t \to j)}) \bigr).$$

The intuition is to minimize the maximum of "current instantaneous regret" plus "the difficulty of the remaining learning task if the type is $j$." It is a minimax optimization against a worst-case future adversary, similar to SOA in classification but using Stackelberg regret.

**3. $\gamma$-SN / $\gamma$-SG Dimensions: Adding a $\gamma$-Threshold for "Costly Disagreements" in PAC Settings**

For the batch PAC setting, simply applying Natarajan or Graph dimensions overestimates difficulty because it fails to capture harmless disagreements (different predictions but same leader strategy). The solution is to add a $\gamma$ threshold to shattered sets, only counting dimensions where disagreements are "costly." A set of $n$ elements is $\gamma$-SN-shattered if there exist two functions $g_0, g_1$ such that: (i) for each $\mathbf z_i$, the leader cannot find a mixed strategy that yields $\le \gamma$ regret for both followers; (ii) any bit pattern $\{0,1\}^n$ is realizable by $\mathcal H$. The $\mathfrak L^*$ algorithm keeps only the subclass $\mathcal H|_S$ consistent with $n$ samples and performs a minimax over the candidate set for new contexts: $\mathbf x^* = \inf_{\mathbf x} \max_{i \in F} r(\mathbf z, \mathbf x, f^{(i)})$.

## Key Experimental Results

As this is a theoretical paper, the results are presented as theorems and constructive counterexamples.

### Main Results

| Setting | Dimension Controlling Regret/Samples | Relationship to Classic Dimensions | Algorithm |
| :--- | :--- | :--- | :--- |
| Online Regret (Upper, Thm 3.9) | $\mathrm{SLdim}_{\mathcal G}(\mathcal H)$ | $\mathrm{SLdim}_{\mathcal G}(\mathcal H) \le \mathrm{Ldim}(\mathcal H)$ | SSOA (Alg. 1) |
| Online Regret (Lower, Thm 3.8) | $\mathrm{SLdim}_{\mathcal G}(\mathcal H) - \epsilon$ | No deterministic algorithm can do better | Adversarial construction |
| PAC Sample Lower (Thm 4.4) | $\Omega\bigl(\frac{\mathrm{SNdim}^{(\gamma)} + \log(1/\delta)}{\epsilon}\bigr)$ | Adapted from Natarajan with $\gamma$ threshold | — |
| PAC Sample Upper (Thm 4.7) | Controlled by $\mathrm{SGdim}^{(\gamma)}_{\mathcal G}(\mathcal H)$ | Corresponds to Graph dim with utility cut-off | $\mathfrak L^*$ (Alg. 2) |

### Strict Separation from Classic Dimensions

| Example | $\mathrm{Ldim}(\mathcal H)$ | $\mathrm{SLdim}_{\mathcal G}(\mathcal H)$ | Explanation |
| :--- | :--- | :--- | :--- |
| Thm 3.5 construction | $\infty$ | $0$ | Two followers induce same optimal strategy near threshold; classification is infinitely hard, but strategy regret is zero. |
| Example 3 ($n$ types + Permutation class) | $n-1$ | $n-H_n$ ($H_n$ is harmonic number) | SLdim is smaller by a harmonic factor; gap diverges with $n$. |
| Thm 3.11 | Large | Small | SOA continues to suffer utility loss while SSOA does not. |

### Key Findings
- The true necessary and sufficient condition for online "learnability" is not finite classifier complexity, but finite *utility-aware* SL dimension.
- In batch settings, "game-aware" Natarajan/Graph dimensions with $\gamma$ thresholds are required to match upper and lower bounds.
- Directly applying the classical SOA is *not* optimal: the authors construct instances where SOA's cumulative regret is strictly higher than SSOA's.
- Analytical solutions for Example 3 show that as the number of follower types $n$ increases, the gap between "classification difficulty" and "game difficulty" is at least a multiplicative factor of $\Omega(\log n)$.
- The framework extends to "learning to bid in auctions" and "Bayesian persuasion," showing SLdim is a unified characterization for commitment-then-response frameworks.

## Highlights & Insights
- A clean paradigm for "utility-aware online learning dimensions" is proposed: replace 0/1 mistake weights in shattered trees with task-specific losses (Stackelberg regret) to obtain instance-optimal dimensions.
- By demonstrating "equivalent labels but identical strategies," the work proves the "over-estimation" of classical Littlestone/Natarajan dimensions.
- The SSOA algorithm is structurally dual to classic algorithms, making it easy to extend existing SOA implementations by adding an SLdim term.

## Limitations & Future Work
- SSOA and $\mathfrak L^*$ require enumerating or optimizing over version spaces, which is computationally expensive for large hypothesis classes.
- The work assumes a realizable setting ($h^* \in \mathcal H$); whether the SL dimension characterizes the agnostic case remains an open question.
- Matching bounds are limited to deterministic algorithms and PAC cut-off losses; optimality for randomized algorithms or expected-utility losses is still open.
- A gap remains between $\gamma$-SN and $\gamma$-SG dimensions; the instance-optimal sample complexity for the batch case is not yet closed.

## Related Work & Insights
- **vs Harris et al. 2024**: Harris proved $\Theta(T)$ regret for two-sided adversaries; this work bypasses unlearnability by introducing the structural mapping $\mathcal H$.
- **vs Balcan et al. 2015 / Harris et al. 2023**: Previous online Stackelberg work used Hedge over mixed strategies; this work replaces i.i.d. context assumptions with structural assumptions and lifts SLdim as a unified measure.
- **vs Ahmadi et al. 2024 (Strategic Littlestone)**: Both extend Littlestone-like dimensions to commitment games, but Ahmadi focuses on strategic classification (followers modify features), while this work focuses on Stackelberg actions.
- **vs Wang et al. 2026b**: They model LLM SFT as contextual Stackelberg with query costs; this work provides the learnability proof using structural assumptions as a tool to replace i.i.d. assumptions.

## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](../../ICLR2026/reinforcement_learning/learning_to_play_multi-follower_bayesian_stackelberg_games.md)
- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](../../ICLR2026/reinforcement_learning/nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)
- [\[ACL 2026\] The Stackelberg Speaker: Optimizing Persuasive Communication in Social Deduction Games](../../ACL2026/reinforcement_learning/the_stackelberg_speaker_optimizing_persuasive_communication_in_social_deduction_.md)
- [\[NeurIPS 2025\] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](../../NeurIPS2025/reinforcement_learning/learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [\[ICLR 2026\] Stackelberg Coupling of Online Representation Learning and Reinforcement Learning](../../ICLR2026/reinforcement_learning/stackelberg_coupling_of_online_representation_learning_and_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->

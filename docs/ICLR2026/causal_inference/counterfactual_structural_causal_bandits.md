---
title: >-
  [Paper Note] Counterfactual Structural Causal Bandits
description: >-
  [ICLR 2026][Causal Inference][Paper Note] This paper elevates Structural Causal Bandits (SCB) from the L1/L2 (observational/interventional) layers of the Pearl Causal Hierarchy to the L3 (counterfactual) layer. It proposes the CTF-SCB framework, incorporating implementable counterfactual actions—such as "what if a downstream variable had received $x'$ instead
tags:
  - ICLR 2026
  - Causal Inference
date: 2026-05-08
content_hash: 2e909185987a63b2
---
# Counterfactual Structural Causal Bandits

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=gjvTNxVd2f](https://openreview.net/forum?id=gjvTNxVd2f)  
**Code**: To be confirmed  
**Area**: Causal Inference / Causal Bandit / Online Learning Theory  
**Keywords**: Structural Causal Models, Counterfactuals, Pearl Causal Hierarchy, Multi-Armed Bandits, Action Space Pruning

## TL;DR
This paper elevates Structural Causal Bandits (SCB) from the L1/L2 (observational/interventional) layers of the Pearl Causal Hierarchy to the L3 (counterfactual) layer. It proposes the CTF-SCB framework, incorporating implementable counterfactual actions—such as "what if a downstream variable had received $x'$ instead of $x$"—into the arm space. Using a set of graph-theoretic characterizations (CTF-MIS / CTF-POMIS / counterfactual regime graphs), the work prunes an exponential arm space down to a representative subset of "possibly optimal" arms, which, when paired with standard bandit solvers, achieves lower cumulative regret.

## Background & Motivation

**Background**: Integrating causal models into Multi-Armed Bandits (MAB) is a major research line in decision theory. Structural Causal Bandits (SCB, Lee & Bareinboim 2018/2019) treat each arm as an intervention $do(X=x)$ on a variable within a Structural Causal Model (SCM). Thus, "which arm to pull" becomes "which variable to intervene on and what value to set." This line of work has demonstrated that treating all interventions as independent arms for a standard bandit algorithm leads to excessive regret; instead, the action space should be pruned using the causal graph before learning.

**Limitations of Prior Work**: However, all existing SCB works are confined to L1 (observational $P(Y \mid x)$) and L2 (interventional $P(Y \mid do(x))$) of the Pearl Causal Hierarchy (PCH), treating each "physically executable intervention" as an arm while counterfactual (L3) quantities are entirely excluded. The reason is the default assumption that counterfactuals are "unsamplable"—once a unit naturally selects $X=x'$, its outcome $Y_x$ under a counterfactual intervention $do(x)$ cannot be simultaneously observed.

**Key Challenge**: Yet, counterfactuals are more fine-grained and informative than L2. An intervention $do(X=x)$ is equivalent to making all children of $X$ receive the same value $x$. In contrast, counterfactuals allow different children of $X$ to receive different "hypothetical inputs"—for example, letting a doctor evaluate a patient's true complaint $x$, while an AI tool receives a reformulated report $x'$. Such a world where the same root variable takes different values across different downstream channels cannot be expressed by standard interventions, yet it may yield higher rewards. Discarding this information means the optimal arm of an SCB might not even exist within its considered action space.

**Key Insight**: The authors seize a critical fact: not all L3 quantities are unsamplable. Counterfactual randomization proposed by Bareinboim et al. and the characterization of "implementable distributions" by Raghavan & Bareinboim (2025) indicate that quantities like $P(Y_x, x')$ involving counterfactual mediators can be sampled through a set of physical operations. Since these counterfactual actions can actually be executed to obtain rewards, there is no reason to exclude them from the bandit's arms.

**Core Idea**: The action space of SCB is expanded from "L2 interventions" to "implementable counterfactual regimes," resulting in CTF-SCB. A set of graph-theoretic tools is developed to prune this exponentially expanding counterfactual arm space down to a minimal representative subset—guaranteed not to lose the optimal arm—allowing standard bandit solvers to directly benefit from lower regret without relying on parametric assumptions.

## Method

### Overall Architecture

CTF-SCB models the environment as an SCM $M=\langle U,V,F,P(U)\rangle$ containing a reward variable $Y$. The agent **knows the causal graph $G$ and $Y$, but not the mechanisms $F$ or the exogenous distribution $P(U)$** (structure is known, parameters are unknown). Unlike traditional SCB which optimizes $\mathbb{E}Y_x=\mathbb{E}[Y\mid do(X=x)]$, this work generalizes arms to **counterfactual events** $X_*=\{X_{1[w_1]},X_{2[w_2]},\dots\}$. Each $X_i$ is no longer fixed to a constant but "behaves as it would in submodel $M_{w_i}$," and this value replaces the natural mechanism $f_{X_i}$. In each round $t$, the agent pulls an arm $X_*$ and observes the counterfactual reward $Y_{X_*}$, aiming to minimize cumulative regret $R_T^{\mathcal{A}}=T\mu_{X_*^\star}-\sum_{t}\mu_{X_*^{(t)}}$.

Searching directly over all counterfactual events of the form $\{X_{i[w_i]}\}$ leads to a combinatorial explosion that is **super-exponential** relative to the number of nodes, with most being redundant or provably suboptimal. Thus, the main thread of the paper is a "layer-by-layer contraction" pruning pipeline (Fig. 2 in the paper):

$$
\underbrace{\mathcal{A}}_{\text{All Implementable Counterfactual Arms}}\;\xrightarrow{\text{Sec. 3.1}}\;\underbrace{\text{CTF-MIS}}_{\text{Remove redundancies with no impact on reward}}\;\xrightarrow{\text{Sec. 3.2}}\;\underbrace{\text{CTF-POMIS}}_{\text{Remove provably suboptimal arms}}\;\xrightarrow{\text{Sec. 3.2.2}}\;\underbrace{\text{Representative CTF-POMIS}}_{\mathcal{A}^\star}
$$

The hard constraint of this chain is $\mathbb{E}R_T^{\mathcal{A}^\star}\le\mathbb{E}R_T^{\mathcal{A}}$: pruning **only removes arms that should not be explored and never removes potentially optimal arms**. After contracting to $\mathcal{A}^\star$, standard solvers with finite-time regret guarantees (KL-UCB, Thompson Sampling) are applied. The four components of the design correspond to: legal and non-redundant arms (Implementability + CTF-MIS), "possibly optimal" arms (CTF-POMIS criteria + counterfactual regime graphs), and the efficient enumeration of these arms (Algorithm 1).

### Key Designs

**1. Implementability Criteria for Counterfactual Actions: Conflict Detection via Ancestry**

Not every arbitrarily written counterfactual event can be sampled in a system. The difficulty lies in constraints such as $X_*=\{W_x, Z_{x'}\}$ where $x\ne x'$. To let $Y$ hear $W_x$, mechanism $f_Z$ must receive $x$; but to let $Y$ also hear $Z_{x'}$, that same $f_Z$ must receive $x'$. A single mechanism cannot receive two mutually exclusive inputs, making the arm physically unimplementable. The authors provide a necessary and sufficient condition for implementability via **ancestral conflict detection** (Prop. 1): defining the ancestor set of a counterfactual variable $X_w$ as $\mathrm{An}(X_w)$, $X_*$ is a legal action if and only if there are **no copies of the same variable $X$ under two different regimes $w\ne t$** in $\mathrm{An}(Y_x, X_*)$. This is grounded in the Counterfactual Unnesting Theorem (CUT): $P(y_{X_*})=\sum_x P(y_x, X_*=x)$, which decomposes the counterfactual reward into a samplable form, finalized by the implementability characterization of Raghavan & Bareinboim (2025). This step prunes non-existent arms using pure graph operations.

**2. CTF-MIS: Eliminating Redundant Counterfactual Variables**

Legal arms still contain many redundancies—a variable $X_w\in X_*$ might not contribute to the reward, meaning $\mu_{X_*}=\mu_{X_*\setminus\{X_w\}}$. The authors use CTF-calculus (the L3 version of do-calculus) and **intervention minimization** (Lemma 1: $\lVert X_w\rVert\triangleq X_{w\cap\mathrm{An}(X)_{G_{\overline{W}}}}$, which cuts unused parts of the subscription) to identify these equivalences. This defines the **Counterfactual Minimal Intervention Set (CTF-MIS)** (Def. 3): it satisfies $X_*=\lVert X_*\rVert$ and no smaller $Z_*\subsetneq X_*$ exists such that $\mu_{X_*}=\mu_{Z_*}$ for all SCMs compatible with $G$. Its graph-theoretic characterization (Thm. 1) is intuitive: (i) each $X_i$ has a directed causal path to $Y$ ($X\subseteq\mathrm{An}(Y)_{G_{\overline{X}}}$); (ii) the subscription $W_i$ of each $X_{i[w_i]}$ must actually "control" the ancestors of $X_i$ ($W_i\cap\mathrm{An}(X_i)_{G_{\overline{X}\setminus\{X_i\}}}\ne\emptyset$). A counter-intuitive point noted is that **different CTF-MIS can be equivalent** (Remark 1), unlike in standard SCB. For instance, $\{W_x,Z_{x'}\}$ and $\{T_x,Z_{x'}\}$ may be equivalent if they propagate values to $Y$ in identical ways, providing a basis for selecting single representatives.

**3. CTF-POMIS + Counterfactual Regime Graph: Determining "Possibly Optimal" via Intervention Boundaries**

A partial order exists among CTF-MIS: some counterfactual interventions are at least as good as others regardless of model parameters. The **Counterfactual Possibly Optimal Minimal Intervention Set (CTF-POMIS)** (Def. 4) is defined as a set for which there exists an SCM compatible with $G$ where the arm is strictly better than all non-equivalent CTF-MIS. The key construction is the **counterfactual regime graph** $H_{X_*}=\langle V^\dagger, E^\dagger\rangle$, where $V^\dagger=V(\mathrm{An}(Y_x,X_*))$. Unlike standard causal graphs, it **retains the true causal relations between counterfactual variables**—based on consistency $X_*=x\Rightarrow Y_x=Y_{X_*}$, it connects each $X_i$ to its children in $G$ (e.g., $W\to T$ denotes $W_x\to T_{W_x}$), preserving information lost in older representations like AMWN. With this graph, CTF-POMIS has a clean graph-theoretic condition (Thm. 2): $X_*$ is a CTF-POMIS if and only if $\mathrm{IB}(H_{X_*}, Y)=\emptyset$, where IB is the **Intervention Boundary** (the set of nodes directly affecting the smallest UC-territory). An important corollary (Prop. 2 / Cor. 2) is that if $Y$ is not coupled with its ancestors by unobserved confounding (e.g., a Markovian graph), the unique CTF-POMIS is $\mathrm{Pa}(Y)_G$—meaning **in unconfounded scenarios, counterfactual actions are unnecessary, and intervening on the parents of $Y$ suffices**. The value of counterfactuals emerges only when $Y$ is confounded.

**4. Algorithm 1: Enumerating Representative CTF-POMIS via Edge Selection**

Verifying every counterfactual event is computationally infeasible. The authors solve this by focusing only on counterfactuals of the form $\{X_{i[\mathrm{Pa}(X_i)_G]}\}$. Each such "representative" can represent dozens of equivalent CTF-(PO)MIS, and together they cover $\mathcal{A}^\star$ (Prop. 3). The algorithm (Algo. 1, complexity $O(n^2\cdot 2^{|E|})$) follows an **edge selection** approach: for each combination of "child selections," edges are deleted from the graph to obtain $H'$. If $\mathrm{IB}(H',Y)=\emptyset$, a counterfactual event $X_*$ is constructed and passed to the `CTF-MISIFY` subroutine (which removes variables violating Thm. 1) to form a legal CTF-MIS. Finally, it is checked for conflicts via Prop. 1. Thm. 3 guarantees that it returns all representative CTF-POMIS without omission or duplication. Standard bandit solvers then operate on $\mathcal{A}^\star$ with a regret bound of $O\!\big(\sum_{X_*\in\mathcal{A}^\star:\Delta_{X_*}>0}\frac{\log T}{\Delta_{X_*}}\big)$.

### Loss & Training
This is a theoretical and simulation-based work with no trainable parameters. The online learning goal is to minimize cumulative regret (Eq. 1), implemented using KL-UCB and Thompson Sampling over the pruned action space $\mathcal{A}^\star$. The regret upper bound $O(\sum_{X_*\in\mathcal{A}^\star:\Delta_{X_*}>0}\frac{\log T}{\Delta_{X_*}})$ shows that reducing the number of arms directly minimizes the regret constant.

## Key Experimental Results

The experiments compare cumulative regret (CR) across three action space strategies: **CTF-POMIS** (pruned as per this paper), **CTF-MIS** (redundancy removed, suboptimality remains), and **POMIS** (standard SCB restricted to L≤2). Each is paired with Thompson Sampling (TS) and KL-UCB.

### Main Results (Cumulative Regret for Three Causal Graph Tasks)

| Task | Solver | CR of CTF-POMIS | Ratio vs CTF-MIS | Conclusion |
|------|--------|-----------------|-------------------|------|
| Task 1 (Fig. 5a) | TS | 66.69 | ≈38.92% | CTF-POMIS is lowest |
| Task 1 | KL-UCB | 148.19 | ≈38.96% | CTF-POMIS is lowest |
| Task 2 (Fig. 3b) | TS | 387.01 | 57.46% | CTF-(PO)MIS consistently outperforms POMIS |
| Task 2 | KL-UCB | 831.24 | 70.17% | Same as above |
| Task 3 (Fig. 9) | TS / KL-UCB | See Appx Table 1–2 | — | POMIS suffers linear regret when optimal arm is not in POMIS |

### Key Findings
- **CTF-POMIS reduces regret to ~39% of CTF-MIS (Task 1)**: Removing "provably suboptimal" counterfactual arms saves significantly more exploration cost than just removing redundancies, showing that the POMIS pruning layer is the primary driver of regret reduction.
- **POMIS suffers linear regret when the optimal arm is outside L≤2**: In Task 3, where a strictly positive gap $\Delta_{L\le2}\triangleq\mu_{X_*^\star}-\mu_{x^\star}>0$ exists, POMIS fails to reach the counterfactual optimal arm, leading to linear regret growth—direct evidence that ignoring L3 is systematically suboptimal.
- **POMIS converges faster only when the optimal arm is in L≤2**: In Task 3 where $\Delta_{L\le2}=0$, POMIS has fewer arms and less exploration, leading it to lead temporarily. However, since the agent cannot know the layer of the optimal arm beforehand, this does not weaken the conclusion to consider counterfactuals by default.

## Highlights & Insights
- **Bridges PCH L3 to sequential decision making**: Unlike previous causal bandits restricted to L1/L2, this work uses implementability theory to show that a class of counterfactual actions is samplable and executable, expanding the arm space to a more granular level.
- **The Counterfactual Regime Graph is the centerpiece**: By preserving the causal edges from $X_i$ to its true children (e.g., $W_x\to T_{W_x}$), it allows the **existing L2 tool** of intervention boundaries ($\mathrm{IB}=\emptyset$) to be ported directly to characterize L3 CTF-POMIS.
- **"Counterfactuals are not needed under Markovian settings" is a useful negative result**: Prop. 2/Cor. 2 informs practitioners that counterfactual actions only offer gains when the reward is contaminated by unobserved confounding; otherwise, intervening on $\mathrm{Pa}(Y)$ is optimal.
- **Representative CTF-POMIS + edge selection enumeration** compresses super-exponential search into an $O(n^2\cdot 2^{|E|})$ process, making the theory actionable through a concrete pruning algorithm.

## Limitations & Future Work
- **SCM Modeling Assumptions**: Requires a fixed causal structure and defined variables, making it difficult to model dynamic, high-dimensional, or partially observable systems.
- **Feasibility of Counterfactual Actions**: Assumes the agent can execute any implementable counterfactual $P(Y_{X_*成就})$, which implicitly requires identifying suitable "counterfactual mediators." This may be constrained by ethics or technology in reality.
- **Known Causal Graph Assumption**: Assumes the agent has the true graph. While causal discovery can help, the authors note that **conservatively assuming a confounding edge (super-model) does not compromise soundness** (it only reduces precision), whereas missing an edge is dangerous.
- Future Directions: Explicitly incorporating causal discovery uncertainty into regret analysis or providing robustness guarantees under graph misspecification would make the framework more practical.

## Related Work & Insights
- **vs Traditional SCB (Lee & Bareinboim 2018/2019)**: They define MIS/POMIS on L≤2. This work elevates those concepts to L3 and identifies that "equivalence between different MIS" occurs in the counterfactual setting, necessitating a new enumeration approach.
- **vs Implementability Characterization (Raghavan & Bareinboim 2025; Bareinboim et al. 2015)**: That line of work identifies which counterfactual distributions are samplable; this paper applies that to determine which of those samplable distributions should be used as bandit arms.
- **vs Standard MAB Solvers**: This work does not change the solvers but the action space they operate on, decoupling causal pruning from the underlying learning algorithm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to integrate PCH L3 counterfactuals into SCB arm spaces with full graph characterization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong verification of regret advantages across three tasks, though based on synthetic SCMs.
- Writing Quality: ⭐⭐⭐⭐ Logical progression from implementability to algorithms, though high notation density presents a learning curve.
- Value: ⭐⭐⭐⭐⭐ Establishes a new direction at the intersection of causal inference and online learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Self-Supervised Learning from Structural Invariance](self-supervised_learning_from_structural_invariance.md)
- [\[NeurIPS 2025\] Root Cause Analysis of Outliers with Missing Structural Knowledge](../../NeurIPS2025/causal_inference/root_cause_analysis_of_outliers_with_missing_structural_knowledge.md)
- [\[ICLR 2026\] On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study](on_the_eligibility_of_llms_for_counterfactual_reasoning_a_decompositional_study.md)
- [\[ICLR 2026\] Counterfactual Explanations on Robust Perceptual Geodesics](counterfactual_explanations_on_robust_perceptual_geodesics.md)
- [\[ICLR 2026\] Multiverse Mechanica: A Testbed for Learning Game Mechanics via Counterfactual Worlds](multiverse_mechanica_a_testbed_for_learning_game_mechanics_via_counterfactual_wo.md)

</div>

<!-- RELATED:END -->

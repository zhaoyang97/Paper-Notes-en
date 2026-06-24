---
title: >-
  [Paper Note] Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition
description: >-
  [ICML 2025][Others (Online Learning][Online Information Acquisition] This paper studies the Best Scoring Rule Identification (BSRI) problem under the principal-agent online information acquisition framework, proposing two algorithms, OIAFC (fixed confidence) and OIAFB (fixed budget). It establishes the first instance-dependent sample complexity upper bound of $\widetilde{O}(MH_\Delta)$ and significantly improves the instance-independent sample complexity from the existing lim…
tags:
  - "ICML 2025"
  - "Others (Online Learning"
  - "Mechanism Design"
  - "Game Theory)"
  - "Online Information Acquisition"
  - "Best Scoring Rule"
  - "Principal-Agent Game"
  - "Multi-Armed Bandit"
  - "Best Arm Identification"
  - "Sample Complexity"
date: 2026-05-08
content_hash: 73e3c1bc7750818f
---

# Provably Efficient Algorithm for Best Scoring Rule Identification in Online Principal-Agent Information Acquisition

**Conference**: ICML 2025  
**arXiv**: [2505.17379](https://arxiv.org/abs/2505.17379)  
**Code**: None  
**Area**: Others (Online Learning / Mechanism Design / Game Theory)  
**Keywords**: Online Information Acquisition, Best Scoring Rule, Principal-Agent Game, Multi-Armed Bandit, Best Arm Identification, Sample Complexity

## TL;DR

This paper studies the Best Scoring Rule Identification (BSRI) problem under the principal-agent online information acquisition framework, proposing two algorithms, OIAFC (fixed confidence) and OIAFB (fixed budget). It establishes the first instance-dependent sample complexity upper bound of $\widetilde{O}(MH_\Delta)$ and significantly improves the instance-independent sample complexity from the existing limit of $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$ to $\widetilde{O}(MK\epsilon^{-2})$.

## Background & Motivation

### Problem Scene

The information acquisition problem studies a hierarchical game scenario: a **principal** (e.g., a manager) hires an **agent** (e.g., a domain expert) to collect information about an unknown state of nature $\omega$. The key difficulty lies in the principal's inability to directly observe the agent's effort level, necessitating the design of appropriate **scoring rules** to incentivize the agent to both choose the optimal action and truthfully report their beliefs about the state.

Typical applications include:
- **Online questionnaire survey**: The principal designs the questionnaire, and the agent chooses the survey method (differing in cost and information quality).
- **Advertising effect evaluation**: The manager evaluates the market fit of an advertising strategy, while the expert chooses a research method to obtain market information.
- **Crowdsourcing verification**: The platform (principal) incentivizes workers (agents) to provide high-quality data through scoring rules.

### Limitations of Prior Work

Only Chen et al. (2023) and Cacciamani et al. (2023) have previously studied online information acquisition:

1. **Chen et al. (2023)** focused on regret minimization. Although they provided results for fixed-confidence BSRI, the sample complexity is $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$, which is significantly larger than the $\widetilde{O}(K\epsilon^{-2})$ of standard multi-armed bandit best arm identification.
2. **Cacciamani et al. (2023)**: The ETC (Explore-Then-Commit) framework is inherently difficult to yield instance-dependent sample complexity.
3. Totally **lacks instance-dependent** sample complexity results.
4. The BSRI under the **fixed-budget** setting is completely unexplored.

### Difference from Standard Best Arm Identification

BSRI is more challenging than standard multi-armed bandit best arm identification:
- The principal cannot directly "pull an arm" but can only **indirectly incentivize** the agent to select a specific action by designing scoring rules.
- The scoring rules must achieve two goals simultaneously: (1) incentivize the agent to choose the optimal action, and (2) incentivize the agent to report beliefs truthfully.
- Additional learning of belief distributions $q_k$ and pairwise cost differences $C(k, k')$ is required.

## Method

### Overall Architecture

The core idea of this paper is to reduce the information acquisition problem to a bandit-like problem, and then design UCB-style algorithms to solve it.

**Interaction Protocol** (for each round $t$):
1. The principal announces the scoring rule $S_t : \Omega \times \Delta(\Omega) \to \mathbb{R}_+$
2. The agent chooses action $k_t$ (observable by the principal), incurring cost $c_{k_t}$
3. The environment generates a hidden state $\omega_t$, sending observation $o_t$ to the agent
4. The agent submits belief report $\hat{\sigma}_t$
5. The environment reveals $\omega_t$, the principal pays according to the scoring rule and obtains utility

**Problem Reduction**: Define $V_k = \{S \in \mathcal{S} \mid g(k, S) \geq g(k', S), \forall k'\}$ as the region of scoring rules where the agent chooses action $k$. The original bi-level optimization problem is reduced to:

$$\max_{k \in \mathcal{A}} h(S_k^*), \quad h(S_k^*) = \sup_{S \in V_k} \mathbb{E}_{\sigma \sim q_k}[u(\sigma) - S(\sigma)]$$

When $V_k$, $q_k$, and pairwise cost differences $C(k,k')$ are known, the inner problem can be formulated as a linear program $\text{LP}_k$.

### Key Design 1: UCB Linear Programming (UCB-LP)

Due to the principal not knowing the belief distribution $q_k$ and pairwise cost difference $C(k,k')$, the algorithm needs to learn these parameters **online**. Define the empirical estimator:

$$\hat{q}_k^t(\sigma) = \frac{1}{N_k^t} \sum_{s=1}^{t-1} \mathbf{1}\{\sigma_s = \sigma, k_s = k\}$$

and the confidence radius (fixed-confidence setting):

$$I_q^t(k) = \sqrt{\frac{2\log(4K^2 M t^2 / \delta)}{N_k^t}}$$

By substituting the estimate and confidence radius into $\text{LP}_k$, we construct the Upper Confidence Bound Linear Program UCB-LP$_{k,t}$:

$$\hat{h}_k^t = \max_{S \in \mathcal{S}} \hat{u}_k^t + B_u I_q^t(k) - v$$
$$\text{s.t.} \quad |v - \hat{v}_S^t(k)| \leq B_S I_q^t(k)$$
$$v - \hat{v}_S^t(k') \geq \hat{C}^t(k,k') - (I_c^t(k,k') + B_S I_q^t(k')), \quad \forall k' \neq k$$

This yields a high-probability upper bound estimate $\hat{h}_k^t$ for $h(S_k^*)$, which is used to select the currently optimal arm $k_t^* = \arg\max_k \hat{h}_k^t$.

### Key Design 2: Adaptive Trade-off Parameter and Stopping Condition

**Conservative Scoring Rules**: To ensure that the agent chooses the expected action $k_t^*$, the algorithm employs a mixed strategy:

$$S_t = \alpha_k^t \tilde{S}_{k_t^*} + (1 - \alpha_k^t) \hat{S}_{k_t^*, t}$$

where $\tilde{S}_k$ comes from an "action-informed oracle" (which guarantees that the agent selects action $k$), and $\hat{S}_{k,t}$ is the learned approximately optimal scoring rule.

**Instance-Dependent Parameter Design** (core contribution of this paper):

$$\alpha_k^t = \min\left(\sqrt{\frac{M}{L_k^t}}, 1\right), \quad \beta_t = \frac{\epsilon^{-2} \alpha_{k_t^*}^t (B_S + B_u)}{1 - \alpha_{k_t^*}^t}$$

Chen et al. (2023) utilized a fixed $\alpha_k^t = \min(K/t^{1/3}, 1)$, which leads to a suboptimal instance-independent bound of sample complexity. The adaptive parameter of this work ensures that the number of forced exploration rounds satisfies $L_k^\tau = \widetilde{O}(M^2(B_S + B_u)^2 \Delta_k^{-2})$, thereby obtaining instance-dependent bounds.

**Stopping Rule**: The algorithm stops when $2(B_S + B_u) I_q^t(k_t^*) \leq \beta_t$, and outputs $\hat{S}^* = S_t$.

### Key Design 3: Binary Search for Learning Cost Differences

When the action actually chosen by the agent $k_t \neq k_t^*$, a convex-combination-based binary search is used to estimate the pairwise cost difference $C(k, k')$:

$$C(k, k') = v_S(k) - v_S(k'), \quad \forall S \in V_k \cap V_{k'}$$

A binary search is conducted between $\tilde{S}_k$ (which triggers action $k$) and $\tilde{S}_{k'}$ (which triggers action $k'$), finding a boundary scoring rule $S \in V_k \cap V_{k'}$, which allows updating the cost difference estimate using Equation (7).

## Theoretical Results

Since this paper is a purely theoretical work with no numerical experiments, the key theoretical results are summarized in the tables below:

### Sample Complexity Comparison

| Algorithm/Result | Setting | Sample Complexity | Type |
|:--|:--|:--|:--|
| Chen et al. (2023) | Fixed Confidence | $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$ | Instance-Independent |
| **OIAFC (Ours)** | Fixed Confidence | $\widetilde{O}(\epsilon^{-2} B_S^2 M H_\Delta)$ | **Instance-Dependent** |
| **OIAFC (Ours)** | Fixed Confidence | $\widetilde{O}(\epsilon^{-2} B_S^2 M K \epsilon^{-2})$ | Instance-Independent |
| **OIAFB (Ours)** | Fixed Budget | $T = \widetilde{O}(\epsilon^{-2} B_S^2 M K \epsilon^{-2})$ | Instance-Independent |
| Standard MAB BAI | Fixed Confidence | $\widetilde{O}(K\epsilon^{-2})$ | Instance-Independent |

Where $H_\Delta = 4(B_S+B_u)^2\epsilon^{-2} + \sum_{k \neq k^*} \Delta_k^{-2}$ and $M \leq K \times C_O$.

### Core Theorem Comparison

| Theorem | Content | Condition |
|:--|:--|:--|
| Theorem 1 (OIAFC) | Outputs an $(\epsilon, \delta)$-optimal scoring rule with a sample complexity of $\widetilde{O}(\epsilon^{-2}B_S^2 M H_\Delta)$ | $\alpha_k^t = \min(\sqrt{M/L_k^t}, 1)$ |
| Corollary 1 (OIAFC) | Instance-independent bound $\widetilde{O}(MK\epsilon^{-2})$ | $\alpha_k^t = \epsilon/(4(B_S+B_u))$ (fixed) |
| Theorem 2 (OIAFB) | Given budget $T$, outputs an $(\epsilon, \tilde{\delta})$-optimal scoring rule | $\alpha_k^t = \epsilon/(4(B_S+B_u))$ |
| Corollary 2 (OIAFB) | When budget $T = \widetilde{O}(MK\epsilon^{-2})$, $\tilde{\delta} \leq \delta$ | Matches the instance-independent bound of OIAFC |

## Key Findings

1. **First Instance-Dependent Result**: OIAFC establishes the first instance-dependent sample complexity upper bound for BSRI, which degenerates to the known optimal bound $\widetilde{O}(H_\Delta)$ of standard MAB BAI under simple settings.
2. **Significantly Improved Instance-Independent Bound**: Improved from Chen et al.'s $\widetilde{O}(C_O^3 K^6 \epsilon^{-3})$ to $\widetilde{O}(MK\epsilon^{-2})$, eliminating the cubic dependence on $\epsilon$.
3. **Unification of Fixed-Budget and Fixed-Confidence**: OIAFB matches the sample complexity of OIAFC under the instance-independent setting, showing that both settings share the same complexity characterization.
4. **Improved Stopping Strategy**: Chen et al. used a fixed iteration stopping rule adapted from Jin et al. (2018), which requires $\widetilde{O}(\epsilon^{-6} K^6 C_O^3)$ rounds; this paper designs an adaptive stopping rule, significantly improving efficiency.

## Highlights & Insights

- **Bridge from Indirect Incentivization to Direct Identification**: The core insight is to reduce the scoring rule design problem to a multi-armed bandit problem with $h(S_k^*)$ as the reward. However, the key difference is that the principal cannot directly "pull an arm" but must indirectly control the agent's behavior via scoring rules.
- **Adaptive Trade-off Parameter $\alpha_k^t$**: This is the most crucial technical contribution of the paper. Unlike Chen et al., who used a uniform fixed decay $\alpha_k^t = \min(K/t^{1/3}, 1)$ for all arms, this paper designs an adaptive parameter $\alpha_k^t = \min(\sqrt{M/L_k^t}, 1)$ for each arm, which depends on the actual exploration times of each arm, thereby achieving instance-dependent complexity.
- **Clever Combination of UCB-LP**: By combining the optimistic estimation of UCB with the linear programming framework, it guarantees the high-probability upper bound property while utilizing the linear structure of the problem for efficient computation.
- **Theoretical Framework Completeness**: Simultaneously covers four combinations (fixed-confidence + fixed-budget, instance-dependent + instance-independent), establishing a complete theoretical landscape.

## Limitations & Future Work

1. **Assumed Observability of Actions**: It requires the principal to observe the actions chosen by the agent (research methods). Although the paper provides practical examples such as online/offline surveys, actions are difficult to observe in many real-world scenarios.
2. **Requirement of Action-Informed Oracle**: The algorithm relies on an "Action-Informed Oracle" (Assumption 1) that provides $K$ scoring rules $\{\tilde{S}_k\}$, which may be difficult to obtain in practice.
3. **No Numerical Experimental Validation**: This is a purely theoretical work, lacking empirical validation to demonstrate the tightness of the theoretical bounds and the practical convergence rate of the algorithms.
4. **Cost of the $M$ Factor**: The presence of $M \leq K \times C_O$ in the sample complexity implies that when the belief space is large, the complexity could be high.
5. **Unexplored Lower Bound**: The information-theoretic lower bound for BSRI is not provided, making it impossible to determine whether the derived upper bound is optimal.

## Related Work & Insights

- **Best Arm Identification** (Even-Dar et al. 2002/2006; Gabillon et al. 2012; Jamieson et al. 2013): This paper generalizes BAI techniques (UCB + elimination) to strategic game environments with indirect control mechanisms.
- **Strategic Online Environment Learning**: Includes Stackelberg games (Sessa et al. 2020), online auctions (Feng et al. 2017), contract design (Ho et al. 2014; Zhu et al. 2022), and Bayesian persuasion (Castiglioni et al. 2020).
- **Offline Scoring Rule Design** (Neyman et al. 2021; Li et al. 2022; Hartline et al. 2023): This work extends offline problems to online learning settings.
- **Insights**: This framework can be generalized to more complex principal-agent scenarios, such as information acquisition with multiple agents, partially observable actions, or continuous scoring rule spaces.

## Rating

⭐⭐⭐

The theoretical contribution is solid, establishing the first instance-dependent sample complexity for the online information acquisition problem with significant improvements. However, the assumptions of the problem setting are relatively strong (action observability + action-informed oracle), and experimental validation is lacking, narrowing its practical applications. Overall, it is a rigorous theoretical work that advances the intersection of online learning and mechanism design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Near Optimal Best Arm Identification for Clustered Bandits](near_optimal_best_arm_identification_for_clustered_bandits.md)
- [\[ICLR 2026\] An Efficient, Provably Optimal Algorithm for the 0-1 Loss Linear Classification Problem](../../ICLR2026/learning_theory/an_efficient_provably_optimal_algorithm_for_the_0-1_loss_linear_classification_p.md)
- [\[ICML 2025\] Improved and Oracle-Efficient Online $\ell_1$-Multicalibration](improved_and_oracle-efficient_online_ell_1-multicalibration.md)
- [\[ICLR 2026\] A Near-Optimal Best-of-Both-Worlds Algorithm for Federated Bandits](../../ICLR2026/learning_theory/a_near-optimal_best-of-both-worlds_algorithm_for_federated_bandits.md)
- [\[NeurIPS 2025\] Efficient Kernelized Learning in Polyhedral Games Beyond Full-Information: From Colonel Blotto to Congestion Games](../../NeurIPS2025/learning_theory/efficient_kernelized_learning_in_polyhedral_games_beyond_full-information_from_c.md)

</div>

<!-- RELATED:END -->

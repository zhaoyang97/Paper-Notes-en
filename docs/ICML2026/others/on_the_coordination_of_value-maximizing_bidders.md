---
title: >-
  [Paper Note] On the Coordination of Value-Maximizing Bidders
description: >-
  [ICML 2026][Value-maximization] This paper formally investigates the "coordination" problem of multiple value-maximizing auto-bidders in online advertising. It proposes a simple coordination mechanism where "only the alliance member with the highest value bids, while others bid 0." It proves that for a large class of auto-bidding algorithms, this mechanism simultaneously reduces the RoS violation for each member and drives the total alliance value to the asymptotic optimum am…
tags:
  - "ICML 2026"
  - "Value-maximization"
  - "Auto-bidding"
  - "Second-price auction"
  - "RoS constraint"
  - "Coordination mechanism"
date: 2026-05-08
content_hash: adf202228dc0c2eb
---

# On the Coordination of Value-Maximizing Bidders

**Conference**: ICML 2026  
**arXiv**: [2511.04993](https://arxiv.org/abs/2511.04993)  
**Code**: None  
**Area**: Online Advertising / Auto-bidding / Mechanism Design  
**Keywords**: Value-maximization, Auto-bidding, Second-price auction, RoS constraint, Coordination mechanism

## TL;DR
This paper formally investigates the "coordination" problem of multiple value-maximizing auto-bidders in online advertising. It proposes a simple coordination mechanism where "only the alliance member with the highest value bids, while others bid 0." It proves that for a large class of auto-bidding algorithms, this mechanism simultaneously reduces the RoS violation for each member and drives the total alliance value to the asymptotic optimum among all coordination mechanisms.

## Background & Motivation

**Background**: The core paradigm of modern search/feed advertising is auto-bidding: advertisers delegate the optimization task of "maximizing total value subject to a Return-on-Spend (RoS) constraint (i.e., at least one unit of value per unit of spend)" to platforms or third-party agents. Algorithms then learn bids online via methods like mirror descent or dual gradients. Most literature assumes each bidder independently optimizes their own objective.

**Limitations of Prior Work**: This independence assumption is fragile in practice. Third-party agents often manage dozens of advertisers simultaneously, and large e-commerce entities (like Amazon, Temu, or Shein) often use the same portfolio to bid on multiple similar ads. This "same-root" competition on the same ad slot drives up clearing prices and breaks RoS constraints, making it a negative-sum game for the alliance. However, the auto-bidding literature lacks a theoretical characterization of "coordination"—aside from a few empirical works or studies on utility-maximizers (e.g., Decarolis et al., Romano et al., Chen et al.).

**Key Challenge**: The behavior of value-maximizers differs from that of utility-maximizers. The former actively overbid (typically $b_{i,t}=(1+1/\lambda_{i,t}) v_{i,t} > v_{i,t}$) to capture volume. Thus, the damage caused by "internal overbidding" within an alliance is more subtle than in classic cartel analysis and requires new modeling.

**Goal**: The objective is decomposed into three sub-questions: (i) Can a simple coordination mechanism strictly outperform independent bidding? (ii) Under what algorithms and distribution conditions does this "superiority" hold? (iii) Can it reach optimality in terms of the known optimal rates of the algorithms?

**Key Insight**: The authors leverage a simple observation: in a second-price auction, if the alliance members allow only the "member with the highest current value" $i^* = \arg\max_i v_{i,t}$ to bid while others stay silent, the alliance effectively eliminates all "internal friction." This collapses the coupled dynamics of $N$ bidders into an equivalent single-bidder problem, providing space for subsequent proofs.

**Core Idea**: Replace independent bidding with a coordination mechanism that grants the "highest value bidder exclusivity." This achieves RoS improvements (conditional) and value improvements (unconditional for mirror-descent class algorithms) from both mechanism and distribution perspectives.

## Method

### Overall Architecture
The setting involves $T$ rounds of repeated second-price auctions. An alliance of $N$ bidders each samples a value $v_{i,t}\in[0,B]$ independently from the same continuous distribution $F$ in each round $t$, facing an external competitive bid $d_t^O\sim D$ (which can incorporate reserve prices). The competitive bid for bidder $i$ is $d_{i,t}=\max\{d_t^O, \max_{j\neq i} b_{j,t}\}$, with the winning indicator $x_{i,t}=\mathbb{I}\{b_{i,t}\ge d_{i,t}\}$ and utility $u_{i,t}=x_{i,t}(v_{i,t}-d_{i,t})$. The goal is $\max \sum_t v_{i,t} x_{i,t}$ s.t. $\sum_t u_{i,t}\ge 0$ (RoS constraint).

Two protocols are compared:
- **Independent Bidding (Alg 1)**: Each bidder runs their own algorithm $A(H_{i,t})$ and bids against each other.
- **Coordinated Bidding (Alg 2)**: Only $i^*=\arg\max_i v_{i,t}$ bids $b_{i^*,t}=A(H_{i^*,t})$, while others bid 0.

The theory is split into two main threads: Section 3 discusses the necessary and sufficient distribution conditions for RoS (utility) improvement; Section 4 addresses (unconditional) value improvement and optimality under mirror-descent algorithms; Section 5 extends this to non-i.i.d. values.

### Key Designs

**1. Highest-Value Only (HVO) Mechanism: Eliminating Internal Friction with Minimal Coordination**

The primary pain point for an alliance is "same-root" bidders driving up clearing prices and breaking RoS constraints through internal competition. The authors observe that in a second-price auction, permitting only the member with the highest current value $i^*=\arg\max_i v_{i,t}$ to bid $b_{i^*,t}=A(H_{i^*,t})$—while others bid 0—completely removes internal competition. The mechanism itself is minimal: a central planner reads $\{v_{i,t}\}$ each round, identifies $i^*$, and lets its auto-bidding algorithm proceed as usual. This is effective because the other $N-1$ members no longer contribute to the competitive bid $d_{i^*,t}=\max\{d_t^O,\max_{j\neq i^*}b_{j,t}\}$; thus, $i^*$ faces only the external price $d_t^O$, saving substantial second-price payments. While the second-price auction is DSIC and bidding $v$ is a baseline, value-maximizers actively overbid ($b_{i,t}=(1+1/\lambda_{i,t})v_{i,t}>v_{i,t}$) to gain volume—this overbidding is exactly why they hurt each other in independent bidding. HVO removes this "mutual harm" term without requiring members to exchange private strategies beyond their values.

**2. Necessary and Sufficient Condition Transition Assumption 3.1: Characterizing "When" Coordination Reduces RoS Violation**

HVO is not unconditionally superior, so the authors provide a precise boundary for when "coordination strictly reduces the RoS violation for each bidder." Let $v_{(N)}, v_{(N-1)}$ be the highest and second-highest values among $N$ i.i.d. samples. Define

$$\Delta := \mathbb{E}_{F,D}\big[(v_{(N-1)}-d^O)_+ - (d^O - v_{(N)})_+\big]\ge 0,$$

Intuitively, this means the "advantage of the second-highest value over the external price" should exceed the "advantage of the external price over the highest value"—i.e., multiple alliance members are strong enough to suppress external competition. Two practical examples are provided: $N=4, F=D=U[0,1]$ yields $\Delta=1/6$; $N=3, F=U[0,1], D=\mathrm{Beta}(3,2)$ yields $\Delta=1/40$. It is proven that any full-support $F, D$ satisfies this as $N$ becomes sufficiently large. This conclusion is derived from two lemmas: Lemma 3.1 ($U^{\mathrm{Truth}}_i\ge U^{I,A}_i$ due to second-price DSIC) and Lemma 3.2 ($\mathbb{E}[U^{C,A}_i]\ge\mathbb{E}[U^{\mathrm{Truth}}_i]+T\Delta/N$). Together they form Theorem 3.1: when $\Delta\ge 0$, $\mathbb{E}[U^{C,A}_i-U^{I,A}_i]\ge T\Delta/N$ for any overbidding algorithm; conversely, when $\Delta<0$, the reverse is tight—there exist overbidding algorithms where coordination is detrimental. Such "necessary and sufficient" characterizations are rare in the auto-bidding literature.

**3. Coordinated Mirror Descent (MD-h) + Asymptotic Optimality: Pushing Value to the Optimum of All Coordination Mechanisms**

Comparing utility alone is insufficient, as platforms care about the "total value won per unit of budget spent." Thus, the third step proves value improvement and optimality under RoS conditions. Bidders use a dual multiplier $\lambda_{i,t}$ to control the overbid ratio, $b_{i,t}=(1+1/\lambda_{i,t})v_{i,t}$, and perform a Bregman projection after observing utility $g_{i,t}$: $\lambda_{i,t+1}=\arg\min_\lambda\{\alpha g_{i,t}\lambda+D_h(\lambda,\lambda_{i,t})\}$ (e.g., multiplicative update $\lambda_{i,t+1}=\lambda_{i,t}\exp(-\alpha g_{i,t})$ with entropy mirror). A key step is reducing the $N$-bidder coordination dynamics to "a virtual bidder seeing $v_{(N)}$": under coordination $\mathbb{E}[g_{i,t}\mid H_{t-1}]=G_{(N)}(\lambda_{i,t})/N$, where $G_{(N)}(\lambda)=\mathbb{E}[(v_{(N)}-d^O)\mathbb{I}[(1+1/\lambda)v_{(N)}>d^O]]$ is monotonically increasing. Thus, the active bidder's $\lambda$ converges to the root $\lambda_\star=\inf\{\lambda:G_{(N)}(\lambda)\ge 0\}$, and total value converges to $V_{(N)}(\lambda_\star)$ (Theorem 4.1). Using a Lagrange envelope to bound the total value of independent bidding at the same $V_{(N)}(\lambda_\star)$, and given $\lambda^C_{i,t}\to 0$ under Assumption 3.1, the authors prove coordinated MD is also superior to any other coordination mechanism (Theorem 4.2). This reduction back to mature single-bidder RoS mirror descent theory is the linchpin for both value improvement and optimality theorems.

### Loss & Training
There is no "training objective" in the traditional sense; the core "online optimization" is the MD-h update of $\lambda_{i,t}$ after observing $g_{i,t}$ in each round, with a learning rate $\alpha=1/\sqrt T$. The algorithm, similar to Feng et al. (2023), has known $O(\sqrt T \log T)$ RoS violation and $O(\sqrt T)$ value regret bounds, which are directly utilized here.

## Key Experimental Results

### Main Results
The authors compare independent (I) vs. coordinated (C) bidding on synthetic data (symmetric/asymmetric distributions, $N\in\{2,3,4,5\}$, $T\in\{4000,10000,20000\}$) and the public iPinYou Season 2 dataset (55 advertisers, 2.5M auction records), averaging over 100 runs. Table 1 (normalized by $T$):

| Setting | $N$ | $T$ | Util (I) | Util (C) | Value (I) | Value (C) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| i.i.d. $U[0,1] / U[0,0.9]$ | 2 | 4000 | -0.011 | **0.220** | 0.643 | **0.666** |
| i.i.d. $U[0,1] / U[0,1]$ | 4 | 4000 | -0.077 | **0.302** | 0.774 | **0.800** |
| i.i.d. $U[0,1] / \mathrm{Beta}(3,2)$ | 3 | 4000 | -0.049 | **0.153** | 0.712 | **0.748** |
| Non-i.i.d. | 5 | 20000 | -0.062 | **0.619** | 0.814 | **0.819** |
| iPinYou Real | 4 | 20000 | -0.040 | **0.155 ± 0.012** | 0.620 ± 0.016 | **0.928 ± 0.003** |
| iPinYou Real | 5 | 20000 | -0.065 | **0.172 ± 0.012** | 0.608 ± 0.012 | **0.958** |

Independent bidding results in nearly all negative per-capita utility (RoS constraints are broken), whereas coordination immediately raises it to +0.15 ~ +0.62. Value improves by 2%-4% on synthetic data and jumps from 0.62 to 0.93 (+50%) on real iPinYou data, because real-world external price distributions are more heavy-tailed, allowing HVO to save even more on second-price payments.

### Ablation Study
While there is no typical ablation, comparisons between i.i.d./non-i.i.d. settings and different external bid distributions ($D$) highlight the contributions of the HVO mechanism.

| Configuration | Utility Change | Value Change | Explanation |
| :--- | :--- | :--- | :--- |
| i.i.d. Symmetric | -0.05 $\rightarrow$ +0.22 | +2~4% | Theorems 3.1 & 4.1 hold simultaneously. |
| Non-i.i.d. Weak Symmetry | -0.01 $\rightarrow$ +0.22 | +0.4% | Theorem 5.1/5.2 holds but value gains are limited. |
| iPinYou Real Data | -0.04 $\rightarrow$ +0.15 | +50% | HVO saves significant payments with heavy-tailed $D$. |
| Assumption 3.1 Fails | — | — | Counter-examples show coordination can be harmful. |

### Key Findings
- HVO **without exception** turns negative utility in the independent mode into positive utility across all 6 synthetic and 2 real settings, validating the non-trivial nature of the $T\Delta/N$ improvement in Theorem 3.1.
- The magnitude of value improvement is determined by the **tail** of the external distribution: only 2%-4% under uniform distributions, but up to 50% under the heavy tails of iPinYou. This aligns with Theorem 4.1's analysis—the heavier the tail, the closer the asymptotic $\lambda_\star$ is to 0, increasing the overbid ratio.
- In non-i.i.d. settings, while value improvement for individual bidders is no longer guaranteed, the total alliance value consistently outperforms the independent mode, matching the predictions of Theorem 5.2 under Assumption 5.1.

## Highlights & Insights
- HVO serves as a minimal executable coordination protocol: it requires no shared private utility functions, no side payments, and no complex auction redesign. It can be deployed on existing third-party platforms with near-zero modification cost.
- Assumption 3.1 provides a clean "necessary and sufficient" condition for improvement, which can be upgraded to a high-probability guarantee of $1-\exp(-T\Delta^2/(32B^2N^2))$ using Azuma's inequality, making it easy to use for A/B trigger logic.
- The reduction of the $N$-bidder coordinated system to a virtual single-bidder problem seeing $v_{(N)}$ is a robust analytical pattern. It supports both Theorem 4.1 and 4.2 and could likely be reused for analyzing more complex mechanisms, such as partial coordination.

## Limitations & Future Work
- The external bid is assumed to be i.i.d., but in real platforms, the "external" world also consists of auto-bidders who react to the alliance's behavior. The authors list "auto-bidding external competitors" as an open problem.
- The coordination mechanism relies on a central planner being able to observe all $v_{i,t}$. This is reasonable for multiple advertisers managed by one agent but requires privacy protection for cross-advertiser alliances (not covered).
- The study only covers second-price auctions. In non-truthful auctions like first-price or GSP, truthful-bidding lemmas fail, and Lemma 3.1 is no longer valid, requiring a new construction for comparison terms.
- While counter-examples show when coordination fails (if Assumption 3.1 is not met), the paper does not provide design guidelines for "more robust" coordination mechanisms in non-balanced markets.

## Related Work & Insights
- **vs. Decarolis et al. (2020) / Romano et al. (2022)**: These works analyze coordination benefits and computational complexity for utility-maximizers under GSP/VCG. This paper focuses on value-maximizers, RoS, and repeated second-price auctions; the finding that "overbidding is the core problem" is a unique feature of this mechanism.
- **vs. Chen et al. (2023)**: Chen et al. discuss coordinated dynamic bidding under budget constraints. This paper replaces budget with RoS constraints and provides necessary/sufficient conditions for improvement; the two lines of work are complementary across the two major constraints for value-maximizers.
- **vs. Feng et al. (2023) / Balseiro et al. (2023)**: These works provide the baseline for single-bidder mirror-descent RoS auto-bidding. This paper reuses their learning rates and regret bounds, proving that the same algorithm achieves total value optimality when inserted into a coordination mechanism—a natural "algorithm-fixed, mechanism-changed" extension.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Learning Safe Strategies for Value Maximizing Buyers in Uniform Price Auctions](../../ICML2025/others/learning_safe_strategies_for_value_maximizing_buyers_in_uniform_price_auctions.md)
- [\[ACL 2025\] Value Residual Learning](../../ACL2025/others/value_residual_learning.md)
- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](../../AAAI2026/others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)
- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)
- [\[ICML 2025\] Prediction via Shapley Value Regression (ViaSHAP)](../../ICML2025/others/prediction_via_shapley_value_regression.md)

</div>

<!-- RELATED:END -->

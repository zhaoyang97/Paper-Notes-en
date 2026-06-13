---
title: >-
  [Paper Note] On the Coordination of Value-Maximizing Bidders
description: >-
  [ICML 2026][Value-Maximization] This paper formally investigates the "coordination" problem among multiple value-maximizing auto-bidders in online advertising. It proposes a simple coordination mechanism—"only allow the…
tags:
  - "ICML 2026"
  - "Value-Maximization"
  - "Auto-bidding"
  - "Second-Price Auction"
  - "RoS Constraint"
  - "Coordination Mechanism"
date: 2026-05-08
content_hash: 02db423a9b31b8c9
---

# On the Coordination of Value-Maximizing Bidders

**Conference**: ICML 2026  
**arXiv**: [2511.04993](https://arxiv.org/abs/2511.04993)  
**Code**: None  
**Area**: Online Advertising / Auto-bidding / Mechanism Design  
**Keywords**: Value-Maximization, Auto-bidding, Second-Price Auction, RoS Constraint, Coordination Mechanism

## TL;DR
This paper formally investigates the "coordination" problem among multiple value-maximizing auto-bidders in online advertising. It proposes a simple coordination mechanism—"only allow the coalition member with the highest value to bid, while others bid 0"—and proves that for a broad class of auto-bidding algorithms, this mechanism simultaneously reduces RoS violations for each member and drives the total coalition value to the asymptotic optimum of all coordination mechanisms.

## Background & Motivation

**Background**: The core paradigm of modern search and feed advertising is auto-bidding: advertisers delegate the optimization task of "maximizing total value under RoS (Return-on-Spend, at least one dollar of value for every dollar spent) constraints" to platforms or third-party agents. Algorithms use online learning methods like mirror descent or dual methods to determine bids per round. Most literature assumes each bidder independently optimizes their own objective.

**Limitations of Prior Work**: In reality, this independence assumption is fragile. Third-party agents often manage dozens of advertisers simultaneously, and large e-commerce entities (such as Amazon, Temu, or Shein) often use the same portfolio to bid on multiple similar ads. Such "related" bidders raising prices against each other for the same ad slot increases the clearing price and breaks RoS constraints, which is negative-sum for the coalition. However, current auto-bidding literature lacks theoretical characterization of "coordination"—with only a few empirical studies or works focused on utility-maximizers (e.g., Decarolis et al., Romano et al., Chen et al.).

**Key Challenge**: The behavior of value maximizers differs from utility maximizers. The former actively overbid (typically $b_{i,t}=(1+1/\lambda_{i,t}) v_{i,t} > v_{i,t}$) to capture volume. Therefore, the damage caused by "mutual overbidding" within a coalition is far more subtle than in classic cartel analysis and requires new modeling.

**Goal**: Decomposed into three sub-problems: (i) Can a simple coordination mechanism strictly outperform independent bidding? (ii) For which algorithms and distribution conditions does this "superiority" hold? (iii) Can it achieve optimality in the sense of known optimal rates for the algorithms?

**Key Insight**: The authors leverage a simple observation: in a second-price auction, if coalition members allow only the "current highest-value member" $i^* = \arg\max_i v_{i,t}$ to bid while others remain silent, the coalition completely eliminates internal bidding competition. Only this single representative faces the external competition $d_t^O$. This collapses the coupled dynamics of $N$ bidders into an equivalent single-bidder problem, providing space for subsequent proofs.

**Core Idea**: Replace independent bidding with a "Highest-Value Only" (HVO) coordination mechanism. This achieves RoS improvement (conditional) through the mechanism and distribution, and value improvement (unconditional for mirror-descent algorithms) simultaneously.

## Method

### Overall Architecture
The setting involves $T$ rounds of repeated second-price auctions. $N$ bidders in a coalition independently sample values $v_{i,t}\in[0,B]$ from the same continuous distribution $F$ each round $t$, and face an external competitive bid $d_t^O\sim D$ (which can also absorb the reserve price). The competitive bid for bidder $i$ is $d_{i,t}=\max\{d_t^O, \max_{j\neq i} b_{j,t}\}$, reward indicator $x_{i,t}=\mathbb{I}\{b_{i,t}\ge d_{i,t}\}$, and utility $u_{i,t}=x_{i,t}(v_{i,t}-d_{i,t})$. The goal is $\max \sum_t v_{i,t} x_{i,t}$ s.t. $\sum_t u_{i,t}\ge 0$ (RoS constraint).

Two compared protocols:
- **Independent Bidding (Alg 1)**: Each bidder runs their own algorithm $A(H_{i,t})$ and competes with others.
- **Coordinated Bidding (Alg 2)**: Only $i^*=\arg\max_i v_{i,t}$ bids $b_{i^*,t}=A(H_{i^*,t})$, while others bid 0.

The theory is split into two main lines: Section 3 discusses necessary/sufficient distribution conditions for RoS (utility) improvement; Section 4 discusses (unconditional) value improvement and optimality under mirror-descent algorithms; Section 5 extends to non-i.i.d. values.

### Key Designs

1. **Highest-Value Only (HVO) Mechanism**:
    - **Function**: A minimal coordination protocol without communication costs or private signal disclosure, eliminating internal bidding by $N-1$ coalition members for the same slot.
    - **Mechanism**: In each round, a central planner reads $\{v_{i,t}\}$, selects $i^*$, and lets their auto-bidding algorithm run as usual; Others set $b_{i,t}=0$. Since the other $N-1$ members no longer contribute to the $\max_{j\neq i^*} b_{j,t}$ term in $d_{i^*,t}$, the competitive bid faced by $i^*$ reduces directly to the external $d_t^O$, saving substantial second-price payments.
    - **Design Motivation**: Second-price is DSIC, and truthful bidding $v$ is a baseline; however, value-maximizers overbid, which is the source of their mutual harm in independent bidding. HVO removes this "mutual harm" term without requiring the exchange of private strategy information beyond values/bids.

2. **Necessary and Sufficient Distribution Condition (Assumption 3.1)**:
    - **Function**: Characterizes when "coordination strictly reduces the RoS violation for each bidder."
    - **Mechanism**: Let $v_{(N)}, v_{(N-1)}$ be the highest and second-highest of $N$ i.i.d. values. Define $\Delta := \mathbb{E}_{F,D}[(v_{(N-1)}-d^O)_+ - (d^O - v_{(N)})_+]\ge 0$. Intuition: The "advantage of the second-highest internal value over the external price" should exceed the "advantage of the external price over the highest value," meaning the coalition members are strong enough to suppress external competition. The authors provide typical examples: $\Delta=1/6$ for $N=4, F=D=U[0,1]$; $\Delta=1/40$ for $N=3, F=U[0,1], D=\mathrm{Beta}(3,2)$; and prove this holds for any full-support $F,D$ when $N$ is sufficiently large.
    - **Design Motivation**: Through Lemma 3.1 ($U^{\mathrm{Truth}}_i \ge U^{I,A}_i$ from second-price DSIC) and Lemma 3.2 ($\mathbb{E}[U^{C,A}_i] \ge \mathbb{E}[U^{\mathrm{Truth}}_i] + T\Delta/N$), the authors derive Theorem 3.1: $\mathbb{E}[U^{C,A}_i - U^{I,A}_i] \ge T\Delta/N$ when $\Delta\ge 0$ for any overbidding algorithm; the reverse is tight when $\Delta<0$. This is a rare necessary and sufficient characterization.

3. **Coordinated Mirror Descent (MD-h) + Asymptotic Optimality**:
    - **Function**: Beyond RoS conditions, proves mirror-descent algorithms strictly increase total coalition value under coordination and achieve asymptotic optimality among all coordination mechanisms under Assumption 3.1.
    - **Mechanism**: Bidders use dual multipliers $\lambda_{i,t}$ to control the overbid ratio, $b_{i,t}=(1+1/\lambda_{i,t})v_{i,t}$. After observing utility $g_{i,t}$, they perform a Bregman projection $\lambda_{i,t+1}=\arg\min_\lambda \{\alpha g_{i,t}\lambda + D_h(\lambda,\lambda_{i,t})\}$ (which reduces to the multiplicative update $\lambda_{i,t+1}=\lambda_{i,t}\exp(-\alpha g_{i,t})$ under entropy mapping $h(\lambda)=\lambda\log\lambda-\lambda$). The authors analyze dynamics in the $y_{i,t}=h'(\lambda_{i,t})$ space: under coordination, $\mathbb{E}[g_{i,t}\mid H_{t-1}]=G_{(N)}(\lambda_{i,t})/N$, where $G_{(N)}(\lambda)=\mathbb{E}[(v_{(N)}-d^O)\mathbb{I}[(1+1/\lambda)v_{(N)}>d^O]]$ is monotonically increasing. This causes the active bidder’s $\lambda$ to converge to the root $\lambda_\star=\inf\{\lambda:G_{(N)}(\lambda)\ge 0\}$, and the total value converges to $V_{(N)}(\lambda_\star)$ for the equivalent single-bidder problem (Theorem 4.1). Using the Lagrange envelope, the upper bound of the independent bidding value is also pushed to the same $V_{(N)}(\lambda_\star)$. Combined with $\lambda^C_{i,t}\to 0$ under Assumption 3.1, this proves coordinated MD outperforms any other coordination mechanism (Theorem 4.2).
    - **Design Motivation**: Comparing utility is insufficient; platforms/advertisers care about "how much value is won with spent budget." Reducing $N$-bidder coordination dynamics to a "virtual bidder observing $v_{(N)}$" is the key to connecting this complex multi-agent learning problem back to mature single-bidder RoS mirror descent theory, enabling the optimality proof of Theorem 4.2.

### Loss & Training
There is no "training objective" per se; the core "online optimization" is MD-h updating $\lambda_{i,t}$ after observing $g_{i,t}$ each round, with a learning rate $\alpha=1/\sqrt T$. The authors reuse the algorithm from Feng et al. (2023), which has known $O(\sqrt T \log T)$ RoS violation and $O(\sqrt T)$ value loss bounds.

## Key Experimental Results

### Main Results
The authors compared independent vs. coordinated bidding on synthetic data (symmetric/asymmetric distributions, $N\in\{2,3,4,5\}$, $T\in\{4000,10000,20000\}$) and public iPinYou Season 2 data (55 advertisers, 2.5M records), running 100 trials per setting. The table below shows Table 1 from the paper (normalized by $T$).

| Setting | $N$ | $T$ | Util (I) | Util (C) | Value (I) | Value (C) |
|------|-----|-----|----------|----------|-----------|-----------|
| i.i.d. $U[0,1] / U[0,0.9]$ | 2 | 4000 | -0.011 | **0.220** | 0.643 | **0.666** |
| i.i.d. $U[0,1] / U[0,1]$ | 4 | 4000 | -0.077 | **0.302** | 0.774 | **0.800** |
| i.i.d. $U[0,1] / \mathrm{Beta}(3,2)$ | 3 | 4000 | -0.049 | **0.153** | 0.712 | **0.748** |
| Non-i.i.d. | 5 | 20000 | -0.062 | **0.619** | 0.814 | **0.819** |
| iPinYou Real Data | 4 | 20000 | -0.040 | **0.155 ± 0.012** | 0.620 ± 0.016 | **0.928 ± 0.003** |
| iPinYou Real Data | 5 | 20000 | -0.065 | **0.172 ± 0.012** | 0.608 ± 0.012 | **0.958** |

Independent per-bidder utility is almost always negative (RoS broken), while coordination immediately pulls it to +0.15 ~ +0.62. Value improves by 2%-4% on synthetic data and jumps from 0.62 to 0.93 (+50%) on iPinYou, because real-world external bid distributions have heavier tails, making the HVO second-price savings more dramatic.

### Ablation Study
The paper lacks a typical ablation, but the comparison between i.i.d./non-i.i.d. and different $D$ illustrates the contribution of the HVO mechanism.

| Config | Utility Change | Value Change | Description |
|------|---------|---------|------|
| i.i.d. Symmetric Dist. | -0.05 → +0.22 | +2~4% | Theorem 3.1 + 4.1 hold simultaneously; cleanest case. |
| Non-i.i.d. Weakly Symm. | -0.01 → +0.22 | +0.4% | Theorem 5.1/5.2 hold but value gain is diminished. |
| iPinYou Real Data | -0.04 → +0.15 | +50% | Heavy-tailed external prices allow significant HVO payment savings. |
| Assumption 3.1 Fails (Theory) | — | — | Counter-examples show coordination can be detrimental. |

### Key Findings
- HVO corrected negative utility in **all** 6 synthetic and 2 real-world settings, validating the non-triviality of the $T\Delta/N$ improvement in Theorem 3.1.
- The magnitude of value improvement is determined by the **tail** of the external distribution: only 2%-4% under uniform distribution, but up to 50% under heavy-tailed iPinYou. This aligns with the Theorem 4.1 analysis of coordination converging to $V_{(N)}(\lambda_\star)$—the heavier the tail, the closer the asymptotic $\lambda_\star$ is to 0, and the larger the overbid ratio.
- In non-i.i.d. cases, while individual value improvement is no longer guaranteed, total coalition value consistently outperforms the independent mode, consistent with Theorem 5.2 under Assumption 5.1.

## Highlights & Insights
- HVO is a minimal actionable coordination protocol: it requires no transfer of private utility functions, no side payments, and no complex auction redesign. It can be launched on existing third-party platforms with near-zero modification costs.
- $\Delta\ge 0$ in Assumption 3.1 is a clean "necessary and sufficient condition" and can be elevated via Azuma’s inequality to a high-probability guarantee of $1-\exp(-T\Delta^2/(32B^2N^2))$, making it easy for engineering A/B triggers.
- Reducing the $N$-bidder coordination system to a virtual single-bidder problem using $v_{(N)}$ serves to prove both Theorem 4.1 and Theorem 4.2. This is a highly reusable analytical pattern that could be replicated for more complex coordination mechanisms (e.g., partial coordination).

## Limitations & Future Work
- Assumes external bids are i.i.d., but in real platforms, the "external" world also consists of auto-bidders who react to coalition behavior; the authors list "auto-bidding external bidders" as an open problem.
- The coordination mechanism relies on a central planner being able to read all $v_{i,t}$, which is reasonable when one agent manages multiple advertisers but requires privacy protection for cross-advertiser coalitions (not covered).
- Only covers second-price auctions; for non-truthful auctions like first-price or GSP, the truthful-bid lemma fails, and Lemma 3.1 does not apply, requiring a new construction for comparison terms.
- When Assumption 3.1 does not hold, the paper only provides counter-examples where coordination fails, without offering design guidance on "which coordination mechanism is more robust"—unbalanced markets remain an open problem.

## Related Work & Insights
- **vs Decarolis et al. (2020) / Romano et al. (2022)**: They analyze coordination benefits and computational complexity for utility-maximizers under GSP/VCG. This paper focuses on value-maximizers + RoS + repeated second-price, where "overbidding is core" is a completely new feature of the results.
- **vs Chen et al. (2023)**: Chen et al. discuss coordinated dynamic bidding under budget constraints. This paper switches to RoS constraints and provides necessary/sufficient improvement theorems; the two lines are complementary regarding the two major constraints for value-maximizers.
- **vs Feng et al. (2023) / Balseiro et al. (2023)**: These serve as the baseline for single-bidder mirror-descent RoS auto-bidding. This paper reuses their learning rates and regret bounds, showing that the same algorithm plugged into a coordination mechanism achieves optimal total value—a natural "fixed algorithm, changed mechanism" extension.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Key and Value Weights Are Probably All You Need: On the Necessity of the Query, Key, and Value Weight Triplet in Self-Attention](../../ICLR2026/others/key_and_value_weights_are_probably_all_you_need_on_the_necessity_of_the_query_ke.md)
- [\[NeurIPS 2025\] Faithful Group Shapley Value](../../NeurIPS2025/others/faithful_group_shapley_value.md)
- [\[AAAI 2026\] Extreme Value Monte Carlo Tree Search for Classical Planning](../../AAAI2026/others/extreme_value_monte_carlo_tree_search_for_classical_planning.md)
- [\[ICML 2026\] Local and Mixing-Based Algorithms for Gaussian Graphical Model Selection from Glauber Dynamics](local_and_mixing-based_algorithms_for_gaussian_graphical_model_selection_from_gl.md)
- [\[ICML 2026\] Return-to-Go is More Than a Number: Q-Guided Alignment for Return-Conditioned Supervised Learning](return-to-go_is_more_than_a_number_q-guided_alignment_for_return-conditioned_sup.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Best-of-Majority: Minimax-Optimal Strategy for Pass@k Inference Scaling
description: >-
  [ICLR 2026][learning_theory][Best-of-N] This paper formalizes LLM Pass@k inference (sampling $N$ candidates, submitting at most $k$, and evaluating the best one) as a regret minimization problem. It proves that both majority voting and Best-of-N are sub-optimal in this setting. The authors propose the Best-of-Majority (BoM) strategy—which "pre-screens by fre
tags:
  - ICLR 2026
  - learning_theory
  - Best-of-N
date: 2026-05-08
content_hash: 5a719153b55e5540
---
# Best-of-Majority: Minimax-Optimal Strategy for Pass@k Inference Scaling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=a6CVQpjbXq](https://openreview.net/forum?id=a6CVQpjbXq)  
**Code**: TBD  
**Area**: Learning Theory / Test-time Compute  
**Keywords**: Pass@k Inference, Inference Scaling, Minimax Optimality, Best-of-N, Majority Voting

## TL;DR
This paper formalizes LLM Pass@k inference (sampling $N$ candidates, submitting at most $k$, and evaluating the best one) as a regret minimization problem. It proves that both majority voting and Best-of-N are sub-optimal in this setting. The authors propose the Best-of-Majority (BoM) strategy—which "pre-screens by frequency and then selects top-$k$ by reward"—and provide a regret upper bound of $\tilde O(\epsilon_{\mathrm{opt}}+\sqrt{C^*\epsilon_{\mathrm{RM}}^2/k})$ with a matching lower bound, establishing the first minimax-optimal algorithm for Pass@k inference.

## Background & Motivation
**Background**: While scaling laws for training are well-studied, training remains expensive. Consequently, researchers have pivoted to "inference-time scaling"—improving performance by increasing sampling with a fixed model. The two most common aggregation strategies are **majority voting** (selecting the most frequent answer) and **Best-of-N** (BoN, selecting the answer with the highest reward model score).

**Limitations of Prior Work**: Evaluation in practice commonly uses the Pass@k metric—allowing the model to submit up to $k$ answers, succeeding if at least one is correct. However, existing inference scaling theories primarily analyze the "single answer output" case (Pass@1): Wu et al. (2024b) provided convergence rates for majority voting, and Huang et al. (2025a) gave upper bounds for BoN at $k=1$. Currently, **no work characterizes the dependence of regret on $k$**, nor answers "which strategy is optimal under Pass@k."

**Key Challenge**: Majority voting only utilizes the distribution information (frequency) of the reference policy $\pi_{\mathrm{ref}}$ and ignores rewards entirely. Conversely, BoN relies solely on the reward model $\hat r$ and ignores the distribution. The former is biased when $\pi_{\mathrm{ref}}$ assigns high probability to incorrect answers, while the latter suffers from reward overoptimization when the reward model is inaccurate. Neither source of information is fully utilized.

**Goal**: To address two fundamental questions—(Q1) What is the optimal scaling of Pass@k inference regret with respect to $k$ and $N$? (Q2) Which strategy achieves this optimal bound while remaining scaling-monotonic (performance does not degrade as $N$ increases)?

**Key Insight**: The authors employ the **pessimism** principle from reinforcement learning—for answers rarely sampled by $\pi_{\mathrm{ref}}$, supervision is sparse and error is high for the reward model, so the reward should not be blindly trusted.

**Core Idea**: Fuse the "frequency screening" of majority voting with the "reward ranking" of BoN—**first use frequency to filter out low-probability candidates where rewards are unreliable, then select the top-$k$ by reward from the credible subset**, thereby benefiting from both sources of information.

## Method

### Overall Architecture
The paper establishes a theoretical framework for the **Pass@k inference scaling problem**, proves the sub-optimality of existing strategies within this framework, proposes BoM, and proves its optimality.

The setting is as follows: given a prompt $x$, the reference policy $\pi_{\mathrm{ref}}(\cdot\mid x)$ independently samples $N$ candidate answers $\hat Y=\{\hat y_1,\dots,\hat y_N\}$. The algorithm selects at most $k$ answers to submit, aiming to make the maximum true reward among these $k$ as close to the optimal as possible. The formalized regret is:

$$\mathrm{Regret}(x)=\mathbb{E}_{\pi^*}\big[r^*(x,\cdot)\big]-\mathbb{E}_{y_1,\dots,y_k}\Big[\max_{1\le i\le k} r^*(x,y_i)\Big],$$

where $r^*:\mathcal X\times\mathcal Y\to[0,1]$ is the true reward and $\pi^*$ is the maximizer of $r^*$. The algorithm only has an **imperfect reward model** $\hat r$, characterized by two quantities: the overall mean squared error $\mathbb{E}_{y\sim\pi_{\mathrm{ref}}}[(r^*-\hat r)^2]\le\epsilon_{\mathrm{RM}}^2$, and the error at the optimal answer $\epsilon_{\mathrm{opt}}=|r^*(x,y^*)-\hat r(x,y^*)|$. In tasks like mathematics with a unique verifiable answer, $r^*$ degrades to a $\{0,1\}$ verifier, and minimizing regret is equivalent to maximizing the probability that "at least one of $k$ is correct," which is exactly Pass@k.

The BoM process is concise: **Sample $N$ candidates → Calculate empirical frequency $\hat\pi(y)$ for each answer → Discard candidates with frequency below a threshold $\alpha$ to obtain $\hat Y_\alpha$ → Sort by reward in $\hat Y_\alpha$ and take top-$k$**. The difference from majority voting is the subsequent reward ranking; the difference from BoN is the preceding frequency threshold—this threshold is the decisive factor in the theory.

### Key Designs

**1. Pass@k Regret Formalization and Coverage Coefficient: Compressing "Distribution Quality" into a Scalar**

Existing inference theories focus only on Pass@1. The first step here is to formulate the "submit $k$, take the best" mechanism into the regret metric and explicitly track its dependence on $k$ (Remark 3.3 highlights this as a new focus compared to sample-and-evaluate frameworks). To characterize the quality of the reference policy, the **L1 coverage coefficient** $C^*(x):=\mathbb{E}_{y\sim\pi^*}[\pi^*(y\mid x)/\pi_{\mathrm{ref}}(y\mid x)]$ is used. Under Assumption 3.2, where the optimal answer $y^*$ is unique and $\pi^*$ is one-hot, the coverage coefficient collapses to a simple quantity: $C^*(x)=1/\pi_{\mathrm{ref}}(y^*\mid x)$—the reciprocal of the probability that the reference policy samples the correct answer. A larger $C^*$ indicates a rarer correct answer and a harder problem; it serves as a "difficulty factor" in all subsequent bounds. Another key property is **scaling-monotonicity** (Definition 3.4): if the reward model is accurate enough and $N$ is large enough, regret can be pushed arbitrarily small and performance will not degrade as $N$ increases—this is a highly desirable property in practice for solving hard problems simply by increasing $N$.

**2. High-frequency Pre-screening + Top-$k$ Reward: Restricting Candidates to the "Credible Region" via Pessimism**

This is the algorithmic core of BoM (Algorithm 3). The pain point is that the error of the reward model $\hat r$ is uncontrollable on answers rare in $\pi_{\mathrm{ref}}$. BoN ranks all $N$ candidates by $\hat r$, essentially trusting everyone equally, which easily leads to being misled by an "overestimated bad answer." BoM's approach is to first calculate empirical frequencies $\hat\pi(y)=\frac1N\sum_{i=1}^N\mathbb 1(\hat y_i=y)$ (an estimate of $\pi_{\mathrm{ref}}$), retain only $\hat Y_\alpha=\{y\in\hat Y:\hat\pi(y)\ge\alpha\}$ with threshold $\alpha=3/(4C^*(x))$, and then select $\hat y_1,\dots,\hat y_k=\text{Top-}k\{y\in\hat Y_\alpha:\hat r(y)\}$. Why it works: high frequency implies $\pi_{\mathrm{ref}}(y\mid x)\ge\alpha$, which provides leverage for the reward error—the key inequality (5.1) in the proof:

$$\min_{i\in[k]}\Delta_i\le\sqrt{\textstyle\sum_{i=1}^k\Delta_i^2/k}\le\sqrt{\textstyle\sum_{i=1}^k\pi_{\mathrm{ref}}(\hat y_i\mid x)\Delta_i^2/(\alpha k)}\le\sqrt{\epsilon_{\mathrm{RM}}^2/(\alpha k)}$$

converts the reward error $\Delta_i=|\hat r-r^*|$ of each candidate into the overall error $\epsilon_{\mathrm{RM}}$ using the frequency lower bound $\alpha$, yielding a $1/\sqrt k$ decay. In short: **the frequency threshold keeps "unreliable rewards" out, so only the remaining rewards are used for ranking**. An added benefit is that rewards are only queried for the screened subset, saving reward model calls.

**3. Matching Upper and Lower Bounds: BoM is Minimax-Optimal**

The authors prove the algorithm is "optimal." Upper bound (Theorem 5.1): Setting $\alpha=3/(4C^*)$, then

$$\mathrm{Regret}(x)\le\epsilon_{\mathrm{opt}}(x)+O\!\Big(\sqrt{C^*(x)\epsilon_{\mathrm{RM}}^2(x)/k}\Big)+O\!\big(C^*(x)e^{-N/(32C^*(x))}\big).$$

When the sampling budget $N\ge 16C^*\log(kC^*/\epsilon_{\mathrm{RM}}^2)$ (i.e., $N=\tilde\Theta(C^*)$), the third term vanishes exponentially, yielding $\mathrm{Regret}\le\epsilon_{\mathrm{opt}}+\tilde O(\sqrt{C^*\epsilon_{\mathrm{RM}}^2/k})$. Lower bound (Theorem 6.1): For **any** Pass@k algorithm, there exists a hard instance such that $\mathrm{Regret}=\Omega(\epsilon_{\mathrm{opt}}+\sqrt{C^*\epsilon_{\mathrm{RM}}^2/k})$. The forms are identical, indicating BoM is minimax-optimal—no algorithm can perform better in the worst case. Two structural observations: the $\epsilon_{\mathrm{opt}}$ term does not decay with $k$ (reward error at the optimal answer is an unavoidable wall), while the $\epsilon_{\mathrm{RM}}$ term decays at $1/\sqrt k$ (submitting more candidates does make the problem easier). A technique in the upper bound proof involves a "squeeze" event $\mathcal E:Y_{1/C^*}\subset\hat Y_{3/(4C^*)}\subset Y_{1/(4C^*)}$ to ensure the screened set neither misses $y^*$ nor admits very low-frequency answers, using Chernoff bounds and a union bound over "buckets."

**4. Why Existing Strategies are not Scaling-monotonic: Constant Regret for Majority Voting and Degradation for BoN**

To highlight BoM, the authors prove lower bounds for two baselines. Majority Voting (Theorem 4.1): Even with a perfect reward model ($\epsilon_{\mathrm{RM}}=\epsilon_{\mathrm{opt}}=0$), by constructing multiple "bad answers" with higher probability than $y^*$, when $N\ge 9C^*\log(2k+2)$, the strategy gets stuck at $\mathrm{Regret}=\Omega(1)$—increasing $N$ or $k$ cannot help because it ignores rewards. BoN (Theorem 4.2): If $N\le C^*$, it is constant regret; otherwise, $\mathrm{Regret}=\Omega(\min\{1,\sqrt{N\epsilon_{\mathrm{RM}}^2/k}\})$—note that this bound **worsens as $N$ increases**, because more sampling increases the chance of hitting a bad answer overestimated by the reward model (reward overoptimization). Neither is scaling-monotonic. In contrast, BoM (Corollary 5.2) is scaling-monotonic: as $N\to\infty$ and $\epsilon_{\mathrm{RM}}\to0$, regret converges to 0.

## Key Experimental Results

### Main Results
The reference policy used is Qwen3-4B-Instruct-2507, and the reward model is AceMath-7B-RM. Comparisons between BoM, majority voting, BoN, and SBoN / SBoN(C) were conducted on GSM8K ($N=2000$), MATH-500, and AIME24 ($N=500$).

| Dataset | Setting | BoM | Majority Voting | BoN | Conclusion |
|--------|------|-----|----------|-----|------|
| MATH-500 | $k\in\{1,2,3,5,10\}$ | Leading throughout | Lower | Middle | BoM is optimal across all $k$ |
| GSM8K | Small $k$ | Leading | Lower | Close | BoM significantly better for small $k$ |
| AIME24 | Small $k$ | Leading | Lowest | Close | Significantly outperforms Majority Voting |

### Scaling with $N$ (Verifying Scaling-monotonicity)
On AIME24, fixing $k\in\{1,3,5\}$ and sweeping $N$ from 100 to 2000:

| Method | Behavior as $N$ increases | Corresponding Theory |
|------|------------------|----------|
| Majority Voting | Remains low, does not benefit from larger $N$ | Theorem 4.1 (Constant regret) |
| BoN | Tends to decrease initially then drop off | Theorem 4.2 (Non-monotonic) |
| BoM | Stable lead after $N\ge200$, no significant drop | Corollary 5.2 (Monotonic) |

### Ablation Study (Threshold $\alpha$, MATH-500 / Qwen3-4B)

| Pass@$k$ | 1 | 2 | 3 | 5 | 10 |
|----------|-----|-----|-----|-----|-----|
| $\alpha=0$ (=BoN) | 83.6 | 87.2 | 88.8 | 89.2 | 90.0 |
| $\alpha=0.005$ | 86.8 | 89.2 | 89.6 | 89.8 | 90.2 |
| $\alpha=0.015$ | 87.4 | 88.8 | 89.4 | 89.4 | 89.8 |
| Majority Voting | 86.4 | 88.0 | 88.4 | 88.8 | 90.0 |

### Key Findings
- When $\alpha=0$, BoM degrades to BoN, confirming that the frequency threshold is the sole critical switch differentiating BoM from BoN.
- Increasing $\alpha$ significantly improves performance for small $k$ (especially $k=1$, 83.6→87.4), but slightly harms large $k$ performance—stricter thresholds are more conservative, reducing the candidate set diversity.
- The behavior of the three methods with $N$ aligns perfectly with theory: Majority Voting is flat, BoN drops, and BoM is stable.

## Highlights & Insights
- **Frequency as a free estimator for $\pi_{\mathrm{ref}}$**: BoM does not require true access to $\pi_{\mathrm{ref}}$ probabilities; it uses empirical frequency $\hat\pi$ as a threshold, achieving the benefits of pessimism with zero extra engineering cost.
- **Collapsing coverage coefficient $C^*$ to $1/\pi_{\mathrm{ref}}(y^*)$**: By assuming a unique optimal answer, the abstract L1 coverage coefficient becomes intuitive, making the difficulty characterization clean.
- **Dual Selling Point of Matching Bounds + Scaling-monotonicity**: It not only provides an optimal algorithm but also pinpoint the failure reasons of the most common baselines (Majority Voting ignores reward, BoN is misled by overestimation) with hard instances.
- **Transferable Design Philosophy**: The "pessimistic filtering" of using distribution info to define a credible zone followed by value info for ranking can be transferred to any "generator + imperfect scorer" selection problem.

## Limitations & Future Work
- **Strong unique optimal answer assumption (Assumption 3.2)**: In many real-world tasks, there are multiple correct answers. If $\pi^*$ is not one-hot, the $C^*$ characterization does not hold.
- **Minimax (Worst-case) Perspective**: Lower bounds are derived from hard instances, reflecting worst-case scenarios which may be too pessimistic for typical instances; instance-dependent lower bounds are left for future work.
- **Coupling of threshold $\alpha$ and $C^*$**: Theory suggests $\alpha=3/(4C^*)$, but $C^*$ is unknown in practice. Experiments rely on manual tuning (range 0.003~0.015), implying a tuning burden.
- **Dependency on Reward Model Assumptions**: Analysis excludes the reward model's training; if $\epsilon_{\mathrm{RM}}$ fails the squared error assumption, the bounds may not hold.

## Related Work & Insights
- **vs Majority Voting (Wang et al. 2022, etc.)**: Majority voting ignores rewards. This paper proves its constant regret and non-monotonicity in the worst case; BoM adds reward ranking after frequency screening.
- **vs Best-of-N (Lightman et al. 2023; Huang et al. 2025a)**: BoN ignores distribution and degrades with large $N$ (overoptimization). This paper recovers the $\Omega(\min\{1,\sqrt{N\epsilon_{\mathrm{RM}}^2}\})$ bound for $k=1$ and reveals the $1/\sqrt k$ factor for $k>1$.
- **vs SBoN (Aminian et al. 2025; Verdun et al. 2025)**: SBoN uses softmax temperature $\beta$ to soften BoN. BoM uses a hard frequency threshold and provides minimax optimality guarantees.
- **vs Pass@k Training (Tang et al. 2025; Chen et al. 2025b)**: While those focus on training, this is the first theoretical characterization of Pass@k as an **inference** problem.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to formalize Pass@k inference and provide a minimax-optimal algorithm, revealing the $1/\sqrt k$ factor.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three math datasets with $k/N/\alpha$ ablations; theoretical predictions match experiments well.
- Writing Quality: ⭐⭐⭐⭐⭐ Comparison tables clearly explain regret and monotonicity for all methods.
- Value: ⭐⭐⭐⭐ Provides a principled algorithm and theoretical baseline for the "generator + imperfect reward" selection problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Minimax-Optimal Aggregation for Density Ratio Estimation](minimax-optimal_aggregation_for_density_ratio_estimation.md)
- [\[ICLR 2026\] Transformers as Measure-Theoretic Associative Memory: A Statistical Perspective and Minimax Optimality](transformers_as_measure-theoretic_associative_memory_a_statistical_perspective_a.md)
- [\[ICLR 2026\] Learning Shrinks the Hard Tail: Training-Dependent Inference Scaling in a Solvable Linear Model](learning_shrinks_the_hard_tail_trainingdependent_inference_scaling_in_a_solvable.md)
- [\[ICLR 2026\] A Near-Optimal Best-of-Both-Worlds Algorithm for Federated Bandits](a_near-optimal_best-of-both-worlds_algorithm_for_federated_bandits.md)
- [\[ICLR 2026\] Variational Inference for Cyclic Learning](variational_inference_for_cyclic_learning.md)

</div>

<!-- RELATED:END -->

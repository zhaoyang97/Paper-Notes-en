---
title: >-
  [Paper Note] Reinforcement Learning for Reachability: Guaranteeing Asymptotic Optimality
description: >-
  [ICML 2026][Reinforcement Learning][Reachability Specification] This paper addresses the problem of learning reachability specifications on unknown MDPs by proposing a direct learning algorithm that refines PAC parameter…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Reachability Specification"
  - "PAC Learning"
  - "Asymptotic Optimality"
  - "Bounded Value Iteration"
date: 2026-05-08
content_hash: 80f27c15878b0fa2
---

# Reinforcement Learning for Reachability: Guaranteeing Asymptotic Optimality

**Conference**: ICML 2026  
**arXiv**: [2605.24740](https://arxiv.org/abs/2605.24740)  
**Code**: https://github.com/amoghp214/asymptotic-ltl-reachability (Available)  
**Area**: Reinforcement Learning / Formal Methods / PAC Learning / Temporal Logic  
**Keywords**: Reachability Specification, PAC Learning, Asymptotic Optimality, Bounded Value Iteration

## TL;DR
This paper addresses the problem of learning reachability specifications on unknown MDPs by proposing a direct learning algorithm that refines PAC parameters in stages. It proves that there exists a finite stage $K_{\mathsf{opt}}$ almost surely, after which the algorithm only outputs optimal policies. This stage is explicitly characterized by intrinsic MDP parameters, and benchmarks empirically confirm that optimal policies appear within very few stages (median $k=2$).

## Background & Motivation

**Background**: Classical RL has a complete theoretical framework for reward-based discounted return objectives (asymptotic convergence of $Q$-learning, PAC bounds for $E^3$/RMAX). Recent works have generalized objectives from rewards to LTL/$\omega$-regular formal specifications to express complex temporal behaviors such as safety and liveness. Reachability is the core primitive for this class of specifications: all $\omega$-regular specifications can be reduced to reachability problems.

**Limitations of Prior Work**: For general LTL specifications, PAC learning has been proven infeasible (Yang 2022; Alur 2022) unless intrinsic MDP parameters (minimum transition probability $p_{\min}$, mixing time, expected distance, etc.) are introduced; however, these quantities are unknown in an RL setting. On the other hand, research on asymptotic convergence is largely limited to Le et al. 2024, which converts LTL into a limit-average reward and solves it using a sequence of discount factors $\gamma \to 1$. In that case, convergence is only characterized by "external parameters" (discount factors) and is decoupled from the original MDP structure, making it impossible to answer practical questions such as "when will the optimal policy emerge."

**Key Challenge**: There is a conflict between using PAC (which requires unrealistic prior knowledge of parameters like $p_{\min}$) and using asymptotic convergence (which is often a byproduct of reward reduction and lacks insight into convergence dynamics). Neither approach directly characterizes "when the optimal policy phase begins" under the original MDP parameters.

**Goal**: To provide a learning algorithm directly targeting reachability specifications without reward conversion, where the convergence stage can be explicitly expressed using intrinsic MDP quantities, and to demonstrate that this stage occurs early in standard benchmarks.

**Key Insight**: The authors observe a crucial fact: although $p_{\min}$ is unknown, it can be "guessed and refined in stages." By geometrically decaying the PAC parameters ($p_k$, $\delta_k$, $\varepsilon_k$) as $1/2^k$ in each stage, there will be a finite stage $K_{\mathsf{PAC}}$ after which $p_k \le p_{\min}$ holds indefinitely. Combined with the summable property $\sum \delta_k < \infty$ and the Borel–Cantelli lemma, "staged PAC" can be upgraded to "asymptotic optimality with probability 1."

**Core Idea**: Use a staged refinement PAC subroutine to approximate an unknown $p_{\min}$ such that the approximation tolerance $\varepsilon_k$ eventually falls below the value gap $\varepsilon_{\mathsf{diff}}$ between optimal and sub-optimal policies, thereby automatically upgrading $\varepsilon_k$-optimality to "strict optimality."

## Method

### Overall Architecture
The Asymptotic algorithm (Algorithm 1) runs in stages $k=1,2,3,\dots$. In stage $k$, three parameters are set as $p_k=\delta_k=\varepsilon_k=1/2^k$ (guessed minimum transition probability, confidence error, approximation tolerance):

1. Execute $N_k$ rollouts using a simulator to update transition counts $\#(s,a)$ and $\#(s,a,s')$, and construct a partial model $PM$.
2. Detect and collapse Maximal End Components (MECs) on $PM$ to obtain $PMC$.
3. Run Bounded Value Iteration (BVI) on $PMC$ to obtain a lower bound $L(s)$ and an upper bound $U(s)$ for the values.
4. Extract a memoryless deterministic policy $\pi_k$ from $L$ and $U$ as the output for that stage.

The inputs consist only of the MDP state/action spaces $S, A$, and the target set $G$, along with a simulator; $p_{\min}$, $K_{\mathsf{opt}}$, and $K_{\mathsf{PAC}}$ are not inputs and are used only for theoretical analysis.

### Key Designs

1. **Geometrically Decaying Triple Parameters + Conservative Transition Estimation**:
    - **Function**: Guarantees the existence of a finite stage $K_{\mathsf{PAC}}$ after which $p_k \le p_{\min}$ persists, allowing the repeated application of the PAC sub-theorems from Ashok et al.
    - **Mechanism**: Applies a Hoeffding deviation $c=\sqrt{\frac{\ln(\delta_P/2)}{-2\cdot\#(s,a)}}$ to the frequency $\frac{\#(s,a,s')}{\#(s,a)}$ to obtain a **lower estimate** $\hat P(s,a,s')=\max\{0,\frac{\#(s,a,s')}{\#(s,a)}-c\}$. Simultaneously, $p_k, \varepsilon_k, \delta_k$ are tightened as $1/2^k$, distributing the three types of errors $\delta_{TP}+\delta_{EC}+\delta_{N_k}=\delta_k$ into each stage.
    - **Design Motivation**: Since $p_{\min}$ is strictly required by PAC formulas but unknown in RL, a monotonic search via $1/2^k$ will eventually drop below the true value within finite steps. Since $\sum_k 1/2^k$ is summable, it fits the Borel–Cantelli requirement to upgrade "probably correct" to "almost surely correct."

2. **Optimal Policy Extraction based on MEC Collapse + BVI**:
    - **Function**: Obtains a credible interval $[L, U]$ and extracts a memoryless deterministic policy $\pi_k$ using only the partial model and lower estimate $\hat P$.
    - **Mechanism**: The BVI update rules $L(s,a)=\sum_{s'} \hat P(s,a,s')L(s')$ and $U(s,a)=\sum_{s'}\hat P(s,a,s')U(s')+(1-\sum_{s'}\hat P(s,a,s'))$ place all "unobserved probability mass" into $U$, ensuring $L(s)\le V(s)\le U(s)$. MECs are collapsed into super-states; MECs without targets are set to $L=U=0$, while those with targets are set to $L=U=1$ to prevent oscillation. Policies are extracted using $\mathsf{Best\_Action}(s)=\arg\max_a U(s,a)$.
    - **Design Motivation**: Reachability RL cannot simply use Q-learning because discounting confuses "reaching the target in $k$ steps" with "the true reachability probability." BVI + MEC collapse is a provably convergent algorithm for reachability that provides credible intervals when paired with a conservative $\hat P$.

3. **Three-stage Proof Chain (PAC → Strict Optimal → Probability 1)**:
    - **Function**: Upgrades "$\varepsilon_k$-optimal with probability $1-\delta_k$ per stage" to "almost surely outputting only optimal policies from some stage onwards."
    - **Mechanism**: Theorem 3.1 utilizes Ashok 2019's PAC lemma to show $\Pr[\pi_k\in \Pi_{\mathsf{opt}}^{\varepsilon_k}] \ge 1-\delta_k$ from $K_{\mathsf{PAC}}$ onwards. Theorem 3.2 notes that since there are finitely many memoryless deterministic policies, the value gap $\varepsilon_{\mathsf{diff}} > 0$. When $\varepsilon_k \le \varepsilon_{\mathsf{diff}}$, $\varepsilon_k$-optimality implies strict optimality ($K_{\mathsf{opt}}$). Theorem 3.3 applies Borel–Cantelli to yield the "almost sure" result. Theorem 4.1 explicitly lower bounds $\varepsilon_{\mathsf{diff}}$ as $(2D)^{-2|A||S|}\cdot 2^{-2|S|}$ using transition complexity $D$.
    - **Design Motivation**: This distinguishes the work from Le et al. 2024 by proving that the policy itself is optimal almost surely from a certain point, rather than just value convergence.

### Loss & Training
There is no explicit loss function (non-gradient method). Each stage uses a budget of $2^k \cdot |S|$ BVI updates on $PMC$. Rollouts use the previous stage's optimal policy with probability $1-\mu$ and random exploration with probability $\mu \in (0,1]$. MEC detection uses a $\delta_C$-confident policy requiring state-action counts $n \ge \ln \delta_C / \ln (1-p_k)$ to ensure exit edges are not missed.

## Key Experimental Results

### Main Results

| Metric | Convergence Stage (Median $k$) | Convergence Stage (Mean $k$) | Remarks |
| :--- | :--- | :--- | :--- |
| Policy Accuracy ($\Pi_{\mathsf{opt}}$ emergence) | 2 | 2.3 | Average of 9 benchmarks |
| Value Upper Bound $U(s_0)$ converges to 1.0 | $\sim 16$ | — | Dining Philosophers |
| Value Lower Bound $L(s_0)$ converges to 1.0 | $\sim 16$ | — | Dining Philosophers |
| Standard Deviation across trials | Low | — | Convergence is robust to randomness |

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
| :--- | :--- | :--- |
| Full Algorithm | Policy converges at $k=2$, value bounds at $k \sim 16$ | Matches theoretical policy emergence |
| Value Bounds only | Significant lag behind policy convergence | Actual $\varepsilon_{\mathsf{diff}}$ is much larger than the worst-case bound |
| Theoretical $N_k$ vs. Empirical $N_k$ | Similar convergence profiles | Truncation/pruning does not break asymptotic properties |
| Zero Reachability Target | Policy accuracy is 0 | Correct behavior: optimal reachability probability is 0 |

### Key Findings
- **Policy converges far earlier than value bounds**: Theoretically, $K_{\mathsf{opt}}$ is determined by the worst-case $\varepsilon_{\mathsf{diff}}$, but in real MDPs, the policy gap is much wider than the $(2D)^{-2|A||S|}$ bound. Thus, the optimal policy stabilizes at stage $k=2$ while bounds tighten slowly.
- **Robustness of Geometric Decay**: The $p_k=1/2^k$ sequence quickly passes the true $p_{\min}$ across all benchmarks without problem-specific tuning.
- **Reachability ⇒ LTL**: Reachability is a fundamental primitive. Any asymptotic algorithm for reachability immediately lifts to LTL, providing a foundational structure for all $\omega$-regular learning.

## Highlights & Insights
- **Turning "Unknown Parameters" into "Refined Parameters"**: A core technique to bypass PAC infeasibility. It admits that $p_{\min}$ is needed but handles it via a monotonic sequence and Borel–Cantelli.
- **Stronger-than-Asymptotic Convergence**: Unlike traditional asymptotic convergence ($J(\pi_n) \to J^*$), which allows infinite sub-optimal outputs, this work guarantees that sub-optimal policies appear only finitely many times.
- **Algebraic Characterization of $\varepsilon_{\mathsf{diff}}$**: Uses the integrality of $\det F(\mathrm{Id} - \mathbf{A})$ and Cramer's rule to link the policy value gap to the MDP transition complexity $D$.

## Limitations & Future Work
- **Ours** is model-based: maintaining a partial model $PM$ and counters is memory-intensive for large state spaces; model-free extensions are identified as future work.
- Restricted to **memoryless deterministic policies**: while these are optimal for reachability, the $\varepsilon_{\mathsf{diff}} > 0$ argument must be revisited for non-Markovian rewards or memory-dependent policies.
- The theoretical bound for $K_{\mathsf{opt}}$ is extremely loose compared to the experimental $k=2.3$.
- Evaluation is limited to 9 quantitative verification benchmarks; scalability to industrial-scale MDPs has not been tested.

## Related Work & Insights
- **vs. Le et al. 2024**: They use reward reduction and discount factor sequences for value convergence. Ours avoids reward conversion, focuses on reachability, and binds convergence to intrinsic quantities like $\varepsilon_{\mathsf{diff}}$.
- **vs. Ashok et al. 2019**: This work adapts their PAC lemma for the staged subroutines but removes the requirement that $p_{\min}$ must be known a priori.
- **vs. Alur et al. 2022**: They proved reachability cannot be reduced to discounted rewards while preserving optimality. This work provides a direct learning path that bypasses this impossibility.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)
- [\[NeurIPS 2025\] Learning in Stackelberg Mean Field Games: A Non-Asymptotic Analysis](../../NeurIPS2025/reinforcement_learning/learning_in_stackelberg_mean_field_games_a_non-asymptotic_analysis.md)
- [\[ICML 2026\] Safe In-Context Reinforcement Learning](safe_in-context_reinforcement_learning.md)
- [\[ICML 2026\] Distributional Inverse Reinforcement Learning](distributional_inverse_reinforcement_learning.md)
- [\[ICML 2026\] Learning in Structured Stackelberg Games](learning_in_structured_stackelberg_games.md)

</div>

<!-- RELATED:END -->

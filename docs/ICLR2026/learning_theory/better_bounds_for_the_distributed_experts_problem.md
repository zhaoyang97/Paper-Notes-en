---
title: >-
  [Paper Note] Better Bounds for the Distributed Experts Problem
description: >-
  [ICLR2026][Learning Theory][Distributed Experts Problem] This paper investigates the distributed online prediction problem where "experts are distributed across multiple servers and losses are aggregated across servers according to the $\ell_p$ norm." It proposes a suite of protocols based on exponential random variable embedding combined with geometric mean variance reduction. It is the first to handle general $\ell_p$ losses in the coordinator (message-passing) model…
tags:
  - "ICLR2026"
  - "Learning Theory"
  - "Online Learning"
  - "Communication Complexity"
  - "Distributed Experts Problem"
  - "$\\ell_p$ Loss"
  - "Geometric Mean Estimation"
date: 2026-05-08
content_hash: 0f29c668bc3df00b
---

# Better Bounds for the Distributed Experts Problem

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=6oiqYdYFn8](https://openreview.net/forum?id=6oiqYdYFn8)  
**Code**: https://github.com/samsonzhou/distributed-experts  
**Area**: Learning Theory / Online Learning / Communication Complexity  
**Keywords**: Distributed Experts Problem, Online Learning, Communication Complexity, $\ell_p$ Loss, Geometric Mean Estimation

## TL;DR
This paper investigates the distributed online prediction problem where "experts are distributed across multiple servers and losses are aggregated across servers according to the $\ell_p$ norm." It proposes a suite of protocols based on exponential random variable embedding combined with geometric mean variance reduction. It is the first to handle general $\ell_p$ losses in the coordinator (message-passing) model, achieving a communication complexity of $\left(\tfrac{n+s}{R^2}\right)\cdot\max(s^{1-2/p},1)\cdot\mathrm{polylog}(nsT)$ bits for a target regret $R$, outperforming prior work that was limited to $\ell_1$.

## Background & Motivation
**Background**: Online expert prediction is a classic framework where there are $n$ experts, and at each time $t\in[T]$, the algorithm selects an expert $i_t$ and observes the entire loss vector $(L_1(t),\dots,L_n(t))$. The goal is to minimize the cumulative regret relative to the "best expert in hindsight," $R=\frac1T\big(\sum_t L_{i_t}(t)-\sum_t L_{i^*}(t)\big)$. Exponential Weights Algorithm (EWA) and Multiplicative Weights Update (MWU) achieve optimal regret $O(\sqrt{\log n/T})$ under full information. However, when the number of experts $n$ and rounds $T$ are both large, centralizing all losses to run MWU becomes expensive in terms of both computation and communication.

**Limitations of Prior Work**: In real-world scenarios, expert losses are naturally "distributed." This paper focuses not on the streaming (memory-constrained) line of work, but on a complementary setting: expert costs are partitioned across $s$ servers, and a central coordinator aims to implement a low-regret algorithm while minimizing communication with the servers. Crucially, the true loss of expert $i$ at time $t$ is not explicitly given but must be aggregated across servers:

$$L_i(t)=\Big(\sum_{j\in[s]}\ell_i(j,t)^p\Big)^{1/p}$$

which is the $\ell_p$ norm of local losses $\ell_i(j,t)$ on $s$ servers. The coordinator cannot access $L_i(t)$ directly and must rely on servers executing a distributed protocol to (approximately) compute it.

**Key Challenge**: There is a trade-off between communication volume and regret—achieving smaller regret requires more frequent and fine-grained synchronization of losses from servers to the coordinator, leading to higher communication costs. Prior work by Jia et al. (2025) had two limitations: (1) its general $\ell_p$ protocol only applies to the broadcast/blackboard model (where messages are visible to all servers), whereas in the more realistic message-passing model, sending the same message to all servers incurs an $s$-fold overhead, presenting a different technical challenge; (2) in the message-passing model, it only solved $\ell_1$ (SUM) losses, with communication scales by an $O(Ts)$ term. $\ell_1$ is manageable because it is **additive** across servers—the total loss is simply the sum of individual losses, allowing for "sampling proportional to loss magnitude." However, $\ell_p$ for $p>1$ is sub-additive or super-additive, rendering these sampling techniques invalid.

**Goal**: To design protocols for general $\ell_p$ losses ($p\ge 1$) in the coordinator (message-passing) model that achieve near-optimal regret while keeping communication at a provably low level, characterizing a clean "communication-regret" trade-off curve.

**Key Insight**: Since $\ell_p$ losses are difficult to sample directly, the **max-stability** of exponential random variables is leveraged to "embed" the $\ell_p$ problem into an $\ell_\infty$ (maximum) problem. Finding a maximum in a distributed environment only requires reporting a few large values, which naturally saves communication.

**Core Idea**: Use exponential random variable scaling to transform $\ell_p$ losses into a "maximum" computation, then use a geometric mean estimator to suppress the unbounded variance introduced by exponential variables, and finally feed this low-variance loss estimate into MWU.

## Method

### Overall Architecture
The global objective is to estimate an approximate loss $\widehat{s_i}(t)\approx L_i(t)$ for each expert $i$ at each time $t$, using it as the MWU weight increment $w_i(t)=w_i(t-1)+\widehat{s_i}(t)$ to select experts with probability proportional to $\exp(-\eta w_i(t))$. The primary difficulty lies in "how to estimate an **unbiased and variance-controlled** $\widehat{s_i}(t)$ without explicitly obtaining $L_i(t)$ and with minimal communication."

The methodology progresses through three layers: first, a warm-up protocol (Algorithm 2, assuming losses fall within a constant interval $[a,b]$ to obtain a fixed regret); second, parameterizing it to achieve a communication-regret trade-off (Algorithm 3, using sampling probability $\varrho$); and finally, removing the $[a,b]$ assumption by allowing servers to report more aggressively based on finer thresholds to obtain the main result (Algorithm 4). All three share the same "exponential scaling + threshold reporting + geometric mean + MWU" framework, gradually relaxing assumptions and tightening communication bounds.

### Key Designs

**1. Exponential Random Variable Scaling: Mapping $\ell_p$ Loss to "Maximum"**

Proportional sampling fails for $\ell_p$ losses because they are non-additive. This paper bypasses this via the **max-stability** of exponential random variables (Lemma 2.3): assigning an independent rate-1 exponential variable $e_i(j,t)$ to each local loss $\ell_i(j,t)$ and scaling as $\ell_i(j,t)/e_i(j,t)^{1/p}$ gives:

$$\max_{j\in[s]}\frac{\ell_i(j,t)}{e_i(j,t)^{1/p}}\ \sim\ \frac{\big(\sum_{j\in[s]}\ell_i(j,t)^p\big)^{1/p}}{e^{1/p}}=\frac{L_i(t)}{e^{1/p}}$$

where $e$ is another rate-1 exponential variable. Thus, the distribution of the "maximum scaled value" is exactly the "true $\ell_p$ loss" divided by a factor from a known distribution. Since $\mathbb{E}[1/e^{1/p}]$ is computable, an unbiased estimate of $L_i(t)$ can be derived if servers report the **maximum scaled value** to the coordinator. This step converts the difficult "aggregate $\ell_p$" operation into "finding a maximum"—a task that requires minimal communication in distributed settings.

**2. Threshold Reporting: Efficiently Finding the Maximum**

Servers do not know the global maximum and cannot directly "send the max." The paper proves that, with high probability, the maximum scaled value will exceed a threshold of approximately $s^{1/p}$, and **very few scaled values exceed this threshold**. The protocol instructs each server to report $q_i^{(b)}(j,t)$ to the coordinator only if $q_i^{(b)}(j,t)\ge \tfrac{s^{1/p}}{100\log(nsT)}$ (Algorithm 2, lines 8–10). This ensures a high probability of capturing the maximum while limiting the number of reports per expert per round to $\mathrm{polylog}$ levels, reducing total communication to $O(sT)+nT\cdot\mathrm{polylog}(nsT)$. The cost is that if the maximum value is below the threshold, it is not reported, introducing a slight bias; however, this event only occurs with probability $1/\mathrm{poly}(nT)$, resulting in only low-order term perturbations to the regret.

**3. Geometric Mean Estimator: Suppressing Unbounded Variance**

While max-stability provides an unbiased estimate, the form $1/e^{1/p}$ has **unbounded variance** (the PDF of exponential variables does not decay fast enough for large values), which would cause MWU to fail. Borrowing from classic norm estimation techniques (Li, 2008), the paper takes the maximum of $B=\lceil 3/p\rceil$ independent scalings and computes their **geometric mean**:

$$\widehat{s_i}(t)=\prod_{b\in[B]}\Big(\max_{j\in[s]}q_i^{(b)}(j,t)\Big)^{1/B}$$

Lemma 3.2 proves there exists a constant $C_{3.2}\in(0,2B]$ such that $\mathbb{E}[Z]=C_{3.2}$ and $\mathbb{E}[Z^2]\le 3^B$, meaning the geometric mean compresses the second moment into a bounded small value. Combined with Lemma 3.3—which states that if the loss estimate $\widehat{s_i}(t)$ is approximately unbiased and has a second moment $\le\rho$, using it in MWU yields a regret of $O(\sqrt{\rho\log n/T})$—controlled variance directly translates to controlled regret. The authors emphasize that "using the geometric mean for variance reduction of exponential scaling estimates" is the core algorithmic and technical novelty compared to Jia et al. (2025), potentially having independent value for other applications of exponential variables.

**4. Probabilistic Sampling + Adaptive Thresholds: Characterizing the Trade-off**

To transition from "fixed regret" to "adjustable target regret $R$," Algorithm 3 has each server independently decide whether to participate in the round with probability $\varrho=\tfrac{1}{R^2T}$. The coordinator uses shared randomness to know which servers are sampled without additional communication. Estimates are multiplied by $1/\varrho$ to maintain unbiasedness. Lemma 4.2 gives $\mathbb{E}[\widehat{s_i}(t)]=L_i(t)+\tfrac{1}{\mathrm{poly}(nT)}$ with a second moment $\le O(TR^2)\cdot 3^B\cdot L_i(t)^2$, resulting in regret $O(Rs^{1/p}\sqrt{\log n})$ and communication $\left(\tfrac{n+s}{R^2}\right)\mathrm{polylog}(nsT)$ (Theorem 1.2). Algorithm 4 further removes the "constant interval" assumption: it selects a level $a$ with probability $\tfrac{\varrho}{2^a}$, instructing sampled servers to report more aggressively based on a smaller threshold $\tfrac{s^{1/p}}{100(2^a)^{1/p}\log(nsT)}$, using $a^*$ (the smallest integer satisfying $\widehat{s_i}(t)\ge (s/2^{a^*})^{1/p}$) for adaptive scaling (line 11, $w_i(t)\mathrel{+}=2^{a^*}R^2T\cdot\widehat{s_i}(t)$). This leads to the main communication result of $\left(\tfrac{n+s}{R^2}\right)\cdot\max(s^{1-2/p},1)\cdot\mathrm{polylog}(nsT)$ (Theorem 1.3), where the $\max(s^{1-2/p},1)$ factor characterizes the extra communication cost required for different $p$ values when the interval assumption is removed.

### Loss & Training
The foundation is standard MWU (Algorithm 1): maintain weights $w_i$, add losses each round, and select experts with probability proportional to $\exp(-\eta w_i)$ with a learning rate $\eta\approx 1/\sqrt T$. This paper does not modify MWU itself; all innovation lies in "how to feed it a high-quality loss estimate $\widehat{s_i}(t)$ in a distributed, low-communication manner." The theoretical lower bound $R\ge 1/\sqrt T$ is an information-theoretic hard limit (Cover, 1966), so aiming for $R\ge 1/\sqrt T$ is standard.

## Key Experimental Results

The experiments serve as a proof-of-concept, run on the HPO-B hyperparameter optimization benchmark. Different model classes are treated as experts, different datasets are assigned to servers, and each search step is a "day." The cost vector is the normalized negative accuracy of models on datasets. Comparators are offline MWU and the protocol from Jia et al. (2025).

### Main Results

| Dimension | Ours (Algorithm 2/3) | Baseline | Observation |
|----------|--------------------------|------|------|
| Communication vs $p$ | Changes with threshold | — | Communication increases with threshold for $p>1$; trend reverses for $p<1$. |
| Reward vs $p$ | Better than MWU | Offline MWU | Our reward is actually higher (attributed by authors to suboptimal MWU learning rate tuning). |
| Comm. vs Regret ($p=1$) | More efficient | Jia et al. (2025) | Communication is better than Jia et al. for $p=1$, replicating the predicted trade-off curve. |

### Comparison of Communication Complexity (Theoretical, Figure 1)

| Source | Loss Function | Communication Cost (bits) |
|------|----------|------------------|
| Jia et al. (2025) | $\ell_1$ | $\left(\tfrac{n}{R^2}+Ts\right)\cdot\mathrm{polylog}(nsT)$ |
| Ours (Theorem 1.2) | $\ell_p$ | $\left(\tfrac{n+s}{R^2}\right)\cdot\mathrm{polylog}(nsT)$ |

### Key Findings
- Even for $p=1$, a special case covered by Jia et al., this work improves the $O(Ts)$ dependency to a parameterized version of the regret $R$: e.g., for $R=O(1)$, Jia is still $O(Ts)$ while this work is $O(s)$, showing a significant gap over large $T$.
- The geometric mean estimator is crucial for achieving the regret bound—it constrains the second moment of the exponential scaling to $\mathbb{E}[Z^2]\le 3^B$, directly determining the second moment $\rho$ fed into MWU.
- The observation that reward exceeds offline MWU is somewhat counter-intuitive; the authors honestly attribute this to poor learning rate tuning for the baseline rather than the protocol being inherently "stronger."

## Highlights & Insights
- **Converting "Aggregation" to "Maximum Finding"**: The max-stability of exponential random variables is the soul of this paper—$\ell_p$ is hard to sample, but $\max$ requires only a few reports in distributed settings. This transformation simultaneously addresses the difficulty of aggregation and the need for communication efficiency.
- **Geometric Mean for Variance Reduction**: Exponential scaling provides an unbiased but high-variance estimate. The geometric mean (rather than arithmetic) precisely suppresses the second moment of such heavy-tailed estimates to the $3^B$ level, bridging the gap between an unbiased estimate and the bounded second moment requirement of MWU.
- **Pedagogical Three-Layer Structure**: Progressing from warm-up (fixed regret) to probabilistic sampling (communication-regret trade-off) to adaptive thresholds (removing interval assumptions) clearly illustrates the communication cost contributions of each assumption (especially the origin of the $\max(s^{1-2/p},1)$ factor).

## Limitations & Future Work
- The experiments are only a proof-of-concept on a small scale (single i7-3770, HPO-B benchmark) and do not verify communication savings in a real-world large-scale distributed system.
- The phenomenon of reward exceeding offline MWU was not strictly explained, and the authors attribute it to baseline tuning, suggesting the control was not optimal.
- The $\max(s^{1-2/p},1)$ factor implies extra communication costs for $p<2$ when removing the $[a,b]$ assumption; whether this is optimal or can be removed remains unproven.
- The authors suggest the geometric mean could be replaced by other variance reduction techniques but did not explore them; they list extensions to submodular objectives and $\ell_\infty$ (max) loss as future directions.

## Related Work & Insights
- **vs Jia et al. (2025) (Direct Comparison)**: They perform general $\ell_p$ in the broadcast/blackboard model (by iterating servers to find the max-loss server per expert) and only $\ell_1$ in the message-passing model (via proportional sampling). This paper is the first to solve general $\ell_p$ in the more realistic coordinator/message-passing model, using a fundamentally different technical route (exponential scaling + geometric mean vs. broadcast iteration/additive sampling) and improving communication even for $p=1$.
- **vs Streaming Online Learning (Srinivas 2022 / Woodruff 2023 / Braverman 2026, etc.)**: That line of work processes losses sequentially under memory constraints to maintain compact sketches. This paper is complementary—the coordinator can hold full expert information; the bottleneck is communication, not memory.
- **vs Exponential Scaling in Norm Estimation (Li 2008; Woodruff & Zhou 2021)**: These techniques were previously used for streaming norm estimation. This paper is the first to transplant them to the distributed online learning scenario, representing a successful cross-domain method migration.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to solve general $\ell_p$ distributed experts in the coordinator model; the combination of exponential scaling and geometric mean is a significant algorithmic innovation.
- Experimental Thoroughness: ⭐⭐⭐ Rigorous theory, but experiments are only small-scale proof-of-concept with loosely controlled baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Three-layer structure is very clear, precisely accounting for communication costs at each step.
- Value: ⭐⭐⭐⭐ Completes the $\ell_p$ puzzle for communication-regret trade-offs in distributed online learning; the methodology is applicable to other distributed estimation tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Two (narrow) heads are better than (an arbitrarily wide) one](two_narrow_heads_are_better_than_an_arbitrarily_wide_one.md)
- [\[ICLR 2026\] Adaptive Conformal Prediction via Mixture-of-Experts Gating Similarity](adaptive_conformal_prediction_via_mixture-of-experts_gating_similarity.md)
- [\[ICLR 2026\] On Coreset for LASSO Regression Problem with Sensitivity Sampling](on_coreset_for_lasso_regression_problem_with_sensitivity_sampling.md)
- [\[ICML 2026\] Robustness of Mixtures of Experts to Feature Noise](../../ICML2026/learning_theory/robustness_of_mixtures_of_experts_to_feature_noise.md)
- [\[ICLR 2026\] Revisiting Tree-Sliced Wasserstein Distance through the Lens of the Fermat–Weber Problem](revisiting_tree-sliced_wasserstein_distance_through_the_lens_of_the_fermatweber_.md)

</div>

<!-- RELATED:END -->

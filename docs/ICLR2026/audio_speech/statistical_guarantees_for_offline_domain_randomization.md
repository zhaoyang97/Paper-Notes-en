---
title: >-
  [Paper Note] Statistical Guarantees for Offline Domain Randomization
description: >-
  [ICLR 2026][Audio & Speech][domain randomization] This paper formalizes offline domain randomization (ODR) as a maximum likelihood estimation problem over a parameterized family of simulators. Under mild regularity and i…
tags:
  - "ICLR 2026"
  - "Audio & Speech"
  - "domain randomization"
  - "sim-to-real transfer"
  - "maximum likelihood estimation"
  - "consistency"
  - "offline RL"
date: 2026-05-08
content_hash: 74f53854581635f6
---

# Statistical Guarantees for Offline Domain Randomization

**Conference**: ICLR 2026
**arXiv**: [2506.10133](https://arxiv.org/abs/2506.10133)
**Code**: None
**Area**: Audio & Speech
**Keywords**: domain randomization, sim-to-real transfer, maximum likelihood estimation, consistency, offline RL

## TL;DR

This paper formalizes offline domain randomization (ODR) as a maximum likelihood estimation problem over a parameterized family of simulators. Under mild regularity and identifiability assumptions, it establishes weak consistency (convergence in probability); with an additional uniform Lipschitz continuity assumption, strong consistency (almost sure convergence) is further proved. These results provide the first theoretical foundation for the empirical success of ODR in sim-to-real transfer.

## Background & Motivation

**Background**: Reinforcement learning agents frequently suffer performance degradation when deployed from simulation to the real world—the so-called "sim-to-real gap." Domain Randomization (DR) is the dominant approach to address this issue: physical parameters (mass, friction coefficients, sensor noise, etc.) are randomly sampled during training to construct a diverse family of simulators, endowing the policy with robustness to environmental variation. DR has enabled zero-shot transfer in quadrotor flight, dexterous manipulation, legged locomotion, and other tasks.

**Limitations of Prior Work**:
   - **Inefficiency of Uniform DR (UDR)**: The standard practice applies broad uniform priors over physical parameters, but the theoretical analysis of Chen et al. (2022) shows that the sim-to-real gap of UDR scales as $O(M^3 \log(MH))$ with respect to the number of candidate simulators $M$—performance guarantees deteriorate rapidly as the simulator count grows.
   - **Neglect of available real-world data**: UDR makes no use of offline data already collected from the real system to guide the choice of parameter distribution.
   - **Lack of theoretical foundation**: Although ODR methods (e.g., DROPO, DROID, BayesSim) demonstrate substantial empirical advantages, it remains theoretically unknown (i) whether the fitted distribution converges to the true dynamics as data grows, and (ii) how much improvement is achieved relative to UDR.

**Key Challenge**: ODR performs well empirically, yet lacks statistical guarantees—it is unclear under what conditions offline data can reliably guide the selection of the domain randomization distribution.

**Goal**:
   - Prove weak consistency of the ODR estimator (convergence in probability to the true parameter).
   - Prove strong consistency of the ODR estimator (almost sure convergence).
   - Analyze the practicality of each assumption and provide relaxed variants.

**Key Insight**: ODR is treated as maximum likelihood estimation (MLE) over a parameterized simulator family, and classical statistical tools—uniform laws of large numbers in the Glivenko–Cantelli sense, the Borel–Cantelli lemma, etc.—are employed to establish rigorous convergence proofs.

**Core Idea**: ODR is essentially a parameterized MLE problem that admits provable statistical consistency under mild assumptions, providing a solid theoretical explanation for its empirical success.

## Method

### Overall Architecture

The central contribution of this paper is not a new algorithm but a theoretical framework for the existing ODR paradigm, organized in three layers:

- **Input**: An offline dataset $\mathcal{D} = \{(s_i, a_i, s_i')\}_{i=1}^N$ of i.i.d. transition tuples collected from the real environment $\mathcal{M}^*$.
- **Parameterization**: A simulator family $\mathcal{U} = \{\mathcal{M}_\xi : \xi \in \Xi \subset \mathbb{R}^d\}$ sharing state/action spaces but with transition probabilities governed by physical parameter $\xi$; the parameter distribution is $p_\phi(\xi) = \mathcal{N}(\mu, \Sigma)$.
- **Objective**: Maximize the mixture likelihood $\phi^* = \arg\max_\phi \sum_{i} \log \mathbb{E}_{\xi \sim p_\phi}[P_\xi(s_i' | s_i, a_i)]$.
- **Output**: The learned distribution $p^*(\xi)$ is used for downstream policy training $\pi_{\text{ODR}}^* = \arg\max_\pi \mathbb{E}_{\xi \sim p^*}[V_{\mathcal{M}_\xi}^\pi(s_1)]$.

### Key Designs

1. **MLE Formalization of ODR**:

    - Function: Reframes ODR as a structured maximum likelihood estimation problem.
    - Mechanism: The empirical log-likelihood is defined as $L_N(\phi) = \frac{1}{N}\sum_{i=1}^N \log q_\phi(s_i' | s_i, a_i)$, where $q_\phi(s'|s,a) = \int p_\xi(s'|s,a) p_\phi(\xi) d\xi$ is the mixture transition kernel. Via a KL divergence decomposition, it is shown that the unique maximizer of the population log-likelihood $L(\phi)$ is $\phi^* = (\xi^*, 0)$ (i.e., the distribution degenerates to the true parameter).
    - Design Motivation: Embedding ODR within the classical MLE framework enables direct application of established consistency tools from statistics.

2. **Weak Consistency (Theorem 1)**:

    - Function: Proves that any measurable maximizer $\hat{\phi}_N$ converges in probability to $\phi^*$.
    - Mechanism: The proof proceeds in three steps: (a) a uniform law of large numbers (ULLN) in the Glivenko–Cantelli class establishes $\sup_\phi |L_N(\phi) - L(\phi)| \to 0$ (Lemma 2); (b) the separation property of the unique maximizer yields a uniform likelihood-loss lower bound $\eta(\epsilon)$ for parameters deviating from $\phi^*$ (Lemma 3); (c) combining these, the probability that $\hat{\phi}_N$ falls outside the $\epsilon$-neighborhood of $\phi^*$ is controlled by $P(\sup |L_N - L| \geq \eta/3)$.
    - Required Assumptions: Assumption 1 (simulator regularity: bounded and continuous density), Assumption 2 (compact parameter space), Assumption 3 (mixture positivity: $q_\phi \geq c > 0$), Assumption 4 (identifiability).

3. **Strong Consistency (Theorem 2)**:

    - Function: Upgrades weak consistency to almost sure convergence.
    - Mechanism: An additional uniform Lipschitz assumption is introduced (Assumption 5): $|a(x,\phi) - a(x,\psi)| \leq L\|\phi - \psi\|$. Compactness of the parameter space is used to construct an $\epsilon/L$-net cover; Hoeffding's inequality yields an exponential probability bound at each grid point $P(|L_N(\phi_i) - L(\phi_i)| > \epsilon) \leq 2\exp(-N\epsilon^2 / 2\tilde{M}^2)$; the Borel–Cantelli lemma then gives $\sum_N P(\sup|L_N - L| > 2\epsilon) < \infty$, establishing almost sure convergence.
    - Distinction from Weak Consistency: Weak consistency requires only the ULLN (probability of convergence $\to 1$); strong consistency requires summability of probabilities (Borel–Cantelli), which the Lipschitz condition provides as a quantitative bridge from pointwise to uniform control.

4. **Definition of $\alpha$-Informativeness**:

    - Function: Defines the ability of an ODR algorithm to "concentrate" the distribution.
    - Mechanism: An algorithm $\mathcal{A}$ is called $(\alpha, \epsilon)$-informative if there exists $N_0$ such that for all $N \geq N_0$, the learned distribution $\hat{\phi}_N$ assigns at least probability mass $\alpha$ within the $\epsilon$-ball around the true parameter $\xi^*$. By strong consistency, Gaussian ODR is $\alpha$-informative for any $\alpha < 1$.
    - Significance: Provides a model-agnostic metric for evaluating and comparing different ODR algorithms.

### Assumption Analysis and Relaxation

The paper analyzes the practicality of each of the five theoretical assumptions and provides relaxations:

| Assumption | Original Form | Relaxation | Applicability |
|---|---|---|---|
| A1 Simulator regularity | Bounded and continuous density | No relaxation needed | Satisfied by finite state spaces and Gaussian transitions |
| A2 Compact parameter space | $\Phi$ compact | No relaxation needed | Physical parameters always have prior bounds |
| A3 Mixture positivity | $q_\phi \geq c > 0$ | Replaced by log-tail condition $P(\inf_\phi q_\phi(X) \leq \epsilon) \leq 1/\log(1/\epsilon)^2$ | Covers Gaussian and other light-tailed families |
| A4 Identifiability | Unique recovery of $\xi^*$ | Relaxed to convergence to identification set $\mathcal{Q}_\mu^*$ | Natural degradation under partial coverage |
| A5 Uniform Lipschitz | $\|a(x,\phi)-a(x,\psi)\| \leq L\|\phi-\psi\|$ | Sufficient if transition kernel is twice differentiable in $\xi$ with bounded gradient (Lemma 7) | Satisfied by smooth physics simulators |

## Key Experimental Results

### Comparison of Theoretical Results

This paper is a purely theoretical contribution; the core results are statistical guarantees rather than empirical performance. The following compares ODR theoretical results with existing UDR theory:

| Method | Convergence Type | Sim-to-real Gap | Data Requirement | Dependence on $M$ |
|---|---|---|---|---|
| UDR (Chen et al., 2022) | Non-adaptive | $O(M^3 \log(MH))$ | No offline data | Cubic growth |
| UDR (improved bound, this paper) | Non-adaptive | $O(M^3 \log(MH))$ (improved log factor) | No offline data | Cubic growth |
| ODR Weak Consistency (Thm 1) | In probability $\to \phi^*$ | Converges to 0 as $N$ grows | i.i.d. or ergodic | Related to identifiability of $\xi^*$ |
| ODR Strong Consistency (Thm 2) | Almost surely $\to \phi^*$ | Converges to 0 as $N$ grows | i.i.d. + Lipschitz | Additional Lipschitz control |

### Hierarchy of Assumptions and Guarantees

| Assumption Set | Guarantee Level | Convergence Mode | Key Tools |
|---|---|---|---|
| A1+A2+A3+A4 | Weak consistency | $\hat{\phi}_N \xrightarrow{P} \phi^*$ | ULLN (Glivenko–Cantelli) |
| A1+A2+A3+A4+A5 | Strong consistency | $\hat{\phi}_N \xrightarrow{a.s.} \phi^*$ | Hoeffding + Borel–Cantelli |
| A1+A2+A3 (no A4) | Set consistency | $\text{dist}(\hat{\phi}_N, \mathcal{Q}_\mu^*) \xrightarrow{P} 0$ | Berge's maximum theorem |
| A1+A2+relaxed A3 | Weak consistency | $\hat{\phi}_N \xrightarrow{P} \phi^*$ | Integrable envelope condition |

### Key Findings

- **Improved UDR gap bound**: Appendix A tightens Chen et al.'s $O(M^3 \log^3(MH))$ to $O(M^3 \log(MH))$ (reducing the logarithmic exponent from three to one) through more refined parameter selection.
- **Adaptive advantage of ODR**: By leveraging offline data to concentrate the distribution near the true parameter, ODR avoids the $M^3$ amplification effect arising from UDR's uniform coverage of the entire simulator family.
- **Sufficient condition for Lipschitz**: It suffices for the transition kernel $p_\xi$ to be twice differentiable in $\xi$ with bounded gradients ($|\nabla_\xi p_\xi| \leq G_1$, $|\nabla_\xi^2 p_\xi| \leq G_2$) to guarantee Assumption 5, with constant $L = (G_1 + G_2/2)/c$.

## Highlights & Insights

- **Elegant use of KL divergence decomposition**: By decomposing the population log-likelihood $L(\phi)$ as $-D_{KL}(p_{\xi^*} \| q_\phi) + H(\xi^*)$, the non-negativity and equality condition of KL divergence directly establishes $\phi^* = (\xi^*, 0)$ as the unique maximizer—an elegant and natural argument that seamlessly connects MLE with information theory.
- **Incremental upgrade strategy from weak to strong**: Weak consistency is established first (requiring only ULLN), then a single Lipschitz assumption upgrades it to strong consistency (requiring Borel–Cantelli), followed by an analysis of the relaxability of each assumption. This layered theoretical construction is transferable to other statistical estimation problems.
- **Model-agnostic definition of $\alpha$-informativeness**: A quality metric for ODR algorithms is proposed that is independent of the choice of parameter distribution (Gaussian being just one instance), enabling comparison across different ODR variants.
- **Concept of identification set**: When data coverage is incomplete, point convergence is not pursued; instead, set convergence $\text{dist}(\hat{\phi}_N, \mathcal{Q}_\mu^*) \to 0$ is established—this is information-theoretically optimal, as no method can distinguish parameters that are observationally equivalent under the data distribution.

## Limitations & Future Work

- **No finite-sample bounds**: The paper establishes only asymptotic consistency ($N \to \infty$) without providing concrete convergence rates or finite-sample error bounds—the practically important question of "how much data is needed" remains unanswered.
- **No experimental validation**: As a purely theoretical work, there is no empirical verification of the correspondence between theoretical predictions and actual performance on any simulation or robotics platform.
- **Restriction to Gaussian parameterization**: Although the authors claim that other parametric families can substitute, all proofs rely on properties specific to Gaussian $\mathcal{N}(\mu, \Sigma)$ (e.g., Lévy's continuity theorem for weak convergence); other families require independent verification.
- **Difficulty of verifying the identifiability assumption**: Assumption 4 requires unique recovery of $\xi^*$ from observations, yet it is difficult in practice to verify a priori whether a given simulator family satisfies this condition.
- **Optimization landscape not analyzed**: Consistency of the global maximizer is proved, but in practice MLE is solved via gradient optimization and may converge to local optima—the non-convexity of the objective function is not analyzed.
- **i.i.d. assumption remains strong**: Although relaxation to ergodic sequences is discussed, the concrete convergence proofs still rely on i.i.d. data.

## Related Work & Insights

- **vs. UDR (Chen et al., 2022)**: UDR trains under a uniform prior with a gap of $O(M^3 \log(MH))$; ODR uses offline data to fit a concentrated distribution, asymptotically eliminating the gap—ODR is strictly superior to UDR given sufficient data.
- **vs. DROPO (Tiboni et al., 2023)**: DROPO is a concrete algorithmic instantiation of ODR (Gaussian MLE + gradient-free optimizer); this paper provides statistical guarantees for the MLE-ODR paradigm that encompasses DROPO—serving as a bridge between theory and practice.
- **vs. BayesSim (Ramos et al., 2019)**: BayesSim employs a conditional density estimator to predict the parameter posterior, representing a Bayesian approach; the frequentist MLE analysis of this paper is complementary and may inspire Bayesian consistency studies for ODR.
- **vs. DROID (Tsai et al., 2021)**: DROID uses CMA-ES to optimize an $L_2$ distance for system identification; the MLE objective in this paper is theoretically better justified, as it directly optimizes data likelihood.

## Rating

- Novelty: ⭐⭐⭐⭐ — The first theoretical work to provide statistical consistency guarantees for ODR, filling an important gap.
- Experimental Thoroughness: ⭐⭐ — A purely theoretical contribution with no experimental validation, finite-sample bounds, or practical benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ — Proof structure is clear, each assumption is analyzed individually with relaxations provided; an exemplary theoretical paper.
- Value: ⭐⭐⭐⭐ — Provides a theoretical foundation for ODR methods in the sim-to-real literature, though the absence of experiments and finite-sample analysis limits direct practical guidance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Alexandria: A Multi-Domain Dialectal Arabic Machine Translation Dataset for Culturally Inclusive and Linguistically Diverse LLMs](../../ACL2026/audio_speech/alexandria_a_multi-domain_dialectal_arabic_machine_translation_dataset_for_cultu.md)
- [\[ICLR 2026\] SyncTrack: Rhythmic Stability and Synchronization in Multi-Track Music Generation](synctrack_rhythmic_stability_and_synchronization_in_multi-track_music_generation.md)
- [\[ICLR 2026\] MMSU: A Massive Multi-task Spoken Language Understanding and Reasoning Benchmark](mmsu_a_massive_multi-task_spoken_language_understanding_and_reasoning_benchmark.md)
- [\[ICLR 2026\] Pay Attention to CTC: Fast and Robust Pseudo-Labelling for Unified Speech Recognition](pay_attention_to_ctc_fast_and_robust_pseudo-labelling_for_unified_speech_recogni.md)
- [\[ICLR 2026\] Toward Complex-Valued Neural Networks for Waveform Generation](toward_complex-valued_neural_networks_for_waveform_generation.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Reevaluating Policy Gradient Methods for Imperfect-Information Games
description: >-
  [ICLR 2026][Reinforcement Learning][PPO] The authors propose the "Policy Gradient Hypothesis" – given proper hyperparameter tuning, general policy gradient methods such as PPO and PPG are not inferior (and often superior) to specialized game-theoretic algorithms based on Fictitious Play, Double Oracle, or Counterfactual Regret Minimization in two-player zero-
tags:
  - ICLR 2026
  - Reinforcement Learning
  - PPO
date: 2026-05-08
content_hash: e844bed7adf2a03f
---
# Reevaluating Policy Gradient Methods for Imperfect-Information Games

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=vClBDezZUo](https://openreview.net/forum?id=vClBDezZUo)  
**Code**: [IIG-RL-Benchmark](https://github.com/nathanlct/IIG-RL-Benchmark) / [exp-a-spiel](https://github.com/gabrfarina/exp-a-spiel)  
**Area**: Reinforcement Learning / Game Theory  
**Keywords**: Imperfect-information games, Policy Gradient, Exploitability, Self-play, PPO

## TL;DR
The authors propose the "Policy Gradient Hypothesis" – given proper hyperparameter tuning, general policy gradient methods such as PPO and PPG are not inferior (and often superior) to specialized game-theoretic algorithms based on Fictitious Play, Double Oracle, or Counterfactual Regret Minimization in two-player zero-sum imperfect-information games. To verify this, they open-sourced tools for calculating exact exploitability in five large games for the first time and conducted the largest-scale comparative experiment to date (7000+ runs), with results overwhelmingly supporting the hypothesis.

## Background & Motivation
**Background**: In two-player zero-sum imperfect-information games (IIGs, e.g., Poker, Dark Hex), the standard metric for policy quality is **exploitability**—how much utility a worst-case opponent can extract. Zero exploitability signifies a Nash equilibrium. Over the past decade, the mainstream approach has been porting tabular algorithms with convergence guarantees to deep reinforcement learning: Fictitious Play (FP, e.g., NFSP), Double Oracle (DO, e.g., PSRO), Counterfactual Regret Minimization (CFR, e.g., ESCHER), and regularization routes (R-NaD).

**Limitations of Prior Work**: These specialized algorithms possess significant drawbacks. FP/DO require calculating a best response in each iteration, which is equivalent to solving an entire RL problem, and suffer from slow convergence. Porting CFR to model-free RL requires importance sampling of trajectories, leading to high-variance feedback that is difficult for function approximators to learn. Furthermore, none of these three exhibit last-iterate convergence, requiring the additional complexity of "extracting" a policy from the historical average.

**Key Challenge**: Simultaneously, simple "naive self-play" has been thought to fail catastrophically in IIGs—imperfect information induces cyclical best-response dynamics (rock-paper-scissors loops), causing single-agent algorithms in self-play to often fail to converge, diverge, or yield maximally exploitable strategies. Consequently, the research community formed a **belief** that general methods (especially PPO) are ineffective in IIGs and that game-theoretic algorithms must be employed. PPO has long appeared in literature only as a "weak baseline."

**Key Insight**: However, Magnetic Mirror Descent (MMD) by Sokota et al. (2023) challenged this impression. MMD is essentially a policy gradient algorithm that achieves performance comparable to CFR. The authors analyze MMD through its components: its updates perform three tasks—maximizing expected return, regularizing the policy, and controlling the update step size. Since TRPO, PPO, and PPG already perform these three tasks, the authors argue that if MMD is effective, general PG methods sharing the same lineage should also be effective.

**Core Idea**: The authors formally propose the **Policy Gradient Hypothesis**—"Appropriately tuned general policy gradient methods in the spirit of MMD can compete with or exceed deep RL methods based on FP/DO/CFR in two-player zero-sum IIGs." They test this using a credible large-scale experimental setup. This work is not about proposing a new algorithm but is an empirical "reevaluation and benchmarking" effort.

## Method

### Overall Architecture
The "method" of this paper is not a new model but an **empirical infrastructure and experimental protocol** designed to fairly test the hypothesis. It consists of three core components: (1) Clarifying the hypothesis by decomposing the isomorphism between MMD and general PG; (2) Overcoming the measurement hurdle by implementing and open-sourcing exact exploitability calculations for five large-scale IIGs; (3) Designing a two-stage experimental protocol without maximization bias to evaluate seven algorithms on a level playing field. These components are interdependent: without exact exploitability, there is no credible benchmark, and without the bias-free protocol, the conclusions would lack authority.

### Key Designs

**1. Policy Gradient Hypothesis: Decomposing MMD into "General PG Components"**

The authors present the MMD update rule and map each term to PPO components to argue for their equivalence. The MMD update is:

$$\pi_{t+1} = \arg\max_{\pi}\; \mathbb{E}_{A\sim\pi}\, q(A) \;-\; \alpha\,\mathrm{KL}(\pi,\rho)\;-\;\frac{1}{\eta}\,\mathrm{KL}(\pi,\pi_t),$$

where $q(A)$ is the action value, $\alpha$ is the regularization temperature, $\eta$ is the step size, and $\rho$ is the "magnet" policy. The three terms respectively: maximize expected return ($\mathbb{E}_{A\sim\pi}q(A)$); regularize the updated policy ($-\alpha\mathrm{KL}(\pi,\rho)$, approximating an entropy reward when $\rho$ is uniform); and constrain the update step size ($-\frac{1}{\eta}\mathrm{KL}(\pi,\pi_t)$). The authors point out that TRPO, PPO, and PPG already perform these functions (maximizing value + encouraging entropy + controlling step size). The primary difference lies in technical details, such as MMD's use of reverse KL for step-size control versus gradient clipping or forward KL in common PG methods. The authors hypothesize that while these differences matter in tabular settings with extreme precision, they are likely not critical in deep learning settings. Thus, the perceived failure of general PG in literature may stem from **hyperparameter tuning or confirmation bias** rather than inherent flaws.

**2. exp-a-spiel: Exact Exploitability for Five Large-Scale Games**

A major obstacle to testing the hypothesis is that exact exploitability implementations for large IIGs are rarely available, even in libraries like OpenSpiel. Researchers often rely on unreliable proxies like "exploitability in small games" or "head-to-head performance." Results in small games (e.g., Leduc Hold'em) do not necessarily generalize to large games where agents must learn strong policies for unvisited information states. Head-to-head results are difficult to interpret due to non-transitivity. The authors implemented exact exploitability for five large games: 2D5F Liar's Dice, 3x3 Dark Hex, Phantom Tic-Tac-Toe, 3x3 Abrupt Dark Hex, and Abrupt Phantom Tic-Tac-Toe, which contain millions of information states. Exploitability is defined as:

$$\mathrm{expl}(\pi)=\frac{\max_{\pi_1'}J(\pi_1',\pi_2)-\min_{\pi_2'}J(\pi_1,\pi_2')}{2},$$

representing the expected utility a worst-case opponent can gain. The implementation utilizes **sequence-form representation** and **dynamic payoff matrix generation** to minimize memory usage. For Liar's Dice, dynamic programming is used to calculate utilities in linear time. For the 3x3 variants, computation is distributed across threads with 18 buffers to avoid concurrency conflicts. Calculations take 30–90 seconds on standard hardware.

**3. Two-Stage, Bias-Free Experimental Protocol**

To ensure fairness, especially since the authors **hypothesized that specialized algorithms would lose**, they adopted a rigorous two-stage launch. The first is the **tuning launch**: for every combination of 5 games and 7 algorithms, 50 hyperparameter configurations were tested with 3 seeds each for 10 million steps. Hyperparameters were sampled by perturbing default values. Entropy coefficients for all PG methods were fixed to the values from Sokota et al. (2023). The second is the **evaluation launch**: the top 5 configurations with the lowest mean last-iterate exploitability from the tuning phase were re-run with 10 entirely new seeds. This **re-evaluation with new seeds** eliminates selection/maximization bias, ensuring that performance is not overestimated due to overfitting the tuning seeds. The experiment involved over 7000 runs and 345,000 CPU hours.

## Key Experimental Results

### Main Results
Seven algorithms (NFSP, PSRO, ESCHER, R-NaD, MMD, PPO, PPG) were compared on last-iterate exploitability and head-to-head expected return.

| Category | Algorithm | Exploitability Performance | Conclusion |
|----------|----------|---------------------------|------------|
| General PG | MMD / PPO / PPG | Lowest and roughly equal | Strongest |
| Regularized | R-NaD | Occasionally close to PG, usually worse | Weaker |
| FP-based | NFSP | Occasionally close, mostly lagging | Weaker |
| CFR-based | ESCHER | Consistently uncompetitive | Among the weakest |
| DO-based | PSRO | Mostly ineffective, rare success in DH3 | Among the weakest |

Head-to-head conclusions align with exploitability: general PG methods tie with each other and typically defeat other algorithms.

### Game Sizes
The scale of the five benchmarks (demonstrating they are "large" enough):

| Game | States | Info States | Nash Value |
|------|--------|-------------|------------|
| LD2D5F | 235.9M | 15.73M | 0.0177 |
| DH3 | 19.12B | 6.07M | 1.0000 |
| ADH3 | 29.31B | 27.33M | 0.3844 |
| PTTT | 19.93B | 5.99M | 0.6667 |
| APTTT | 27.12B | 23.31M | 0.5502 |

### Key Findings
- **Overwhelming Evidence**: FP/DO/CFR algorithms not only failed to outperform general PG but were largely "not competitive."
- **Internal Consistency of General PG**: MMD, PPO, and PPG performed similarly, suggesting the advantage resides in the "general PG ethos" rather than specific algorithmic nuances.
- **Large Games vs. Small Games**: Success in large games depends on generalization across massive state spaces rather than exact convergence in small state spaces, explaining why specialized algorithms struggle.

## Highlights & Insights
- **Isomorphism Analysis**: By decomposing MMD into value maximization, regularization, and step-size control, the authors turned a mystery into a falsifiable hypothesis.
- **Infrastructure as a Contribution**: Providing exact exploitability tools for large games allows the community to move beyond proxy metrics toward reproducible comparisons.
- **Rigorous Experimental Integrity**: The use of a two-stage launch with separate tuning and evaluation seeds proactively addresses potential concerns regarding unfair tuning of baselines.
- **Transferable Methodology**: When a specific class of methods is widely dismissed in a field, aligning an outlier success (like MMD) with the dismissed methods (like PPO) can reveal if the failure was fundamental or merely due to tuning bias.

## Limitations & Future Work
- **Game Diversity**: The experiments were limited to five games, four of which are somewhat homogeneous variants. The authors do not claim these rankings generalize to all IIGs.
- **Algorithm Representatives**: Only one representative was chosen for each category (e.g., PSRO for DO). Various modern variants of these algorithms might yield different results.
- **Fixed Constraints**: Results depend on the chosen hyperparameter ranges, 10-million-step budget, and network architecture. Specialized algorithms might require different scales of hyperparameters or longer training times to reach their potential.
- **Honest Restraint**: The authors explicitly state they have not "proven" the hypothesis to be true, only that they have provided strong evidence for it.

## Related Work & Insights
- **vs. MMD (Sokota et al., 2023)**: MMD is the spiritual prototype; this paper argues that because MMD works, general PG should also work.
- **vs. CFR/ESCHER**: While ESCHER addresses variance in CFR, it requires a uniform behavior policy, which may fail to cover critical trajectories in these large games.
- **vs. FP/DO (NFSP/PSRO)**: These require slow iterative best-response calculations and lacks last-iterate convergence, making them less efficient in deep learning settings.
- **vs. Naive Self-play**: While traditionally thought to diverge, this work shows that properly tuned general PG with entropy and step-size control is not only stable but state-of-the-art, echoing findings in cooperative IIGs like those in Yu et al. (2022).

## Rating
- Novelty: ⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Look-ahead Reasoning with a Learned Model in Imperfect Information Games](look-ahead_reasoning_with_a_learned_model_in_imperfect_information_games.md)
- [\[ICLR 2026\] SPG: Sandwiched Policy Gradient for Masked Diffusion Language Models](spg_sandwiched_policy_gradient_for_masked_diffusion_language_models.md)
- [\[ICLR 2026\] On the $O(1/T)$ Convergence of Alternating Gradient Descent-Ascent in Bilinear Games](on_the_o1t_convergence_of_alternating_gradient_descent-ascent_in_bilinear_games.md)
- [\[ICLR 2026\] Nearly-Optimal Bandit Learning in Stackelberg Games with Side Information](nearly-optimal_bandit_learning_in_stackelberg_games_with_side_information.md)
- [\[NeurIPS 2025\] On the Global Optimality of Policy Gradient Methods in General Utility Reinforcement Learning](../../NeurIPS2025/reinforcement_learning/on_the_global_optimality_of_policy_gradient_methods_in_general_utility_reinforce.md)

</div>

<!-- RELATED:END -->

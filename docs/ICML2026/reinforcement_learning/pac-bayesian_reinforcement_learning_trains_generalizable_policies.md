---
title: >-
  [Paper Note] PAC-Bayesian Reinforcement Learning Trains Generalizable Policies
description: >-
  [ICML2026][Reinforcement Learning][PAC-Bayes bounds] Ours presents the first PAC-Bayesian RL generalization bound that **explicitly depends on the mixing time of Markov chains** and reduces the lung-horizon $1/(1-\gamma)…
tags:
  - "ICML2026"
  - "Reinforcement Learning"
  - "PAC-Bayes bounds"
  - "mixing time"
  - "Soft Actor-Critic"
  - "deployment certificates"
  - "posterior-guided exploration"
date: 2026-05-08
content_hash: d6ccac090c5222ed
---

# PAC-Bayesian Reinforcement Learning Trains Generalizable Policies

**Conference**: ICML2026  
**arXiv**: [2510.10544](https://arxiv.org/abs/2510.10544)  
**Code**: None  
**Area**: Reinforcement Learning / Generalization Theory / PAC-Bayes  
**Keywords**: PAC-Bayes bounds, mixing time, Soft Actor-Critic, deployment certificates, posterior-guided exploration  

## TL;DR
Ours presents the first PAC-Bayesian RL generalization bound that **explicitly depends on the mixing time of Markov chains** and reduces the lung-horizon $1/(1-\gamma)$ dependency to a linear scale. By embedding this as an "alive" training objective within SAC, the resulting PB-SAC algorithm provides non-vacuous deployment certificates and competitive performance on MuJoCo continuous control tasks.

## Background & Motivation

**Background**: Deploying reinforcement learning in safety-critical scenarios requires "formal generalization guarantees"—ensuring the trained policy performs well on unseen trajectories. The PAC-Bayes framework provides non-vacuous high-confidence certificates in supervised learning (Pérez-Ortiz 2021), and the certificates themselves can serve as training objectives. However, applying PAC-Bayes to RL faces a fatal issue: trajectory data is **temporally correlated**, where $S_{t+1}$ is determined by $(S_t, A_t)$, violating the i.i.d. assumption required for classical PAC bounds.

**Limitations of Prior Work**: (1) Seldin et al. (2011, 2012) used martingale methods for sequential dependence, but RL data does not naturally form a martingale and requires artificial construction (e.g., Bellman residuals). (2) Fard et al. (2011) applied PAC-Bayes via Bellman errors, resulting in a sample complexity scaled to $\mathcal{O}((1-\gamma)^{-4})$, which is numerically vacuous in modern deep RL settings (e.g., $\gamma=0.99$). (3) Recent works like Tasdighi et al. (2025) either inherit this poor horizon dependency or use PAC-Bayes only as a regularizer (e.g., deep exploration in PBAC, lifelong learning in Zhang 2025), abandoning the original intent of "computable certificates."

**Key Challenge**: To make PAC-Bayes certificates useful in deep RL, three issues must be resolved simultaneously: (a) selection of concentration inequalities for sequential dependence; (b) the exponential "horizon gap" regarding $1/(1-\gamma)$; (c) the instability of critics caused by non-convex PAC-Bayes objectives and periodic posterior updates. No prior work has addressed all three.

**Goal**: (1) Derive a PAC-Bayes RL bound with explicit $\tau_{\min}$ (mixing time) dependence and horizon dependence reduced to $\mathcal{O}((1-\gamma)^{-1})$; (2) make it numerically non-vacuous on MuJoCo; (3) design the PB-SAC algorithm to stably optimize this bound as an alive objective.

**Key Insight**: The authors abandon the two-step derivation involving Bellman residuals and perform a **bounded-differences** analysis directly on the discounted return. They then utilize the results from Paulin (2018), which extends McDiarmid’s inequality to Markov chains—this naturally provides concentration with explicit constants for "Markovian functions satisfying bounded-differences."

**Core Idea**: Directly derive transition-level sensitivity for the bounded-differences condition of discounted returns as $c_{(h,j)} = \gamma^{h-1}R_{\max}/T$. This leads to $\|c\|_2^2 = R_{\max}^2(1-\gamma^{2H})/(T(1-\gamma^2))$. Applying the Markov chain McDiarmid concentration with a standard PAC-Bayes change-of-measure yields a clean certificate containing only $\tau_{\min} \cdot \|c\|_2^2$.

## Method

### Overall Architecture
The work is divided into two layers:

1.  **Theoretical Layer (Section 3)**: Establishes the main PAC-Bayes RL theorem (Theorem 3.3), providing the form $\mathbb{E}_{\theta\sim\rho}[L(\theta)] \le \mathbb{E}_{\theta\sim\rho}[\hat{L}_D(\theta)] + \sqrt{\frac{R_{\max}^2(1-\gamma^{2H})}{T(1-\gamma^2)}\tau_{\min}(\mathrm{KL}(\rho\|\mu) + \ln\sqrt{2}/\delta)}$, where $L(\theta) = -\mathbb{E}_{\xi\sim M}[\frac{\pi_\theta(\xi)}{\pi_b(\xi)}G(\xi)]$ is the off-policy true loss written via importance sampling. The horizon dependence is only $1/(1-\gamma^2)$ (equivalent to linear), and $\tau_{\min}$ is the mixing time of the policy-induced Markov chain.
2.  **Algorithmic Layer (Section 4)**: Constructs PB-SAC (PAC-Bayes Soft Actor-Critic), maintaining a diagonal Gaussian posterior $\rho(\theta) = \mathcal{N}(\upsilon, \mathrm{diag}(\sigma^2))$. Standard SAC gradient updates handle the "fast path," while the PAC-Bayes objective is updated every 20k steps as a "slow path." Four mechanisms (posterior-guided exploration / PAC-Bayes-$\lambda$ variational relaxation / policy-level REINFORCE / adaptive sampling) are used to stabilize the optimization of the alive bound.

### Key Designs

1.  **Transition-level bounded-differences + Paulin concentration**:
    - **Function**: Explicitly quantifies the impact of perturbations to adjacent transitions within a single trajectory to extend McDiarmid to Markov chains.
    - **Mechanism**: For a fixed parameter $\theta$ and two sets of trajectories $D, \bar{D}$, it is proven that $|\hat{L}_D(\theta) - \hat{L}_{\bar{D}}(\theta)| \le \sum_{h',j'} c_{(h',j')} \mathbb{1}[\xi_{h'}^{(j')} \neq \bar{\xi}_{h'}^{(j')}]$, where $c_{(h,j)} = \gamma^{h-1} R_{\max}/T$. The coefficients $c_{(h,j)}$ show that transitions near the start have a large impact, while later ones decay exponentially. Paulin (2018)'s Markov chain McDiarmid is then applied: the deviation of statistics satisfying this bounded-difference on a Markov chain is controlled by $\tau_{\min} \cdot \|c\|_2^2$.
    - **Design Motivation**: Fard et al. (2011) derived $(1-\gamma)^{-4}$ via Bellman residuals because they first converted value error to Bellman error. Ours bypasses this to treat returns directly, controlling the sensitivity sum $\sum_h \gamma^{2h}$ as $(1-\gamma^2)^{-1}$, immediately removing two powers of $1/(1-\gamma)$. This is the key to shrinking the bound from $10^8$ to $10^2$ when $\gamma=0.99$.

2.  **PAC-Bayes-$\lambda$ variational relaxation + alternating optimization**:
    - **Function**: Converts non-convex square-root objectives like "$\hat{L}_D + \sqrt{\text{KL term}}$" into stable convex subproblems.
    - **Mechanism**: Uses the identity $\sqrt{x} = \inf_{\lambda > 0}\bigl(\frac{x}{2\lambda} + \frac{\lambda}{2}\bigr)$ to introduce an auxiliary parameter $\lambda$, rewriting the objective as $\mathcal{J}(\rho, \lambda) = \mathbb{E}_{\theta\sim\rho}[\hat{L}_D(\theta)] + \frac{\|c\|_2 \tau_{\min}}{2\lambda}(\mathrm{KL}(\rho\|\mu) + \ln\sqrt{2}/\delta) + \frac{\lambda}{2}$. This objective is convex w.r.t. $\rho$ and has a closed-form solution for $\lambda^* = \sqrt{\|c\|_2 \tau_{\min}(\mathrm{KL}+\ln\sqrt{2}/\delta)}$. The algorithm alternates between optimizing $(\upsilon, \sigma)$ via gradient descent and solving for $\lambda^*$.
    - **Design Motivation**: This trick was used by Thiemann et al. (2017) to relax non-convexities in majority vote learning; Ours is the first to bring it to RL. This step is critical—optimizing the square-root objective directly leads to training divergence and "$\rho \to \mu$ collapse."

3.  **Adaptive sampling to eliminate actor-critic mismatch**:
    - **Function**: Resolves the issue where sudden PAC-Bayes posterior updates cause critic mismatch, leading to "sawtooth" performance oscillations.
    - **Mechanism**: SAC typically samples 1 posterior (mean policy) for efficiency. **Immediately after** each PAC-Bayes update, the actor is frozen and the sample count is increased to 256. This allows the critic to perform high-frequency evaluations over the entire new posterior distribution, "recalibrating" the critic before resuming 1-sample training. Exploration uses Posterior-Guided Exploration (PGE): sample $|\mathcal{T}|$ candidate parameters from the posterior and select $\arg\max_{\theta_i\in\mathcal{T}} Q(s, \pi_{\theta_i}(s))$.
    - **Design Motivation**: Ablations (Figure 8a) show that without adaptive sampling, performance drops sharply after updates. This mechanism bridges the gap between theoretical alive bounds and deployable algorithms.

### Loss & Training
The SAC main path performs standard $\pi$ and $Q$ updates. PB-SAC triggers a PAC-Bayes-$\lambda$ update every 20k steps: (a) collecting **full** trajectories using the mean policy; (b) sampling multiple $\theta$ from $\rho$ to estimate $\hat{L}_D(\theta)$ via importance sampling; (c) applying a policy-level REINFORCE trick $\nabla_{(\upsilon,\sigma)} \mathbb{E}_{\theta\sim\rho}[\hat{L}_D(\theta)] = \mathbb{E}_{\theta\sim\rho}[\nabla_{(\upsilon,\sigma)}\log P_{\upsilon,\sigma}(\theta) \cdot \hat{L}_D(\theta)]$ for unbiased gradients. The prior $\mu$ is updated toward $\rho$ via a moving average. The mixing time $\tau_{\min}$ is estimated using the autocorrelation decay of rewards/features, taking the maximum value to avoid underestimation (which would lead to overconfident bounds).

## Key Experimental Results

### Main Results: MuJoCo Continuous Control Comparison (1M steps)

| Task | PB-SAC (Ours) | SAC baseline | PBAC (Tasdighi 2025) | PAC-Bayes Certificate |
|------|----------------|--------------|----------------------|----------------|
| HalfCheetah-v5 | ≈10–11k return | ≈10–11k return | Significantly behind | Non-vacuous within 100k steps |
| Ant-v5 | ≈5–6k return | ≈4–5k return | Significantly behind | Meaningful lower bound within 1M steps |
| Hopper-v5 | On par with SAC | baseline | Behind | Similar tightening curve |
| Walker2d-v5 | On par with SAC | baseline | Behind | Similar tightening curve |
| Ant-v5 (delayed, sparse) | **Exceeds SAC + PBAC**| baseline | Targeted, but weaker | PGE excels in sparse rewards |

Core Message: In dense-reward tasks, PB-SAC ≈ SAC (no performance penalty for certification). In sparse-reward tasks, it outperforms PBAC. Certificates tighten monotonically and respond reasonably to performance fluctuations.

### Ablation Study: Necessity of Core Components

| Configuration | Phenomenon | Description |
|------|------|------|
| Full PB-SAC | Smooth learning + tightening bound | Complete model |
| w/o Adaptive Sampling | Obvious "sawtooth", drops after updates | Critic mismatch; slow recovery |
| w/o PGE | Performance degrades to SAC levels in sparse tasks | Exploration lacks uncertainty guidance |
| w/o moving-avg prior | KL explosion | Posterior shifts too fast in early stages |
| Fixed $\tau_{\min}=1$ | Tightest bound but theoretically invalid | Equivalent to i.i.d. McDiarmid |
| Fixed $\tau_{\min}=1000$ | Loose bound but training still converges | Supports "overestimation is safer" philosophy |

### Key Findings
- Reducing horizon dependency from $(1-\gamma)^{-4}$ to $(1-\gamma)^{-1}$ is not merely decorative; it is the critical property that moves certificates from "numerically infinite" to "readable" at $\gamma=0.99$.
- Adaptive sampling is a practical discovery: while conventional wisdom suggests small posterior updates to avoid disturbing the critic, Ours finds that a large jump followed by 256-sample recalibration is much more stable.
- Overestimating mixing time is safer: $\tau_{\min}$ values between 1 and 1000 have limited impact on final performance, but underestimation invalidates the bound.

## Highlights & Insights
- **Alive bound paradigm**: The primary methodological contribution is elevating PAC-Bayes from "post-hoc reporting" to an "optimizable target" during training, supported by a suite of techniques (variational relaxation, PGE, adaptive sampling, moving-average priors).
- **Physical grounding of dependencies**: Using $\tau_{\min}$ allows the penalty to directly correspond to MDP physical properties—if the agent moves through state space quickly (small $\tau_{\min}$), the data is closer to i.i.d. This allows practitioners to intuitively understand the "value" of their trajectory data.
- **Policy-level REINFORCE trick**: Applying the log-likelihood trick to policy parameters $\theta$ rather than actions $a$ keeps PAC-Bayes training costs at the same order of magnitude as standard actor-critic algorithms.

## Limitations & Future Work
- **Limitations**: Underestimating mixing time leads to overconfident bounds (currently defended by max-autocorrelation + monotonic increase); Gaussian posteriors do not respect the geometry of deep network parameter spaces; importance weight clipping introduces pessimistic bias.
- **Ours' own observations**: Experiments are limited to 4 MuJoCo tasks without pixel inputs; PB-SAC overhead comes from periodic 256-sample recalibration; theoretical assumptions require $\tau_{\min} < +\infty$, posing challenges for non-ergodic policies (e.g., absorbing states).
- **Future Work**: Using normalizing flows or Stein variational methods for flexible posteriors; replacing $\tau_{\min}$ with pseudo-spectral gaps; extending the alive bound to model-based RL to simultaneously certify dynamics and policy.

## Related Work & Insights
- **vs Fard et al. (2011) / Tasdighi et al. (2025)**: They derive $(1-\gamma)^{-4}$ horizon dependency via Bellman residuals, causing numerical vacuity. Ours uses return-based bounded differences to reach $(1-\gamma)^{-1}$.
- **vs Seldin et al. (2011, 2012)**: Their martingale paths are elegant but less native to RL structures compared to Ours' Markovian McDiarmid approach.
- **vs PBAC (Tasdighi 2025)**: PBAC uses PAC-Bayes for exploration via an ensemble of critics; Ours focuses on "alive certificates" with a simpler single-critic structure and better dense-reward performance.
- **vs Zhang et al. (2025)**: While they use PAC-Bayes as a prior for lifelong learning, Ours promotes it back to its original purpose—certification.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Simultaneously achieving explicit $\tau_{\min}$, linear horizon dependency, non-vacuity, and an alive training objective is a major breakthrough.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid MuJoCo results and ablations; lacks pixel-based or large-scale task comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Well-structured; Section 3 clearly delineates theoretical improvements, and Section 4 justifies engineering techniques through a "challenge-solution" lens.
- Value: ⭐⭐⭐⭐⭐ The "alive PAC-Bayes" paradigm is a directly applicable tool for safety/medical RL and sets a template for future work on Markovian data certificates.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICML 2026\] Chebyshev Policies and the Mountain Car Problem: Reinforcement Learning for Low-Dimensional Control Tasks](chebyshev_policies_and_the_mountain_car_problem_reinforcement_learning_for_low-d.md)
- [\[ICML 2026\] Learning Unmasking Policies for Diffusion Language Models](learning_unmasking_policies_for_diffusion_language_models.md)
- [\[AAAI 2026\] Explaining Decentralized Multi-Agent Reinforcement Learning Policies](../../AAAI2026/reinforcement_learning/explaining_decentralized_multi-agent_reinforcement_learning_policies.md)
- [\[ICLR 2026\] Learning to Play Multi-Follower Bayesian Stackelberg Games](../../ICLR2026/reinforcement_learning/learning_to_play_multi-follower_bayesian_stackelberg_games.md)

</div>

<!-- RELATED:END -->

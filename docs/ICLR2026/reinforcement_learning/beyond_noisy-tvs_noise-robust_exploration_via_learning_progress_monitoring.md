---
title: >-
  [Paper Note] Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring
description: >-
  [ICLR 2026][Reinforcement Learning][Noisy-TV] To address the classic "Noisy-TV" trap in intrinsic motivation exploration, this paper proposes **Learning Progress Monitoring (LPM)**: using "how much the model improved this round compared to the last" as an intrinsic reward instead of prediction error or novelty. Since unlearnable random transitions yield zero progr
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Noisy-TV
date: 2026-05-08
content_hash: 553a03d5b00dfd21
---
# Beyond Noisy-TVs: Noise-Robust Exploration Via Learning Progress Monitoring

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=wzm38DRLhC](https://openreview.net/forum?id=wzm38DRLhC)  
**Code**: https://github.com/Akuna23Matata/LPM_exploration  
**Area**: Reinforcement Learning / Intrinsic Motivation Exploration  
**Keywords**: Noise-Robust Exploration, Noisy-TV, Learning Progress, Intrinsic Reward, Information Gain  

## TL;DR
To address the classic "Noisy-TV" trap in intrinsic motivation exploration, this paper proposes **Learning Progress Monitoring (LPM)**: using "how much the model improved this round compared to the last" as an intrinsic reward instead of prediction error or novelty. Since unlearnable random transitions yield zero progress, the agent is naturally immune to noise. LPM achieves faster convergence, higher state coverage, and superior extrinsic returns across MNIST, 3D mazes, and Atari compared to SOTA.

## Background & Motivation
**Background**: In sparse-reward reinforcement learning, environments provide extrinsic rewards infrequently, making pure random exploration highly inefficient. A common solution is to provide the agent with an **intrinsic reward** $r_t = r^e_t + \beta r^i_t$. Curiosity-based methods (ICM, RND, Ensemble) use **prediction error/uncertainty** as $r^i_t$ to encourage visiting transitions where the "model prediction is inaccurate"; episodic novelty methods (EME, EDT, etc.) reward "states not seen in the current episode."

**Limitations of Prior Work**: This entire class of methods suffers from an old problem—**Noisy-TV**. If the environment contains an unlearnable source of randomness (a TV playing static noise, a remote that generates random images), curiosity agents are trapped: because static noise is "permanently unpredictable," the prediction error/novelty remains high, causing the agent to stare at the TV rather than exploring useful areas.

**Key Challenge**: The essence of Noisy-TV is that intrinsic rewards **cannot distinguish between two types of uncertainty**. **Epistemic uncertainty** arises from a lack of data and can be reduced by collecting more samples; **aleatoric uncertainty** stems from environmental randomness (sensor noise, static) and cannot be eliminated. Existing "noise-robust" methods (AMA, EME, EDT) attempt to **separate these two uncertainties** and only reward the former. However, separation is extremely difficult, often requiring strong priors or massive datasets, and **rewards in early training stages remain dominated by noise**, leads to significant sample inefficiency.

**Key Insight**: The authors draw from a finding in neuroscience—humans **monitor their own learning progress** during exploration, tending to favor transitions that "let them learn the most." Watching an unlearnable transition **produces no learning progress**, so this strategy is inherently immune to Noisy-TV without requiring explicit uncertainty separation.

**Core Idea**: Shift the intrinsic reward from "prediction error/novelty" to "**model improvement**"—rewarding "how much my world model improved this round compared to the previous one" rather than "how poorly I predict." Static noise can never be learned, and the model never improves; thus, its intrinsic reward **is directly zero**, eliminating the problem at its source.

## Method

### Overall Architecture
LPM operates under a **model-based RL** setting: the agent maintains a dynamics model $f_\theta$ that predicts the next observation $\hat o_{t+1}$ given the current observation $o_t$ and action $a_t$. Let $t$ denote environment steps and $\tau$ denote model update steps, where the model is updated every $N$ environment steps (i.e., $f^{(\tau)}_\theta$ is the model after the $\tau$-th update).

The **Mechanism** is as follows: the **error reduction** of the dynamics model in this round compared to the previous one represents "how much was learned" and should serve as the intrinsic reward. Instead of re-running the saved $f^{(\tau-1)}_\theta$ on current samples, this paper introduces a separate **error model $g_\phi$** to predict the "**expected error of the previous model**." The pipeline follows: dynamics model prediction → compute current error → error model provides expected previous error → subtract the two to get intrinsic reward → combine with extrinsic reward to update policy, with both models synchronized every $N$ steps. This choice of "expected error over single-point error" is critical for theoretical validity.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Observation o_t, Action a_t"] --> B["Dynamics Model f_θ<br/>Predict Next Obs ô_{t+1}"]
    B --> C["Learning Progress Reward<br/>Current Error ε^(τ)"]
    A --> D["Dual-Network Error Model g_φ<br/>Predict Prev. Expected Error"]
    D --> C
    C -->|r_i = g_φ − ε^(τ)| E["Combined Reward r = r_e + β·r_i<br/>Update Policy π"]
    E -->|Sync update f_θ, g_φ every N steps| B
```

### Key Designs

**1. Learning Progress Reward: Rewarding "Model Improvement" instead of "Prediction Error"**

This marks the fundamental paradigm shift targeted at Noisy-TV. Define the log-MSE error of the dynamics model at round $\tau$ and time $t$:

$$\varepsilon^{(\tau)}_t(o_{t+1}) = \log\left(\frac{1}{\dim(\Omega)}\,\big\|o_{t+1} - f^{(\tau)}_\theta(o_t, a_t)\big\|_F^2\right)$$

Intuitively, $\varepsilon^{(\tau-1)}_t - \varepsilon^{(\tau)}_t$ characterizes "how much prediction accuracy improved on this transition after the $\tau$-th update." LPM uses this **error reduction** as the intrinsic reward. The key difference: curiosity methods reward $\varepsilon^{(\tau)}_t$ (high error = high reward), so static noise is always rewarded; LPM rewards the **change in error**—since static noise is unlearnable, the error never decreases, the progress is zero, and the intrinsic reward automatically drops to zero. The agent does not need to "judge" if it is noise; as long as the noise yields no progress, it loses appeal.

**2. Dual-Network Error Model $g_\phi$: Replacing "Single-point Old Error" with "Expected Error"**

To compute error reduction, the "error of the previous model on the current sample" must be known. Instead of caching $f^{(\tau-1)}_\theta$ for inference, an independent **error model** $g_\phi: \mathcal{O}\times\mathcal{A}\to\mathbb{R}$ is used to **regress the expected error of the previous model**:

$$g^{(\tau)}_\phi(o_t, a_t) \approx \mathbb{E}_D\!\left[\varepsilon^{(\tau-1)}_t(o_{t+1})\right]$$

A fixed-size replay queue $D$ of size $d$ stores records $(o_t, a_t, \varepsilon^{(\tau)}_t)$; $g_\phi$ is trained on "errors generated by the previous model" in $D$. The final intrinsic reward is:

$$r^i_t = \mathbb{E}_D\!\left[\varepsilon^{(\tau-1)}_t(o_{t+1})\right] - \varepsilon^{(\tau)}_t(o_{t+1}) = g^{(\tau)}_\phi(o_t, a_t) - \varepsilon^{(\tau)}_t(o_{t+1})$$

The use of **expectation** rather than a single-point old error is necessary because single-point errors contain aleatoric jitter, which can cause rewards to oscillate or even become negative despite true information gain. By using $g_\phi$ to maintain a smooth estimate of "expected error," the reward stably tracks actual learning progress.

**3. Monotonic Correspondence with Information Gain: Proving "Progress = Information"**

The authors define Information Gain (IG) as the KL divergence between posterior and prior $\mathrm{IG} := \mathrm{KL}(p(\theta|D)\,\|\,p(\theta))$. Under i.i.d. Gaussian observation assumptions and using log-MSE as a proxy for likelihood, **Theorem 4.1** proves the LPM intrinsic reward satisfies $r^i \ge \tfrac{1}{c}\,\mathrm{IG}$ and $\mathrm{IG}=0 \Leftrightarrow r^i=0$—meaning $r^i$ is a **zero-equivariant monotonic indicator** of IG. Reward is positive if and only if the model learns something new; it is exactly zero when nothing is learned (e.g., staring at static). **Theorem 4.2** further proves that the **expectation** operation in the error model is essential: using single-point rewards $r^{i,\text{point}} = \log\mathrm{MSE}(\theta) - \log\mathrm{MSE}(\theta_D)$ can result in $r^{i,\text{point}}<0$ even when $\mathrm{IG}>0$, breaking monotonicity.

### Loss & Training
The dynamics model $f_\theta$ fits the next observation using $(o_t,a_t,o_{t+1})$ from replay buffer $B$; the error model $g_\phi$ fits the "previous model error" using the fixed queue $D$. Both are synchronized every $N$ steps. The policy $\pi$ is trained using $r_t = r^e_t + \beta r^i_t$ with any RL algorithm (e.g., PPO). Intrinsic rewards are only issued after the queue $|D|=d$ is full to avoid unreliable early estimates.

## Key Experimental Results

Experiments address three questions: Does LPM converge faster and resist aleatoric uncertainty? Does it cover more states in pure exploration? Does it achieve higher returns in reward-bearing tasks? Environments scale from Noisy MNIST → MiniWorld 3D Maze → MountainCar Continuous → Atari, with increasing complexity and various noise conditions (state/action noise).

### Main Results: MountainCar Continuous State Coverage (%)

| Method | Deterministic Coverage | Stochastic Coverage | Gain (Decrease) |
|------|-----------|-----------|---------|
| **LPM (Ours)** | 76.50 ± 9.08 | **67.04 ± 14.60** | **-12.4%** |
| Ensemble | 91.22 ± 2.04 | 61.02 ± 5.03 | -33.1% |
| EDT | 82.16 ± 13.57 | 53.52 ± 10.53 | -34.9% |
| EME | 89.16 ± 3.40 | 32.46 ± 11.31 | -63.6% |
| RND | 45.50 ± 14.53 | 28.00 ± 10.10 | -38.5% |
| AMA | 33.00 ± 12.31 | 13.20 ± 4.62 | -60.0% |
| IDF | 90.92 ± 5.13 | 12.80 ± 3.19 | -85.9% |

Key Observation: Many baselines (Ensemble/EME/IDF) have higher coverage in **deterministic** environments but crash under noise—IDF drops by 85.9%, EME by 63.6%; whereas LPM only drops by 12.4%, outperforming all methods in stochastic conditions. This directly demonstrates noise robustness.

### Environment Comparisons

| Environment | Metric | LPM Performance | Comparison |
|------|------|---------|------|
| Noisy MNIST | Intrinsic Reward Conv. | Converges to 0 in ≈150 steps | AMA takes ≈400 steps; EDT never converges (finds noise interesting) |
| MiniWorld 3D Maze | Avg. States Visited | **1347.6** | +95.3 states compared to next best (+7.6%), stable across noise tiers |
| Atari (6 games) | Extrinsic Return | Best in 4/6 games | Space Invader drops only 3.9% with noise; EME crashes 100% with noise |
| Montezuma's Revenge | Steps to Non-trivial Return | 20M steps | RND needs 50M; NGU remains at 0 after 50M |

### Key Findings
- **Paradigm shift is the core source of gain**: Rewarding "improvement" rather than "error/novelty" zeros out noise rewards at the source without expensive uncertainty separation.
- **Expected error in dual networks is vital**: Both theory and experiments show point-wise old errors lead to instability and negative rewards; $g_\phi$ provides a stable estimate.
- **Robustness is shown by "minimal performance decay"**: LPM may not be #1 in deterministic settings, but its degradation under noise is far smaller than all baselines.
- **Computational overhead is manageable**: While using dual networks, it is comparable to AMA and much lower than Ensemble or EME (multi-model).

## Highlights & Insights
- **The "reward progress not error" view is a profound simplification**: A single formula $r^i_t = g^{(\tau)}_\phi - \varepsilon^{(\tau)}_t$ renders Noisy-TV irrelevant without explicitly distinguishing epistemic/aleatoric uncertainty.
- **Clean mapping from neuroscience to algorithms**: The observation of "monitoring learning progress" in humans directly maps to "rewarding model improvement."
- **Theoretical justification for dual networks**: While some auxiliary networks are empirical tricks, Theorem 4.2 proves that "taking expectation" is a theoretical necessity to maintain monotonicity.
- **Transferability**: This "improvement as reward" logic can generalize to any active learning or exploration scenario involving a world model.

## Limitations & Future Work
- **Reliance on i.i.d. Gaussian assumptions**: The authors admit that robustness is harder to analyze theoretically outside these assumptions, relying on empirical data instead.
- **Model-based setting**: LPM requires a dynamics model $f_\theta$ and error model $g_\phi$ with periodic updates, making it less than "plug-and-play" for pure model-free pipelines.
- **Hyperparameter sensitivity**: Sensitivity to update period $N$, queue size $d$, and weight $\beta$ across different environments requires more systematic characterization.
- **Long-horizon tasks remain challenging**: While better than RND on Montezuma, it still only achieves "non-trivial" scores, indicating a gap for long-range sparse rewards.

## Related Work & Insights
- **vs. Curiosity (ICM/RND/Ensemble)**: These reward prediction error or model disagreement; noise keeps errors high, trapping the agent. LPM rewards error reduction; noise gives no reduction, so reward goes to zero.
- **vs. AMA (Aleatoric Uncertainty Filtering)**: AMA explicitly estimates and subtracts noise, requiring large data for reliable separation and being noise-dominated early on. LPM converges much faster (150 vs 400 steps on MNIST).
- **vs. Episodic Novelty (EME/EDT)**: These reward "new/similar" states; random transitions are naturally "the most novel," attracting the agent to noise. LPM uses learning progress to bypass the hijacking of novelty by noise.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframing intrinsic reward as learning progress is a paradigm-level shift for Noisy-TV.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage (MNIST to Montezuma), though hyperparameter analysis is somewhat sparse.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from motivation to theory; tight mapping between design and theory.
- Value: ⭐⭐⭐⭐⭐ Conceptually simple, theoretically grounded, and low overhead; highly valuable for robust exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Diversity-Incentivized Exploration for Versatile Reasoning](diversity-incentivized_exploration_for_versatile_reasoning.md)
- [\[ICLR 2026\] AlphaSAGE: Structure-Aware Alpha Mining via GFlowNets for Robust Exploration](alphasage_structure-aware_alpha_mining_via_gflownets_for_robust_exploration.md)
- [\[ICML 2025\] Robust Noise Attenuation via Adaptive Pooling of Transformer Outputs](../../ICML2025/reinforcement_learning/robust_noise_attenuation_via_adaptive_pooling_of_transformer_outputs.md)
- [\[ICML 2025\] Learning Progress Driven Multi-Agent Curriculum](../../ICML2025/reinforcement_learning/learning_progress_driven_multi-agent_curriculum.md)
- [\[ICLR 2026\] Beyond Distributions: Geometric Action Control for Continuous Reinforcement Learning](beyond_distributions_geometric_action_control_for_continuous_reinforcement_learn.md)

</div>

<!-- RELATED:END -->

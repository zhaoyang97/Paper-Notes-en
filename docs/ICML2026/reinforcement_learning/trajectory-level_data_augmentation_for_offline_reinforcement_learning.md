---
title: >-
  [Paper Note] Trajectory-Level Data Augmentation for Offline Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Offline RL] This paper proposes LIFT: In active alignment tasks, it leverages trajectory geometric properties to transform redundant zig-zag trajectories from sub-optimal logging polic…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Offline RL"
  - "Trajectory Augmentation"
  - "Shortcut"
  - "CQL"
  - "Active Alignment"
date: 2026-05-08
content_hash: 791ead9667c774e1
---

# Trajectory-Level Data Augmentation for Offline Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.13401](https://arxiv.org/abs/2605.13401)  
**Code**: https://github.com/HS-Kempten/lift  
**Area**: Reinforcement Learning / Offline RL / Data Augmentation / Active Alignment  
**Keywords**: Offline RL, Trajectory Augmentation, Shortcut, CQL, Active Alignment

## TL;DR
This paper proposes LIFT: In active alignment tasks, it leverages trajectory geometric properties to transform redundant zig-zag trajectories from sub-optimal logging policies into "shortcuts." These synthesized transitions are fed into a lightweight augmentor to replace logging actions during data collection. This enables offline CQL to significantly outperform standard offline RL and warm-start SAC across various settings, including high-dimensional spaces and partial observations.

## Background & Motivation
**Background**: The mainstream of offline RL focuses on "conservative updates + behavioral regularization" (BC loss, CQL pessimistic critic, IQL expectile policy extraction). These algorithmic methods assume the dataset is already "good enough." However, evidence suggests that dataset quality (coverage, expertise, and trajectory structure) often impacts final performance more than algorithmic differences.

**Limitations of Prior Work**: In industrial active alignment scenarios (optical alignment, camera/telescope assembly, robotic arm coarse positioning), logging policies are typically scripted "coordinate-descent" methods—converging from coarse to fine and dimension by dimension. These are reliable but severely sub-optimal, producing circuitous paths. Existing approaches either use pure offline RL (limited by data quality) or offline-to-online fine-tuning (requiring expensive online interaction). The "middle ground"—improving data directly during logging—has been largely ignored. Furthermore, hard injection of superior actions triggers the "hand-off problem": once the script is interrupted, it cannot resume and requires a full reset.

**Key Challenge**: To inject superior actions during collection, one must address: (i) the augmentor must provide reliable suggestions with minimal data; (ii) subsequent logging policy progression must not be disrupted; (iii) theoretical criteria for both dynamics $f$ and value functions $V^\pi$ are needed to determine when a shortcut is truly superior. Simply summing multi-step actions $a = \sum a_k$ guarantees neither reaching $s_j$ nor value stability near $s_j$.

**Goal**: (1) Provide sufficient conditions to identify shortcuts in logged trajectories; (2) Train an augmentor using these shortcuts to replace logging actions during collection; (3) Verify if this "middle ground" approach is more data-efficient than pure offline or warm-start RL.

**Key Insight**: It is observed that distance-improving logging policies in geometrically structured alignment tasks have a strong prior—posterior states are always closer to the target than anterior states. Thus, potential shortcut value can be inferred from value differences without re-execution to synthesize transitions.

**Core Idea**: Verifiable inequalities for "$\sum a_k$ is a $\pi$-shortcut" are derived using "distance-improving + LPE (Linear Positional Error) + $L_V$-Lipschitz value function" conditions. This is instantiated as Algorithm 1 to scan logged trajectories for shortcut synthesis, followed by training an augmentor to replace logging actions with probability $p$ during collection.

## Method

### Overall Architecture
Active alignment is modeled as a contextual POMDP: state $(s, W) \in \mathcal{P} \times \mathcal{W}$, action $a \in \mathcal{A}$, dynamics $s' = f(s, a, W)$, and reward $R = -\|f(s,a,W) - s_W\|$. Typical $f(s,a,W) = s + W \cdot a$ (linear error) or forms with non-linear perturbations. The pipeline consists of two layers: (1) Offline shortcut synthesis (Algorithm 1) extracts $(o_i, \hat{a}, r_{j-1}, o_j)$ triplets satisfying theoretical conditions from logged trajectories; (2) Online LIFT collection (Algorithm 2) replaces logging actions with a $Q_\theta$-based augmentor $a_\theta(o) = \arg\max_a Q_\theta(o,a)$ with probability $p$. Upon replacement, the logging policy internal state is reset to ensure proper hand-off. Finally, CQL is trained on the dataset containing shortcut transitions (CQL-SC), and the combined system is LIFT-SC.

### Key Designs

1. **Theoretical Criteria for Shortcuts (Theorem 3.6 + Corollary 3.8)**:

    - **Function**: Formalizes when summing actions $\sum a_k$ leads to a value improvement into an inequality checkable on logged data.
    - **Mechanism**: Defines "distance-improving policies" where rewards strictly increase. It introduces the LPE (Linear Positional Error) property $\|f(s_0, \sum a_i, W) - s_k\| \le L_f \cdot \sum \|a_i\|$ to bound accumulated action shifts and requires $V^\pi$ to be $L_V$-Lipschitz. It is then proved that if $\gamma V^\pi(s_j, W) - V^\pi(s_i, W) - \|s_j - s_W\| \ge (\gamma L_V + 1) L_f \sum_{k=i}^{j-1} \|a_k\|$, then $\sum a_k$ is a shortcut. Linear dynamics $f(s,a,W) = s + Wa$ is a special case where $L_f = 0$, making any accumulation valid.
    - **Design Motivation**: Summing multi-step actions directly almost always results in a miss. This criterion identifies $(i, j)$ pairs with "large value difference and short paths," which correspond to zig-zag segments in logged trajectories. This serves as the theoretical foundation for filtering shortcuts via Algorithm 1.

2. **Algorithm 1: Scanning and Sampling Shortcuts in Logged Trajectories**:

    - **Function**: For a trajectory with returns $G_i = V^{\pi_\beta}(s_i, W) = \sum_{k=i}^n \gamma^{k-i} r_k$, it iterates $j > i$ starting from position $i$. For each candidate $\hat{a} = \sum_{k=i}^{j-1} a_k$, it checks $\gamma G_j - G_i + r_{j-1} \ge C \sum \|a_k\|$. Validated $(o_i, \hat{a}, r_{j-1}, o_j)$ transitions enter a candidate set $S$, sampled according to normalized rewards $\rho \propto \hat{r} - \min \hat{r}$.
    - **Mechanism**: $C$ collapses Theorem 3.6 constants into a single hyperparameter (default $C=0$). Linear time scanning allows it to be plugged into d3rlpy as a "transition picker."
    - **Design Motivation**: Implements the theoretical criterion into a plug-and-play interface. Any d3rlpy algorithm (besides CQL) can utilize shortcuts by switching the picker. Reward-based sampling preserves diversity while favoring shortcuts closer to the goal.

3. **Algorithm 2: Augmentation during LIFT Collection (Augmentor + Reset)**:

    - **Function**: Replaces $\pi_\beta$ actions with $a_\theta(o)$ with probability $p$ during collection, creating a hybrid between pure offline and warm-start RL. Resets $\pi_\beta$ internal states upon replacement for hand-off.
    - **Mechanism**: Logged trajectories train $a_\theta$ (augmented by Algorithm 1). New episodes allow $a_\theta$ to intervene with $p=0.6$. Defines $\pi_{\text{aug}}(o) = a_\theta(o)$ if $a_\theta(o)$ is a $\pi_\beta$-shortcut else $\pi_\beta(o)$. By Proposition A.1, $V^{\pi_{\text{aug}}} \ge V^{\pi_\beta}$.
    - **Design Motivation**: Explicitly integrates "hand-off friendliness"—once the augmentor takes over, the logging policy internal progress (step size, optimized dimensions) is refreshed to prevent inconsistent states. This nuance is the core of LIFT (Logging Improvement via Fine-tuned Trajectories).

### Loss & Training
No new losses are introduced; all training follows standard CQL objectives. Algorithm 1 transitions are injected via the d3rlpy picker interface. $Q_\theta$ is trained on an initial small dataset (50-100 trajectories) before entering the main collection loop. Hyperparameters: $C=0, p=0.6$, max 20 augmentations per trajectory.

## Key Experimental Results

### Main Results

| Scenario | logging | CQL | CQL-SC | LIFT | LIFT-SC | warm-start SAC |
|---|---|---|---|---|---|---|
| $(\mathcal{O}_{\text{PO}}, f_{\text{blend}})$, $d=5$ | Highly sub-optimal | Fair | Gain | Further Gain | **Best** | Lags behind |
| Lens alignment $\mathcal{O}_{\text{LP}}$ (Image) | Same | Medium | High | High | **Best** | Weaker than LIFT-SC |
| Fetch Reach $\mathcal{O}_{\text{Fetch}}$ | Same | Medium | High | High | **Best** | Slightly weaker |
| Polarized channel $\mathcal{O}_{\text{LT}}$ (Image) | Same | Weak | Medium | Medium | **Best** | Weak |
| $d=2$ Low-dim $\mathcal{O}_{\text{PO}}$ | — | — | — | Flat | Flat | **Better** |

LIFT-SC leads across high-dimensional/PO/image observations (Figure 7, Appendix E). Diffusion-based GTA and Diffusion-QL do not consistently win.

### Ablation Study

| Config | Observation | Interpretation |
|---|---|---|
| Add Shortcut (CQL → CQL-SC) | Consistent gains | Offline shortcuts alone extract latent potential from logged data |
| Add LIFT Collection (CQL → LIFT) | Better than CQL | Improving data distribution during collection is stronger than pure offline RL |
| LIFT-SC = LIFT + shortcut | Optimized in nearly all cases | Gains from both steps are additive |
| $f_{\text{regrot}}$ (violates contraction) | Shortcut fails | Validates that Corollary 3.8 constraints are physically necessary |
| $f_{\text{sqrt}}$ (violates LPE) | Shortcut remains effective | LPE is sufficient but not necessary |
| Noise injection | LIFT-SC remains superior | Indicates lack of reliance on structured coordinate-descent scripts |

### Key Findings
- Shortcuts provide the most gain in high-dimensional and image-based observations, exactly where standard offline RL is most fragile. Expanding data coverage via task geometry is more effective than stricter algorithmic regularization.
- Shortcuts fail on $f_{\text{regrot}}$ which violates contraction; theoretical assumptions must be verified for LPE/contraction in real dynamics.
- LIFT datasets show "high average reward, low exploration," contrasting with IORL (high exploration, poor hand-off), which yields lower trajectory quality.
- Improving data during collection is more direct than complex offline algorithms. Warm-start SAC excels in low dimensions, showing LIFT's advantage in mid-to-high dimensions and partial observability.

## Highlights & Insights
- The formal "shortcut = multi-step sum + geometric conditions" criterion transforms engineering intuition into computable inequalities. This applies to any task with "distance-improving + smooth dynamics."
- Hand-off design is a practical detail—many hybrid methods fail when scripted logging is interrupted. Explicitly including "reset on hand-off" shows deep understanding of industrial deployment.
- Enabling augmentors with small data relies on high-quality supervision from synthesized shortcut transitions rather than massive online interaction.

## Limitations & Future Work
- Theoretical guarantees depend on distance-improving/LPE/Lipschitz properties, failing under discontinuous dynamics like $f_{\text{regrot}}$.
- Evaluations are limited to simulations; sim-to-real gaps are not validated on physical platforms.
- Setting $C=0$ might include false shortcuts in noisy trajectories. Adaptive schemes for $C$ are needed when $L_f$ is large.
- Integration with model-based/world-models is open; shortcuts could merge with Dyna-style RL.
- Only validated on CQL; benefits for IQL/BCQ etc. are not systematically reported.

## Related Work & Insights
- **vs HER (Hindsight Experience Replay)**: Both augment transitions; HER rewrites goals/states for sparse rewards, while LIFT compresses action chains for shortcuts.
- **vs IORL (Zhang et al. 2023)**: Both augment during collection; IORL explores for coverage, whereas LIFT injects exploitative shortcuts.
- **vs GuDA (Corrado et al. 2024)**: GuDA relies on human intervention, whereas LIFT uses algorithms (augmentor + shortcut criteria) to reduce cost.
- **vs Diffusion Augmentation (GTA, Diffusion-QL)**: Diffusion generates synthetic transitions with weak physical consistency; LIFT ensures transitions hold via geometric conditions.
- **vs warm-start SAC / Ball et al. 2023**: Warm-start requires larger online interaction budgets; LIFT excels within fixed trajectory budgets.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic shortcut synthesis and reset-friendly hand-off are novel for active alignment.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various observations and dynamics, though only in simulation.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical structure and well-coordinated algorithms.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play toolset for industrial alignment and advances the "augmentation vs regularization" discussion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Offline Reinforcement Learning with Generative Trajectory Policies](offline_reinforcement_learning_with_generative_trajectory_policies.md)
- [\[ICML 2026\] Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)
- [\[ICML 2026\] Beyond the Proxy: Trajectory-Distilled Guidance for Offline GFlowNet Training](beyond_the_proxy_trajectory-distilled_guidance_for_offline_gflownet_training.md)
- [\[ICML 2026\] Offline Reinforcement Learning with Universal Horizon Models](offline_reinforcement_learning_with_universal_horizon_models.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)

</div>

<!-- RELATED:END -->

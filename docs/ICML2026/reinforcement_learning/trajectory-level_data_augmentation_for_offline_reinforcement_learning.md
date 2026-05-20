---
title: >-
  [Paper Note] Trajectory-Level Data Augmentation for Offline Reinforcement Learning
description: >-
  [ICML 2026][Reinforcement Learning][Offline RL] This paper proposes LIFT: in active localization tasks, it leverages the geometric properties of trajectories to "shortcut" redundant zig-zag paths left by suboptimal loggi…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Offline RL"
  - "trajectory augmentation"
  - "shortcut"
  - "CQL"
  - "active localization"
date: 2026-05-08
content_hash: 7ad35039b70b7014
---

# Trajectory-Level Data Augmentation for Offline Reinforcement Learning

**Conference**: ICML 2026  
**arXiv**: [2605.13401](https://arxiv.org/abs/2605.13401)  
**Code**: https://github.com/HS-Kempten/lift  
**Area**: Reinforcement Learning / Offline RL / Data Augmentation / Active Localization  
**Keywords**: Offline RL, trajectory augmentation, shortcut, CQL, active localization

## TL;DR
This paper proposes LIFT: in active localization tasks, it leverages the geometric properties of trajectories to "shortcut" redundant zig-zag paths left by suboptimal logging policies, synthesizes these transitions, and feeds them to a lightweight augmentor that replaces logging actions during data collection. This enables offline CQL to significantly outperform standard offline RL and warm-start SAC across low- to high-dimensional, partial observation, and other settings.

## Background & Motivation
**Background**: Mainstream offline RL relies on "conservative updates + behavior regularization" (BC loss, pessimistic critic in CQL, expectation quantile policy extraction in IQL), all assuming the dataset is "good enough." However, substantial evidence shows that dataset quality (coverage, expertise, trajectory structure) often impacts final performance more than algorithmic differences.

**Limitations of Prior Work**: In industrial-scale active localization scenarios (optical alignment, camera/telescope assembly, robotic coarse positioning), the logging policy is typically a stateful scripted "coordinate walk"—coarse-to-fine, converging dimension by dimension, reliable but highly suboptimal, producing many detours. Existing approaches either stick to pure offline (limited by data quality) or offline-to-online fine-tuning (requiring expensive online interaction). The middle ground—"improving data during logging"—is largely overlooked. Directly injecting better actions triggers the "hand-off problem": once the script is interrupted, it cannot resume and must reset the entire segment.

**Key Challenge**: The goal is to inject better actions during collection, but (i) the augmentor must provide reliable suggestions with very little data; (ii) must not disrupt the logging policy's subsequent progress; (iii) must provide a theoretical criterion for when a shortcut is truly better, considering both dynamics perturbation $f$ and value function $V^\pi$. Simply summing multi-step actions $a = \sum a_k$ neither guarantees reaching $s_j$ nor value stability near $s_j$.

**Goal**: (1) Provide sufficient conditions for identifying shortcuts on existing logged trajectories; (2) use these shortcuts during data collection to train an augmentor that replaces some logging actions; (3) verify whether this "middle ground" is more data-efficient than pure offline + warm-start RL.

**Key Insight**: It is observed that distance-improving logging policies in geometrically structured localization tasks have a strong prior—the later state is always closer to the goal than the earlier state. Thus, the value difference between states can infer the potential value of a shortcut, enabling synthetic transitions without re-execution.

**Core Idea**: Using "distance improvement + LPE (linear position error) + $L_V$-Lipschitz value function" as three conditions, a verifiable inequality is derived for "$\sum a_k$ is a $\pi$-shortcut." This is instantiated as Algorithm 1, which linearly scans logged trajectories to synthesize shortcut transitions, which are then used to train an augmentor that probabilistically replaces logging actions during collection.

## Method

### Overall Architecture
Active localization is modeled as a contextual POMDP: state $(s, W) \in \mathcal{P} \times \mathcal{W}$, action $a \in \mathcal{A}$, dynamics $s' = f(s, a, W)$, reward $R = -\|f(s,a,W) - s_W\|$; typically $f(s,a,W) = s + W \cdot a$ (linear error) or with nonlinear perturbations. The pipeline has two layers: (1) Offline shortcut synthesis (Algorithm 1) identifies $(o_i, \hat{a}, r_{j-1}, o_j)$ tuples from a logged trajectory that meet theoretical conditions and adds them to the training set; (2) Online LIFT collection (Algorithm 2) replaces logging actions with augmentor actions $a_\theta(o) = \arg\max_a Q_\theta(o,a)$ with probability $p$, resetting the logging policy's internal state upon replacement to ensure hand-off. CQL is then trained on the dataset with shortcut transitions (CQL-SC), and combined with LIFT forms LIFT-SC.

### Key Designs

1. **Theoretical Criterion for Shortcuts (Theorem 3.6 + Corollary 3.8)**:

    - **Function**: Formalizes "when summing actions $\sum a_k$ truly improves value" as an inequality checkable on logged data.
    - **Mechanism**: Defines "distance-improving policy"—reward strictly increases along the trajectory; introduces LPE property (linear position error) $\|f(s_0, \sum a_i, W) - s_k\| \le L_f \cdot \sum \|a_i\|$ to bound cumulative action deviation; requires $V^\pi$ to be $L_V$-Lipschitz. It is then proven that if $\gamma V^\pi(s_j, W) - V^\pi(s_i, W) - \|s_j - s_W\| \ge (\gamma L_V + 1) L_f \sum_{k=i}^{j-1} \|a_k\|$, then $\sum a_k$ is a shortcut. Linear dynamics $f(s,a,W) = s + Wa$ is a special case with $L_f = 0$, where any sum is valid.
    - **Design Motivation**: Directly summing multi-step actions almost always misses; this criterion guides the algorithm to select $(i, j)$ pairs with "large value difference, short path"—corresponding to zig-zag segments in logged trajectories. This is the theoretical foundation, turning "empirical shortcuts" into something Algorithm 1 can systematically extract.

2. **Algorithm 1: Scanning and Sampling Shortcuts on Logged Trajectories**:

    - **Function**: For a trajectory with return $G_i = V^{\pi_\beta}(s_i, W) = \sum_{k=i}^n \gamma^{k-i} r_k$, for each position $i$, traverse $j > i$, for each candidate $\hat{a} = \sum_{k=i}^{j-1} a_k$, check $\gamma G_j - G_i + r_{j-1} \ge C \sum \|a_k\|$; passing synthetic transitions $(o_i, \hat{a}, r_{j-1}, o_j)$ enter candidate set $S$, and one is sampled according to normalized reward $\rho \propto \hat{r} - \min \hat{r}$.
    - **Mechanism**: $C$ consolidates the right-hand constant in Theorem 3.6 into a single hyperparameter (default $C=0$, i.e., all value-increasing candidates are included); linear time scan, can be plugged into d3rlpy as a "transition picker."
    - **Design Motivation**: Implements the theoretical criterion as a plug-and-play interface—any d3rlpy algorithm beyond CQL can use shortcuts by swapping the picker; using reward as sampling weight preserves diversity while favoring shortcuts closer to the goal.

3. **Algorithm 2: LIFT Collection-Time Augmentation (Augmentor + Reset)**:

    - **Function**: During data collection, with probability $p$, replaces $\pi_\beta$'s action with $a_\theta(o)$, creating a hybrid between pure offline and warm-start RL; upon replacement, resets $\pi_\beta$'s internal state to ensure the script can continue.
    - **Mechanism**: Collects a small number of trajectories with the logging policy to train $a_\theta$ (augmented with Algorithm 1), then in each new episode, $a_\theta$ takes over with $p=0.6$; defines $\pi_{\text{aug}}(o) = a_\theta(o)$ if $a_\theta(o)$ is a $\pi_\beta$-shortcut else $\pi_\beta(o)$, and by Proposition A.1, $V^{\pi_{\text{aug}}} \ge V^{\pi_\beta}$.
    - **Design Motivation**: Explicitly encodes "hand-off friendliness"—once the augmentor takes over, the logging policy's internal progress (current step size, optimized dimensions) is refreshed, preventing the script from running in an inconsistent state. This detail, unsolved by pure exploration augmentations like IORL, is central to LIFT ("Logging Improvement via Fine-tuned Trajectories").

### Loss & Training
No new loss is introduced; all training follows the standard CQL (Conservative Q-Learning) objective. Algorithm 1 transitions are injected via d3rlpy's picker interface. $Q_\theta$ is trained on a small early dataset (after 50-100 trajectories), then collection enters the main loop. Hyperparameters: $C=0$, $p=0.6$, per-trajectory augmentation cap 20.

## Key Experimental Results

### Main Results

| Scenario | logging | CQL | CQL-SC | LIFT | LIFT-SC | warm-start SAC |
|---|---|---|---|---|---|---|
| $(\mathcal{O}_{\text{PO}}, f_{\text{blend}})$, $d=5$ | Highly suboptimal | Moderate | Improved | Further improved | **Best** | Lagging |
| Lens alignment $\mathcal{O}_{\text{LP}}$ (image) | Same | Medium | Higher | Higher | **Best** | Weaker than LIFT-SC |
| Fetch Reach $\mathcal{O}_{\text{Fetch}}$ | Same | Medium | Higher | Higher | **Best** | Slightly weaker |
| Polarization channel $\mathcal{O}_{\text{LT}}$ (image) | Same | Weak | Medium | Medium | **Best** | Weak |
| $d=2$ low-dimensional $\mathcal{O}_{\text{PO}}$ | — | — | — | Flat | Flat | **Better** |

In Figure 7 and multiple comparisons in Appendix E, LIFT-SC leads almost universally in high-dimensional/partial observable/image settings; diffusion-based GTA and Diffusion-QL did not consistently outperform.

### Ablation Study

| Configuration | Phenomenon | Interpretation |
|---|---|---|
| Add shortcut (CQL → CQL-SC) | Consistent improvement in all scenarios | Offline shortcut alone already exploits logged data potential |
| Add LIFT collection (CQL → LIFT) | Better than CQL | Improving data distribution during collection is stronger than pure offline |
| LIFT-SC = LIFT + shortcut | Nearly always optimal | Two-step gains are additive |
| $f_{\text{regrot}}$ (violates contraction) | shortcut fails | Confirms Corollary 3.8's constraints are physically necessary |
| $f_{\text{sqrt}}$ (violates LPE) | shortcut still works but less advantage | LPE is "sufficient" not "necessary" |
| Noise injection disrupts logging structure | LIFT-SC still superior | Shows it does not rely on coordinate-walk scripts |

### Key Findings
- Shortcut yields the largest gains in high-dimensional and image observation settings, precisely where standard offline RL is weakest; this shows that "expanding data coverage via task geometry" is more effective than simply adding more conservative algorithmic regularization.
- On $f_{\text{regrot}}$ (violating contraction), shortcut fails outright: theoretical assumptions are not just decorative, and practical deployment requires checking if dynamics meet LPE/contraction.
- LIFT's dataset metrics (per Schweighofer et al. 2022) show "high mean return, low exploration"—in contrast to IORL, which has high exploration but poor hand-off, resulting in lower final trajectory quality.
- TBPTT-style "improving data during collection" addresses data issues more directly than "using more complex offline algorithms"; warm-start SAC still excels in low dimensions, indicating LIFT's advantage is focused on mid/high-dimensional + partially observable settings.

## Highlights & Insights
- The formal criterion "shortcut = multi-step action sum + geometric condition" is elegant, turning the engineering intuition of "obvious shortcuts" into a computable inequality; this approach can be transferred to any task with "distance improvement + smooth dynamics" (robotic coarse positioning, autonomous parking, AFM tip alignment).
- The hand-off design is a truly practical detail—many augmentation/hybrid methods look good on paper, but scripted logging collapses once interrupted; LIFT explicitly encodes "reset on hand-off" in pseudocode, reflecting a deep understanding of industrial deployment.
- The fact that "the augmentor is usable with little data" relies on shortcut-synthesized transitions providing high-quality supervision, not massive online interaction—making the "middle ground" a practical, evidence-based route.

## Limitations & Future Work
- Theoretical guarantees depend on the trio of distance-improving / LPE / $V$ Lipschitz; for non-continuous dynamics like $f_{\text{regrot}}$, this fails outright. Many real-world robotics tasks (contact assembly) have discontinuous dynamics.
- All evaluations are in semi-simulated environments, not on real optical/mechanical platforms; the sim-to-real gap is untested.
- Setting $C=0$ is equivalent to "taking all value-increasing segments," which may include too many false shortcuts in noisy trajectories; when $L_f$ is large, $C$ needs tuning, and the paper acknowledges the lack of an adaptive scheme.
- Integration with model-based/world-model approaches is an open direction—shortcuts are essentially simplified local models and should naturally combine with Dyna-style RL.
- Only validated on CQL; whether IQL/BCQ and other offline RL methods benefit similarly is not yet systematically reported.

## Related Work & Insights
- **vs HER (Hindsight Experience Replay)**: Both are transition augmentations, but HER rewrites goal/state to create sparse reward successes, while this work compresses action chains to generate shortcuts; they are complementary—HER addresses "sparse rewards," LIFT addresses "redundant trajectories."
- **vs IORL (Zhang et al. 2023)**: Both augment during collection; IORL injects exploratory actions to expand coverage, LIFT does the opposite by injecting exploitative shortcuts; experiments show IORL explores well but has poor hand-off, resulting in lower final trajectory quality than LIFT.
- **vs GuDA (Corrado et al. 2024)**: Both use expert-guided collection; GuDA relies on human intervention, LIFT replaces humans with pure algorithms (augmentor + shortcut criterion), saving costly annotation.
- **vs Diffusion Augmentation (GTA, Diffusion-QL)**: Diffusion methods generate synthetic transitions but are less consistent with real dynamics; this work uses geometric conditions to ensure synthetic transitions are valid under the original dynamics, with stronger interpretability.
- **vs warm-start SAC / Ball et al. 2023**: Warm-start requires substantial online interaction budget, while LIFT surpasses it with a fixed trajectory budget, embodying the idea that "better data" is more economical than "more online steps."

## Rating
- Novelty: ⭐⭐⭐⭐ "Synthesizing shortcuts during collection + hand-off friendly reset" is a novel and systematic approach in active localization.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers low-dimensional intuitive, high-dimensional, and image observations + five dynamics + multiple baselines, but only in simulation.
- Writing Quality: ⭐⭐⭐⭐ Theoretical sections (Definition→Proposition→Theorem→Corollary) are clearly structured; Figure 1 overview and Algorithm 1/2 are well integrated.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play d3rlpy-compatible toolset for industrial active localization, advancing the methodological debate of "data augmentation vs algorithmic regularization."

## Related Papers

- [\[AAAI 2026\] Know your Trajectory -- Trustworthy Reinforcement Learning Deployment through Importance-Based Trajectory Analysis](../../AAAI2026/reinforcement_learning/know_your_trajectory_--_trustworthy_reinforcement_learning_deployment_through_im.md)
- [\[NeurIPS 2025\] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](../../NeurIPS2025/reinforcement_learning/noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)
- [\[ICML 2026\] CPMöbius: Iterative Coach–Player Reasoning for Data-Free Reinforcement Learning](cpmobius_iterative_coach-player_reasoning_for_data-free_reinforcement_learning.md)
- [\[ICML 2026\] Long-Horizon Model-Based Offline Reinforcement Learning Without Explicit Conservatism](long-horizon_model-based_offline_reinforcement_learning_without_explicit_conserv.md)
- [\[ICML 2025\] Zero-Shot Generalization of Vision-Based RL Without Data Augmentation](../../ICML2025/reinforcement_learning/zero-shot_generalization_of_vision-based_rl_without_data_augmentation.md)

</div>

<!-- RELATED:END -->

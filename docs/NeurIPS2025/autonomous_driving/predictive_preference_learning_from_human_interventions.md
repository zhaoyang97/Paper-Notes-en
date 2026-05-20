---
title: >-
  [Paper Note] Predictive Preference Learning from Human Interventions
description: >-
  [NeurIPS 2025][Autonomous Driving][Interactive Imitation Learning] PPL leverages a trajectory prediction model to anticipate the agent's future states and "bootstraps" each human intervention signal across the predicted…
tags:
  - "NeurIPS 2025"
  - "Autonomous Driving"
  - "Interactive Imitation Learning"
  - "Preference Learning"
  - "Human Interventions"
  - "Trajectory Prediction"
  - "DPO"
date: 2026-05-08
content_hash: eecda7ee91ddb47e
---

# Predictive Preference Learning from Human Interventions

**Conference**: NeurIPS 2025
**arXiv**: [2510.01545](https://arxiv.org/abs/2510.01545)  
**Code**: [https://metadriverse.github.io/ppl](https://metadriverse.github.io/ppl)  
**Area**: Autonomous Driving / Imitation Learning
**Keywords**: Interactive Imitation Learning, Preference Learning, Human Interventions, Trajectory Prediction, DPO

## TL;DR
PPL leverages a trajectory prediction model to anticipate the agent's future states and "bootstraps" each human intervention signal across the predicted future horizon to construct contrastive preference data. Combined with a dual-loss training strategy of behavior cloning and preference optimization, PPL substantially reduces the number of required human interventions and demonstration data.

## Background & Motivation

Interactive imitation learning (IIL) allows human experts to monitor and correct agent behavior in real time during training, effectively alleviating the distribution shift problem compared to offline imitation learning. However, existing IIL methods face three key limitations:

**Correcting only the current state**: Methods such as HG-DAgger apply behavior cloning solely at the state where the expert intervenes, yet the agent may repeat similar errors at subsequent steps $t+1, \cdots, t+L$, requiring repeated expert corrections.

**High cognitive burden**: Experts must continuously monitor the entire training process, anticipate the agent's future trajectory, and intervene at safety-critical states in a timely manner.

**Low data efficiency**: Only the demonstration data at the moment of intervention is utilized, while the preference information implicit in the intervention itself is ignored—the mere fact that "an expert chose to intervene" already signals that the agent's action was undesirable.

**Core Idea**: A trajectory prediction model visualizes the agent's future states to help experts intervene proactively. Simultaneously, each intervention is bootstrapped over $L$ future steps to construct a preference dataset, and contrastive preference optimization (CPO) is applied to propagate the expert's corrective intent into safety-critical regions.

## Method

### Overall Architecture

The PPL pipeline proceeds as follows: (1) the agent proposes action $a_n$ at each decision step → (2) a trajectory prediction model $f(s, a_n, H)$ generates $H$-step future states and visualizes them → (3) the human expert observes the predicted trajectory and decides whether to intervene → (4) if intervention occurs, the resulting demonstration is stored in $\mathcal{D}_h$, while preference pairs are constructed over the predicted $L$ future states and stored in $\mathcal{D}_{pref}$ → (5) a dual-loss training strategy optimizes the policy network.

### Key Designs

1. **Trajectory Prediction and Visualization**:

    - Given the current state $s$ and agent action $a_n$, the trajectory prediction model produces $f(s, a_n, H) = (s, \tilde{s}_1, \cdots, \tilde{s}_H)$.
    - Predictions are visualized to the expert in real time (e.g., a red predicted trajectory in a driving scenario), enabling proactive intervention when the trajectory indicates imminent danger such as a collision.
    - Implementation options include simulator rollout (1000 fps) or a kinematic bicycle model (3000 fps, no simulator required).
    - **Design Motivation**: Reduce the expert's cognitive burden by offloading future anticipation to the system.

2. **Preference Bootstrapping**:

    - When an expert intervenes at state $s$, preference triples $(\tilde{s}_i, a^+ = a_h, a^- = a_n)$ are constructed for the first $L$ predicted future states, $i = 1, \cdots, L$.
    - The preference horizon $L \leq H$ controls the bootstrapping length and is a critical hyperparameter.
    - **Core Assumption**: The expert's corrective action $a_h$ at state $s$ remains preferable to the agent's action $a_n$ in the near-future states $\tilde{s}_i$.
    - **Design Motivation**: A single intervention generates $L$ preference samples, substantially improving data efficiency and propagating the corrective signal into safety-critical regions the agent may explore.

3. **Dual-Loss Training Strategy**:

    - Behavior cloning loss: $\mathcal{L}_{BC} = -\mathbb{E}_{(s,a_h) \sim \mathcal{D}_h}[\log \pi_\theta(a_h|s)]$
    - Contrastive preference optimization (CPO) loss: $\mathcal{L}_{pref} = -\mathbb{E}_{(s,a^+,a^-) \sim \mathcal{D}_{pref}}[\log \sigma(\beta \log \pi_\theta(a^+|s) - \beta \log \pi_\theta(a^-|s))]$
    - Total loss: $\mathcal{L} = \mathcal{L}_{pref} + \mathcal{L}_{BC}$
    - The BC loss regularizes the policy to stay close to expert demonstrations, while the CPO loss leverages preference signals to suppress dangerous behaviors.

### Loss & Training
- Training hyperparameters: $\beta = 0.1$; $L = 4$ for MetaDrive; $L = 6$ for Nut Assembly; $H = 10$.
- CPO requires no reference policy (an advantage over DPO) and no pre-training.
- The predicted trajectory is updated every $H = 10$ steps (approximately 1 second); experts intervene via an Xbox controller or keyboard.

## Key Experimental Results

### Main Results

**MetaDrive (Human-in-the-loop experiment, 10K steps)**

| Method | Human Data | Success Rate | Return | Route Completion |
|--------|------------|--------------|--------|-----------------|
| BC | 20K (offline) | 0.0 | 53.5 | 0.16 |
| PVP | 4.9K (0.49) | 0.46 | 267.3 | 0.71 |
| Ensemble-DAgger | 3.8K (0.38) | 0.36 | 233.8 | 0.70 |
| **PPL (Ours)** | **2.9K (0.29)** | **0.76** | **324.8** | **0.90** |
| Human Expert | 20K | 0.95 | 349.2 | 0.98 |

PPL achieves the highest success rate (76%) with the least human data (2.9K), completing training in 12 minutes on a single RTX 4080.

### Ablation Study

| Configuration | Success Rate | Route Completion | Notes |
|---------------|-------------|-----------------|-------|
| PPL (full) | 0.81 | 0.92 | Best |
| Imitate $a^+$ only | 0.36 | 0.65 | Contrastive preference information is critical |
| Random $a^+$ | 0.45 | 0.73 | Quality of $a^+$ matters |
| Random $a^-$ | 0.38 | 0.69 | Quality of $a^-$ also matters |
| BC loss only | 0.42 | 0.72 | Missing preference optimization |
| CPO loss only | 0.04 | 0.31 | Missing BC regularization |
| PPL + DPO | 0.80 | 0.91 | DPO requires a reference policy; comparable performance |
| Rule-based trajectory prediction | 0.78 | 0.91 | Does not rely on simulator rollout |

### Key Findings
- **The choice of preference horizon $L$ is critical**: $L = 4$ is optimal for MetaDrive; a small $L$ provides insufficient coverage (high $\delta_{dist}$), while a large $L$ degrades label quality (high $\delta_{pref}$).
- PPL outperforms all baselines even when the trajectory predictor is noisy ($\epsilon \leq 0.25$).
- PPL produces smoother control sequences and trajectories more consistent with human preferences.
- Theoretical upper bound (Theorem 4.1): $J(\pi_h) - J(\pi_n) = O(\sqrt{\epsilon + \delta_{pref}} + \delta_{dist})$, indicating that $L$ must balance the two error terms.
- The method generalizes effectively to RoboSuite tabletop wiping and nut assembly tasks, demonstrating broad applicability.

## Highlights & Insights
- The RLHF/DPO paradigm is elegantly transferred to real-time control by using trajectory prediction to construct preference data without manual preference annotation.
- Preference bootstrapping is a concise and effective idea: a single intervention generates multiple training samples, significantly improving sample efficiency.
- Trajectory visualization simultaneously benefits the human expert (reduced cognitive load) and the algorithm (preference data construction), yielding a mutually beneficial design.
- The theoretical analysis is rigorous and provides principled guidance for selecting the preference horizon.

## Limitations & Future Work
- The approach assumes experts always know and can accurately execute the optimal corrective action, whereas real human demonstrations may be suboptimal or inconsistent.
- All experiments are conducted in simulation; performance in real-robot and physical environments remains unknown.
- The core assumption underlying preference bootstrapping—that $a_h$ remains preferable to $a_n$ in future states—weakens as $L$ increases.
- A usable trajectory prediction model or physics-based model is required, which may not be readily available in all settings.

## Related Work & Insights
- HG-DAgger and PVP are direct predecessors in IIL; PPL addresses their limitation of correcting only the current state.
- DPO/CPO preference optimization methods are adapted from the LLM alignment literature.
- Trajectory prediction has a wealth of mature work in autonomous driving that PPL can leverage.
- The preference bootstrapping idea is broadly applicable to other settings with sparse human feedback.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of trajectory prediction, preference bootstrapping, and IIL is entirely novel and conceptually elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual-domain evaluation (driving + robotic manipulation), both human and simulated experts, comprehensive ablation and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear, the method is intuitive, and theoretical analysis is complete.
- Value: ⭐⭐⭐⭐ Provides a practical and data-efficient approach to IIL, bridging preference-based RL and imitation learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Towards Predicting Any Human Trajectory in Context](towards_predicting_any_human_trajectory_in_context.md)
- [\[NeurIPS 2025\] Towards Physics-Informed Spatial Intelligence with Human Priors: An Autonomous Driving Perspective](towards_physics-informed_spatial_intelligence_with_human_priors_an_autonomous_dr.md)
- [\[CVPR 2026\] Towards Balanced Multi-Modal Learning in 3D Human Pose Estimation](../../CVPR2026/autonomous_driving/towards_balanced_multi-modal_learning_in_3d_human_pose_estimation.md)
- [\[NeurIPS 2025\] HoloLLM: Multisensory Foundation Model for Language-Grounded Human Sensing and Reasoning](holollm_multisensory_foundation_model_for_language-grounded_human_sensing_and_re.md)
- [\[ICCV 2025\] DCHM: Depth-Consistent Human Modeling for Multiview Detection](../../ICCV2025/autonomous_driving/dchm_depth-consistent_human_modeling_for_multiview_detection.md)

</div>

<!-- RELATED:END -->

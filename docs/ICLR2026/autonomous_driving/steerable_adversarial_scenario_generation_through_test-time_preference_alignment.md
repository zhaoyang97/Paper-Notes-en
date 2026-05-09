---
title: >-
  [Paper Note] Steerable Adversarial Scenario Generation through Test-Time Preference Alignment (SAGE)
description: >-
  [ICLR 2026][Autonomous Driving][adversarial scenario generation] SAGE reformulates adversarial scenario generation for autonomous driving as a multi-objective preference alignment problem. By training two preference expert models and performing weight interpolation at inference time, it enables a continuous and steerable trade-off between adversariality and realism—without retraining—generating a full spectrum of scenarios from mild to aggressive, substantially improving closed-loop training performance.
tags:
  - ICLR 2026
  - Autonomous Driving
  - adversarial scenario generation
  - preference alignment
  - multi-objective optimization
  - linear mode connectivity
  - closed-loop training
date: 2026-05-08
content_hash: c2dfb33c2b1004b3
---

# Steerable Adversarial Scenario Generation through Test-Time Preference Alignment (SAGE)

**Conference**: ICLR 2026
**arXiv**: [2509.20102](https://arxiv.org/abs/2509.20102)
**Code**: [https://tongnie.github.io/SAGE/](https://tongnie.github.io/SAGE/)
**Area**: Autonomous Driving / AI Safety
**Keywords**: adversarial scenario generation, preference alignment, multi-objective optimization, linear mode connectivity, closed-loop training

## TL;DR
SAGE reformulates adversarial scenario generation for autonomous driving as a multi-objective preference alignment problem. By training two preference expert models and performing weight interpolation at inference time, it enables a continuous and steerable trade-off between adversariality and realism—without retraining—generating a full spectrum of scenarios from mild to aggressive, substantially improving closed-loop training performance.

## Background & Motivation

**Background**: Safety validation for autonomous driving requires large quantities of safety-critical scenarios to test and train driving policies. Adversarial scenario generation—which efficiently produces long-tail corner cases by perturbing real driving trajectories—has become the dominant paradigm.

**Limitations of Prior Work**: Existing methods (RL-based, diffusion-based, direct optimization) all face a fundamental tension between adversariality and realism. Methods that optimize purely for adversariality tend to produce physically implausible trajectories (e.g., vehicles spinning in place to intercept the ego), while those that balance multiple objectives via linear weighting are highly sensitive to hyperparameter tuning.

**Key Challenge**: Each training run locks in a single fixed trade-off point on the Pareto frontier. Generating scenarios of varying intensity for different use cases (extreme stress testing vs. data augmentation) requires full retraining, which is highly inefficient.

**Goal**: (a) How to efficiently learn the trade-off between adversariality and realism? (b) How to continuously control the attack intensity of generated scenarios at inference time without retraining? (c) How to ensure map compliance (a hard constraint) is not diluted by soft preferences?

**Key Insight**: Inspired by multi-objective alignment in LLMs (e.g., the 3H principles) and model weight interpolation (linear mode connectivity), the authors reframe adversarial scenario optimization as a preference alignment problem. Two expert models biased toward opposite extremes are trained, and at inference time, linear weight interpolation traverses the entire Pareto frontier.

**Core Idea**: Transform adversarial scenario generation from "manually designing weighted objectives" to "learning a steerable preference landscape," achieving continuously adjustable test-time control via expert weight interpolation.

## Method

### Overall Architecture
The input is a real-world driving scene (road map and historical trajectories); the output is an adversarially perturbed trajectory for a designated opponent vehicle. The pipeline consists of three stages: (1) defining a multi-objective optimization problem over a pretrained motion generation model; (2) training two preference expert models via HGPO (Hierarchical Group Preference Optimization); (3) generating a continuously controllable scenario spectrum through weight interpolation at inference time.

### Key Designs

1. **Hierarchical Group Preference Optimization (HGPO)**:

    - **Function**: Fine-tunes a pretrained motion model using a DPO-style offline alignment approach, simultaneously handling hard constraints (map compliance) and soft preferences (adversariality vs. realism).
    - **Mechanism**: Map compliance is decoupled from the reward function and treated as a binary feasibility precondition $F(\tau, \mathcal{M}) \in \{0,1\}$ rather than a continuous penalty term. For each scene, $N$ trajectories are sampled and grouped by feasibility to construct two-level preference pairs: (a) feasible trajectories always preferred over infeasible ones; (b) feasible trajectories ranked internally by $R_{\text{pref}} = w_{\text{adv}} R_{\text{adv}} - w_{\text{real}} P_{\text{real}}$. Standard DPO loss is then applied over all preference pairs.
    - **Design Motivation**: Linearly penalizing map violations conflates hard and soft constraints—passing through a wall is not "slightly undesirable" but "entirely invalid"—causing the model to exploit out-of-map shortcuts. Group-based sampling is substantially more data-efficient than single winner-loser pairs.
    - **Novelty vs. Prior Work**: Standard DPO selects a single best/worst pair per scene; HGPO extracts multiple preference pairs from a group of samples, greatly improving sample utilization.

2. **Test-Time Steerable Generation (Mixture of Preferences)**:

    - **Function**: Enables continuous control over scenario attack intensity at inference time through linear interpolation of two expert model weights.
    - **Mechanism**: Two expert models are trained—$\pi_{\theta_{\text{adv}}}$ (adversariality-biased) and $\pi_{\theta_{\text{real}}}$ (realism-biased)—fine-tuned from the same pretrained model using opposing preference weights $w^*$. At inference time, a mixed model is constructed as $\theta(\lambda) = (1-\lambda)\theta_{\text{real}} + \lambda\theta_{\text{adv}}$; users slide continuously along the Pareto frontier by adjusting $\lambda \in [0,1]$, and can even extrapolate to $\lambda > 1$ to generate more extreme scenarios.
    - **Design Motivation**: Since both experts are fine-tuned from the same pretrained model on related tasks, the linear mode connectivity (LMC) assumption guarantees they reside in the same low-loss basin, so linear weight interpolation does not traverse high-loss regions.
    - **Theoretical Guarantees**: Theorem 1 proves that the sub-optimality of the interpolated model scales with the squared distance between the two expert weights (smaller distance implies smaller gap); Proposition 1 proves that, under concavity of the reward landscape, weight-space mixing outperforms output-space ensembling.

3. **Dual-Axis Curriculum for Closed-Loop Adversarial Training**:

    - **Function**: Integrates SAGE into closed-loop RL training of the ego policy via a progressive curriculum to prevent catastrophic forgetting.
    - **Mechanism**: Two dimensions are progressively increased simultaneously: (a) scenario intensity (increasing $\lambda$ from mild to aggressive); (b) frequency of adversarial scenario exposure. This ensures the ego model does not forget normal driving due to excessive exposure to extreme scenarios.

### Loss & Training
The HGPO loss is an extended DPO loss that takes expectations over all grouped preference pairs:
$$\mathcal{L}_{\text{HGPO}}(\theta) = \mathbb{E}\left[-\log\sigma\left(\beta\left(\log\frac{\pi_\theta(\tau^w|c)}{\pi_{\text{ref}}(\tau^w|c)} - \log\frac{\pi_\theta(\tau^l|c)}{\pi_{\text{ref}}(\tau^l|c)}\right)\right)\right]$$
where $\beta$ controls alignment strength and $(\tau^w, \tau^l)$ are drawn from hierarchical group sampling.

## Key Experimental Results

### Main Results
Evaluated on the MetaDrive simulator with the Waymo Open Motion Dataset, compared against 6 state-of-the-art baselines.

| Method | Attack Success Rate↑ | Adversarial Reward↑ | Behavioral Realism Penalty↓ | Kinematic Penalty↓ | Lane Violation Penalty↓ |
|--------|---------------------|---------------------|-----------------------------|--------------------|------------------------|
| Rule | 100.00% | 5.048 | 2.798 | 5.614 | 7.724 |
| CAT | 94.85% | 3.961 | 8.941 | 3.143 | 9.078 |
| GOOSE | 36.07% | 2.378 | 4.718 | 21.32 | 14.48 |
| SAGE (w=1.0) | 76.15% | 4.121 | **1.429** | **2.479** | **1.084** |

Closed-loop training evaluation (ego policy quality):

| Training Method | Reward↑ | Completion Rate↑ | Collision Rate↓ |
|----------------|---------|-----------------|----------------|
| SAGE | **45.14** | **0.69** | **0.31** |
| CAT | 37.70 | 0.58 | 0.37 |
| Replay | 41.32 | 0.62 | 0.44 |
| Rule-based | 32.99 | 0.50 | 0.33 |

### Ablation Study

| Configuration | Key Effect | Notes |
|--------------|-----------|-------|
| HGPO (full) | Fast convergence + high reward | Group preference pairs provide rich signal |
| Replace with standard DPO | Slow convergence, low sample efficiency | Only one preference pair per scene |
| Remove map hard constraint | Map feasibility collapses | Model learns to exploit shortcuts |
| Map as weighted penalty | Feasibility improves but remains suboptimal | Hard/soft constraints conflated |

### Key Findings
- SAGE reduces map violation penalties by 85%+ while maintaining high attack success rates, validating the effectiveness of decoupling hard constraints.
- The Pareto frontier generated by weight interpolation strictly dominates logit-space and trajectory-space mixing, empirically confirming LMC theory and Proposition 1.
- In closed-loop training, ego policies trained with SAGE exhibit the best generalization under cross-evaluation (maintaining high completion rates under different attack distributions).
- Weight extrapolation ($\lambda > 1$) can generate more extreme scenarios beyond the training convex hull.

## Highlights & Insights
- **The hard constraint decoupling design is particularly elegant**: Elevating map compliance from a continuous penalty to a binary precondition fundamentally prevents the model from learning shortcut behaviors. This idea transfers directly to any multi-objective optimization setting with mixed hard and soft constraints.
- **Validation of LMC in motion generation models**: This work is the first to empirically verify linear mode connectivity in motion generation models and leverage it to theoretically justify weight interpolation, providing a principled foundation for multi-objective control in other generative models (e.g., image, text).
- **Dual-axis curriculum against catastrophic forgetting**: Progressively adjusting both scenario intensity and frequency allows the ego policy to learn to handle extreme scenarios without forgetting normal driving—a trick that can be directly applied to other adversarial training pipelines.

## Limitations & Future Work
- The current framework considers only two objectives (adversariality vs. realism); scaling to additional objectives (e.g., scenario novelty, complexity) and the resulting growth in weight-space dimensionality remain unexplored.
- Linear interpolation relies on the LMC assumption, which may break down when expert models diverge significantly.
- The physical fidelity of the MetaDrive simulator is limited; performance in higher-fidelity simulators or real-world settings requires further validation.
- Potential improvements include adaptive curricula based on ego policy learning progress (replacing manual annealing) and more advanced model merging techniques.

## Related Work & Insights
- **vs. CAT (Zhang et al., 2023)**: CAT achieves adversarial generation through candidate resampling; while its attack success rate is high, behavioral realism penalties are substantially larger (8.941 vs. SAGE's 1.429), and test-time control is unavailable.
- **vs. GOOSE (Ransiek et al., 2024)**: GOOSE adopts RL for adversarial generation but incurs very high kinematic penalties (21.32), producing physically implausible trajectories.
- **vs. DPO/RLHF in LLMs**: SAGE is the first to transfer multi-objective preference alignment from LLM alignment to motion generation, demonstrating the cross-domain feasibility of this paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to introduce test-time multi-objective preference alignment into adversarial scenario generation, with tight integration of theory and practice.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Open-loop + closed-loop + cross-evaluation + ablation + theoretical validation; comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, rigorous theoretical derivations, and highly informative figures and tables.
- Value: ⭐⭐⭐⭐⭐ Provides an efficient and theoretically grounded new paradigm for autonomous driving safety testing.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](../../CVPR2026/autonomous_driving/composing_driving_worlds_through_disentangled_cont.md)
- [\[CVPR 2026\] CompoSIA: Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation](../../CVPR2026/autonomous_driving/composing_driving_worlds_through_disentangled_control_for_adversarial_scenario_g.md)
- [\[CVPR 2026\] Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](../../CVPR2026/autonomous_driving/drive_my_way_preference_alignment_of_vision-language-action_model_for_personaliz.md)
- [\[CVPR 2026\] TT-Occ: Test-Time 3D Occupancy Prediction](../../CVPR2026/autonomous_driving/test-time_3d_occupancy_prediction.md)
- [\[CVPR 2026\] MetaDAT: Generalizable Trajectory Prediction via Meta Pre-training and Data-Adaptive Test-Time Updating](../../CVPR2026/autonomous_driving/metadat_generalizable_trajectory_prediction_via_me.md)

<!-- RELATED:END -->

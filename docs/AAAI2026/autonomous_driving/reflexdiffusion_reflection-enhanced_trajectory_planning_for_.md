---
title: >-
  [Paper Note] ReflexDiffusion: Reflexion-Enhanced Trajectory Planning for High Lateral Acceleration in Autonomous Driving
description: >-
  [AAAI 2026][Autonomous Driving][Diffusion Planning] This paper proposes ReflexDiffusion, which introduces a physics-aware reflection mechanism during the inference stage of diffusion models. By injecting gradients to enforce curvature-velocity-acceleration coupling constraints ($a_y = \kappa v^2$), the method achieves a 14.1% improvement in driving score on nuPlan high-lateral-acceleration long-tail scenarios. The architecture-agnostic design allows direct deployment on exist…
tags:
  - "AAAI 2026"
  - "Autonomous Driving"
  - "Diffusion Planning"
  - "Reflection Mechanism"
  - "High Lateral Acceleration"
  - "Curvature-Velocity Coupling"
  - "Classifier-Free Guidance"
date: 2026-05-08
content_hash: 644266616c69689f
---

# ReflexDiffusion: Reflexion-Enhanced Trajectory Planning for High Lateral Acceleration in Autonomous Driving

**Conference**: AAAI 2026
**arXiv**: [2601.09377](https://arxiv.org/abs/2601.09377)  
**Code**: [https://github.com/Luminous2028/ReflexDiffusion.git](https://github.com/Luminous2028/ReflexDiffusion.git)  
**Area**: Autonomous Driving / Trajectory Planning / Diffusion Models
**Keywords**: Diffusion Planning, Reflection Mechanism, High Lateral Acceleration, Curvature-Velocity Coupling, Classifier-Free Guidance

## TL;DR

This paper proposes ReflexDiffusion, which introduces a physics-aware reflection mechanism during the inference stage of diffusion models. By injecting gradients to enforce curvature-velocity-acceleration coupling constraints ($a_y = \kappa v^2$), the method achieves a 14.1% improvement in driving score on nuPlan high-lateral-acceleration long-tail scenarios. The architecture-agnostic design allows direct deployment on existing diffusion planners.

## Background & Motivation

High lateral acceleration maneuvers (sharp curves, U-turns, high-curvature ramps) represent the highest fatality-risk yet most underrepresented long-tail scenarios in autonomous driving training data. When $|a_y| \geq 4.0\ \text{m/s}^2$ persists for $\geq 0.5\text{s}$, the vehicle approaches its dynamic limits, and trajectory planning must precisely satisfy the centripetal force constraint $a_y = \kappa v^2$.

**Limitations of Prior Work:**
- **Rule-based methods** (IDM/PDM-Closed): Manually designed constraints cannot generalize to novel scenarios; IDM scores only 36.74 on Test14-hard high lateral acceleration scenarios.
- **Learning-based methods** (imitation learning/RL): Difficulty in capturing multimodal driving behavior; UrbanDriver scores only 26.09, producing suboptimal unimodal trajectories.
- **Diffusion planners** (Diffusion Planner): Perform well in common scenarios but suffer from curvature-velocity decoupling in high lateral acceleration scenarios due to training data imbalance — planned curvature $\kappa$ and vehicle speed $v$ fail to satisfy the centripetal force constraint, resulting in a driving score of 0 in U-turn scenarios.
- **Classifier Guidance**: Requires manually designed continuously differentiable guidance functions; multi-objective constraint design is complex and struggles with non-differentiable safety constraints.

**Key Challenge**: How can inference-stage compensation for training data sparsity enable diffusion models to generate safe trajectories near physical limits?

## Method

### Overall Architecture

A four-module architecture: (a) Training Module — 10% conditional Dropout for robustness; (b) Denoising Module — Classifier-Free Guidance for initial trajectory generation; (c) Reflection Module — physics-aware gradient injection for iterative refinement; (d) Trajectory Confidence Module — multi-factor confidence evaluation for dynamic reflection triggering.

Inputs include ego state $(x, y, \cos\theta, \sin\theta)$, neighbor history (21 past timesteps), HD map (lane polylines + traffic lights + speed limits), static obstacles, and navigation route. Output is an 8-second @ 10Hz joint trajectory matrix for ego and $M$ neighbors: $x^{(0)} \in \mathbb{R}^{(M+1)\times 80\times 4}$.

### Key Designs

1. **Conditional Dropout Training Strategy**: With 10% probability, the full condition $c_\text{full} = [c_\text{neighbors}, c_\text{lanes}, c_\text{nav}, c_\text{static\_obj}]$ is degraded to a decoupled condition $c_\text{decouple} = [c_\text{nav}]$ (navigation only, removing lane curvature $R$ and vehicle speed $v$). This forces the model to learn robust representations capable of generating reasonable trajectories even without physical coupling information, while providing an "unconditional baseline" for CFG inference. Applying conditional Dropout to Diffusion Planner without CFG leads to a large performance drop (14.04 points), confirming that Dropout must be used in conjunction with CFG and reflection.

2. **Classifier-Free Guidance Denoising**: Leveraging the conditional Dropout from training, inference amplifies key physical conditioning signals via the difference between conditional and unconditional predictions. CFG formula:
$$\hat{\epsilon}_\theta^t = \epsilon_\theta(x|c_\text{decouple}) + \lambda_1 \cdot [\epsilon_\theta(x|c_\text{full}) - \epsilon_\theta(x|c_\text{decouple})]$$
where $\lambda_1 = 0.9$. A DDIM scheduler ensures deterministic denoising paths.

3. **Physics-Aware Reflection Mechanism**: Triggered when confidence $C(x_{t-1}) < \gamma$. Core steps:
    - Recover $x'_t$ from $x_{t-1}$ via reverse noising, approximating $\epsilon_\theta^t(x_t) \approx \epsilon_\theta^t(x_{t-1})$
    - Compute conditional gradient difference $\Delta_\text{couple}$ encoding road curvature $\kappa$ and vehicle speed $v$ coupling
    - Map gradients onto the centripetal force constraint manifold $a_y \approx \kappa v^2$ via projection matrix $P = \begin{bmatrix} v^2 & 2\kappa v \\ 0 & 1 \end{bmatrix}$
    - Obtain physically consistent correction: $x'_t = \sqrt{\alpha_t} \cdot x_{t-1} + b \cdot \Delta_\text{proj}$
    - Projection matrix $P$ design: the upper row exploits $\partial(\kappa v^2)/\partial\kappa = v^2$ and $\partial(\kappa v^2)/\partial v = 2\kappa v$ to amplify coupling; the lower row $[0, 1]$ preserves free motion.

4. **Trajectory Confidence Module**: Three-factor evaluation:
    - $D_\text{kin}$ (kinematic consistency): checks deviation of $a_y^\text{traj}$ vs. $a_y^\text{ref}$, and whether lateral jerk $j_\text{lat}$ exceeds limits
    - $G_\text{align}$ (geometric alignment): deviation between trajectory curvature $\kappa_\tau$ and road curvature $\kappa_\text{road}$, plus maximum lateral offset check
    - $S_\text{margin}$ (safety margin): TTC $\geq 2.5\text{s}$, out-of-drivable-area probability $p_\text{ODA}$, heading deviation $\Delta\psi$
    - Weighted combination compared against threshold $\gamma = 0.8$ to determine whether to trigger reflection

### Loss & Training

- Training: $\mathcal{L}_\theta = \mathbb{E}[||x_\text{gt} - x_t||^2]$ with 10% conditional Dropout and DDPM noise schedule
- Inference hyperparameters: CFG scale $\lambda_1 = 0.9$, reflection scale $\lambda_2 = 0.0$ (physical correction solely from projection matrix), confidence threshold $\gamma = 0.8$
- Reflection is triggered in $\leq 0.5\%$ of actual driving scenarios, adding only ${\sim}0.4\text{ms}$ to average runtime

## Key Experimental Results

### Main Results: High Lateral Acceleration Driving Score

| Type | Method | Test14-Hard NR | Test14-Hard R | Test14-Random NR | Test14-Random R |
|------|--------|:-:|:-:|:-:|:-:|
| Rule | IDM | 36.74 | 62.42 | 67.61 | 64.66 |
| Rule | PDM-Closed | 32.55 | 53.03 | 75.69 | 82.17 |
| Hybrid | Gameformer | 53.12 | 57.46 | 82.60 | 79.49 |
| Hybrid | SAH-Drive | 43.08 | 57.40 | 91.18 | 89.27 |
| Learning | Diffusion-es | 44.63 | 52.72 | 88.20 | 84.20 |
| Learning | Pluto | 42.21 | 45.98 | 81.67 | 75.95 |
| Learning | Diffusion Planner | 58.47 | 57.41 | 71.60 | 82.88 |
| Learning | DP + Dropout | 12.60 | 14.04 | 38.45 | 44.88 |
| Learning | DP + Dropout + CFG | 44.40 | 55.55 | 76.44 | 70.21 |
| **Learning** | **ReflexDiffusion** | **59.94** | **65.53** | **86.40** | **71.57** |

### Ablation Study

| Ablation Setting | Test14-Hard Driving Score (R) |
|-----------------|:-:|
| Full ReflexDiffusion | **65.53** |
| Remove Conditional Dropout | 23.86 |
| Remove CFG Denoising | 59.85 |
| Remove Reflection Mechanism | 53.21 |

### Runtime & Generalization

| Planner | Per-Step Latency (ms) | End-to-End Latency (ms) |
|---------|:-:|:-:|
| Diffusion-es | — | 7612.7 |
| Diffusion Planner | — | 35.7 |
| ReflexDiffusion (reflection step) | 6.3 | 36.1 |

| Generalization Test | Original Score | + ReflexDiffusion |
|--------------------|:-:|:-:|
| Diffusion Planner (Test14-hard R) | 57.41 | **65.53** (+14.1%) |
| Diffusion-es (Test14-hard R) | 31.88 | **39.04** (+22.5%) |

### Key Findings

- **U-Turn scenario**: Diffusion Planner scores 0 (trajectory leaves the lane); ReflexDiffusion scores 100, with confidence rising from 0.48 to 0.87.
- The reflection trigger rate is extremely low ($\leq 0.5\%$), adding negligible average inference latency (35.7 → 36.1 ms) while maintaining real-time control frequency of $> 20\text{Hz}$.
- Conditional Dropout is the most critical module — removing it drops the score to 23.86; the reflection mechanism ranks second (53.21); CFG has the smallest individual impact (59.85).
- Optimal hyperparameter combination: Dropout rate = 10%, $\lambda_1 = 0.9$, $\lambda_2 = 0.0$, $\gamma = 0.8$ (sensitive range for $\gamma$: $[0.75, 0.85]$).

## Highlights & Insights

- **First application of LLM-style reflection to trajectory planning**: The generate-evaluate-refine paradigm crosses from NLP into autonomous driving, representing a conceptual breakthrough.
- **Physics prior embedded in the generative process**: Projection matrix $P$ maps gradients onto the centripetal force manifold $a_y = \kappa v^2$ rather than applying it as an external penalty.
- **Architecture-agnostic**: As an inference-stage plugin, training requires only adding 10% Dropout with no modification to model architecture.
- Insight into why reflection works: the information dilution and error accumulation inherent in standard denoising are especially severe in long-tail scenarios; the reflection mechanism amplifies weak physical signals through a noise-then-denoise loop.

## Limitations & Future Work

- $\lambda_2 = 0.0$ means the "conditional gradient" term in reflection is effectively unused; physical correction relies entirely on the projection matrix — the reflection mechanism may be oversimplified.
- Validation is limited to the nuPlan benchmark; more complex real-world scenarios (e.g., icy roads, extreme weather) are untested.
- The weight assignment for the three factors in the confidence module is not elaborated.
- Performance on Test14-Random R (71.57) falls below some baselines (SAH-Drive: 89.27) despite significant U-turn improvements.

## Related Work & Insights

- Reflection mechanisms (Reflexion/Self-Refine for LLMs) → refine loops in diffusion denoising
- CFG (Classifier-Free Guidance) transferred from image generation to trajectory planning
- Centripetal force constraint $a_y = \kappa v^2$ is generalizable to other generative tasks requiring physical consistency

## Rating

- Novelty: ⭐⭐⭐⭐ (first combination of reflection and physics-constraint injection)
- Technical Depth: ⭐⭐⭐⭐ (complete theoretical derivation, elegant projection matrix design)
- Experimental Thoroughness: ⭐⭐⭐⭐ (main results + ablation + generalization + runtime + visualization)
- Practical Value: ⭐⭐⭐⭐⭐ (plug-and-play, real-time capable)

### Main Results: nuPlan Test14-Hard & Test14-Random

| Benchmark & Mode | Metric | ReflexDiffusion | Diffusion Planner | PDM-Closed | GameFormer | Gain |
|-----------------|--------|----------------|-------------------|------------|------------|------|
| Test14-Hard High Lat. Accel. R | Driving Score | **65.53** | 57.41 | 63.41 | 63.47 | +14.1% |
| Test14-Hard NR | Driving Score | **59.94** | 58.47 | 56.07 | 50.73 | +2.5% |
| Test14-Random NR | Driving Score | **86.40** | 71.60 | 76.59 | 75.80 | +20.7% |
| U-Turn Scenario | Driving Score | **100.0** | 0.0 | — | — | Failure → Perfect |

### Ablation Study

| Module | Test14-Hard R Score | Notes |
|--------|-------------------|-------|
| Full ReflexDiffusion | **65.53** | — |
| Remove Conditional Dropout | 23.86 | Most critical component; 63% degradation |
| Remove CFG Denoising | 59.85 | Minor drop |
| Remove Reflection Mechanism | 53.21 | Significant drop |
| Dropout rate 5% | 56.62 | Insufficient |
| Dropout rate 10% | **65.53** | Optimal |
| Dropout rate 20% | 61.70 | Excessive |
| $\gamma = 0.75$ | 63.14 | Moderate sensitivity |
| $\gamma = 0.85$ | 64.82 | Moderate sensitivity |

### Key Findings

- Conditional Dropout is the **most critical component** — removing it causes the score to collapse from 65.53 to 23.86, as the "unconditional baseline" is required for computing physics-aware gradients.
- The reflection mechanism is triggered in $\leq 0.5\%$ of scenarios yet yields dramatic improvements in long-tail cases (U-Turn: 0 → 100).
- Runtime overhead is minimal: 36.1 ms vs. baseline 35.7 ms (+1.1%), maintaining $> 20\text{Hz}$ real-time requirements.
- Per-step latency during triggering increases from 3.3 ms to 6.3 ms, but the negligible trigger rate renders the overall impact insignificant.
- Architecture-agnostic: also improves Diffusion-es planner by 22.5% (Test14-Hard NR Score: 72.60 → 88.96).
- Confidence decreases then increases during reflection, indicating an active explore-then-correct process.

## Highlights & Insights

- First introduction of the LLM-style "reflection" (generate-evaluate-refine) paradigm into autonomous driving trajectory planning.
- Physics projection matrix $P$ maps abstract gradient corrections onto a concrete centripetal force constraint, providing strong physical interpretability.
- Architecture-agnostic plug-and-play design — no modification to any underlying diffusion planner architecture required.
- The combination of conditional Dropout and CFG constitutes a general technique for handling imbalanced training data.
- The qualitative improvement in U-turn scenarios from score 0 to perfect is highly compelling.

## Limitations & Future Work

- The physics projection matrix $P$ assumes a simplified single-track (bicycle) dynamic model; more complex conditions may require higher-fidelity models.
- The confidence threshold $\gamma$ requires cross-dataset tuning (sensitive range: $[0.75, 0.85]$).
- The reflection mechanism increases determinism (DDIM), potentially sacrificing diversity in multimodal sampling.
- Validation is limited to nuPlan closed-loop testing; broader real-world deployment testing is needed.

## Related Work & Insights

- vs. Diffusion Planner (ICLR 2025): inference-stage enhancement vs. training-stage optimization — fully complementary.
- vs. Classifier Guidance: no need for manually designed differentiable guidance functions; physical constraints are embedded within the gradient.
- vs. Rule-based methods (PDM-Closed): substantial lead in long-tail scenarios with improvements in common scenarios as well.
- The inference-stage reflection/correction paradigm is generalizable to other safety-critical generative tasks (medical image generation, chemical molecule design, etc.).

## Rating

⭐⭐⭐⭐⭐ (5/5)
The method demonstrates strong novelty (first application of reflection mechanism to trajectory planning), with an elegant and practical architecture-agnostic plug-and-play design. nuPlan experiments are comprehensive, with significant improvements in long-tail safety-critical scenarios. The work directly serves safety-critical requirements in autonomous driving.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] HD2-SSC: High-Dimension High-Density Semantic Scene Completion for Autonomous Driving](hd2-ssc_high-dimension_high-density_semantic_scene_completion_for_autonomous_dri.md)
- [\[AAAI 2026\] DriveSuprim: Towards Precise Trajectory Selection for End-to-End Planning](drivesuprim_towards_precise_trajectory_selection_for_end-to-end_planning.md)
- [\[AAAI 2026\] DiffRefiner: Coarse to Fine Trajectory Planning via Diffusion Refinement with Semantic Interaction for End to End Autonomous Driving](diffrefiner_coarse_to_fine_trajectory_planning_via_diffusion_refinement_with_sem.md)
- [\[ICLR 2026\] BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving](../../ICLR2026/autonomous_driving/bridgedrive_diffusion_bridge_policy_for_closed-loop_trajectory_planning_in_auton.md)
- [\[AAAI 2026\] WorldRFT: Latent World Model Planning with Reinforcement Fine-Tuning for Autonomous Driving](worldrft_latent_world_model_planning_with_reinforcement_fine-tuning_for_autonomo.md)

</div>

<!-- RELATED:END -->

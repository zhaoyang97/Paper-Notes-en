---
title: >-
  [Paper Note] When would Vision-Proprioception Policies Fail in Robotic Manipulation?
description: >-
  [ICLR 2026][Robotics][vision-proprioception policy] This paper identifies why vision-proprioception manipulation policies fail during motion-transition phases—proprioceptive signals dominate optimization and suppress vis…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "vision-proprioception policy"
  - "modality temporality"
  - "gradient adjustment"
  - "motion-transition phase"
  - "robotic manipulation"
date: 2026-05-08
content_hash: fb2ef5c1665003d0
---

# When would Vision-Proprioception Policies Fail in Robotic Manipulation?

**Conference**: ICLR 2026
**arXiv**: [2602.12032](https://arxiv.org/abs/2602.12032)  
**Code**: [Project Page](https://gewu-lab.github.io/GAP/)  
**Area**: Robotics
**Keywords**: vision-proprioception policy, modality temporality, gradient adjustment, motion-transition phase, robotic manipulation

## TL;DR
This paper identifies why vision-proprioception manipulation policies fail during motion-transition phases—proprioceptive signals dominate optimization and suppress visual learning—and proposes the Gradient Adjustment with Phase-guidance (GAP) algorithm, which adaptively attenuates proprioceptive gradients to restore visual modality learning, achieving significant generalization improvements in both simulated and real-world environments.

## Background & Motivation

Proprioception provides real-time joint state information and is considered essential for precise servoing control. In learned manipulation policies, combining proprioception with vision is widely believed to enhance performance on complex tasks. However, existing studies have reported **contradictory findings**:
- HPT demonstrates that vision + proprioception significantly outperforms vision-only policies.
- Octo finds that adding proprioception actually degrades performance.

**Key Challenge**: Although proprioception should theoretically be a beneficial complement, vision-proprioception policies in practice often generalize worse than vision-only policies. **What is the root cause of this contradiction, and under what conditions does it occur?**

**Key Findings**: Through temporally controlled experiments, the authors identify the problem as occurring specifically during **motion-transition phases**. Robot motion in manipulation tasks can be decomposed into *motion-consistent phases* (e.g., continuous forward movement) and *motion-transition phases* (e.g., reorienting toward a new target and changing movement direction).

- **Motion-consistent phases**: Proprioceptive signals are effective and policies perform normally.
- **Motion-transition phases**: The visual modality should play a critical role (target localization), but visual learning is suppressed in vision-proprioception policies.

**Analysis**: From an optimization perspective, proprioceptive signals are compact and low-dimensional, providing faster loss reduction during training and thus dominating optimization. Although visual signals carry critical information such as target localization, their learning is suppressed because pixel-level variations are subtler than proprioceptive signals—a manifestation of modality competition/modality laziness.

## Method

### Overall Architecture
The GAP pipeline proceeds as follows:
1. **Input**: Expert demonstration trajectories (containing visual and proprioceptive information).
2. **Motion Representation**: Robot motion is defined based on proprioceptive signals.
3. **Motion-Transition Phase Estimation**: Change Point Detection (CPD) + LSTM predict the probability $\rho$ that each timestep belongs to a motion-transition phase.
4. **Gradient-Adjusted Policy Learning**: During training, the gradient magnitude of the proprioceptive branch is reduced according to $\rho$.
5. **Output**: A vision-proprioception policy with improved generalization.

### Key Designs

1. **Motion Representation**: Robot motion is defined based on proprioceptive signals. The motion from timestep $i$ to $j$ is defined as $m_{i:j} = \{p_{i:j}, \theta_{i:j}, g_{i:j}\}$, where $p_{i:j}$ denotes gripper position change, $\theta_{i:j}$ denotes orientation change, and $g_{i:j}$ denotes gripper aperture change. These three dimensions fully characterize the motion of the robot arm.

2. **Change Point Detection (CPD) Segmentation**: Dynamic programming is used to identify timesteps at which the direction of motion undergoes a fundamental change. A motion consistency distance is defined as:
    $d(m_{t_1:t_2}, m_{i:i+1}) = -\cos(p_{t_1:t_2}, p_{i:i+1}) - \alpha\cos(\theta_{t_1:t_2}, \theta_{i:i+1}) - \beta(\text{sgn}(g_{t_1:t_2}) == \text{sgn}(g_{i:i+1}))$
   where $\alpha=1$ and $\beta=2\times10^{-3}$ balance the contributions of position, orientation, and gripper aperture. CPD minimizes the total cost to segment trajectories into motion-consistent phases.

3. **LSTM Temporal Modeling for Transition Probability Prediction**: CPD produces discrete breakpoints, but motion transitions are continuous processes. An LSTM network is therefore used to model the temporal differences of proprioceptive signals $\Delta s_i = s_{i+1} - s_i$ and predict a continuous transition probability $\rho_i \in [0,1]$ for each timestep. The LSTM uses CPD output as supervision and applies reduced penalties to timesteps near transition points to better capture gradual transitions. This outperforms directly using discrete CPD labels (verified by ablation studies).

4. **Gradient Adjustment**: The core technique—during each training epoch, the parameters $\omega_s$ of the proprioceptive feature extractor are modulated as:
    $\omega_s^{j+1} = \omega_s^j - \lambda \cdot (1-\rho) \cdot \eta \nabla_{\omega_s^j} \mathcal{L}_{BC}(\omega_s^j)$
   where $\lambda=0.3$ controls the degree of adjustment and $\rho$ is the transition probability. When $\rho$ is high (motion-transition phases), proprioceptive gradients are substantially suppressed, forcing the network to rely more heavily on visual signals to learn behavior during these phases. This constitutes a **fine-grained, phase-aware modality balancing strategy**.

5. **GAP Applied Only During Early Training**: Gradient adjustment is applied only during the first 50 of 100 total epochs, avoiding training instability and policy collapse caused by excessive modulation. Ablation experiments confirm the robustness of this choice.

### Loss & Training
- Behavior Cloning (BC) paradigm with MSE loss as the baseline.
- Visual branch: ResNet-18 → 512-dimensional representation → 4-layer temporal transformer.
- Proprioceptive branch: 3-layer MLP.
- Features from both branches are fused via concatenation and passed to a 3-layer MLP policy head that outputs an action sequence of length $L=9$.
- Training: Adam optimizer, learning rate 3e-4, batch size 128, single RTX 3090, 100 epochs.
- Meta-World uses 100 expert demonstrations; RoboSuite uses 500 synthesized trajectories.

## Key Experimental Results

### Main Results (Simulation + Real World)

| Method | Meta-World (4-task avg.) | RoboSuite (4-task avg.) | Notes |
|--------|--------------------------|--------------------------|-------|
| Vision-only | 80.5% | 64.6% | Vision-only baseline |
| Concatenation | 71.8% | 53.8% | Vision-proprioception, performance **drops** |
| MS-Bot | 83.5% | 69.4% | Uses semantic phase information |
| Aux Loss | 83.5% | 56.3% | Auxiliary visual prediction loss |
| Mask | 84.5% | 62.0% | Random proprioception masking |
| **GAP (Ours)** | **88.4%** | **73.0%** | **Best overall** |

Real-world experiments (20 rollouts each):

| Method | Single-arm (3 tasks) | Bimanual (3 tasks) |
|--------|----------------------|--------------------|
| Vision-only | 41/60 | 35/60 |
| Concatenation | 28/60 | 24/60 |
| **GAP (Ours)** | **50/60** | **49/60** |

### OOD Generalization

| Method | assembly | bin-picking | stack | threading | cube | handover |
|--------|----------|-------------|-------|-----------|------|----------|
| Vision-only | 78% | 59% | 63% | 32% | 12/20 | 12/20 |
| Concatenation | 62% | 32% | 49% | 28% | 7/20 | 9/20 |
| **GAP** | **88%** | **67%** | **72%** | **49%** | **15/20** | **15/20** |

### Compatibility with VLA Models (Octo Fine-tuning)

| Method | disassemble | push-wall | put hammer | threading |
|--------|-------------|-----------|------------|-----------|
| Octo-V (vision-only) | 95% | 77% | 92% | 69% |
| Octo-VP (vision+proprioception) | 82% | 65% | 88% | 57% |
| Octo-VP + **GAP** | **100%** | **85%** | **97%** | **78%** |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Human annotation vs. CPD | GAP (CPD) superior | Automatic detection more comprehensive than manual |
| HDBSCAN clustering | Performance drops | Destroys temporal trajectory structure |
| CoTPC | Below GAP | Simple cosine distance insufficient |
| Fixed $\rho$ vs. LSTM | LSTM significantly better | Continuous probability outperforms discrete labels |
| $\lambda$ = 0.1/0.2/0.3/0.4/0.5 | 0.2–0.4 all viable | Insensitive to $\lambda$ |
| Epochs with GAP applied | 50 epochs optimal | Too few is insufficient; too many causes over-suppression |

### Key Findings
- Standard vision-proprioception concatenation **consistently underperforms** vision-only policies (average drop ~15%), confirming the generality of the OOD generalization problem.
- Intervention experiments precisely localize the failure: visual modality learning is suppressed during **motion-transition phases**, preventing effective target localization.
- GAP restores visual branch learning via gradient adjustment; linear-probing experiments directly confirm improved visual feature quality (e.g., assembly task improves from 61% to 74%).
- GAP is compatible with multiple fusion strategies (Concatenation, Summation, FiLM) and policy architectures (MLP head, Diffusion head, VLA/Octo).
- GAP is effective in both single-arm and bimanual real-world tasks, demonstrating practical deployment value.

## Highlights & Insights
- **Precise Diagnosis**: Temporally controlled intervention experiments pinpoint *motion-transition phases* as the critical failure mode, rather than offering a vague critique of multimodal learning.
- **Deep Optimization Perspective**: The suppression of visual learning is explained through the lens of gradient competition and modality laziness, connecting to theoretical analyses of multimodal learning.
- **Simple Yet Effective Method**: Only the gradient magnitude of the proprioceptive branch is modified—no architectural changes, no additional modules, no extra loss terms.
- **Strong Adaptability**: Compatible with CNN/Transformer/Diffusion/VLA architectures, single-arm/bimanual setups, and simulated/real environments.
- **Thorough Ablation**: Motion detection methods, LSTM vs. alternatives, and hyperparameters $\alpha$/$\beta$/$\lambda$/training epoch count are all ablated.
- **Counter-Intuitive Finding**: Proprioception should help the robot, yet in deep learning optimization it suppresses visual learning—revealing a previously overlooked pitfall in multimodal learning.

## Limitations & Future Work
- All experiments are conducted on a single embodiment; cross-embodiment generalization remains unvalidated.
- The motion consistency distance in CPD relies on hand-crafted metrics (cosine similarity + sign function), which may not generalize to all manipulation task types.
- LSTM training depends on CPD-provided labels; systematic biases in CPD segmentation would propagate to the LSTM.
- GAP is applied only during early training (first 50/100 epochs); this threshold may require adjustment across different tasks and data scales.
- The combination of more sophisticated modality fusion strategies (e.g., attention-based fusion) with GAP remains unexplored.
- A uniform $\lambda$ is applied to gradient adjustment across all timesteps; adaptive $\lambda$ scheduling could be considered.
- Only vision and proprioception are considered; additional modalities such as tactile sensing are not addressed.

## Related Work & Insights
- **Modality Competition/Modality Laziness** (Huang et al. 2022, Fan et al. 2023): A pervasive problem in multimodal learning where one modality dominates optimization; GAP provides a concrete solution in the robotic manipulation domain.
- **HPT (Wang et al. 2024)**: Demonstrates the positive effect of proprioception.
- **Octo**: Reports the negative effect of proprioception; this paper explains and resolves that contradiction.
- **Modality Temporality (Feng et al.)**: Introduces the concept that modality importance varies over time; this paper specializes it as *motion-transition phases*.
- **Diffusion Policy (Chi et al. 2023)**: GAP is compatible with diffusion heads and yields further improvements.
- Insight: The gradient adjustment approach can be generalized to any multimodal learning scenario in which one modality excessively dominates optimization.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel diagnostic perspective; method is simple but effective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (14+ tasks, multiple architectures, simulation + real world, comprehensive ablations, OOD, VLA compatibility)
- Writing Quality: ⭐⭐⭐⭐ (Problem-driven, logically clear, rich figures and tables)
- Value: ⭐⭐⭐⭐⭐ (Addresses a practical and pervasive problem in robotic manipulation)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MemoryVLA: Perceptual-Cognitive Memory in Vision-Language-Action Models for Robotic Manipulation](memoryvla_perceptual-cognitive_memory_in_vision-language-action_models_for_robot.md)
- [\[ICLR 2026\] VLBiMan: Vision-Language Anchored One-Shot Demonstration Enables Generalizable Bimanual Robotic Manipulation](vlbiman_vision-language_anchored_one-shot_demonstration_enables_generalizable_bi.md)
- [\[ICLR 2026\] RoboInter: A Holistic Intermediate Representation Suite Towards Robotic Manipulation](robointer_a_holistic_intermediate_representation_suite_towards_robotic_manipulat.md)
- [\[ICML 2026\] RoboMME: Benchmarking and Understanding Memory for Robotic Generalist Policies](../../ICML2026/robotics/robomme_benchmarking_and_understanding_memory_for_robotic_generalist_policies.md)
- [\[AAAI 2026\] SemanticVLA: Semantic-Aligned Sparsification and Enhancement for Efficient Robotic Manipulation](../../AAAI2026/robotics/semanticvla_semantic-aligned_sparsification_and_enhancement_for_efficient_roboti.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning
description: >-
  [ICLR 2026][Reinforcement Learning][multitask RL] This paper proposes ScaleZero, which incorporates a Mixture-of-Experts (MoE) architecture into a unified world model to address gradient conflict and plasticity collapse…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "multitask RL"
  - "world model"
  - "MoE"
  - "MCTS"
  - "plasticity collapse"
  - "dynamic parameter scaling"
date: 2026-05-08
content_hash: b70318927808b304
---

# One Model for All Tasks: Leveraging Efficient World Models in Multi-Task Planning

**Conference**: ICLR 2026
**arXiv**: [2509.07945](https://arxiv.org/abs/2509.07945)  
**Code**: [https://github.com/opendilab/LightZero](https://github.com/opendilab/LightZero)  
**Area**: Multi-Task Reinforcement Learning / World Models
**Keywords**: multitask RL, world model, MoE, MCTS, plasticity collapse, dynamic parameter scaling

## TL;DR
This paper proposes ScaleZero, which incorporates a Mixture-of-Experts (MoE) architecture into a unified world model to address gradient conflict and plasticity collapse in multi-task learning. Combined with a Dynamic Parameter Scaling (DPS) strategy that adaptively allocates model capacity, a single multi-task model achieves performance comparable to single-task expert models across three benchmarks (Atari/DMC/Jericho) while reducing environment interactions by approximately 28.5%.

## Background & Motivation
**Background**: Unified world models (e.g., UniZero) integrate representation learning, dynamics prediction, and policy learning into a single Transformer with MCTS planning, achieving state-of-the-art performance on individual tasks. Extending such models to heterogeneous multi-task learning (with diverse observation/action spaces and varying complexity) is a critical step toward general-purpose agents.

**Limitations of Prior Work**: UniZero suffers from severe **plasticity collapse** during multi-task training — simple tasks (e.g., Pong) converge rapidly and their gradients dominate shared parameter updates, causing complex tasks (e.g., Seaquest) to first learn and then degrade. This manifests as a sharp increase in the proportion of dormant neurons and uncontrolled expansion of latent state norms.

**Key Challenge**: **Gradient conflict** across heterogeneous tasks sharing a common backbone — gradient directions beneficial to task $i$ may be harmful to task $j$ (i.e., $\cos(g_i, g_j) < 0$). Furthermore, static resource allocation treats all tasks equally regardless of their complexity, leading to computational waste.

**Goal**: (1) How can gradient conflict and plasticity collapse be mitigated within a shared world model? (2) How can model capacity be dynamically allocated according to each task's learning progress?

**Key Insight**: Two complementary perspectives are adopted — at the **architecture level**, dense FFN layers are replaced with MoE to route different tasks through different expert paths, reducing interference; at the **process level**, DPS progressively injects LoRA adapters based on learning progress to expand capacity.

**Core Idea**: MoE mitigates gradient conflict within each training iteration + DPS optimizes resource allocation across the entire training process = an efficient multi-task world model.

## Method

### Overall Architecture
ScaleZero upgrades the UniZero architecture in three ways: (1) the encoder is replaced from ResNet to ViT; (2) dense FFN layers in the Transformer backbone are replaced with sparse MoE layers (comprising shared experts and task-routed specialized experts); (3) latent representation normalization uses SimNorm. Training employs a unified loss $\mathcal{L} = \sum_{t}(\mathcal{L}_{\text{value}} + \mathcal{L}_{\text{policy}} + \mathcal{L}_{\text{reward}} + \mathcal{L}_{\text{dynamics}})$. The DPS strategy triggers LoRA injection during training based on the rate of change in each task's return.

### Key Designs

1. **Sparse MoE Backbone (Core Architectural Innovation)**:

    - **Function**: Replaces dense FFN layers in the Transformer with MoE layers containing $N$ parallel experts, where only the top-k experts are activated per input.
    - **Mechanism**: $\text{MoE}(x) = \sum_{i=1}^{N} G_i(x) \cdot \text{Expert}_i(x)$, where the gating network $G(x)$ automatically selects a subset of experts. A hybrid design is adopted — 1 shared expert captures cross-task general knowledge, while multiple routed experts handle task-specific dynamics.
    - **Design Motivation**: Different tasks are routed to distinct parameter subsets, physically isolating gradient conflicts. Experiments confirm that MoE yields the largest gain among all architectural choices explored, and directly correlates with maintaining healthy dormant neuron ratios and latent state norms.

2. **Dynamic Parameter Scaling (DPS)**:

    - **Function**: Injects lightweight LoRA adapters in a stage-wise manner according to each task's learning progress to expand model capacity.
    - **Mechanism**: Each task's return curve is monitored; when a task is detected to have converged (stagnated progress), its parameters are frozen and new LoRA modules are injected into the paths of tasks that have not yet converged. The weight update takes the form $(W_0 + \alpha BA)x$, training only the small low-rank matrices $(A, B)$.
    - **Design Motivation**: Conventional methods allocate computational resources uniformly across tasks, ignoring large differences in task complexity. DPS realizes a form of "task curriculum" — simpler tasks are mastered and frozen first, after which resources are concentrated on harder tasks.

3. **Systematic Architectural Search**:

    - Controlled experiments are conducted along 5 dimensions: task conditioning method, encoder (ResNet vs. ViT), latent normalization (LayerNorm vs. SimNorm), backbone (dense vs. MoE), and optimization strategy (gradient correction MoCo vs. standard).
    - Result: The MoE backbone yields the largest and most consistent gains; SimNorm provides partial benefits; other modifications show marginal effects.

### Loss & Training
- Pure online RL training, without reliance on expert data or offline datasets.
- A shared inverse dynamics controller assists MCTS planning.
- The injection layers and rank of LoRA in DPS are configured adaptively based on the size of the task set.

## Key Experimental Results

### Main Results (Atari 100k, 26 Games, HNS)

| Method | Type | Norm. Mean | Norm. Median |
|--------|------|-----------|-------------|
| UniZero | 26 single-task models | 0.38 | 0.21 |
| UniZero | 1 multi-task model | 0.31 | 0.16 |
| **ScaleZero** | **1 multi-task model** | **0.39** | 0.16 |

### DMC Suite (18 Continuous Control Tasks)

| Method | Type | Mean Return | Median Return |
|--------|------|------------|--------------|
| UniZero (ST) | 18 single-task | 787.2 | 875.1 |
| **ScaleZero (MT)** | **1 multi-task** | 769.7 | **887.3** |

### DPS Efficiency Gains

| Configuration | Env. Interactions | Performance | Note |
|---------------|------------------|-------------|------|
| ScaleZero (full) | 100% | baseline | Full training |
| ScaleZero + DPS | **71.5%** | ≈ baseline | 28.5% reduction |

### Key Findings
- ScaleZero's single multi-task model surpasses the average normalized score of 26 single-task expert models (0.39 vs. 0.38), demonstrating positive knowledge transfer.
- MoE yields the largest gain among all architectural modifications: it directly elevates multi-task UniZero from severe collapse to parity with single-task performance.
- On previously collapsed hard tasks such as Seaquest, ScaleZero achieves stable learning with dormant neuron ratios maintained at healthy levels.
- DPS reduces environment interactions by approximately 28.5% while preserving performance, achieving a better performance–efficiency trade-off.
- Median scores on DMC surpass single-task performance (887.3 vs. 875.1), indicating that most tasks benefit from multi-task learning.

## Highlights & Insights
- **Quantitative Diagnosis of Plasticity Collapse**: Rather than vaguely attributing failure to "multi-task learning being difficult," the paper concretely quantifies failure modes via dormant neuron ratios and latent norms, tracing them to gradient competition and representation interference. This diagnostic framework is transferable to other multi-task learning settings.
- **First Successful Application of MoE in RL World Models**: Prior MoE usage has been largely confined to LLMs and supervised learning; this paper demonstrates its effectiveness in online RL world models, supported by both theoretical justification (gradient direction separation) and empirical evidence (improved plasticity metrics).
- **DPS as an Automatic Task Curriculum**: Rather than a predefined curriculum, DPS automatically discovers task difficulty ordering from real-time return feedback and adaptively schedules computational resources accordingly — a principle generalizable to any multi-task learning setting.

## Limitations & Future Work
- The 100k-step constraint in Atari 100k may be insufficient to fully reveal DPS benefits; more pronounced gains may emerge under longer training.
- MoE introduces additional gating networks and expert parameters; while total compute is controlled by sparsity, overall parameter count increases.
- Cross-modal inputs (visual vs. text vs. state) still require separate encoders, preventing truly unified input processing.
- DPS's task convergence detection relies on heuristic rules over return curves, which may be sensitive to reward noise.
- Validation is limited to RL benchmarks; extension to more complex real-world robotics tasks remains unexplored.

## Related Work & Insights
- **vs. UniZero**: UniZero unifies representation, dynamics, and prediction in a monolithic Transformer but collapses under multi-task training due to gradient conflict; ScaleZero replaces dense layers with MoE to address the root cause.
- **vs. L2M**: L2M also performs multi-task RL but relies on supervised learning from offline expert data; ScaleZero operates via pure online RL without requiring expert demonstrations.
- **vs. Gradient Correction Methods (PCGrad/NashMTL)**: Optimization-level corrections show limited effectiveness in this paper's experiments, proving less impactful than architectural-level improvements (MoE).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of MoE and world models is novel, and DPS offers a creative adaptive scaling mechanism, though individual components are not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three benchmarks across different modalities (Atari/DMC/Jericho), systematic architectural ablation, and plasticity metric analysis.
- **Writing Quality**: ⭐⭐⭐⭐ The logical chain from problem diagnosis → solution design → experimental validation is clear and complete.
- **Value**: ⭐⭐⭐⭐ Provides a practical architectural blueprint for multi-task world models with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Estimation of Kernel Surrogate Models for Task Attribution](efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [\[ICLR 2026\] From Observations to Events: Event-Aware World Model for Reinforcement Learning](from_observations_to_events_event-aware_world_model_for_reinforcement_learning.md)
- [\[ICLR 2026\] Deep SPI: Safe Policy Improvement via World Models](deep_spi_safe_policy_improvement_via_world_models.md)
- [\[ICLR 2026\] WIMLE: Uncertainty-Aware World Models with IMLE for Sample-Efficient Continuous Control](wimle_uncertainty-aware_world_models_with_imle_for_sample-efficient_continuous_c.md)
- [\[ICLR 2026\] Model Predictive Adversarial Imitation Learning for Planning from Observation](model_predictive_adversarial_imitation_learning_for_planning_from_observation.md)

</div>

<!-- RELATED:END -->

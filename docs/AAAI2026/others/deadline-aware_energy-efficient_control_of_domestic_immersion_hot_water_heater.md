---
title: >-
  [Paper Note] Deadline-Aware, Energy-Efficient Control of Domestic Immersion Hot Water Heaters
description: >-
  [AAAI 2026][Energy Efficiency] This paper proposes a deadline-aware energy-efficient control method for domestic hot water heaters. Using a Gymnasium-based simulation environment, it benchmarks a bang-bang baseline…
tags:
  - "AAAI 2026"
  - "Energy Efficiency"
  - "Reinforcement Learning"
  - "Water Heater Control"
  - "PPO"
  - "MCTS"
date: 2026-05-08
content_hash: 15e7f1c59eefdf9f
---

# Deadline-Aware, Energy-Efficient Control of Domestic Immersion Hot Water Heaters

**Conference**: AAAI 2026
**arXiv**: [2601.18123](https://arxiv.org/abs/2601.18123)  
**Code**: None  
**Area**: Other
**Keywords**: Energy Efficiency, Reinforcement Learning, Water Heater Control, PPO, MCTS

## TL;DR

This paper proposes a deadline-aware energy-efficient control method for domestic hot water heaters. Using a Gymnasium-based simulation environment, it benchmarks a bang-bang baseline, an MCTS planner, and a PPO policy, demonstrating that PPO achieves up to 69% energy savings under identical physical conditions.

## Background & Motivation

Domestic water heating constitutes a significant share of household energy consumption. In practice, hot water demand tends to concentrate in predictable time windows (e.g., before work or in the evening), yet most controllers still rely on simple on/off thermostat rules, ignoring factors such as tank water volume, heat dissipation rate, and the actual time at which hot water is needed. This leads to:

**Premature or excessive heating**: Energy is wasted as heated water continuously loses heat to the environment.

**Peak load contribution**: Load is not shifted to lower-carbon time periods.

**Neglect of environmental factors**: Ambient temperature, water volume, and other physical parameters are disregarded.

The central problem addressed in this paper is: **heating water to a target temperature before a given deadline while minimizing total energy consumption**. This "just-in-time" control framework is directly relevant to household resource management and environmental objectives.

The key insight is that when the deadline is generous or the initial temperature is already high, a forward-looking controller (e.g., one that delays heating) can achieve significant energy savings by reducing unnecessary heat losses, whereas a conventional bang-bang controller wastes substantial energy by heating at full power and then maintaining the target temperature.

## Method

### Overall Architecture

The paper models the domestic water heater control problem as a finite-horizon Markov Decision Process (MDP) and evaluates it in a lightweight simulation environment based on the Gymnasium API. Three controllers are compared under identical physical conditions.

### Key Designs

1. **Physical Environment Modeling**: A first-order thermodynamic equation models the tank as a lumped thermal mass with Newtonian cooling. The governing equation is:

    $mc_p \frac{dT}{dt} = \eta P(t) - hA[T(t) - T_a]$

   Forward Euler discretization is applied with a time step of $\Delta t = 120\,\text{s}$. Key parameters include water mass $m = 50\,\text{kg}$, heating power 6000 W, efficiency $\eta = 0.95$, and ambient temperature $T_a = 20°\text{C}$. The observation space is $o_t = [T_t, T_{\text{target}}, T_a, \tau_t]$, and the action space is discrete $\{0, 6000\}\,\text{W}$.

2. **Reward Function Design**: The reward consists of two components — a per-step energy penalty and a terminal temperature deviation penalty:

    $r_t = -\alpha E_t + \begin{cases} -\beta |T_{\text{target}} - T_{t+1}|, & \text{if } |T - T^*| \leq \tau \\ 0, & \text{otherwise} \end{cases}$

   where $\alpha = 1.86 \times 10^{-8}$ and $\beta = 0.03$. The design principle ensures that the cost of heating one additional step is lower than the benefit of reducing terminal error by 1°C ($\beta \cdot 1°\text{C} = 0.03 > \alpha E_{\text{step}} \approx 0.0128$), preventing the policy from optimizing purely for energy savings while neglecting the temperature objective.

3. **Three Controllers**:

    - **Bang-bang Baseline**: Heats at full power until the target temperature is reached, then maintains it within the target range. Time-optimal but least energy-efficient.
    - **MCTS Planner**: Uses UCB1 ($c = \sqrt{2}$) for node selection with 25,000 simulations per episode. Leverages the known deterministic dynamics model for online lookahead search; applicable zero-shot without training.
    - **PPO Policy**: Trained for 2.5M steps using default Stable-Baselines3 hyperparameters, converging at approximately 2.1M steps. Initial states are randomly sampled at the start of each training episode to improve generalization.

### Loss & Training

PPO employs a multi-layer perceptron architecture with a discrete action head and is trained on CPU. Generalization is ensured by training across diverse initial states. MCTS serves as a training-free baseline, performing online search at each step. The primary evaluation metric is total energy consumption (Wh) under identical physical and temporal conditions.

## Key Experimental Results

### Main Results

Experiments cover parameter sweeps across three dimensions:

| Setting | PPO Energy | Bang-bang Energy | MCTS Energy | PPO Savings |
|---|---|---|---|---|
| Representative trajectory (20→60°C, 60 steps) | Lowest | Highest | Intermediate | ~54% vs. BB, ~33% vs. MCTS |
| 30-step deadline | ~3230 Wh | ~4370 Wh | ~4180 Wh | ~26% |
| 60-step deadline | ~3230 Wh | — | — | — |
| 90-step deadline | ~3230 Wh | ~10450 Wh | ~6460 Wh | ~69% |

### Ablation Study

| Parameter Dimension | PPO | Bang-bang | MCTS |
|---|---|---|---|
| Initial temperature (10–30°C) | Low sensitivity, low variance | High energy, linear decrease | Moderate, non-monotonic |
| Deadline (30–90 steps) | Nearly constant ~3230 Wh | Linear growth 4370→10450 Wh | Moderate 4180→6460 Wh |
| Target temperature (40–80°C) | Consistently lowest | Steepest increase | Moderate |

### Key Findings

1. **PPO establishes an energy lower bound**: PPO consistently achieves the lowest energy consumption and smallest variance across all parameter settings.
2. **Deadline is the key differentiating factor**: As available time increases, PPO's energy savings advantage grows from 26% to 69%.
3. **Delayed heating strategy**: PPO learns to delay heating — activating the heater only near the deadline — thereby avoiding heat losses that follow premature heating.
4. **MCTS is zero-shot effective but limited**: MCTS provides partial energy savings without training, but underperforms PPO due to search stochasticity and the absence of learned priors.
5. **Bang-bang is most costly when time is plentiful**: Full-power heating followed by temperature maintenance results in energy waste that grows linearly with available time.

## Highlights & Insights

- **Anticipation is essential**: Controllers capable of anticipating future demand — whether through learning or planning — consistently outperform reactive controllers.
- **Practical trade-offs are clear**: MCTS offers training-free improvement at the cost of online computation; PPO incurs near-zero inference cost after training, making it suitable for large-scale embedded deployment.
- **Environment design is concise yet effective**: A first-order physical model, Gymnasium interface, and compact state space yield a fair and reproducible benchmark.
- **Reward function is carefully calibrated**: By precisely specifying the relationship between $\alpha$ and $\beta$, the design ensures the policy remains energy-efficient without neglecting the temperature objective.

## Limitations & Future Work

1. **Modeling simplification**: A uniform temperature assumption is adopted, neglecting thermal stratification within the tank.
2. **Lack of real-world deployment validation**: Evaluation is confined to simulation; no physical experiments are conducted.
3. **Dynamic factors not considered**: Time-of-use electricity pricing, carbon emission signals, and realistic water usage patterns are not incorporated.
4. **Overly simplified action space**: Only binary on/off control is considered; multi-level power settings may be available in practice.
5. **Limited parameter sweep resolution**: Only five data points are evaluated per dimension.

## Related Work & Insights

This paper applies the "optimal start" intuition from building control to device-level deadline-aware control. Unlike day-scale cost-driven scheduling, the focus is on energy minimization for a single device. By placing MCTS and PPO within the same environment, the paper reveals practical trade-offs between online search and zero-cost inference, in contrast to MPC and forward search approaches. Future extensions may include continuous on-demand control, time-varying electricity pricing, and richer actuator models.

## Rating

- Novelty: ⭐⭐⭐ (Meaningful problem formulation, but methodologically standard)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Systematic parameter sweeps, but no physical validation)
- Writing Quality: ⭐⭐⭐⭐ (Clear and thorough, with detailed explanation of reward design)
- Value: ⭐⭐⭐ (Clear practical application scenario, but limited technical contribution)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Enhancing Control Policy Smoothness by Aligning Actions with Predictions from Preceding States](enhancing_control_policy_smoothness_by_aligning_actions_with_predictions_from_pr.md)
- [\[AAAI 2026\] PIPHEN: Physical Interaction Prediction with Hamiltonian Energy Networks](piphen_physical_interaction_prediction_with_hamiltonian_energy_networks.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[NeurIPS 2025\] Frequency-Aware Token Reduction for Efficient Vision Transformer](../../NeurIPS2025/others/frequency-aware_token_reduction_for_efficient_vision_transformer.md)
- [\[AAAI 2026\] Area-Optimal Control Strategies for Heterogeneous Multi-Agent Pursuit](area-optimal_control_strategies_for_heterogeneous_multi-agen.md)

</div>

<!-- RELATED:END -->

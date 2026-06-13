---
title: >-
  [Paper Note] Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control
description: >-
  [ICML 2026][Reinforcement Learning][Active Flow Control (AFC)] This paper introduces FluidGym—the first RL benchmark for active flow control implemented entirely in PyTorch without external CFD solver dependencies. It is…
tags:
  - "ICML 2026"
  - "Reinforcement Learning"
  - "Active Flow Control (AFC)"
  - "RL Benchmark"
  - "Differentiable Simulation"
  - "Multi-agent RL"
  - "GPU CFD"
date: 2026-05-08
content_hash: 6cc9adc600ba00f5
---

# Plug-and-Play Benchmarking of Reinforcement Learning Algorithms for Large-Scale Flow Control

**Conference**: ICML 2026  
**arXiv**: [2601.15015](https://arxiv.org/abs/2601.15015)  
**Code**: https://github.com/safe-autonomous-systems/fluidgym  
**Area**: Reinforcement Learning / Benchmarking / Active Flow Control / Differentiable Simulation  
**Keywords**: Active Flow Control (AFC), RL Benchmark, Differentiable Simulation, Multi-agent RL, GPU CFD

## TL;DR
This paper introduces FluidGym—the first RL benchmark for active flow control implemented entirely in PyTorch without external CFD solver dependencies. It is end-to-end differentiable, natively supports multi-agent and 3D flow fields, and providing standardized results from over 25k GPU hours across 13 2D/3D environments using PPO, SAC, TD-MPC, and DPC.

## Background & Motivation
**Background**: RL has demonstrated potential in various active flow control (AFC) tasks, such as aerodynamic drag reduction, heat transfer enhancement, and Tokamak plasma stabilization. However, the environments, sensor layouts, reward definitions, and hyperparameters used in the community vary significantly, making horizontal comparisons difficult. Furthermore, over 75% of AFC literature defaults to PPO, while newer continuous control methods (SAC, TD-MPC, DPC) have rarely been systematically evaluated.

**Limitations of Prior Work**: Existing AFC benchmarks (DRLinFluids, drlFoam, DRLFluent, Gym-preCICE, Beacon, HydroGym) suffer from at least one of the following: (i) reliance on external CFD solvers like OpenFOAM/Fluent/FEniCS, which have fragile installation chains and require CFD expertise; (ii) lack of differentiability, preventing the use of gradient-based methods like DPC; (iii) lack of multi-agent APIs; (iv) restriction to 2D cases. Beacon is a rare pure-Python solution but is limited to 2D single-agent and is non-differentiable; HydroGym covers 3D but has inconsistent backends for different environments.

**Key Challenge**: AFC tasks naturally involve large action spaces and spatially distributed actuators (hundreds to thousands of jets or heaters). The most natural modeling is MARL + 3D + differentiable gradients; however, existing engineering stacks decouple these three attributes. Researchers are often forced to choose between differentiability, 3D support, or MARL.

**Goal**: To build a unified benchmark that is (1) pure PyTorch and "pip installable," (2) fully differentiable, (3) natively supporting both SARL and MARL modes, and (4) covers 2D/3D with three difficulty levels for each environment, accompanied by standardized train/val/test protocols and multi-algorithm baselines.

**Key Insight**: Based on the team's self-developed GPU-accelerated PICT solver (implemented as PyTorch CUDA operators for incompressible Navier-Stokes), the CFD solver and RL interface are encapsulated into a single Python package, allowing autograd to pass directly through the CFD stepping.

**Core Idea**: Replacing the traditional "external CFD + Python wrapper" glue stack with a PyTorch-native GPU CFD solver to fundamentally eliminate engineering debt in differentiability and MARL integration.

## Method

### Overall Architecture
FluidGym encapsulates the PICT GPU solver into a `FluidEnv` abstraction layer. It exposes RL interfaces such as Gymnasium (SARL), PettingZoo (MARL), Stable-Baselines3, and TorchRL at the top level while reusing PyTorch autograd at the bottom level. The same physical task can support three interaction modes: Single-Agent (global observation + global action), Multi-Agent (local observation + local action, aggregated and smoothed to boundary conditions by a mapping function $\Gamma$), and Gradient-based (autograd through the entire rollout to calculate reward gradients directly for policy parameters). The environment supports multi-GPU parallel rollouts, with experiments totaling over 25k GPU hours.

The environment family includes 4 categories of basic physics and 13 specific configurations: flow over a cylinder (CylinderRot2D / CylinderJet2D / CylinderJet3D), Rayleigh-Bénard convection (RBC2D / RBC2D-wide / RBC3D / RBC3D-wide), flow over an airfoil (Airfoil2D / Airfoil3D), and turbulent channel flow (TCFSmall3D / TCFLarge3D × both/bottom). Each environment provides 3 difficulty levels (adjusted by Reynolds or Rayleigh numbers), with maximum action dimensions up to 4096 (TCFLarge3D) and observation dimensions near 900,000 (RBC3D-wide).

### Key Designs

1.  **PyTorch-native Differentiable CFD Stack**:
    *   **Function**: Supports environment stepping, automatic differentiation, and RL training within a single Python package, eliminating the burden of installing or coupling external CFD solvers.
    *   **Mechanism**: The underlying PICT solver is a GPU-accelerated incompressible Navier-Stokes solver where all operators are implemented using PyTorch CUDA kernels. Consequently, every flow field update can be tracked by autograd. `FluidEnv.step(a)` does not just return standard `(obs, reward, done)`; it allows gradients to propagate from the reward back to the action $a$ and the policy parameters $\theta$. This is crucial for Differentiable Predictive Control (DPC). On CylinderJet2D-easy, DPC training is approximately 10x faster than PPO and 100x faster than SAC in terms of sample efficiency.
    *   **Design Motivation**: Existing solutions like Beacon are Python-only but non-differentiable, and HydroGym is partially JAX-based but fragmented across backends. Writing the solver in PyTorch is the only path to satisfy "Pure Python + Fully Differentiable + Unified Scenarios," at the cost of maintaining a custom GPU CFD kernel.

2.  **SARL/MARL Dual-mode + Control Mapping $\Gamma$**:
    *   **Function**: Allows the same physical task to run in both single-agent and multi-agent modes without rewriting the environment.
    *   **Mechanism**: In SARL mode, the agent outputs a global action vector $\vec{a}_t$. In MARL mode, each agent $i$ outputs a local action $\vec{a}_t^i$ based on local observation $\vec{o}_{t+1}^i$. Internally, the environment uses a mapping function $\Gamma$ to aggregate local actions into global boundary conditions (e.g., via normalization and spatial smoothing). For example, in RBC2D with 12 heaters, SARL uses one policy to output a 12D vector, while MARL deploys a shared policy across 12 agents. The reward is a weighted combination of local and global metrics, such as $r_t=\mathrm{Nu}_{\mathrm{ref}}-\mathrm{Nu}_{\mathrm{instant}}$ for RBC, where $\mathrm{Nu}_{\mathrm{instant}}=\sqrt{\mathrm{Ra}\cdot\mathrm{Pr}}\langle u_y T\rangle_V$.
    *   **Design Motivation**: Real-world AFC involves spatially distributed actuators. SARL becomes infeasible as action dimensions explode (e.g., 4096 in TCFLarge3D), whereas MARL's translational equivariance fits spatial boundaries perfectly. Making dual-mode a first-class citizen allows direct comparison between the two approaches.

3.  **Standardized Training/Evaluation Protocol**:
    *   **Function**: Eliminates incomparability caused by differences in initial conditions, seeds, and episode lengths in existing AFC literature.
    *   **Mechanism**: Each environment provides 3 fixed splits (train/val/test), each containing 10 pre-generated random initial fields. `env.reset()` adds random perturbations and rollout steps to these fields, ensuring reproducibility via seeds. Each algorithm is run with 5 seeds × 10 test episodes. Performance is reported using performance profiles (Agarwal et al., 2021) with confidence intervals estimated via 2k stratified bootstrap samples. Metrics are reported per-step rather than cumulatively to avoid pollution by episode duration.
    *   **Design Motivation**: Many AFC papers use a single seed or only compare against uncontrolled baselines, which can significantly exaggerate variance and differences. A unified protocol constitutes half the value of a benchmark.

### Loss & Training
PPO and SAC use default hyperparameters from Stable-Baselines3, while MARL variants (MA-PPO/MA-SAC) use shared policies. DPC is demonstrated only on CylinderJet2D and RBC2D due to the computational weight of rollout backpropagation in 3D. All experiments were run on NVIDIA A100 GPUs, with per-step execution times ranging from 1.17 seconds (RBC3D) to 52.89 seconds (Airfoil3D).

## Key Experimental Results

### Main Results: Environment Overview
| Environment Prefix | Control Objective | #Sensors | #Actuators | SARL | MARL | Step Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CylinderJet2D | Drag Reduction | 302 | 1 | ✓ | × | 2.01 |
| CylinderJet3D | Drag Reduction | 4832 | 8 | ✓ | ✓ | 9.52 |
| RBC2D | Heat Enhancement | 768 | 12 | ✓ | ✓ | 1.92 |
| RBC3D | Heat Enhancement | 221184 | 64 | × | ✓ | 1.17 |
| RBC3D-wide | Heat Enhancement | 884736 | 256 | × | ✓ | 1.71 |
| Airfoil3D | Lift-to-Drag | 2508 | 12 | ✓ | ✓ | 52.89 |
| TCFLarge3D-both | Drag Reduction | 4096 | 4096 | × | ✓ | 0.56 |

Action dimensions range from 1 to 4096, and observation dimensions span three orders of magnitude.

### Key Findings
*   **Challenging the Default Status of PPO**: SAC achieved the highest normalized test scores across all difficulties, while PPO converged slowly. This contradicts the choice of PPO in 75% of AFC literature, highlighting the selection bias caused by the lack of unified benchmarks.
*   **Trading Engineering for Differentiable Performance**: DPC utilizes reward gradients to achieve 10–100x acceleration in sample efficiency. However, because it requires an extra backward CFD pass per step, its wall-clock time is 1.5–2x higher than RL, representing a clear trade-off between sample efficiency and per-step overhead.
*   **Emergent MARL Cooperation**: On RBC3D-easy, MA-PPO learned heating patterns that formed two stable convection rolls, consistent with physical findings (Vasanth 2024), suggesting RL can learn spatially invariant collaborative strategies without explicit coordination.
*   **Robustness of Transfer**: 2D-to-3D transfer and small-to-large domain transfer yielded results comparable to or better than native training, indicating that physical similarities in AFC tasks provide significant room for transfer learning.

## Highlights & Insights
*   **The Engineering Bet of a PyTorch CFD Solver**: Developing the PICT GPU solver was a high-investment decision, but it was the only way to achieve "Differentiability + Single Python Stack + GPU Acceleration." This choice defines the benchmark's capabilities and sets a new paradigm.
*   **Performance Profiles Over Mean Scores**: Using Agarwal's performance profiles and bootstrap CIs directly addresses common issues in the AFC community, such as single-seed reporting. This represents a valuable transfer of rigorous RL evaluation practices to an application domain.
*   **Transferrable Tricks**: The combination of MARL + Shared Policy + Spatial Equivariance allows training on small domains and deployment on large ones, saving significant CFD computation.
*   **"Plug-and-Play" Philosophy**: A single `pip install` and Gym-compatible API lowers the entry barrier for machine learning researchers to a level acceptable for non-experts in CFD.

## Limitations & Future Work
*   The authors acknowledge that: (i) due to CFD costs, only 5 seeds per algorithm were run; (ii) CUDA GPUs are required as CPU-only paths are unsupported; (iii) DPC was only demonstrated in 2D; (iv) baselines used default SB3 hyperparameters.
*   **Hidden Constraints**: All 13 environments are based on incompressible Navier-Stokes, excluding compressible flow, magnetohydrodynamics (MHD), combustion, or phase changes.
*   **Future Directions**: Plans include adding MHD physics, increasing seed counts, and integrating differentiable RL for horizontal comparison with DPC.
*   **Evaluation Gaps**: While transfer between dimensions and sizes was tested, transfer across Reynolds numbers or out-of-distribution (OOD) physical parameters—common in real-world deployment—was not explored.

## Related Work & Insights
*   **vs. HydroGym (Lagemann 2025)**: HydroGym supports 3D/MARL but uses fragmented backends (FEniCS, m-AIA, JAX); FluidGym unifies everything with a single PICT/PyTorch stack, offering a shorter installation chain.
*   **vs. Beacon (Viquerat 2024)**: Beacon is Python-only but limited to 2D single-agent non-differentiable tasks; FluidGym expands all three dimensions.
*   **vs. DRLinFluids / drlFoam / Gym-preCICE**: These rely on external solvers like OpenFOAM; FluidGym avoids high maintenance costs by embedding the solver.
*   **vs. PDE Control Benchmarks (Bhan 2024, Zhang 2024)**: Those focus on low-dimensional PDEs, whereas FluidGym focuses on high-dimensional turbulent fields.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ The self-developed PyTorch CFD solver is a significant engineering contribution.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ 25k+ GPU hours across 13 environments is substantial, though seed count and DPC coverage could be higher.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear arguments and high information density.
*   **Value**: ⭐⭐⭐⭐⭐ A critical step for the RL-for-AFC community, providing an easy-to-use tool that will likely catalyze progress.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Vulnerable Agent Identification in Large-Scale Multi-Agent Reinforcement Learning](vulnerable_agent_identification_in_large-scale_multi-agent_reinforcement_learnin.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](../../ICLR2026/reinforcement_learning/verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICML 2026\] Adaptive Bandit Algorithms for Contextual Matching Markets](adaptive_bandit_algorithms_for_contextual_matching_markets.md)
- [\[ICML 2026\] Perceptual Flow Network for Visually Grounded Reasoning](perceptual_flow_network_for_visually_grounded_reasoning.md)
- [\[ICLR 2026\] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models](../../ICLR2026/reinforcement_learning/spell_self-play_reinforcement_learning_for_evolving_long-context_language_models.md)

</div>

<!-- RELATED:END -->

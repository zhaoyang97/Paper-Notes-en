---
title: >-
  [Paper Note] G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration
description: >-
  [ICML 2025][LLM Evaluation][LLM-driven Simulation] This paper proposes G-Sim, a hybrid framework that utilizes LLMs to automatically design the causal structures of simulators (submodules and connectivity), and then calibrates empirical numerical parameters using gradient-free optimization (GFO) or simulation-based inference (SBI) within an iterative refinement loop to generate reliable, intervenable, and general-purpose simulators.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "LLM-driven Simulation"
  - "Gradient-free Calibration"
  - "Automated Simulator Construction"
  - "Simulation-based Inference"
  - "Causal Structure"
date: 2026-05-08
content_hash: c7aaecb5c638701c
---

# G-Sim: Generative Simulations with Large Language Models and Gradient-Free Calibration

**Conference**: ICML 2025  
**arXiv**: [2506.09272](https://arxiv.org/abs/2506.09272)  
**Code**: [GitHub](https://github.com/samholt/generative-simulations)  
**Area**: LLM Evaluation  
**Keywords**: LLM-driven Simulation, Gradient-free Calibration, Automated Simulator Construction, Simulation-based Inference, Causal Structure

## TL;DR

This paper proposes G-Sim, a hybrid framework that utilizes LLMs to automatically design the causal structures of simulators (submodules and connectivity), and then calibrates empirical numerical parameters using gradient-free optimization (GFO) or simulation-based inference (SBI) within an iterative refinement loop to generate reliable, intervenable, and general-purpose simulators.

## Background & Motivation

Constructing high-quality simulators is crucial for "what-if" decision analysis in safety-critical domains such as healthcare, supply chains, and logistics. However, existing methods suffer from a polarized dilemma:

**Purely Data-Driven Methods** (e.g., world models): Although capable of fitting in-distribution data, they generalize poorly when facing sparse/fragmented data and out-of-distribution (OOD) interventions. Furthermore, they lack causal structural priors, making system-level intervention experiments difficult.

**Pure LLM-Generated Simulators**: Although LLMs possess broad domain knowledge enabling them to propose reasonable modular structures, they lack quantitative calibration mechanisms. Consequently, their numerical parameters are often unreliable, leading to discrepancies between simulated trajectories and real-world data.

The authors point out that a truly general-purpose simulator must simultaneously satisfy four core properties:
- **(P0) System-level Experimentation**: Supports submodule-level intervention and stress testing.
- **(P1) Plausible Generalization**: Maintains reasonable behavior under out-of-distribution conditions.
- **(P2) Empirical Alignment**: Highly aligned with observed data.
- **(P3) Data Modality Consistency**: Preserves continuous, discrete, or stochastic data modalities.

Existing methods fail to satisfy all of these properties concurrently. Thus, a hybrid framework that integrates LLM domain knowledge with rigorous empirical calibration is highly warranted.

## Method

### Overall Architecture

The core idea of G-Sim is to decompose the simulator parameter space into **structural parameters $\lambda$** (which submodules exist and how they are connected) and **numerical parameters $\omega$** (rates, coefficients, thresholds, etc.), allowing both to co-evolve within an iterative loop.

The framework consists of three alternating phases:

1. **Propose**: The LLM proposes simulator code $\lambda \sim p_{\text{LLM}}(\lambda \mid \mathcal{K})$ based on domain knowledge $\mathcal{K}$ and historical feedback, including the selection of submodule templates (e.g., SIR model) and causal connection rules among modules.
2. **Calibrate**: Treating the LLM-generated structure as a black-box, gradient-free and likelihood-free methods are utilized to calibrate the numerical parameters $\omega$, aligning the simulation trajectories with the observed data $\mathcal{D}$.
3. **Refine**: Diagnostic evaluations are performed on the calibrated simulator. The identified issues are then converted into natural language feedback, guiding the LLM via in-context learning to propose improved structures in the next iteration.

This loop continues iteratively (default $m=16$ rounds) or until the diagnostic metrics converge.

### Key Designs

#### LLM-driven Structural Design (Satisfying P1, P3)

- The LLM acts as a **generative engine** searching within the structural configuration space.
- Inputs include: textual domain descriptions, known constraints, and feedback from previous iterations.
- Outputs: Python simulator code containing modular-level subprocess definitions and causal connections.
- For example, given a hospital workflow description, the LLM might propose: a patient arrival module, a bed allocation module, and a discharge module, establishing reasonable connection relationships among them.
- This approach directly injects domain-level causal hypotheses into the simulator structure, providing a strong inductive bias.

#### Composite Submodule Structure

The system is decomposed into $K$ submodules $\mathcal{M} = \{\mathcal{M}_1, \ldots, \mathcal{M}_K\}$, where each submodule defines a local mapping:

$$F^k: \mathcal{X} \times \mathcal{U} \times \Theta^k \to \mathcal{Y}^k$$

The global transition operator produces the next state by combining the outputs of each submodule:

$$\mathbf{x}_{t+1} = F_0(F^1(\mathbf{x}_t, \mathbf{u}_t; \theta^1), \ldots, F^K(\mathbf{x}_t, \mathbf{u}_t; \theta^K), \theta^0)$$

Where $\theta^0$ captures cross-submodule coupling (such as shared constraints and resource balancing).

#### Dual-Path Calibration Strategy (Satisfying P2)

**Path 1: Gradient-Free Optimization (GFO) — Point Estimation**

- Implemented using Evolution Strategies (ES) via EvoTorch.
- Minimizes a fitness function $\mathcal{J}(\omega, \lambda)$ that measures the discrepancy between simulation trajectories and real-world data (such as MSE or MMD).
- Advantages: Does not require the simulator to be differentiable and can handle non-smooth loss landscapes.
- Suitable for rapidly obtaining optimal parameter point estimates.

**Path 2: Simulation-Based Inference (SBI) — Bayesian Posterior**

- Uses Neural Posterior Estimation (NPE) to train a neural network to approximate the posterior distribution $p(\omega \mid \mathcal{D}, \lambda)$.
- Provides not only parameter point estimates but also complete uncertainty quantification.
- Extremely critical for high-stakes scenarios where model confidence must be evaluated.
- **Crucial Note**: The theoretical guarantees of SBI assume that the simulator structure $\lambda$ is correct. During the G-Sim search process, the posterior $p(\omega \mid \mathcal{D}, \lambda^{(g)})$ is conditioned on potentially misspecified models, and therefore does not capture structural uncertainty.

#### Diagnostics-Driven Iterative Refinement

The diagnostics function $\text{Diag}(\lambda, \omega^*)$ aggregates multiple mismatch signals:

- **Predictive Discrepancy $\delta_{\text{predictive}}$**: such as Wasserstein distance or MSE, comparing simulation trajectories with held-out data.
- **Domain Violation $\delta_{\text{domain}}$**: checking compliance with known rules (such as capacity bounds and conservation laws).

Diagnostic results are synthesized into natural language summary feedback for the LLM, such as: "The simulator overestimates ICU occupancy over weekends and fails to capture the weekly seasonality present in the data. Consider adding a time-dependent factor in the arrival or discharge modules."

### Loss & Training

- GFO Path: The fitness function $\mathcal{J}(\omega, \lambda)$ measures the statistical distance (MSE or MMD) between simulation trajectories and observed data.
- SBI Path: The training objective of NPE is to maximize the log-likelihood of the posterior approximation.
- Iteration Termination Condition: Reaching the maximum iteration limit $m=16$ or dropping below a convergence threshold $\epsilon$ for diagnostic metrics.
- Prompting Strategy: Uses a generic, reusable core prompt supplemented by concise, environment-specific details, thereby reducing prompt engineering overhead.

## Key Experimental Results

### Main Results

Evaluated on three real-world-inspired simulation tasks, with Wasserstein distance used as the primary metric (lower is better):

| Method | COVID-19 | Supply Chain | Hospital Beds |
|------|----------|--------------|---------------|
| DyNODE | 65.1±2.21 | 38.3±0.40 | 231±0.14 |
| SINDy | 23.9±0.40 | 18.2±0.24 | 199±0.04 |
| RNN | 16.7±1.61 | 9.71±2.21 | 199±2.49 |
| Transformer | 3.30±0.15 | 2.29±0.06 | 199±0.25 |
| Genetic Program | 63.6±7.64 | 30.7±1.41 | 231±0.04 |
| G-Sim-ES ZeroShot | 1.17±0.71 | 2.63±2.79 | 102±1.01 |
| G-Sim-ES ZeroShotOptim | 0.469±0.107 | 9.89±15.3 | 103±2.06 |
| **G-Sim – SBI** | **0.351±0.094** | **1.22±1.68** | **5.24±2.70** |
| G-Sim – ES | 0.405±0.060 | 1.55±1.39 | 101±17.4 |

G-Sim-SBI achieves the best performance across all three environments, demonstrating a dominant advantage particularly on the complex Hospital Beds task (5.24 vs. 199+ for data-driven methods).

### Ablation Study

| Configuration | COVID-19 | Description |
|------|----------|------|
| ZeroShot (No Calibration) | 1.17 | The LLM generates code in a single-shot manner with no parameter optimization |
| ZeroShotOptim (No Structural Iteration) | 0.469 | Optimizes only numerical parameters without adjusting the structure |
| G-Sim – ES (Full Iteration) | 0.405 | Co-evolution of structure and parameters |
| G-Sim – SBI (Full + Bayesian) | 0.351 | Uses SBI for uncertainty quantification |

The ablation results demonstrate that: (1) even zero-shot LLM simulators outperform most data-driven methods; (2) parameter calibration further boosts performance; (3) iterative structural refinement yields additional gains; (4) the SBI path outperforms the GFO path.

### Key Findings

1. **Out-of-Distribution Generalization**: In the COVID-19 lockdown intervention experiments, G-Sim successfully predicted the effects of unseen lockdown policies (with varying $\alpha$ values: 0.05, 0.1, 0.15, 0.3) on infection curves during training, which all baseline methods failed to analyze.
2. **Policy Optimization**: In the Hospital Beds task, the optimal policy discovered by G-Sim (lockdown start day $\tau=15$, extra beds $\Delta B=2500$, cost = 32703) is highly consistent with the true optimal policy ($\tau=10$, $\Delta B=2500$, cost = 29274).
3. **Supply Chain Resource Optimization**: The cost heatmaps generated by G-Sim (extra capacity $\Delta C$ vs. lead time $\ell$) are highly consistent with the global structure of the ground-truth environment.

## Highlights & Insights

1. **Precise Problem Formulation**: Mentally decomposing the simulator parameter space into structural parameters $\lambda$ and numerical parameters $\omega$, assigned to the LLM and the calibration algorithm respectively, maximizes their respective advantages.
2. **Plug-and-Play Dual Calibration Paths**: Users can choose between fast GFO or SBI for uncertainty quantification depending on current needs, making the framework design highly flexible.
3. **Natural Language as a Feedback Bridge**: Diagnostic results are parsed into natural language feedback provided to the LLM, leveraging LLM's in-context learning capabilities for structural improvement in a simple yet effective manner.
4. **Outperforming Data-driven Models Zero-Shot**: Even without calibration, the zero-shot simulator generated by the LLM on the COVID-19 task (1.17) already surpasses Transformer (3.30), vividly demonstrating the power of domain knowledge priors.
5. **Compositional Submodule Design**: Supports asynchronous/continuous-time formulations, allowing substitution with differential equations or event-driven frameworks for broad applicability.

## Limitations & Future Work

1. **Scalability to High-Dimensional Systems**: The current experimental environments are of relatively low dimensionality, and scalability to extremely high-dimensional systems has yet to be verified.
2. **Unmodeled Structural Uncertainty**: The posterior of SBI is only conditioned on the currently proposed structure $\lambda$ and does not capture the uncertainty of the structural search process itself, which represents an important theoretical gap.
3. **LLM Structural Diversity**: The system relies on the LLM's ability to propose sufficiently diverse structural candidates; if the LLM's prior bias is excessively strong, it might neglect critical structural setups.
4. **Suboptimal ES Performance on Hospital Beds**: G-Sim-ES yields a Wasserstein distance of 101 on this task, greatly underperforming compared to SBI's 5.24, highlighting the insufficient robustness of the GFO path in certain complex scenarios.
5. **Computational Cost**: Each iteration loop requires both LLM inference and parameter calibration, which can lead to non-trivial computational overhead in large-scale settings.

## Related Work & Insights

- **WorldCoder** (Tang et al., 2024): Uses LLMs to generate environment code for MBRL, but only processes deterministic discrete logic and lacks rigorous numerical calibration.
- **Hybrid Digital Twins** (Holt et al., 240b): Integrates mechanistic models with data-driven corrections, but assumes continuous physical processes, making it unsuitable for discrete/stochastic scenarios.
- **Foundation Models as World Models** (Gao et al., 2024 et al.): Directly simulates environments using LLMs, but trajectory bias accumulates over time.
- **Data-Driven World Models** (Hafner et al., 2023 et al.): Perform well in-distribution but fail in out-of-distribution scenarios.
- **Insights**: G-Sim's paradigm of "LLM-proposed structure + black-box calibration" can be extended to broader scientific discovery domains, such as drug discovery and climate modeling, where domain knowledge-guided structural searches are of value.

## Rating

| Dimension | Rating (1-5) | Description |
|------|-----------|------|
| Novelty | ⭐⭐⭐⭐ | Structural/numerical parameter separation + novel LLM-calibration iterative loop design |
| Practicality | ⭐⭐⭐⭐ | Open-source code with dual calibration paths selectable on demand |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Three environments + Ablation + Policy Optimization + OOD Intervention experiments |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clear problem formulation, systematic methodology, and rich visualizations |
| Value | ⭐⭐⭐⭐ | A solid hybrid framework piece that compromises both theoretical depth and practical utility |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Entropy Calibration of Language Models](../../NeurIPS2025/llm_evaluation/on_the_entropy_calibration_of_language_models.md)
- [\[ICML 2025\] Correlated Errors in Large Language Models](correlated_errors_in_large_language_models.md)
- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[ICCV 2025\] Generative Zoo](../../ICCV2025/llm_evaluation/generative_zoo.md)
- [\[ICML 2025\] Fleet of Agents: Coordinated Problem Solving with Large Language Models](fleet_of_agents_coordinated_problem_solving_with_large_language_models.md)

</div>

<!-- RELATED:END -->

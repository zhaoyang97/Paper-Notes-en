---
title: >-
  [Paper Note] Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents
description: >-
  [ICML 2025][Physics & Scientific Computing][Gravitational physics] Proposes **Gravity-Bench-v1**, an **interactive environment** benchmark based on gravitational dynamics simulation to evaluate the capability of AI agents to make scientific discoveries (including OOD physics scenarios) under restricted observation budgets. The results show that current models possess significant shortcomings in observational planning and budget utilization.
tags:
  - "ICML 2025"
  - "Physics & Scientific Computing"
  - "Gravitational physics"
  - "scientific discovery"
  - "AI Agent"
  - "benchmark"
  - "partially observable environment"
  - "observational planning"
  - "binary systems"
date: 2026-05-08
content_hash: f757476a566b5e53
---

# Gravity-Bench-v1: A Benchmark on Gravitational Physics Discovery for Agents

**Conference**: ICML 2025

**arXiv**: [2501.18411](https://arxiv.org/abs/2501.18411)

**Authors**: Nolan Koblischke, Hyunseok Jang, Kristen Menou, Mohamad Ali-Dib

**Area**: Physics / AI for Science / Benchmarking

**Keywords**: Gravitational physics, scientific discovery, AI Agent, benchmark, partially observable environment, observational planning, binary systems

## TL;DR

Proposes **Gravity-Bench-v1**, an **interactive environment** benchmark based on gravitational dynamics simulation to evaluate the capability of AI agents to make scientific discoveries (including OOD physics scenarios) under restricted observation budgets. The results show that current models possess significant shortcomings in observational planning and budget utilization.

## Background & Motivation

Modern science originated from repeated observation and reasoning of planetary motion. Existing AI evaluation benchmarks focus primarily on knowledge assessment (e.g., GPQA, MMLU) or general problem-solving (e.g., ARC, HellaSwag), lacking evaluation of AI capacity in the **real scientific discovery process**—including planning observations, reasoning under uncertainty, and discovering new phenomena.

Design Motivation for Gravity-Bench-v1:

- **Simulating the full scientific process**: Agents must not only analyze data but also **actively plan observations** and collect data within a limited budget.
- **Including OOD scenarios**: Modifying gravitational laws ($F_G \propto r^{-(2+\alpha)}$) and introducing drag forces to test **scientific generalization capabilities**.
- **Open-ended solution space**: The solution methods are not restricted, allowing agents to discover strategies superior to human baselines.
- **Providing PhD-level reference solutions**: Utilizing human expert solutions as upper bounds to calibrate AI performance.

## Method

### Overall Architecture

Gravity-Bench-v1 consists of three components:

1. **Simulation environment**: Binary star system simulation based on Rebound (a gravitational N-body simulator).
2. **Observation protocol**: full-obs (complete data) and budget-obs-100 (at most 100 observations).
3. **Task design**: 16 binary simulations $\times$ 50 tasks = **206 task-simulation pairs**.

### Key Designs

**Environment Design**: All orbits are in the Cartesian $(x, y)$ plane. Simulations use WHFast (an energy-conserving integrator) or IAS15 (an adaptive 15th-order integrator, used for modified gravity scenarios). The time step is $1/5000$ of the orbital period.

**Observation Tool**: Agents collect data via the `observe` tool, with each call retrieving at most 10 data points and a total budget of $N_{\text{obs}} = 100$. Agents can choose the timing of observations, encouraging strategic planning.

**Symmetry-Breaking Strategies**:
- Center of mass offset from the origin.
- Introduction of proper motion.
- Simulating the "messiness" of real astronomical observations.

**OOD Scenarios** (6):
- 3 drag scenarios: $\ddot{x}_i = -v_i / \tau$, requiring inference of the drag timescale $\tau$.
- 3 modified gravity scenarios: $F_G \propto r^{-(2+\alpha)}$, requiring inference of the deviation $\alpha$.

**Task Types**

Tasks cover: stellar mass inference, orbital period, eccentricity, semi-major axis, verification of conservation of energy, verification of Kepler's Third Law, maximum velocity, Roche lobe radius, angular momentum, modified gravity power index, etc.

**Evaluation Method**

Standard for correct answers: relative error below a **task-specific threshold** (5%–70%), which is set based on the performance degradation of the PhD-level solution under 100 uniform samples.

**Baseline Agent**

ReAct-style agents are used, equipped with the `observe` tool and a Python interpreter (containing `numpy`, `scipy`, `pandas`), supporting multi-step reasoning and code execution.

## Key Experimental Results

### Main Results

| Model | Accuracy (budget-obs-100) | Accuracy (full-obs) | Total Cost ($) | Avg. Observations Used |
|:---|:---|:---|:---|:---|
| o1-2024-12-17 | — | 64.0%† | $100.07 | — |
| Claude 3.5 Sonnet | 21.5% ± 2.5% | 39.5% ± 3.2% | $15.88 | 24.3 |
| Claude 3.5 Haiku | 16.1% ± 2.3% | 34.1% ± 3.1% | $3.33 | 12.6 |
| GPT-4o | 15.5% ± 2.1% | 36.1% ± 3.2% | $9.60 | 12.2 |
| GPT-4o-mini | 8.3% ± 1.5% | 26.7% ± 2.8% | $0.60 | 13.4 |
| **PhD-level Solution** | **82.5%** | **100.0%** | — | 100.0 |

### Key Findings

| Finding | Details |
|:---|:---|
| **Severe Underutilization of Observation Budget** | GPT-4o used only 12/100 observations on average, while Claude used 24/100 |
| **OOD Tasks are Highly Challenging** | Only o1 could consistently solve modified gravity tasks (2/6), and Claude 3.5 Sonnet solved 1/6 |
| **Mass Assumption as Primary Failure Mode** | GPT-4o assumed mass = 1 in 33% of incorrect solutions, compared to only 5% in correct solutions |
| **Agents Tend to "Settle Quick"** | Stopping upon finding a seemingly plausible answer without further verification |
| **Significant Variance in Planning Capabilities** | Claude 3.5 Sonnet occasionally achieved <1% error through meticulous planning, but inconsistently |

### Planning Case Study

In the maximum velocity estimation task (with 40 observations):
- **Success case**: Claude first coarsely sampled to locate the high-velocity region, then iteratively refined the temporal resolution, achieving a final error of 2%.
- **Failure case**: The same model failed to record the epoch of peak velocity, misinterpreting the resolution increase as a velocity increase, leading to a final error of 45%.

## Highlights & Insights

- **Interactive Environment Evaluation Paradigm**: Closer to the actual scientific discovery process compared to static QA benchmarks.
- **Ingenious OOD Design**: The task of inferring the modified gravitational index $\alpha$ is almost impossible to solve by memorization, serving as a genuine test of generalization.
- **Unveils "Scientific Hallucinations" in Agents**: Models tend to assume symmetries and simplified conditions rather than deriving them from data.
- **Open-Ended Solution Space**: In theory, agents can discover better observation strategies than humans.

## Limitations & Future Work

1. **Limited to Two-Body Gravity**: Physical complexity is limited, omitting more complex systems like three-body dynamics or hydrodynamics.
2. **2D Orbits**: All orbits are planar, without considering projection effects.
3. **Single Agent Framework**: Only ReAct-style agents are tested, without exploring other architectures (e.g., Tree-of-Thought).
4. **Incomplete evaluation for o1**: Due to API content policy restrictions, 17/206 questions were rejected, and only a single run was conducted.
5. **High Cost**: Complete evaluation with o1 costs $100+, restricting large-scale experimentation.

## Related Work & Insights

- **SWE-bench, RE-bench, BrowserGym**: Interactive environment agent benchmarks, but oriented towards software/web domains.
- **DiscoveryBench, DiscoveryWorld**: Data-driven discovery benchmarks, but relatively static.
- **The AI Scientist (Lu et al., 2024)**: Automated scientific research workflows, but not focused on physical discoveries.
- **ScienceAgentBench**: Scientific research agent evaluation, but focused on the coding level.

**Insights**: This work demonstrates the value of the **partially observable environment + budget constraint** evaluation paradigm for assessing the scientific capability of agents. The shortcomings of models in modern aspects such as "when to stop observing" and "how to verify answers" warrant close attention.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Interactive physical discovery benchmark represents a new evaluation paradigm.
- **Technical Depth**: ⭐⭐⭐⭐ — Rigorous Rebound simulation with well-thought-out task design.
- **Utility**: ⭐⭐⭐⭐ — Highly valuable reference for evaluating and improving AI scientific discovery capabilities.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear case studies and in-depth discussion on failure modes.
- **Overall Rating**: 8/10 — Fills an important gap in evaluating physical discovery agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Causal Discovery of Latent Variables in Galactic Archaeology](causal_discovery_of_latent_variables_in_galactic_archaeology.md)
- [\[NeurIPS 2025\] Unsupervised Discovery of High-Redshift Galaxy Populations with Variational Autoencoders](../../NeurIPS2025/physics/unsupervised_discovery_of_high-redshift_galaxy_populations_with_variational_auto.md)
- [\[NeurIPS 2025\] Score-informed Neural Operator for Enhancing Ordering-based Causal Discovery](../../NeurIPS2025/physics/score-informed_neural_operator_for_enhancing_ordering-based_causal_discovery.md)
- [\[ICML 2025\] Differentiable Stellar Atmospheres with Physics-Informed Neural Networks](differentiable_stellar_atmospheres_with_physics-informed_neural_networks.md)
- [\[NeurIPS 2025\] POLARIS: A High-contrast Polarimetric Imaging Benchmark Dataset for Exoplanetary Disk Representation Learning](../../NeurIPS2025/physics/polaris_a_high-contrast_polarimetric_imaging_benchmark_dataset_for_exoplanetary_.md)

</div>

<!-- RELATED:END -->

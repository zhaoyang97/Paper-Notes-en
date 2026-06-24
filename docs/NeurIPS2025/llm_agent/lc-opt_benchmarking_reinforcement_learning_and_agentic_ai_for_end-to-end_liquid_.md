---
title: >-
  [Paper Note] LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers
description: >-
  [NeurIPS 2025][LLM Agent][Liquid cooling optimization] This paper presents LC-Opt, a liquid cooling benchmark environment built upon a high-fidelity digital twin of the cooling system of the ORNL Frontier supercomputer. It supports end-to-end liquid cooling optimization via RL control policies, encompassing centralized/decentralized multi-agent RL, policy distillation into interpretable decision trees, and an LLM-driven agentic mesh architecture.
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Liquid cooling optimization"
  - "reinforcement learning"
  - "digital twin"
  - "multi-agent RL"
  - "sustainable data centers"
date: 2026-05-08
content_hash: 2e30071ad469d286
---

# LC-Opt: Benchmarking Reinforcement Learning and Agentic AI for End-to-End Liquid Cooling Optimization in Data Centers

**Conference**: NeurIPS 2025

**arXiv**: [2511.00116](https://arxiv.org/abs/2511.00116)

**Code**: None (Modelica-based simulation environment)

**Area**: Reinforcement Learning / Agentic AI / Data Center Optimization

**Keywords**: Liquid cooling optimization, reinforcement learning, digital twin, multi-agent RL, sustainable data centers

## TL;DR

This paper presents LC-Opt, a liquid cooling benchmark environment built upon a high-fidelity digital twin of the cooling system of the ORNL Frontier supercomputer. It supports end-to-end liquid cooling optimization via RL control policies, encompassing centralized/decentralized multi-agent RL, policy distillation into interpretable decision trees, and an LLM-driven agentic mesh architecture.

## Background & Motivation

The explosive growth of AI workloads has made thermal management in high-density data centers increasingly critical:

**Necessity of liquid cooling**: Conventional air cooling can no longer meet the heat dissipation demands of high-performance computing (HPC) systems, making liquid cooling an essential technology.

**Limitations of rule-based control**: Traditional PID controllers and static rules are inadequate for handling dynamically varying workloads and complex multi-objective optimization.

**Lack of standard benchmarks**: The ML/RL community lacks detailed, customizable liquid cooling simulation environments for developing and validating control policies.

**Multi-objective optimization challenges**: Simultaneously balancing local thermal regulation (preventing overheating) and global energy efficiency (reducing energy consumption) remains an open problem.

LC-Opt aims to fill this gap by providing a standardized platform for research on RL and agentic AI in data center liquid cooling optimization.

## Method

### Overall Architecture

LC-Opt constructs an end-to-end liquid cooling simulation environment covering all hierarchical levels of a data center cooling system:

1. **Site-level cooling tower (CT)**: Controls chilled water temperature setpoints
2. **Data center rack level**: Controls liquid supply temperature, flow rate, and valve actuation
3. **Server blade group level**: Fine-grained thermal management
4. **Additional components**: Heat recovery units (HRU), etc.

A standard RL interaction interface is provided via Gymnasium, supporting dynamic workload variations.

### Key Designs

**1. High-Fidelity Digital Twin**

- Constructed using the **Modelica** modeling language
- Baseline derived from the ORNL Frontier supercomputer cooling system
- End-to-end model covers the complete physical process from cooling tower to rack
- Incorporates detailed physical models of fluid dynamics, heat transfer, pump power, etc.

**2. Multi-Level RL Control**

- **Rack-level control**: RL agents optimize liquid supply temperature, flow rate, and fine-grained valve actuation
- **Cooling tower-level control**: Optimizes cooling tower temperature setpoints
- **Multi-objective**: Simultaneously minimizes energy consumption and temperature violations

**3. Centralized vs. Decentralized Multi-Agent RL**

- **Centralized**: A single RL agent controls all actuators
- **Decentralized**: Multiple RL agents each control local devices and coordinate via communication
- Compares different MARL architectures (e.g., MAPPO, QMIX) in complex control scenarios

**4. LLM-Driven Agentic Mesh Architecture**

- LLM agents interpret control actions in natural language
- The agentic mesh design promotes user trust
- Simplifies system management, enabling non-expert operators to understand decisions

**5. Policy Distillation**

- Distills trained RL policies into **decision trees** and **regression trees**
- Provides interpretable control rules
- Facilitates deployment and auditing

### Loss & Training

The RL reward function is designed for multi-objective optimization:
- **Energy consumption penalty**: Pump power and cooling tower energy consumption
- **Temperature violation penalty**: Severe penalty for exceeding safe temperature thresholds
- **Efficiency reward**: Improvement in PUE (Power Usage Effectiveness)
- Standard RL algorithms (PPO, SAC, etc.) are used for training

## Key Experimental Results

### Main Results

**Table 1: Energy Efficiency Comparison Across Control Strategies**

| Control Strategy | PUE (↓) | Temperature Violations (↓) | Energy Savings (%) | Interpretability |
|:---|:---:|:---:|:---:|:---|
| Baseline PID | ~1.30 | Baseline | 0% | High |
| Centralized PPO | ~1.15 | Greatly reduced | ~12% | Low |
| Centralized SAC | ~1.14 | Greatly reduced | ~13% | Low |
| Decentralized MAPPO | ~1.16 | Reduced | ~11% | Low |
| Policy Distillation (Decision Tree) | ~1.18 | Reduced | ~10% | High |
| LLM Agent | ~1.20 | Reduced | ~8% | Highest |

**Table 2: Robustness Under Different Workload Patterns**

| Workload Pattern | PID PUE | RL Optimal PUE | Improvement |
|:---|:---:|:---:|:---:|
| Steady-state high load | 1.28 | 1.12 | 12.5% |
| Periodic fluctuation | 1.32 | 1.16 | 12.1% |
| Bursty peak | 1.35 | 1.18 | 12.6% |
| Mixed dynamic | 1.33 | 1.17 | 12.0% |

### Ablation Study

**Table 3: Effect of Control Granularity on Performance**

| Control Level | Number of Control Variables | PUE | Energy Savings |
|:---|:---:|:---:|:---:|
| Cooling tower setpoint only | 1–2 | 1.22 | ~6% |
| Cooling tower + rack temperature | 5–10 | 1.17 | ~10% |
| Full end-to-end (including valves) | 20+ | 1.14 | ~13% |

**Table 4: Policy Distillation Performance Retention**

| Distilled Model | Tree Depth | PUE Relative to RL Policy | Interpretability |
|:---|:---:|:---:|:---|
| Decision tree (shallow) | 5 | ~95% | Each path manually auditable |
| Decision tree (medium) | 10 | ~97% | Good |
| Regression tree | 8 | ~96% | More precise for continuous control |

### Key Findings

1. **RL substantially outperforms traditional control**: Across all workload patterns, RL policies save 10–13% more energy than PID control.
2. **End-to-end control is optimal**: Fine-grained valve control yields an additional 3–7% energy savings.
3. **Policies are distillable**: Decision trees retain 95–97% of RL policy performance while being fully interpretable.
4. **Centralized slightly outperforms decentralized**: Centralized RL achieves marginally better optimality, whereas decentralized approaches scale more readily.
5. **Trust advantage of LLM agents**: Despite slightly lower performance, natural language explanations significantly improve operator trust.

## Highlights & Insights

1. **Real-system benchmark**: The cooling system of the world's leading supercomputer (Frontier, ranked first) serves as the basis — not a fictitious simplified model.
2. **Full-stack coverage**: End-to-end modeling from cooling tower to server blade captures inter-system coupling effects.
3. **Interpretability pathway**: RL → decision tree distillation + LLM explanation provides a complete black-box-to-white-box pipeline for industrial deployment.
4. **Open environment**: Detailed liquid cooling models are made available to the ML community, lowering the barrier to entry for research.
5. **Multidisciplinary integration**: Integrates RL, multi-agent systems, LLMs, and thermodynamic engineering.

## Limitations & Future Work

1. **Sim-to-real gap**: Despite high fidelity, the digital twin still differs from the real system and requires domain adaptation.
2. **Computational overhead**: The simulation speed of Modelica models may constrain RL training efficiency.
3. **Scalability**: The current benchmark primarily targets a single data center; multi-data-center coordinated optimization warrants further exploration.
4. **Safety constraints**: Real-world deployment demands stricter safety guarantees (e.g., temperature never exceeding thresholds).
5. **Integration with predictive control**: Incorporating workload prediction models could further improve anticipatory control.

## Related Work & Insights

- **DC Cooling RL** (Chi et al., 2022): RL-based control for data center cooling
- **Gymnasium**: Standard RL interaction interface
- **Modelica**: Multi-physics domain modeling language
- **MAPPO** (Yu et al., 2022): Multi-agent PPO
- **Google DeepMind data center optimization**: Pioneering industrial practice
- Insight: Deploying RL in industrial control requires interpretability and safety guarantees; policy distillation serves as an effective bridge.

## Rating

| Dimension | Score (1–5) | Note |
|:---|:---:|:---|
| Novelty | 4 | Full-stack end-to-end benchmark + LLM agents; novel combination |
| Technical Quality | 4 | High-fidelity simulation, multi-method benchmarking |
| Experimental Thoroughness | 4 | Comprehensive comparison: centralized/decentralized/distillation/LLM |
| Practicality | 5 | Benchmark environment directly targeting industrial deployment |
| Writing Quality | 3 | Systems-level paper with considerable detail |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Unlocking Long-Horizon Agentic Search with Large-Scale End-to-End RL](../../ICLR2026/llm_agent/unlocking_long-horizon_agentic_search_with_large-scale_end-to-end_rl.md)
- [\[ACL 2026\] MemSearcher: Training LLMs to Reason, Search and Manage Memory via End-to-End RL](../../ACL2026/llm_agent/memsearcher_training_llms_to_reason_search_and_manage_memory_via_end-to-end_rein.md)
- [\[NeurIPS 2025\] MAT-Agent: Adaptive Multi-Agent Training Optimization](mat-agent_adaptive_multi-agent_training_optimization.md)
- [\[NeurIPS 2025\] What AI Speaks for Your Community: Polling AI Agents for Public Opinion on Data Center Projects](what_ai_speaks_for_your_community_polling_ai_agents_for_public_opinion_on_data_c.md)
- [\[NeurIPS 2025\] Ground-Compose-Reinforce: Grounding Language in Agentic Behaviours using Limited Data](ground-compose-reinforce_grounding_language_in_agentic_behaviours_using_limited_.md)

</div>

<!-- RELATED:END -->

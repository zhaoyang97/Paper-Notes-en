---
title: >-
  [Paper Note] AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture
description: >-
  [AAAI 2026][LLM Agent][Urban pipeline networks] This paper proposes AquaSentinel, a physics-informed AI system that achieves network-wide pipeline leak detection using only 20–30% node coverage through sparse sensor deployment, physics-augmented virtual sensors, a MoE spatiotemporal GNN ensemble, a dual-threshold RTCA detection algorithm, causal flow localization, and LLM-based report generation. The system achieves 100% detection rate across 110 leak scenarios.
tags:
  - AAAI 2026
  - LLM Agent
  - Urban pipeline networks
  - leak detection
  - sparse sensing
  - physics-informed AI
  - MoE spatiotemporal forecasting
date: 2026-05-08
content_hash: 91d2d73cad59d353
---

# AquaSentinel: Next-Generation AI System Integrating Sensor Networks for Urban Underground Water Pipeline Anomaly Detection via Collaborative MoE-LLM Agent Architecture

**Conference**: AAAI 2026
**arXiv**: [2511.15870v1](https://arxiv.org/abs/2511.15870v1)
**Code**: [https://github.com/VV123/STEPS](https://github.com/VV123/STEPS) (dataset)
**Area**: AI for Infrastructure / Spatiotemporal Graph Neural Networks / Anomaly Detection
**Keywords**: Urban pipeline networks, leak detection, sparse sensing, physics-informed AI, MoE spatiotemporal forecasting

## TL;DR
This paper proposes AquaSentinel, a physics-informed AI system that achieves network-wide pipeline leak detection using only 20–30% node coverage through sparse sensor deployment, physics-augmented virtual sensors, a MoE spatiotemporal GNN ensemble, a dual-threshold RTCA detection algorithm, causal flow localization, and LLM-based report generation. The system achieves 100% detection rate across 110 leak scenarios.

## Background & Motivation
The United States loses 2.1 trillion gallons of treated water annually due to pipeline leaks. Recent major water main failures in Houston (2024) and Detroit (2023) highlight three critical issues: (1) **background leaks** (small and persistent) in underground pipelines frequently escalate into catastrophic ruptures; (2) **manual inspection** offers limited coverage and delayed response; (3) **dense sensor networks** are economically infeasible for aging infrastructure.

Core trade-off: dense sensors = high coverage, high cost vs. sparse sensors = low cost, low detection capability.

## Core Problem
**How can network-wide real-time pipeline anomaly detection and precise localization be achieved with only 20–30% sensor coverage?**

## Method

### Overall Architecture
AquaSentinel is an end-to-end multi-stage system:
1. **Sparse sensor deployment**: Sensors installed at high-centrality nodes (manhole-mounted, no pipeline entry required)
2. **Physics augmentation**: Mass conservation and energy conservation laws used to create "virtual sensors" at unmonitored nodes
3. **MoE forecasting**: Weighted ensemble of six spatiotemporal GNN expert models to predict normal-state behavior
4. **RTCA detection**: Dual-threshold (instantaneous + cumulative) anomaly detection algorithm
5. **Causal localization**: Upstream tracing of anomaly sources along the pipeline network
6. **LLM reporting**: Generation of actionable maintenance instructions

### Key Designs

1. **Intelligent sensor placement**: A three-dimensional scoring scheme combining topological betweenness centrality $C_B(v)$, hydraulic importance (flow × pressure differential), and risk factors (historical failure rate + infrastructure age), with a minimum-distance constraint to ensure uniform coverage. Sensors are mounted on the underside of manhole covers, requiring no pipeline entry.

2. **Physics-augmented virtual sensors**: The Hazen-Williams equation (pipe head loss $h_f = 10.67 L(Q/CD^{2.63})^{1.852}$) and mass conservation (nodal flow balance) are applied to solve for the states of unmonitored nodes from sparse sensor readings, effectively injecting physical priors into data augmentation.

3. **RTCA dual-threshold detection**: Both the instantaneous error $e_t^{RT}$ and the sliding-window cumulative error $e_t^C$ are monitored simultaneously. Adaptive thresholds track statistical changes via exponential moving averages. An anomaly is confirmed only when **both thresholds are exceeded for $T$ consecutive steps**, substantially reducing false alarm rates.

4. **Causal flow localization**: Network topology is used to identify anomaly sources as nodes with no anomalous upstream neighbors: $v^* = \{v \in \mathcal{A}: \text{Upstream}(v) \cap \mathcal{A} = \emptyset\}$. The leaking pipe segment lies between the source node and its nearest normal upstream neighbor.

### Loss & Training
Six MoE experts (CaST, GMAN, ST-SSL, STG-MAMBA, STGCN, HydroNet) are each trained independently on normal-condition data. A gating network dynamically weights contributions using exponential smoothing of recent per-expert losses. Sensor data are drawn from a real sewer network at Texas A&M University, sampled at 10-minute intervals.

## Key Experimental Results

| Leak Type | Detection Rate | Avg. Latency (steps) | Detection within 10 Steps |
|-----------|--------------|----------------------|--------------------------|
| Constant <5% | 100% | 12 | 81.8% |
| Constant 5–15% | 100% | 8 | 100% |
| Constant 15–25% | 100% | 5 | 100% |
| Constant >25% | 100% | 3 | 100% |
| Dynamic (0→35%) | 100% | 15 | 73.6% |
| **Total (110 cases)** | **100%** | — | **90.91%** |

Among MoE experts, HydroNet achieves the best performance (MAE 0.0085), followed by STGCN.

### Ablation Study
- **100% detection rate**: All 110 leak scenarios (22 pipes × 5 leak types) are successfully detected.
- **Rapid detection of large leaks**: Leaks exceeding 25% are detected within 3 steps (30 minutes); leaks below 5% require 12 steps (120 minutes of cumulative evidence).
- **Value of physics augmentation**: Data quality from sparse sensors combined with physics augmentation is comparable to that of full-coverage sensor deployment.
- **Dual-threshold superiority**: Cumulative error captures small leaks; instantaneous error captures large leaks — the two are complementary.

## Highlights & Insights
- **System-level innovation**: Rather than improving a single algorithm, this work presents a complete end-to-end system spanning sensor deployment to maintenance reporting.
- **Physics–data-driven integration**: Physical laws serve as data augmentation (virtual sensors), allowing AI models to benefit from physical priors without being constrained by physical models.
- **Real-world deployment validation**: Sensors are installed on an actual university campus sewer network for data collection, rather than relying solely on simulation.
- **Engineering practicality of RTCA**: The dual-threshold combined with persistence confirmation is simple yet highly practical — the industrial community frequently faces the sensitivity–false-alarm trade-off, and RTCA provides an effective resolution framework.
- **Cost-awareness**: The entire design targets minimum sensor count and minimum cost, oriented toward retrofitting aging infrastructure.

## Limitations & Future Work
- **Simulated leak scenarios**: The 110 leak events are generated via PCSWMM simulation; no real-world leaks occurred during the deployment period.
- **Sewer network only**: The hydraulic characteristics of water supply networks differ from sewer networks; applicability to water distribution systems remains unvalidated.
- **Small network scale**: The campus network comprises only 23 nodes; scalability to city-scale networks (thousands of nodes) is not demonstrated.
- **Only two sensor features**: Flow velocity and water depth are collected; water quality sensors (pH, turbidity, etc.) are absent, potentially missing certain anomaly types.
- **LLM reports not quantitatively evaluated**: The accuracy and utility of generated reports are demonstrated only qualitatively.

## Related Work & Insights

| Method | Sensor Coverage | Physical Constraints | Detection Approach | Difference from AquaSentinel |
|--------|----------------|---------------------|--------------------|------------------------------|
| **Traditional manual inspection** | Very low | None | Visual | Low coverage, delayed response |
| **Dense sensor networks** | High (100%) | None | Threshold / statistical | Cost prohibitive for aging infrastructure |
| **Purely data-driven** | Medium | None | ML models | Poor performance under incomplete observations |
| **Purely physics-based models** | Medium | Strong | Simulation comparison | Poor adaptability to changing conditions; difficult parameter calibration |
| **AquaSentinel** | Low (20–30%) | Augmented | MoE + RTCA | Physics–AI integration, sparse → full network, 100% detection |

The "virtual sensor" concept (inferring unobserved states from limited observations via physical models) is generalizable to other infrastructure monitoring domains (power grids, natural gas networks, etc.). The RTCA dual-threshold and persistence confirmation paradigm is applicable to any time-series anomaly detection task requiring a balance between sensitivity and false alarm rate. The MoE-based spatiotemporal model ensemble approach is simple and effective, with promising applicability in domains such as traffic forecasting.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Innovative at the system integration level (sparse sensing + physics augmentation + AI detection + LLM reporting), though individual components are not entirely novel in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐ — Real sensor deployment is conducted, but leak events are simulated; the network scale is small, and comparisons with other pipeline anomaly detection methods are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is compelling (grounded in real accident cases); system description is clear.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to smart city and infrastructure monitoring; the cost advantage of 20–30% coverage is attractive.

## Additional Remarks
- The methodology and experimental design of this work offer valuable reference for related domains.
- Future work should validate generalizability and scalability across larger networks and more diverse deployment scenarios.
- Potential research value exists in combining this work with recent advances in RL/MCTS and multimodal approaches.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] AgentSense: Virtual Sensor Data Generation Using LLM Agents in Simulated Home Environments](agentsense_virtual_sensor_data_generation_using_llm_agents_i.md)
- [\[AAAI 2026\] FinRpt: Dataset, Evaluation System and LLM-based Multi-agent Framework for Equity Research Report Generation](finrpt_dataset_evaluation_system_and_llm-based_multi-agent_framework_for_equity_.md)
- [\[NeurIPS 2025\] PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer](../../NeurIPS2025/llm_agent/panda_towards_generalist_video_anomaly_detection_via_agentic_ai_engineer.md)
- [\[AAAI 2026\] COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)
- [\[AAAI 2026\] EcoAgent: An Efficient Device-Cloud Collaborative Multi-Agent Framework for Mobile Automation](ecoagent_an_efficient_device-cloud_collaborative_multi-agent.md)

<!-- RELATED:END -->

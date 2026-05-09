---
title: >-
  [Paper Note] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation
description: >-
  [ACL 2026][Other][Supply Chain Simulation] Using LLM agents (DeepSeek/GPT series) to simulate a multi-stage supply chain in the classic Beer Distribution Game, systematically studying how cognitive heterogeneity (reasoning capability differences) affects system behavior; findings show LLM agents reproduce human-like bullwhip effects and myopic behavior, and that information sharing can effectively mitigate these adverse effects.
tags:
  - ACL 2026
  - Other
  - Supply Chain Simulation
  - Cognitive Heterogeneity
  - Bullwhip Effect
  - LLM Agents
  - Beer Distribution Game
date: 2026-05-08
content_hash: 1630d97389aeb8b4
---
# Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation

**Conference**: ACL 2026
**arXiv**: [2604.17220](https://arxiv.org/abs/2604.17220)
**Code**: None
**Area**: Other
**Keywords**: supply chain simulation, cognitive heterogeneity, bullwhip effect, LLM agents, beer distribution game

## TL;DR

This paper deploys LLM agents (DeepSeek/GPT series) in the classic beer distribution game to simulate multi-stage supply chains, systematically investigating how cognitive heterogeneity (differences in reasoning capability) affects system behavior. The findings demonstrate that LLM agents can reproduce human-observed bullwhip effects and myopic behaviors, and that information sharing effectively mitigates these adverse effects.

## Background & Motivation

**Background**: Behavioral experiments (e.g., the beer distribution game) have revealed supply chain inefficiencies caused by cognitive biases, such as the bullwhip effect. However, traditional human-subject experiments face constraints in scalability, cost, and experimental control. The potential of LLMs as behavioral proxies is an emerging area of exploration.

**Limitations of Prior Work**: (1) Most LLM multi-agent studies focus on static or structurally simple settings, without exploring highly dynamic multi-period environments; (2) existing studies typically deploy homogeneous agents, neglecting the impact of cognitive heterogeneity (mixtures of agents with varying reasoning capabilities) on collective behavior; (3) rigorous statistical validation is often absent.

**Key Challenge**: Strategic diversity is both pervasive and consequential in real organizations, yet its interaction effects in synthetic environments remain insufficiently studied.

**Goal**: To establish an LLM-driven supply chain simulation paradigm and systematically investigate how cognitive heterogeneity shapes collective behavior.

**Key Insight**: Agents with different reasoning capabilities — base models vs. reasoning-enhanced models — are used to represent distinct cognitive levels, and heterogeneous agents are deployed at different positions within the supply chain.

**Core Idea**: LLM agents can reproduce human behavioral biases; cognitive heterogeneity exacerbates systemic inefficiency; and information sharing serves as an effective mitigation mechanism.

## Method

### Overall Architecture

LLM agents are deployed in the classic beer distribution game (a 4-echelon supply chain: Retailer → Wholesaler → Distributor → Manufacturer), with each agent deciding order quantities at every period. Experiments include homogeneous conditions (all shallow/deep agents) and stratified conditions (a single deep agent placed at different positions), with 32 independent replications per configuration over 20 periods.

### Key Designs

1. **Hierarchical Reasoning Framework**:

    - Function: Systematically model agents with varying cognitive depths.
    - Mechanism: Cognition is divided into two levels — shallow (DeepSeek-V3, GPT-4.1) and deep (DeepSeek-R1, GPT-5). Deep models consistently outperform their base counterparts on reasoning benchmarks such as AIME and GPQA. A dual-family design (DeepSeek series + GPT series) controls for architectural differences while validating cross-family consistency.
    - Design Motivation: Provides empirically grounded justification for the cognitive stratification, ensuring a scientific basis for the experimental categorization.

2. **Cognitive Heterogeneity Experimental Design**:

    - Function: Isolate the effect of cognitive depth on supply chain behavior.
    - Mechanism: Six configurations are used — homogeneous conditions (Original: all shallow; R-Overall: all deep) and stratified conditions (R-S1 through R-S4, placing a single deep agent at one echelon). Each configuration is crossed with two information conditions (with/without information sharing), and chain-of-thought (CoT) prompting supports structured decision-making.
    - Design Motivation: Systematically varying a single variable (the position of cognitive depth) enables identification of causal effects.

3. **Information Sharing Mechanism**:

    - Function: Test the effectiveness of information transparency in mitigating behavioral biases.
    - Mechanism: Under the information-sharing condition, each agent receives inventory and backlog information from all other echelons. Order variance, total cost, and bullwhip effect intensity are compared across information conditions.
    - Design Motivation: Information asymmetry is a classical driver of the bullwhip effect; this design validates whether LLM agents similarly benefit from information sharing.

### Loss & Training

No model training is involved. Standard statistical tests (sign test, t-test, Mann-Whitney test) are used to verify the significance of results.

## Key Experimental Results

### Main Results

Bullwhip effect replication (homogeneous conditions, no information sharing):

| Configuration | Order Variance Amplification | p-value | Note |
|---|---|---|---|
| DeepSeek-Original | 82.3% | <0.001 | Significant bullwhip effect |
| DeepSeek-R-Overall | 79.8% | <0.001 | Persists after reasoning enhancement |
| GPT-Original | 74.2% | <0.001 | Consistent across families |
| GPT-R-Overall | 74.3% | <0.001 | Cross-family validation |

### Ablation Study

Mitigation effect of information sharing:

| Condition | Total Cost (No IS) | Total Cost (With IS) | Reduction |
|---|---|---|---|
| DeepSeek-Original | 39.43 | 20.15 | ~49% |
| DeepSeek-R-Overall | 29.43 | 17.71 | ~40% |

### Key Findings

- LLM agents successfully reproduce the bullwhip effect observed in human experiments (p<0.001), validating the credibility of LLMs as behavioral proxies.
- Compared to human data, LLM agents exhibit more stable decisions (lower variance), yielding cleaner statistical signals.
- Cognitive enhancement (R1/GPT-5) reduces total cost but does not eliminate the bullwhip effect — even "smarter" agents still exhibit myopic behavior.
- Information sharing is the most effective intervention, consistently reducing costs by 40–50% across all configurations.
- Self-interested behavior (each agent minimizing its own cost) is identified as the root cause of systemic inefficiency.

## Highlights & Insights

- Using LLMs to simulate behavioral experiments is a highly promising paradigm: compared to human-subject experiments, the approach reduces costs by orders of magnitude, supports large-scale replication, and enables precise variable control. This has transformative implications for operations management and behavioral economics research.
- The finding that cognitive enhancement cannot eliminate the bullwhip effect is particularly insightful: the problem lies not in individual intelligence, but in information structure and incentive design — a conclusion that closely mirrors dynamics in real organizations.
- The dual-family validation design (DeepSeek + GPT) ensures the cross-platform robustness of the findings.

## Limitations & Future Work

- Whether the "cognitive biases" exhibited by LLM agents are fundamentally equivalent to those of humans remains uncertain — they may reflect behavioral patterns learned from training data rather than genuine cognitive limitations.
- While the beer distribution game is a classic benchmark, it is highly simplified; real supply chains involve far greater complexity (multiple products, stochasticity, contractual constraints).
- The temperature parameter is fixed at 1; behavior may differ under other temperature settings (though prior work on stability is cited).
- Only 4-echelon linear supply chains are studied; networked supply chain behavior may differ substantially.

## Related Work & Insights

- **vs. Kirshner (2024)**: A pioneering work deploying LLM agents in supply chains, but using homogeneous settings; this paper is the first to introduce cognitive heterogeneity.
- **vs. Park et al. (2023) (Generative Agents)**: Focuses on social interaction simulation; this paper extends LLM agents to structured economic environments.
- **vs. traditional RL methods (IPPO/MAPPO)**: Require strict state space definitions and extensive training; LLM agents exhibit human-like behavior with zero training.

## Rating
- Novelty: ⭐⭐⭐⭐ A novel perspective combining cognitive heterogeneity with supply chain simulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 32 replications × 6 configurations × 2 information conditions, with rigorous statistical validation.
- Writing Quality: ⭐⭐⭐⭐ Experimental design is clear; statistical analysis is sound.
- Value: ⭐⭐⭐⭐ Opens a new direction for applying LLM agents to organizational behavior research.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] A Phase Transition for Opinion Dynamics with Competing Biases](../../AAAI2026/others/a_phase_transition_for_opinion_dynamics_with_competing_biase.md)
- [\[NeurIPS 2025\] A Differentiable Model of Supply-Chain Shocks](../../NeurIPS2025/others/a_differentiable_model_of_supply-chain_shocks.md)
- [\[AAAI 2026\] DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks](../../AAAI2026/others/ds-atgo_dual-stage_synergistic_learning_via_forward_adaptive_threshold_and_backw.md)
- [\[ICLR 2026\] LPWM: Latent Particle World Models for Object-Centric Stochastic Dynamics](../../ICLR2026/others/latent_particle_world_models_self-supervised_object-centric_stochastic_dynamics_.md)
- [\[ACL 2025\] Consistent Client Simulation for Motivational Interviewing-based Counseling](../../ACL2025/others/consistent_client_simulation_for_motivational_interviewing-based_counseling.md)

<!-- RELATED:END -->

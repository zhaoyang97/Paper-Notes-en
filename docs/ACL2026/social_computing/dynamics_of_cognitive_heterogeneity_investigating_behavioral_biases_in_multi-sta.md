---
title: >-
  [Paper Note] Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation
description: >-
  [ACL 2026][Social Computing][Supply chain simulation] This study utilizes LLM agents (DeepSeek/GPT series) to simulate multi-stage supply chains in the classic beer distribution game. It systematically investigates the i…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Supply chain simulation"
  - "cognitive heterogeneity"
  - "bullwhip effect"
  - "LLM agents"
  - "beer distribution game"
date: 2026-05-08
content_hash: 955d7ef34d84fa33
---

# Dynamics of Cognitive Heterogeneity: Investigating Behavioral Biases in Multi-Stage Supply Chains with LLM-Based Simulation

**Conference**: ACL 2026  
**arXiv**: [2604.17220](https://arxiv.org/abs/2604.17220)  
**Code**: None  
**Area**: Others  
**Keywords**: Supply chain simulation, cognitive heterogeneity, bullwhip effect, LLM agents, beer distribution game

## TL;DR

This study utilizes LLM agents (DeepSeek/GPT series) to simulate multi-stage supply chains in the classic beer distribution game. It systematically investigates the impact of cognitive heterogeneity (differences in reasoning capabilities) on system behavior, finding that LLM agents can replicate human bullwhip effects and myopic behavior, while information sharing effectively mitigates these adverse effects.

## Background & Motivation

**Background**: Behavioral experiments (such as the beer distribution game) have revealed supply chain inefficiencies (e.g., the bullwhip effect) caused by cognitive biases. However, traditional human experiments face limitations in scalability, cost, and experimental control. The potential of LLMs as behavioral agents is currently being explored.

**Limitations of Prior Work**: (1) Most LLM multi-agent research focuses on static or structurally simple settings, neglecting highly dynamic multi-period environments; (2) existing studies typically deploy homogeneous agents, ignoring the impact of cognitive heterogeneity (mixing agents with different reasoning capabilities) on collective behavior; (3) there is a lack of rigorous statistical validation.

**Key Challenge**: Strategy diversity is both prevalent and critical in real-world organizations, yet its interactive effects within synthetic environments have not been sufficiently studied.

**Goal**: To construct an LLM-driven supply chain simulation paradigm and systematically study how cognitive heterogeneity affects collective behavior.

**Key Insight**: Utilize LLMs with varying reasoning capabilities (base vs. reasoning-enhanced) to represent different cognitive levels, deploying heterogeneous agents at different positions within the supply chain.

**Core Idea**: LLM agents can replicate human behavioral biases; cognitive heterogeneity exacerbates system inefficiency, while information sharing serves as an effective mitigation mechanism.

## Method

### Overall Architecture

LLM agents are deployed in the classic beer distribution game (a 4-stage supply chain: Retailer → Wholesaler → Distributor → Manufacturer), where each agent decides the order quantity in each period. The experiments include homogeneous conditions (all shallow or all deep agents) and tiered conditions (a single deep agent placed at different positions). Each configuration involves 32 independent repetitions over 20 periods.

### Key Designs

1.  **Hierarchical Reasoning Framework**:
    - **Function**: To systematically model agents with varying cognitive depths.
    - **Mechanism**: Cognition is divided into two levels: shallow (DeepSeek-V3, GPT-4.1) and deep (DeepSeek-R1, GPT-5). Deep models consistently outperform their corresponding base versions on reasoning benchmarks such as AIME and GPQA. A dual-family design (DeepSeek series + GPT series) is employed to control for architectural differences and verify cross-family consistency.
    - **Design Motivation**: To provide an empirically supported basis for cognitive stratification, ensuring that experimental classifications are scientifically grounded.

2.  **Cognitive Heterogeneity Experimental Design**:
    - **Function**: To isolate the impact of cognitive depth on supply chain behavior.
    - **Mechanism**: Six configurations are tested: homogeneous conditions (Original all-shallow, R-Overall all-deep) and tiered conditions (R-S1 to R-S4, where a deep agent is placed at only one position). Each configuration is tested under two information conditions (with/without information sharing), using Chain-of-Thought (CoT) prompting to support structured decision-making.
    - **Design Motivation**: To identify causal effects by systematically varying a single variable (the position of cognitive depth).

3.  **Information Sharing Mechanism**:
    - **Function**: To test the effectiveness of information transparency in alleviating behavioral biases.
    - **Mechanism**: Under the information sharing condition, each agent is provided with the inventory and backlog information of other stages. Order fluctuations, total costs, and the intensity of the bullwhip effect are compared between conditions with and without information sharing.
    - **Design Motivation**: Information asymmetry is a classic cause of the bullwhip effect; this design verifies whether LLM agents also benefit from information sharing.

### Loss & Training

This study does not involve model training. Standard statistical tests (sign test, t-test, Mann-Whitney test) are used to verify the significance of the results.

## Key Experimental Results

### Main Results

Replication of the bullwhip effect (homogeneous conditions, no information sharing):

| Configuration | Order Variance Increase | $p$-value | Description |
| :--- | :--- | :--- | :--- |
| DeepSeek-Original | 82.3% | $<0.001$ | Significant bullwhip effect |
| DeepSeek-R-Overall | 79.8% | $<0.001$ | Persists after reasoning enhancement |
| GPT-Original | 74.2% | $<0.001$ | Consistent across families |
| GPT-R-Overall | 74.3% | $<0.001$ | Consistency verification |

### Ablation Study

Mitigation effect of information sharing:

| Condition | Total Cost w/o IS | Total Cost w/ IS | Reduction |
| :--- | :--- | :--- | :--- |
| DeepSeek-Original | 39.43 | 20.15 | ~49% |
| DeepSeek-R-Overall | 29.43 | 17.71 | ~40% |

### Key Findings

- LLM agents successfully replicate the bullwhip effect observed in human experiments ($p < 0.001$), validating the credibility of LLMs as behavioral proxies.
- Compared to human data, LLM agents' decisions are more stable (lower variance), yielding clearer statistical signals.
- While cognitive enhancement (R1/GPT-5) reduces total costs, it does not eliminate the bullwhip effect—even "smarter" agents continue to exhibit myopic behavior.
- Information sharing is the most effective intervention: it consistently reduces costs by 40-50% across all configurations.
- Self-interested behavior (each agent minimizing its own cost) is the fundamental cause of system inefficiency.

## Highlights & Insights

- Using LLMs for behavioral experiment simulation is a highly promising paradigm: compared to human experiments, the costs are several orders of magnitude lower, allowing for large-scale repetition and precise variable control. This has transformative implications for operations management and behavioral economics.
- The insight that cognitive enhancement cannot eliminate the bullwhip effect is profound: the issue lies not in individual intelligence deficits but in the information structure and incentive mechanisms—this aligns closely with observations in real-world organizations.
- The dual-family verification design (DeepSeek + GPT) ensures the robustness of the findings across different platforms.

## Limitations & Future Work

- Whether the "cognitive biases" of LLM agents are essentially identical to those of humans remains questionable—they may represent behavioral patterns learned from training data rather than true cognitive constraints.
- While classic, the beer distribution game is highly simplified; the complexity of real supply chains (multi-product, stochasticity, contractual constraints) far exceeds this setup.
- The temperature parameter was fixed at 1; behavior might vary under different temperatures (though stability results from prior work were cited).
- Only 4-stage linear supply chains were investigated; the behavior of networked supply chains may be entirely different.

## Related Work & Insights

- **vs Kirshner (2024)**: A pioneer in deploying LLM agents in supply chains, but focused on homogeneous settings; this paper is the first to introduce cognitive heterogeneity.
- **vs Park et al. (2023) (Generative Agents)**: Focuses on social interaction simulation; this work extends LLM agents into structured economic environments.
- **vs Traditional RL methods (IPPO/MAPPO)**: These require strict state space definitions and extensive training; LLM agents exhibit human-like behavior with zero training.

## Rating
- Novelty: ⭐⭐⭐⭐ New perspective on cognitive heterogeneity + supply chain simulation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 32 repetitions × 6 configurations × 2 information conditions, with rigorous statistical validation.
- Writing Quality: ⭐⭐⭐⭐ Clear experimental design and solid statistical analysis.
- Value: ⭐⭐⭐⭐ Opens a new direction for the application of LLM agents in organizational behavior research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Probing Multimodal Large Language Models on Cognitive Biases in Chinese Short-Video Misinformation](probing_multimodal_large_language_models_on_cognitive_biases_in_chinese_short-vi.md)
- [\[ACL 2026\] Why Are We Moral? An LLM-based Agent Simulation Approach to Study Moral Evolution](why_are_we_moral_an_llm-based_agent_simulation_approach_to_study_moral_evolution.md)
- [\[ACL 2026\] Point of Order: Action-Aware LLM Persona Modeling for Realistic Civic Simulation](point_of_order_action-aware_llm_persona_modeling_for_realistic_civic_simulation.md)
- [\[ACL 2026\] Investigating Counterfactual Unfairness in LLMs towards Identities through Humor](investigating_counterfactual_unfairness_in_llms_towards_identities_through_humor.md)
- [\[ACL 2026\] To Lie or Not to Lie? Investigating The Biased Spread of Global Lies by LLMs](to_lie_or_not_to_lie_investigating_the_biased_spread_of_global_lies_by_llms.md)

</div>

<!-- RELATED:END -->

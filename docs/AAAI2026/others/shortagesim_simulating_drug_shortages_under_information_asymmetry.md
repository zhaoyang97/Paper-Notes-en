---
title: >-
  [Paper Note] ShortageSim: Simulating Drug Shortages under Information Asymmetry
description: >-
  [AAAI 2026][drug shortage] This paper proposes ShortageSim, the first **LLM-based multi-agent** simulation framework for drug shortages. It models strategic decision-making among FDA regulators, manufacturers…
tags:
  - "AAAI 2026"
  - "drug shortage"
  - "multi-agent simulation"
  - "large language models"
  - "information asymmetry"
  - "regulatory policy"
date: 2026-05-08
content_hash: 3cf819cc5a9618a6
---

# ShortageSim: Simulating Drug Shortages under Information Asymmetry

**Conference**: AAAI 2026
**arXiv**: [2509.01813](https://arxiv.org/abs/2509.01813)  
**Code**: [https://github.com/Lemutisme/ShortageSim](https://github.com/Lemutisme/ShortageSim)  
**Area**: Other
**Keywords**: drug shortage, multi-agent simulation, large language models, information asymmetry, regulatory policy

## TL;DR

This paper proposes ShortageSim, the first **LLM-based multi-agent** simulation framework for drug shortages. It models strategic decision-making among FDA regulators, manufacturers, and buyers under information asymmetry, achieving an 84% improvement in predicting resolution lag time on historical shortage data, and provides a controlled testbed for evaluating regulatory strategies.

## Background & Motivation

1. **Background**: Drug shortages represent a global healthcare crisis — the United States has averaged 130+ new shortage events annually over the past decade, with mean duration growing from 9 months in 2011 to 14 months in 2016. U.S. hospitals spend at least $359 million per year managing shortage-related logistics.
2. **Limitations of Prior Work**: (a) Severe information asymmetry — manufacturers conceal capacity data as trade secrets, regulators observe only aggregate shortage levels, and buyers cannot distinguish temporary from permanent supply disruptions; (b) Traditional game-theoretic models assume perfect rationality and complete information, failing to capture how real decision-makers interpret ambiguous regulatory signals and dynamically update beliefs; (c) FDA regulatory interventions (e.g., issuing shortage alerts) may trigger hoarding behavior, paradoxically exacerbating shortages.
3. **Key Challenge**: Regulatory signals are designed to promote transparency and alleviate shortages, yet they may trigger panic hoarding and coordination failures from overcapacity expansion. Evaluating these counterfactual effects requires a controlled experimental environment.
4. **Goal**: To construct a framework capable of simulating the impact of regulatory interventions on competitive dynamics, particularly under conditions of information asymmetry.
5. **Key Insight**: LLMs serve as heterogeneous agents that simulate subjective interpretation of partial information and suboptimal decision-making by manufacturers, buyers, and the FDA.
6. **Core Idea**: LLMs inherently possess the capacity for "bounded rationality and subjective interpretation," making them more suitable than game-theoretic assumptions for simulating human decision-making under information asymmetry.

## Method

### Overall Architecture

The framework comprises four core components: (1) an **Environment Module** managing market dynamics, state transitions, and information isolation; (2) an **Agent System** of LLM-driven, role-specific decision-makers (manufacturers, buyers, FDA); (3) an **Information Flow** layer controlling inter-agent communication to simulate information asymmetry; and (4) a **Simulation Controller** orchestrating execution flow and logging decisions.

### Key Designs

1. **Two-Stage Decision Pipeline (Analyze → Decide)**

    - **Function**: Simulates the process from information processing to strategy formation in real-world decision-making.
    - **Mechanism**: Each agent proceeds through two steps — (a) **Collection & Analysis**: receives unstructured context (FDA announcement text, aggregated market indicators, historical trends) and generates a structured situational representation; (b) **Decision Generation**: produces a decision and detailed reasoning from the structured representation. Crucially, decisions need not be optimal — reflecting the reality that some stakeholders distrust competitor rationality and make suboptimal or even irrational choices.
    - **Design Motivation**: The two-stage design allows inspection of FDA communication effects through reasoning outputs and captures differences in reasoning processes across agent types.

2. **Information Asymmetry Modeling**

    - **Function**: Rigorously simulates information barriers present in pharmaceutical supply chains.
    - **Mechanism**: Manufacturers know their own capacity and recovery time but not competitors' states, inferring the latter only from FDA announcements and their own allocated demand. Buyers observe only total supply without knowing allocations or capacity. The FDA observes only aggregate shortage levels and mandatorily reported disruptions, without knowing specific capacities, investments, or inventories. The temporal sequence is: disruption → FDA decides whether to issue an announcement → manufacturers make simultaneous investment decisions → buyers make simultaneous procurement decisions → market clears.
    - **Design Motivation**: Directly reflects real supply chain structure — manufacturer capacity is a trade secret, U.S. reporting is decentralized and voluntary, and buyer inventories are opaque.

3. **Role-Specific Agent Design**

    - **Function**: Each role has a distinct objective function and information availability.
    - **Mechanism**: **Manufacturers** — aim to maximize profit, balancing the opportunity to gain from competitor disruptions against the risk of overcapacity. **Buyers** — face a variant of the newsvendor problem under supply uncertainty, prioritizing patient safety while balancing hoarding costs. **FDA** — operates within a reactive policy framework, issuing announcements to alleviate shortages without disclosing private information, balancing urgency against market stability.
    - **Design Motivation**: Heterogeneous agents with multiple perspectives and objectives more closely approximate real market behavior than homogeneous assumptions.

### Loss & Training

No training is required. GPT-4o, Gemini 2.5 Flash, Claude Sonnet 4.5, and DeepSeek V3.2 serve as LLM backbones at temperature=0.3. Evaluation is conducted on 2,925 FDA shortage events and 51 resolved historical trajectories.

## Key Experimental Results

### Main Results

| Model | Dataset | RLP(%) | FIP(%) |
|-------|---------|--------|--------|
| ShortageSim (GPT-4o) | FDA-Disc | **4.5±3.4** | 82.6±3.0 |
| Zero-shot (GPT-4o) | FDA-Disc | -28.3±0.2 | 92.9±0.1 |
| ShortageSim (Claude) | FDA-Disc | -9.4±1.4 | 69.1±2.3 |
| Zero-shot (Claude) | FDA-Disc | -32.7±0.5 | 75.9±0.7 |
| ShortageSim (GPT-4o) | FDA-NR | -34.8±8.7 | 30.0±5.5 |
| Zero-shot (GPT-4o) | FDA-NR | -23.1±1.6 | 90.4±18.6 |

### Ablation Study

| Configuration | RLP(%) | Notes |
|---------------|--------|-------|
| ShortageSim (full) | 4.5 | FDA announcements enabled |
| w/o FDA announcements | 19.1 | Resolution time significantly prolonged |
| Zero-shot baseline | -28.3 | No iterative decision-making; predicts premature resolution |

### Key Findings

- ShortageSim consistently outperforms zero-shot baselines in RLP across all 4 LLM providers (Wilcoxon test, $p < 0.05$).
- GPT-4o achieves an RLP of only 4.5%, nearly perfectly reproducing historical resolution timelines.
- Disabling FDA announcements raises RLP from 4.5% to 19.1%, **confirming the effectiveness of regulatory signals in alleviating shortages**.
- Proactive policies induce more hoarding behavior and worse shortage resolution outcomes.
- With more than 5 manufacturers, investment willingness declines (free-rider effect), prolonging shortages.
- Performance is weaker on the FDA-NR dataset, as disruption causes in this subset are unclear and the simulation assumptions do not align well with actual conditions.

## Highlights & Insights

- **LLMs as "Boundedly Rational" Agents**: Classical game theory assumes perfect rationality; LLMs inherently support subjective interpretation of ambiguous signals and suboptimal decision-making, more closely approximating human behavior.
- **Counterfactual Policy Evaluation Platform**: The framework enables systematic comparison of different FDA communication strategies (reactive vs. proactive, severity framing, etc.) — controlled experiments that are infeasible in reality.
- As the **first LLM multi-agent framework in the drug shortage domain**, the open-sourced dataset of 2,925 FDA shortage events addresses the gap created by prior reliance on synthetic or proprietary data.

## Limitations & Future Work

- Intermediate entities such as distributors and GPOs are not included, potentially omitting logistics-level dynamics.
- The manufacturer symmetry assumption is noted as relaxed in the introduction but is not sufficiently explored in experiments.
- Reproducibility of LLM reasoning is limited (temperature=0.3 still introduces stochasticity); averaging over multiple runs may be insufficient.
- Weaker performance on the FDA-NR dataset indicates that the framework is sensitive to prior assumptions about disruption causes.

## Related Work & Insights

- **vs. Traditional Operations Research Models**: These assume rational decisions and complete information, failing to capture dynamic behavior under information asymmetry. ShortageSim directly models information barriers.
- **vs. EconAgent**: EconAgent simulates macroeconomic phenomena, whereas ShortageSim focuses on micro-level interactions within a specific supply chain.
- **vs. OptiGuide**: Microsoft's supply chain optimization framework uses LLMs for optimization; ShortageSim uses LLMs for behavioral simulation.
- The framework is generalizable to other supply chain crisis simulations (semiconductor shortages, energy supply disruptions, etc.).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First application of LLM multi-agents to drug shortage simulation; information asymmetry modeling is elegant
- Experimental Thoroughness: ⭐⭐⭐⭐ Historical data validation + multiple LLM providers + counterfactual analysis
- Writing Quality: ⭐⭐⭐⭐⭐ Problem motivation is well-developed; framework diagrams are clear
- Value: ⭐⭐⭐⭐⭐ Significant potential impact on public health policy formulation

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] On the Information Processing of One-Dimensional Wasserstein Distances with Finite Samples](on_the_information_processing_of_one-dimensional_wasserstein_distances_with_fini.md)
- [\[AAAI 2026\] Optimal Welfare in Noncooperative Network Formation under Attack](optimal_welfare_in_noncooperative_network_formation_under_attack.md)
- [\[AAAI 2026\] Predict and Resist: Long-Term Accident Anticipation under Sensor Noise](predict_and_resist_long-term_accident_anticipation_under_sensor_noise.md)
- [\[AAAI 2026\] Data Complexity of Querying Description Logic Knowledge Bases under Cost-Based Semantics](data_complexity_of_querying_description_logic_knowledge_bases_under_cost-based_s.md)
- [\[NeurIPS 2025\] Neural Network for Simulating Radio Emission from Extensive Air Showers](../../NeurIPS2025/others/neural_network_for_simulating_radio_emission_from_extensive_air_showers.md)

</div>

<!-- RELATED:END -->

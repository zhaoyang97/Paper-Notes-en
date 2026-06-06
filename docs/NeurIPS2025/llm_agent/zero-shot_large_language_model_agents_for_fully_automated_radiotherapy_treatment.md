---
title: >-
  [Paper Note] Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning
description: >-
  [NeurIPS 2025][LLM Agent][radiotherapy treatment planning] This paper proposes a zero-shot LLM Agent-based workflow for automated radiotherapy treatment planning…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "radiotherapy treatment planning"
  - "zero-shot reasoning"
  - "IMRT"
  - "inverse optimization"
date: 2026-05-08
content_hash: 7ab4b5bbe8d85013
---

# Zero-Shot Large Language Model Agents for Fully Automated Radiotherapy Treatment Planning

**Conference**: NeurIPS 2025
**arXiv**: [2510.11754](https://arxiv.org/abs/2510.11754)  
**Code**: None  
**Area**: LLM Agent / Medical AI
**Keywords**: LLM Agent, radiotherapy treatment planning, zero-shot reasoning, IMRT, inverse optimization

## TL;DR

This paper proposes a zero-shot LLM Agent-based workflow for automated radiotherapy treatment planning, in which the LLM directly interacts with a commercial treatment planning system (Eclipse TPS). By iteratively extracting dose-volume histogram (DVH) metrics and objective function losses and reasoning about constraint adjustment strategies, the approach achieves dose distribution quality comparable to or better than clinical manual planning on 20 head-and-neck cancer IMRT cases.

## Background & Motivation

**Background**: Radiotherapy is a cornerstone of modern cancer management; approximately 50–70% of cancer patients worldwide require at least one course of radiotherapy, involving millions of new cases annually. Manual treatment planning in IMRT (Intensity-Modulated Radiation Therapy) and VMAT (Volumetric Modulated Arc Therapy) is extremely time-consuming and exhibits substantial inter-planner variability—even within the same institution, planners may differ considerably in their approaches to target coverage and organs-at-risk (OAR) sparing.

**Limitations of Prior Work**: Current automated planning methods fall into four paradigms: (i) knowledge-based planning (KBP)—requiring large volumes of high-quality annotated data; (ii) protocol-driven planning—lacking flexibility for complex or atypical anatomical structures; (iii) multi-criteria optimization (MCO)—demanding significant planner involvement and expertise; and (iv) reinforcement learning (RL)—computationally intensive and reliant on expert-designed reward functions. None of these approaches readily achieves generalizable clinical deployment.

**Key Challenge**: Radiotherapy planning is a highly specialized domain in which publicly available training data are extremely scarce, placing inherent limits on approaches that depend on large-scale training datasets. Meanwhile, a severe global shortage of radiotherapy workforce persists—dosimetrist recruitment difficulties in the UK, ongoing medical physicist shortages in the US, and widespread understaffing in European radiotherapy departments—creating an urgent need for generalizable automation that does not rely on institution-specific data.

**Goal**: To design a zero-shot LLM Agent-based radiotherapy planning workflow that enables an LLM to autonomously complete IMRT inverse planning optimization through direct interaction with a commercial TPS, without any prior treatment planning training or fine-tuning.

**Key Insight**: The general reasoning capabilities of LLMs are leveraged to decompose the complex treatment planning task into domain-agnostic subtasks (arithmetic computation, trend inference, and constraint adjustment), guided by chain-of-thought prompting for multi-step decision-making.

**Core Idea**: An LLM Agent is embedded within a clinical TPS to simulate the iterative workflow of a human planner—"observe DVH → analyze deviation → adjust constraints"—generating high-quality IMRT plans without any annotated data.

## Method

### Overall Architecture

The workflow comprises two key components: (1) an LLM Agent that interacts directly with a commercial TPS via the Eclipse Scripting API (ESAPI) to extract intermediate planning states (DVH metrics and objective function values) and modify inverse planning constraints; and (2) an LLM that leverages current planning states and historical iteration information to apply general reasoning capabilities in proposing clinically meaningful constraint modifications. The entire process simulates the manual workflow of a human planner: observe key dose endpoints → analyze objective function feedback → reason about and propose constraint adjustments.

### Key Designs

1. **Direct TPS Integration Module**:

    - **Function**: The LLM Agent is directly embedded into Eclipse TPS (version 15.6) via ESAPI, enabling programmatic access to the treatment planning environment. The agent can retrieve intermediate planning states (DVH metrics and objective function values) and modify inverse planning constraints in a manner identical to that of a human planner.
    - **Mechanism**: All interactions occur within the native TPS environment rather than through a surrogate optimization engine or approximate planning platform, ensuring consistency with the clinical workflow.
    - **Design Motivation**: Prior work (Liu et al. 2025) relied on proprietary platforms, limiting clinical transferability. Direct integration with a commercial TPS ensures portability and clinical usability, given that Eclipse is one of the most widely used TPS platforms globally.

2. **Arithmetic Tools and History Tracking Module**:

    - **Function**: Dedicated arithmetic tools are developed to compute numerical deviations among current dose endpoints, clinical objectives, and optimization constraints. All historical iterations of constraints, dose outcomes, and deviations are compiled into structured data for LLM trend reasoning.
    - **Mechanism**: Since LLMs are insufficiently reliable for precise numerical computation, external tools handle arithmetic tasks, allowing the LLM to focus on reasoning and decision-making where it excels. Accumulated historical data enables the agent to identify trends—for example, when reducing an OAR constraint no longer decreases dose but sharply increases loss, indicating limited further optimization potential.
    - **Design Motivation**: Treatment planning requires three core capabilities: (1) arithmetic ability to quantify deviations, (2) domain understanding of the optimization system, and (3) reasoning ability to interpret trends and propose adjustments. LLMs naturally possess capability (3) but require external support for (1) and (2).

3. **Optimization Prior Injection and Chain-of-Thought Reasoning**:

    - **Function**: Domain knowledge of inverse planning is injected into the LLM via prompts, including the semantics and scale of objective function losses, the relationship between constraint deviations and improvement headroom, the numerical ranges of optimization constraints, and the directional effects of tunable parameters on dose distributions. Chain-of-thought prompting is also employed, requiring the LLM to explicitly articulate its reasoning before proposing new constraint values.
    - **Mechanism**: The Eclipse optimization engine uses weighted quadratic penalty objective functions, and effective optimization often requires setting objectives below the desired dose to create "driving force"—a "hidden rule" that is critical for the LLM but not available as prior knowledge. Chain-of-thought reasoning enables the agent to mimic the logical process of a human planner: consider clinical trade-offs → evaluate constraint violations in context → prioritize adjustments based on historical trends.
    - **Design Motivation**: Ablation experiments demonstrate that removing optimization priors leads to significant deterioration in plan quality (with generally increased OAR doses), confirming that structured injection of domain knowledge is a prerequisite for the LLM to successfully perform treatment planning.

4. **Iterative Optimization Strategy**:

    - **Function**: The agent employs a "coarse exploration → fine-grained refinement" iterative strategy. Larger constraint adjustment steps are used in early iterations to probe the dose-sparing potential of each organ, while smaller steps are used in later iterations for precise tuning to avoid over-protection.
    - **Mechanism**: For organs with numerical constraints (e.g., parotid gland median dose 16 Gy), the agent initializes near the clinical objective to accelerate convergence; for range constraints (e.g., 25–30 Gy), boundary values are chosen as starting points; for organs without explicit numerical constraints (e.g., pharynx "minimize as much as possible"), the agent autonomously selects a reasonable starting value (e.g., 45 Gy). When dose stagnation is observed alongside sharply increasing loss, the agent proactively relaxes constraints to protect target coverage.
    - **Design Motivation**: This strategy simulates the actual practice of an experienced dosimetrist—first "probing" the optimization headroom for each organ, then carefully navigating trade-offs. The entire process typically completes within a few optimization steps and takes less than 5 minutes in total.

### Loss & Training

The LLM Agent in this work involves no training or fine-tuning of any kind. Internally, Eclipse TPS employs a weighted sum of quadratic penalties as its objective function, penalizing constraint violations for each structure and objective. The LLM Agent uses the values and trends of this objective function loss to guide constraint adjustment decisions, but does not directly modify the objective function itself.

## Key Experimental Results

### Main Results: LLM Plans vs. Clinical Manual Plans (20 Head-and-Neck IMRT Cases)

| Metric | Clinical | GPT-4.1-WP | GPT-4.1-mini-WP | GPT-4.1-WOP | GPT-4.1-mini-WOP |
|--------|---------|------------|-----------------|-------------|-----------------|
| Plan D_max (Gy) | 76.22±1.44 | 74.53±1.48 | 74.19±1.07 | 74.17±1.20 | **73.87±0.93** |
| Brainstem D_max (Gy) | **22.13±6.65** | 24.56±7.21 | 24.21±6.63 | 27.57±7.27 | 28.08±7.26 |
| Spinal cord+5mm D_max (Gy) | 44.91±2.82 | **44.46±3.47** | 44.58±3.97 | 48.87±3.03 | 49.59±3.06 |
| Mandible D_max (Gy) | 72.06±6.94 | **70.86±6.94** | 71.17±6.96 | 71.66±6.69 | 71.62±6.42 |
| Left parotid D50 (Gy) | 22.66±11.22 | **19.21±3.09** | 21.93±5.71 | 23.18±3.97 | 22.99±3.92 |
| Right parotid D50 (Gy) | 22.52±10.17 | **20.47±3.64** | 20.70±5.42 | 24.94±3.75 | 25.42±5.97 |
| Oral cavity D50 (Gy) | 36.14±12.44 | 34.95±10.98 | **33.26±11.45** | 38.48±9.09 | 39.41±9.88 |
| Larynx D50 (Gy) | 33.16±14.42 | **29.43±8.02** | 31.29±9.96 | 36.24±9.36 | 37.83±11.49 |
| Pharynx D50 (Gy) | 47.54±11.50 | **39.85±9.62** | 44.37±9.04 | 49.18±7.20 | 49.43±8.34 |
| PTV_primary CI | 1.88±0.29 | 1.82±0.17 | 1.83±0.17 | 1.92±0.19 | 1.93±0.17 |
| PTV_boost CI | 1.39±0.19 | 1.18±0.10 | 1.17±0.09 | 1.17±0.09 | **1.16±0.09** |
| PTV_boost HI | 0.061±0.021 | 0.062±0.021 | 0.058±0.013 | 0.059±0.020 | **0.055±0.019** |

### Ablation Study: Effect of Optimization Priors

| Comparison | Key Observations |
|-----------|-----------------|
| GPT-4.1 with prior (WP) vs. without prior (WOP) | Removal of priors leads to generally elevated OAR doses: brainstem +3.01 Gy, spinal cord +4.41 Gy, parotid glands +3–4 Gy, larynx +6.81 Gy, pharynx +9.33 Gy |
| GPT-4.1-mini WP vs. WOP | Similarly, OAR sparing deteriorates markedly without priors; although CI/HI may marginally improve, this results from an unfavorable trade-off due to insufficient OAR protection |
| GPT-4.1 WP vs. GPT-4.1-mini WP | GPT-4.1 achieves numerically superior results on most metrics, reflecting stronger reasoning ability and planning efficiency |
| LLM plans vs. clinical plans: consistency | LLM plans exhibit narrower interquartile ranges, particularly for target conformity indices and parotid doses, indicating lower inter-patient variability |

### Key Findings

- **Superior hotspot control**: GPT-4.1-WP achieves a Plan D_max of 74.53 Gy (106.5% of prescribed dose) vs. 76.22 Gy (108.8%) for clinical plans—a notable improvement.
- **Superior conformity**: PTV_boost CI 1.18 vs. 1.39 (clinical); PTV_primary CI 1.82 vs. 1.88 (clinical).
- **Comparable or superior OAR sparing**: Particularly significant improvements are observed for the parotid glands (left: 19.21 vs. 22.66 Gy; right: 20.47 vs. 22.52 Gy) and pharynx (39.85 vs. 47.54 Gy).
- **Optimization priors are necessary**: Removal of priors uniformly degrades all OAR doses, confirming that structured domain knowledge injection is key to success.
- **Highly efficient planning**: Single-case planning completes in under 5 minutes on an Intel Xeon CPU with 32 GB RAM, substantially faster than manual planning.

## Highlights & Insights

- **Viability of zero-shot approaches in specialized domains**: This work provides the first demonstration that an LLM can autonomously generate clinically acceptable IMRT plans within a commercial TPS without any prior treatment planning data—a significant finding for data-scarce specialized fields.
- **Simple yet effective agent design**: The approach does not require complex multi-agent architectures or retrieval-augmented generation; high-quality decision-making is driven by the straightforward combination of arithmetic tools, domain prior prompts, chain-of-thought reasoning, and history tracking.
- **Interpretable reasoning chains**: The agent explicitly articulates its reasoning at each optimization step (e.g., "observations from the preceding steps indicate that further tightening the constraint only increases loss without reducing dose, suggesting that further protection may compromise PTV coverage"), enhancing clinical trustworthiness.
- **Consistency advantage**: LLM plans exhibit lower inter-patient variability, which is particularly valuable for clinical quality control.

## Limitations & Future Work

1. **Disease site scope**: Validation is limited to head-and-neck IMRT; other common disease sites such as lung, cervical, and prostate cancers, as well as other planning modalities such as VMAT, have not been tested.
2. **Small sample size**: Only 20 patients are included, limiting statistical power; moreover, all patients share the same prescription scheme (70 Gy + 44 Gy).
3. **API cost and dependency**: The approach relies on GPT-4.1 API calls; the costs and latency associated with multiple iterations per case are not quantified. Dependence on a closed-source LLM also limits reproducibility.
4. **Single-institution data**: All cases originate from a single institution (Duke University); cross-institutional generalizability has not been validated, and clinical constraint conventions and physician preferences may vary substantially across institutions.
5. **Absence of comparison with RL/KBP baselines**: Evaluation is conducted only against clinical manual plans, without direct comparison to existing automated methods such as reinforcement learning or knowledge-based planning.
6. **Institution-specificity of optimization priors**: The optimization priors injected into prompts (e.g., the characteristics of Eclipse's quadratic loss function) are TPS-specific; migration to other TPS platforms (e.g., Pinnacle, RayStation) would require rewriting these priors.

## Related Work & Insights

- **Wang et al. (2025)**: Proposes a few-shot LLM planning method for lung and cervical cancers, requiring prior plans as reference. The present work advances this to fully zero-shot, eliminating dependence on historical plans.
- **Liu et al. (2025)**: Uses GPT-4Vision to guide radiotherapy planning, but relies on a proprietary platform. This work directly integrates with a commercial TPS, enhancing clinical feasibility.
- **RL-based methods (Yang et al. 2024; Shen et al. 2021)**: Train virtual planners via reinforcement learning to iteratively adjust parameters. The LLM Agent approach proposed here requires no training, offering a complementary zero-training-data pathway.
- **Broader implication**: The paradigm of "LLM + domain tools + structured priors" is broadly transferable to other domains requiring expert iterative decision-making—such as pharmaceutical formulation optimization or architectural design parameter tuning. The zero-shot capability means new institutions can deploy the system rapidly without accumulating local training data.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The first fully automated zero-shot LLM-based treatment planning approach within a commercial TPS; eliminating dependence on annotated data represents an important breakthrough.
- **Experimental Thoroughness**: ⭐⭐⭐ — The validation scale of 20 cases, a single disease site, and a single institution is limited; the ablation study is well-designed but lacks direct comparison with RL/KBP baselines.
- **Writing Quality**: ⭐⭐⭐⭐ — The methodology is described clearly and in detail; presentation of the agent's reasoning chains in case analyses enhances both readability and credibility.
- **Value**: ⭐⭐⭐⭐ — The work has direct practical value for radiotherapy automation; the zero-shot characteristic addresses the pain points of data scarcity and cross-institutional deployment, with considerable translational potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks](agenttts_large_language_model_agent_for_testtime_computeopti.md)
- [\[NeurIPS 2025\] TrajAgent: An LLM-Agent Framework for Trajectory Modeling via Large-and-Small Model Collaboration](trajagent_an_llm-agent_framework_for_trajectory_modeling_via_large-and-small_mod.md)
- [\[NeurIPS 2025\] Are Large Language Models Sensitive to the Motives Behind Communication?](are_large_language_models_sensitive_to_the_motives_behind_communication.md)
- [\[AAAI 2026\] AutoTool: Efficient Tool Selection for Large Language Model Agents](../../AAAI2026/llm_agent/autotool_efficient_tool_selection_for_large_language_model_agents.md)
- [\[NeurIPS 2025\] Automated Composition of Agents: A Knapsack Approach for Agentic Component Selection](automated_composition_of_agents_a_knapsack_approach_for_agentic_component_select.md)

</div>

<!-- RELATED:END -->

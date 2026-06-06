---
title: >-
  [Paper Note] Context-Value-Action Architecture for Value-Driven Large Language Model Agents
description: >-
  [ACL 2026 (Findings)][LLM Agent][Value-Driven Agents] This paper proposes the CVA (Context-Value-Action) architecture based on the S-O-R psychological model and Schwartz theory of basic human values. By training a Value…
tags:
  - "ACL 2026 (Findings)"
  - "LLM Agent"
  - "Value-Driven Agents"
  - "Behavioral Simulation"
  - "Schwartz Theory of Values"
  - "Behavioral Polarization"
  - "Verifier"
date: 2026-05-08
content_hash: 5c4df246892e6e7b
---

# Context-Value-Action Architecture for Value-Driven Large Language Model Agents

**Conference**: ACL 2026 (Findings)  
**arXiv**: [2604.05939](https://arxiv.org/abs/2604.05939)  
**Code**: None  
**Area**: LLM Agent / Interpretability  
**Keywords**: Value-Driven Agents, Behavioral Simulation, Schwartz Theory of Values, Behavioral Polarization, Verifier

## TL;DR
This paper proposes the CVA (Context-Value-Action) architecture based on the S-O-R psychological model and Schwartz theory of basic human values. By training a Value Verifier on real human data to decouple behavioral generation from cognitive reasoning, it effectively mitigates behavioral polarization in LLM agents, significantly outperforming baselines on CVABench, which contains over 1.1 million real interaction trajectories.

## Background & Motivation

**Background**: LLM-based human-like agents (e.g., game NPCs, social simulators, task assistants) need to faithfully capture the complexity, diversity, and stochasticity of human behavior. Existing methods primarily rely on psychological prompting (such as role-playing or CoT reasoning) to simulate human cognitive processes.

**Limitations of Prior Work**: Existing LLM agents frequently exhibit behavioral rigidity and stereotypes. Critically, this issue is masked by current evaluation methods: "LLM-as-a-judge" assessments suffer from self-reference bias, where the evaluator models share the same pre-training biases as the agents, tending to approve of polarized behaviors rather than penalizing the lack of realism.

**Key Challenge**: Increasing the intensity of prompt-driven reasoning does not improve behavioral faithfulness but instead exacerbates value polarization—LLMs simplify nuanced value dimensions into "caricatured" prototypes (e.g., extremizing an "irritable" personality into consistently aggressive responses), leading to a collapse of population diversity.

**Goal**: Construct agents capable of faithfully reproducing human behavioral diversity, using real human data as the gold standard for evaluation rather than LLM self-assessment.

**Key Insight**: Drawing on the psychological S-O-R (Stimulus-Organism-Response) model and the Schwartz theory of basic human values—human behavior is not a static output of personality but a dynamic process where situational contexts activate specific value dimensions.

**Core Idea**: Replace the LLM's internal value judgment with an external Value Verifier (trained on real human data) to decouple behavior generation from cognitive reasoning, thereby avoiding polarization caused by self-reference bias.

## Method

### Overall Architecture
CVA adopts a "Generation-Verification" paradigm: it first calibrates the base LLM's value-behavior mapping through SFT+DPO (VMC phase), and then uses an independently trained Value Verifier to select the behavior most consistent with current activated values from multiple candidates (VDR phase).

### Key Designs

1.  **Value-behavior Mapping Calibration (VMC)**:
    - **Function**: Corrects intrinsic value distortions within the LLM.
    - **Mechanism**: A two-step process—SFT fine-tunes the model on CVABench real trajectories to align the probability space with the real conditional distribution $P(A|C,V)$; DPO further reinforces authentic value-behavior associations using preference pairs (nuanced and consistent vs. caricatured and exaggerated) to suppress distorted reasoning paths.
    - **Design Motivation**: To learn directly from real-world data and prevent the LLM from simplifying value $V$ into a caricatured prototype $V'$.

2.  **Value-Driven Verifier (Value Verifier)**:
    - **Function**: Acts as an independent discriminator to evaluate the consistency between candidate behaviors and activated values.
    - **Mechanism**: A verifier trained on real $(C, V, A)$ triplets. During inference, it follows a "Generate-then-Select" protocol—the calibrated model samples $N$ candidate behaviors, the verifier calculates a consistency score $s_i = f_{ver}(a_i, C, V)$ for each, and the one with the highest score is selected as the final output.
    - **Design Motivation**: Using the model itself as a verifier creates a self-referential loop that amplifies bias; an independent verifier breaks this cycle.

3.  **CVABench Benchmark**:
    - **Function**: A training and evaluation framework based on real human behavioral data.
    - **Mechanism**: Aggregates over 1.1 million interaction trajectories across three domains (Yelp reviews 54K + Reddit dialogues 155K + Foursquare mobility 871K), covering 15,571 users. GPV (General Psychometric Verification) is used to map user behavior into the Schwartz 10-dimensional value space.
    - **Design Motivation**: To replace LLM self-evaluation with real data, establishing an objective benchmark for behavioral faithfulness.

### Loss & Training
SFT: Standard auto-regressive loss on real trajectories. DPO: Preference optimization favoring nuanced and consistent behavior over polarized and exaggerated behavior. Verifier: Discriminative model training on real $(C,V,A)$ data.

## Key Experimental Results

### Main Results

| Method | Behavioral Fidelity | Diversity Preservation | Level of Value Polarization |
| :--- | :--- | :--- | :--- |
| Raw LLM | Low | Low | High |
| Role Play Agent | Low | Low | High |
| Prompt-Reasoning Agent | Lower | Lower | **Higher** |
| CVA (VMC) | Medium | Medium | Medium |
| CVA (VMC + VDR) | **Highest** | **Highest** | **Lowest** |

### Key Findings

| Finding | Description |
| :--- | :--- |
| Reasoning Intensity vs. Polarization | Contrary to intuition, enhancing prompt-based reasoning exacerbates polarization. |
| Verifier Peak Phenomenon | Behavioral fidelity does not increase monotonically with candidate count $N$; an optimal peak exists. |
| Interpretability | Verifier attention can transparently show which value dimensions determined the selection. |

### Key Findings
*   Increasing reasoning intensity (more CoT steps) fails to improve faithfulness and instead intensifies value polarization and collapses population diversity.
*   A peak for the optimal number of candidates exists for behavioral fidelity, simulating the phenomenon of limited evaluation ranges in human cognitive constraints.
*   CVA significantly outperforms baselines across all three domains (reviews/dialogue/mobility).

## Highlights & Insights
*   **The discovery of "more reasoning leads to more polarization"** is crucial—it directly challenges the "more thinking = better performance" intuition and reveals a core flaw in LLMs for human simulation tasks.
*   **The verifier peak effect** elegantly maps to the concept of "bounded rationality" in cognitive science.
*   **Correction of the evaluation paradigm**: Moving from "LLM-as-a-judge" to "real-data-as-benchmark" sets a new standard for agent evaluation.

## Limitations & Future Work
*   The three data sources in CVABench (Yelp/Reddit/Foursquare) may not represent all patterns of human behavior.
*   While classic, the Schwartz 10-dimensional value model may lack sufficient granularity—certain behaviors might be influenced by factors not yet modeled.
*   Verifier training relies on large amounts of real data, and its effectiveness in data-scarce scenarios remains unknown.

## Related Work & Insights
*   **vs. Park et al. (Generative Agents)**: Rely on persona-prompt simulation which causes behavioral rigidity; CVA replaces this with a verifier trained on real data.
*   **vs. VLA Systems**: VLA focuses on embodied task execution, whereas CVA focuses on socio-psychological behavioral faithfulness.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Deeply integrates psychological value theory with LLM agents; the decoupling verification approach is novel.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Uses 1.1 million real-world data points with in-depth multi-paradigm comparisons.
*   Writing Quality: ⭐⭐⭐⭐ Solid theoretical foundation with profound findings.
*   Value: ⭐⭐⭐⭐⭐ Makes fundamental contributions to LLM human simulation and agent evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)
- [\[AAAI 2026\] AutoTool: Efficient Tool Selection for Large Language Model Agents](../../AAAI2026/llm_agent/autotool_efficient_tool_selection_for_large_language_model_agents.md)
- [\[AAAI 2026\] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](../../AAAI2026/llm_agent/agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[AAAI 2026\] Time, Identity and Consciousness in Language Model Agents](../../AAAI2026/llm_agent/time_identity_and_consciousness_in_language_model_agents.md)

</div>

<!-- RELATED:END -->

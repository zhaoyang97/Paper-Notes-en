---
title: >-
  [Paper Note] MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis
description: >-
  [ACL 2025 (main)][LLM Agent][Differential Diagnosis] This paper proposes the MEDDxAgent framework, which coordinates three modules—a History Taking Simulator, a Knowledge Retrieval Agent, and a Diagnosis Strategy Agent—via a central orchestrator, DDxDriver, to perform iterative differential diagnosis (DDx). It achieves over 10% accuracy improvement in interactive diagnostic scenarios while providing comprehensive reasoning explainability.
tags:
  - "ACL 2025 (main)"
  - "LLM Agent"
  - "Differential Diagnosis"
  - "Multi-Modular Agents"
  - "Iterative Learning"
  - "Explainable Reasoning"
  - "Medical LLMs"
date: 2026-05-08
content_hash: 64f65396fc18ff30
---

# MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis

**Conference**: ACL 2025 (main)  
**arXiv**: [2502.19175](https://arxiv.org/abs/2502.19175)  
**Code**: [https://github.com/nec-research/meddxagent](https://github.com/nec-research/meddxagent)  
**Area**: Agent / Medical Diagnosis  
**Keywords**: Differential Diagnosis, Multi-Modular Agents, Iterative Learning, Explainable Reasoning, Medical LLMs

## TL;DR

This paper proposes the MEDDxAgent framework, which coordinates three modules—a History Taking Simulator, a Knowledge Retrieval Agent, and a Diagnosis Strategy Agent—via a central orchestrator, DDxDriver, to perform iterative differential diagnosis (DDx). It achieves over 10% accuracy improvement in interactive diagnostic scenarios while providing comprehensive reasoning explainability.

## Background & Motivation

**Background**: Differential diagnosis (DDx) is a core component of clinical decision-making, where physicians iteratively narrow down the potential range of diseases based on symptoms, history, and medical knowledge. Although LLMs have shown potential in auxiliary diagnosis in recent years, existing solutions still fall short of real-world clinical scenarios.

**Limitations of Prior Work**: Existing approaches have five key limitations: (1) evaluation is confined to a single dataset, lacking generalization; (2) diagnostic components (e.g., only the diagnosis strategy) are optimized in isolation, lacking multi-stage integration; (3) they assume prior access to a complete patient profile, which is unrealistic in practices where physicians typically start with limited information such as age, gender, and chief complaints; (4) they lack iterative learning mechanisms to update diagnostic conclusions through multi-turn interactions; (5) they rely excessively on medical QA benchmarks, which fail to reflect the complexity of real DDx tasks.

**Key Challenge**: Real differential diagnosis is a progressive process of information collection and reasoning, whereas existing methods simplify this into a one-off "given complete information $\rightarrow$ output diagnosis" paradigm, thereby losing the most critical iterative exploration capability of the diagnostic process.

**Goal**: To design a modular and explainable multi-agent DDx framework that supports iterative diagnosis in interactive scenarios where patient information is progressively acquired.

**Key Insight**: Drawing inspiration from the three-stage structure of real clinical workflows (history-taking, knowledge retrieval, and diagnosis strategy), this work models them as three composable agent modules, coordinated symmetrically by a ReAct-based orchestrator.

**Core Idea**: To model differential diagnosis as an orchestrator-driven multi-agent iterative interaction process, where each module is independently replaceable, and they collectively collaborate to enhance diagnostic accuracy and explainability.

## Method

### Overall Architecture

MEDDxAgent adopts a centralized architecture: the DDxDriver orchestrator serves as the core hub, managing three functional modules: the History Taking Simulator, the Knowledge Retrieval Agent, and the Diagnosis Strategy Agent. All modules communicate solely with the DDxDriver, which is responsible for information maintenance, scheduling, and iteration control. The entire system operates under the ReAct (thought-action-observation) paradigm.

### Key Designs

1. **History Taking Simulator**:

    - **Function**: To simulate the doctor-patient interaction process, progressively collecting diagnostic data when patient information is incomplete.
    - **Mechanism**: Two LLMs are used to play the roles of the patient and the physician, respectively. The patient LLM holds the complete profile, and the physician LLM asks questions based on the initial profile and the conversation goals provided by the DDxDriver. The interaction continues until the goal is achieved or the maximum turn limit is reached, after which the conversation history is returned to the DDxDriver.
    - **Design Motivation**: In real-world scenarios, physicians cannot possess all information at the outset and must gather it through follow-up questions; this module enables the system to handle realistic diagnostic scenarios with incomplete information.

2. **Knowledge Retrieval Agent and Diagnosis Strategy Agent**:

    - **Function**: The Knowledge Retrieval Agent retrieves medical knowledge from external databases (e.g., Wikipedia, PubMed) to assist diagnosis, while the Diagnosis Strategy Agent generates and ranks candidate diagnoses based on current information.
    - **Mechanism**: The Knowledge Retrieval Agent extracts key medical concepts from search queries constructed by the DDxDriver, retrieves them from external databases, and generates evidence summaries. The Diagnosis Strategy Agent supports three modes: zero-shot, fixed few-shot, and dynamic few-shot based on embedding similarity, and can be integrated with CoT reasoning using BioClinicalBERT or BGE embeddings for patient similarity matching.
    - **Design Motivation**: External knowledge is critical for rare disease diagnosis (internal knowledge of LLMs may be outdated or insufficient). The dynamic few-shot strategy guides diagnosis by retrieving the most similar historical cases, offering greater flexibility than a fixed-shot paradigm.

3. **DDxDriver Orchestrator and Iterative Learning Mechanism**:

    - **Function**: To act as a unified coordinating center, managing patient profiles, scheduling module execution, recording reasoning trajectories, and controlling stopping conditions.
    - **Mechanism**: It supports two iterative modes: fixed iteration (sequentially looping through the three modules for a preset number of rounds) and dynamic iteration (where the DDxDriver autonomously decides which module to invoke next based on current observations). The patient profile and candidate diagnosis rankings are updated after each round of iteration.
    - **Design Motivation**: Fixed iteration ensures structured information gathering, while dynamic iteration allows the system to flexibly respond to special circumstances (such as performing additional searches upon suspecting a rare disease). The two modes complement each other.

### Loss & Training

MEDDxAgent is a training-free inference framework that does not involve model parameter updates. All modules work in coordination through prompt engineering and the ReAct paradigm; the primary computational overhead lies in the multi-turn LLM calls during inference.

## Key Experimental Results

### Main Results (Non-interactive Setting, Complete Patient Profiles)

| Dataset | Method | GTPA@1 | GTPA@5 | Avg Rank |
|--------|------|--------|--------|----------|
| DDxPlus | Few-shot (CoT, Dyn_BAII) | 0.97 | 1.00 | 1.03 |
| iCraft-MD | Few-shot (CoT, Dyn_BERT) | 0.64 | 0.73 | 3.68 |
| RareBench | Few-shot (CoT, Dyn_BAII) | 0.82 | 0.88 | 2.11 |

### Interactive Diagnosis Comparison (GPT-4o, No Complete Profiles)

| Method | DDxPlus GTPA@1 | iCraft-MD GTPA@1 | RareBench GTPA@1 |
|------|----------------|-------------------|------------------|
| KR (n=0, No Interaction) | 0.18 | 0.15 | 0.07 |
| DS (n=5) | 0.72 | 0.40 | 0.50 |
| MEDDx (iter=1, n=5) | 0.74 | 0.52 | 0.51 |
| MEDDx (iter=3, n=15) | **0.86** | **0.54** | 0.50 |
| Llama3.1-70B MEDDx (iter=2) | 0.71 | 0.37 | **0.48** |
| Llama3.1-8B MEDDx (iter=2) | 0.56 | 0.14 | 0.09 |

### Key Findings

- Without interactive information ($n=0$), diagnostic accuracy drops drastically (RareBench GTPA@1 decreases from 0.45 to 0.07), verifying the impracticality of prior methods assuming complete profiles.
- The benefits of history-taking question-and-answer turns tend to saturate after 10-15 rounds, indicating an optimal balance between info collection and diagnostic efficiency.
- MEDDxAgent even outperforms the zero-shot baseline using complete profiles on DDxPlus (0.86 vs 0.69), demonstrating that iterative gathering and multi-modular collaboration can compensate for incomplete information.
- Fixed iteration consistently outperforms dynamic iteration because smaller models tend to rely excessively on the history-taking module while ignoring knowledge retrieval, leading to redundant inquiry.
- Model scaling effects are evident: GPT-4o > Llama3.1-70B >> Llama3.1-8B, with smaller models struggling to utilize the iteration effectively on iCraft-MD and RareBench.

## Highlights & Insights

- **Scalability of Modular Design**: Each module can be independently replaced and upgraded. This "Lego-style" architecture allows the framework to easily adapt to new datasets and LLMs, serving as an excellent paradigm for engineering medical AI systems.
- **Practical Value of Explainability**: DDxDriver records all intermediate reasoning steps and module invocation decisions, which is particularly critical in medical scenarios—physicians must understand the AI's diagnostic logic to trust its recommendations.
- **Promoting the Interactive Evaluation Paradigm**: The paper reveals the false optimism of the "complete profile assumption" and systematically evaluates interactive DDx for the first time, providing a valuable benchmark contribution to the community.

## Limitations & Future Work

- The superiority of fixed iteration over dynamic iteration exposes the limitations of current LLMs in long-term planning and intelligent scheduling. Future work could introduce reinforcement learning to optimize scheduling strategies.
- Evaluation is only conducted on English datasets and specific disease categories; generalizability to multilingual settings and a broader disease spectrum has yet to be verified.
- The inference cost is relatively high, requiring multiple LLM calls per patient. Actual deployment will necessitate latency and cost optimization.
- Both patients and physicians are played by LLMs in the simulator, which may not fully mimic the complexity of real-world doctor-patient interactions.

## Related Work & Insights

- **vs AMIE (Tu et al., 2024)**: AMIE also performs interactive diagnosis but focuses on demonstrating the capabilities of a monolithic LLM, whereas MEDDxAgent decomposes the problem into multi-module collaboration, offering higher modularity.
- **vs iCRAFT-MD (Li et al., 2024)**: iCRAFT-MD provides a dermatology dataset and interactive evaluation framework. MEDDxAgent builds upon this by further integrating multi-modular coordination and iterative learning.
- This paper’s approach to the modular design of medical AI Agents is highly referable—breaking complex tasks down into independently optimizable subsystems.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of interactive DDx and multi-modular agents is innovative, though individual module designs are relatively conventional.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The systematic evaluation across three datasets, three model scales, and multiple settings is highly solid.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, but the length is somewhat long and some contents are repetitive.
- Value: ⭐⭐⭐⭐ It provides a substantial advancement to the medical AI Agent field and highlights the importance of interactive evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] MAM: Modular Multi-Agent Framework for Multi-Modal Medical Diagnosis via Role-Specialized Collaboration](mam_modular_multi-agent_framework_for_multi-modal_medical_diagnosis_via_role-spe.md)
- [\[ACL 2025\] METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling](metal_a_multi-agent_framework_for_chart_generation_with_test-time_scaling.md)
- [\[ACL 2026\] LiTS: A Modular Framework for LLM Tree Search](../../ACL2026/llm_agent/lits_a_modular_framework_for_llm_tree_search.md)
- [\[ACL 2025\] Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)
- [\[ICML 2025\] xChemAgents: Agentic AI for Explainable Quantum Chemistry](../../ICML2025/llm_agent/xchemagents_agentic_ai_for_explainable_quantum_chemistry.md)

</div>

<!-- RELATED:END -->

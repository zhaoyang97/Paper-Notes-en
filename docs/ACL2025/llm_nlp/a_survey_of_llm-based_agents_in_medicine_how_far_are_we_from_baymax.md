---
title: >-
  [Paper Note] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?
description: >-
  [ACL 2025][LLM (Other)][LLM medical Agent] This paper systematically reviews the four-layer architecture (Profile, Clinical Planning, Medical Reasoning, and External Capacity Enhancement), four major application scenarios, and evaluation frameworks of LLM-based agents in medicine. Covering 60 studies from 2022 to 2024, it proposes four agent operational paradigms and identifies key challenges such as hallucination management, multimodal integration, and ethical concerns.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "LLM medical Agent"
  - "clinical decision-making"
  - "multi-agent collaboration"
  - "medical reasoning"
  - "architectural survey"
date: 2026-05-08
content_hash: 5ed721206aba594c
---

# A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?

**Conference**: ACL 2025  
**arXiv**: [2502.11211](https://arxiv.org/abs/2502.11211)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: LLM medical Agent, clinical decision-making, multi-agent collaboration, medical reasoning, architectural survey

## TL;DR

This paper systematically reviews the four-layer architecture (Profile, Clinical Planning, Medical Reasoning, and External Capacity Enhancement), four major application scenarios, and evaluation frameworks of LLM-based agents in medicine. Covering 60 studies from 2022 to 2024, it proposes four agent operational paradigms and identifies key challenges such as hallucination management, multimodal integration, and ethical concerns.

## Background & Motivation

**Background**: LLMs possess powerful capabilities in text comprehension, generation, and reasoning. LLM-based agent systems have achieved significant success in various fields such as creative writing and complex decision-making. In the medical field, LLM-based agents have been applied to tasks like diagnostic support (Kim 2024a), patient communication (Mukherjee 2024), and medical education (Yu 2024).

**Limitations of Prior Work**: Standard LLMs primarily process text, whereas medical scenarios require deep integration with external knowledge bases, clinical guidelines, and healthcare systems. Through external knowledge retrieval, task planning, and tool invocation, LLM-based agents transition from "answering questions" to "solving problems," but they face systemic challenges such as hallucination management, safety, and deployment implementation.

**Key Challenge**: The medical field imposes four unique and strict requirements on agents: multimodal integration (processing heterogeneous data such as text, images, and laboratory results), clinical collaboration (facilitating multidisciplinary information sharing and human-AI collaboration), accuracy and reliability (directly impacting patient safety), and transparency and traceability (clinical decisions must be auditable and interpretable). A significant gap exists between current technology and these requirements.

**Goal** -> There is a lack of a comprehensive and systematic survey on agent architectural design, application scenarios, and evaluation frameworks in this field.

**Key Insight**: Using the metaphor of "how far are we from Baymax" as an entry point, this study backtracks from an idealized, omnipotent medical AI assistant to identify current technological gaps and challenges.

**Core Idea**: To comprehensively examine the capability boundaries and directions for improvement of medical LLM agents across three dimensions—architecture, application, and evaluation—anchoring long-term technological development with the Baymax vision.

## Method

### Overall Architecture

This paper proposes a four-layer architectural model for LLM-based medical agents, comprising Profile, Clinical Planning, Medical Reasoning, and External Capacity Enhancement, and defines four agent operational paradigms: Single Agent, Sequential Task Chain, Collaborative Experts, and Iterative Evolution.

### Key Designs

1. **Clinical Planning: From Task Decomposition to Adaptive Collaboration**:

    - **Function**: Decomposes complex medical tasks into executable subtasks, supporting interaction between agents, clinical tools, and databases.
    - **Mechanism**: Proposes four progressive planning architectures: **task decomposition** (decomposing high-level objectives into steps like data ingestion, hypothesis generation, treatment planning, and risk assessment); **multi-agent cross-department collaboration** (allocating specialized agents for radiology, pathology, laboratory tests, etc., aggregating findings and refining diagnoses via standardized protocols, e.g., EHRagent Tang 2024); **adaptive planning** (the MDAgents framework Kim 2024a dynamically adjusts decisions based on real-time data and task complexity, with federated learning enhancing adaptability); and **iterative self-evolution** (Agent Hospital Li 2024b maintains an experience base of past cases, autonomously integrating new data and learning from historical outcomes to continuously improve accuracy and reliability).
    - **Design Motivation**: The complexity of medical tasks demands hierarchical planning capabilities, transitioning from single-agent processing for simple tasks to multi-agent collaboration for complex cases.

2. **Medical Reasoning: From Chain-of-Thought to Collective Consensus**:

    - **Function**: Structures the logical reasoning process to enhance diagnostic accuracy and transparency.
    - **Mechanism**: Four reasoning paradigms: **multi-step diagnostic reasoning** (Chain-of-Thought generates step-by-step reasoning, and Tree-of-Thought explores multiple hypotheses in parallel, discarding low-probability options); **reflective decision-making** (inspired by the ReAct framework, alternating between reasoning and action to detect inconsistencies); **collaborative collective reasoning** (multi-agent frameworks allocate primary care physicians and specialists to analyze independently before aggregating via consensus to reduce bias); and **memory-augmented reasoning** (long-term memory modules accumulate knowledge and clinical experience to enable continuous learning and personalized insights).
    - **Design Motivation**: The uncertainty and complexity of clinical reasoning require multi-layered safeguards—a single reasoning chain is insufficient, necessitating the joint enhancement of reflection, collaboration, and memory.

### Loss & Training

This is a survey paper. The summarized main training/alignment strategies include: reinforcement learning for continuous improvement in iterative self-evolution frameworks (Agent Hospital), federated learning for cross-institutional adaptive planning (MDAgents), RLHF alignment for patient interaction safety (Polaris), and retrieval-augmented generation based on knowledge graphs and clinical guidelines.

## Key Experimental Results

### Main Results

| Application Scenario | Representative System | Framework Type | Key Features |
|---------|---------|---------|---------|
| Clinical Decision-Making | MDAgents (Kim 2024c) | Collaborative Experts | Multi-agent structured discussion improves diagnosis |
| Clinical Decision-Making | Dutta & Hsiao 2024 | Adaptive Planning | Simulates doctor-patient interaction to refine reasoning, outperforming baselines on MedQA |
| Data Analysis | ColaCare (Wang 2024b) | Collaborative Experts | Mortality prediction and readmission analysis on MIMIC-III/IV |
| Documentation | Sporo AI Scribe (Lee 2024) | Single Agent | Solves clinical documentation variability and complexity |
| Training Simulation | Agent Hospital (Li 2024b) | Iterative Evolution | Large-scale repeated training generates complete interactions |
| Training Simulation | SurgBox (Wu 2024) | Collaborative Experts | Surgical workflow training environment, validated by real surgical records |
| Service Optimization | Polaris (Mukherjee 2024) | Sequential Task Chain | General communication + task-specific agents ensure safe interaction |

### Ablation Study

| Benchmark Category | Representative Benchmark | Characteristics | Limitations |
|---------|---------|------|------|
| Static Q&A | MedQA, MedMCQA, MMLU | Preset answers, tests knowledge | Does not reflect interactive clinical decision-making |
| Workflow Simulation | MedChain (12163 cases), ClinicalLab (150 diseases in 24 departments) | Multi-stage clinical reasoning | High difficulty in standardization |
| Automated Evaluation | AI-SCE, RJUA-SPs | Reduces dependence on humans | May be inaccurate in complex scenarios |

### Key Findings

- **Four Agent Paradigms Fit Different Complexities**: Single Agent is used for simple tasks, Sequential Task Chain for medium complexity, Collaborative Experts for high complexity, and Iterative Evolution for scenarios requiring continuous learning.
- **Hallucination is the Largest Technical Risk**: Errors can propagate and amplify during collaboration in multi-agent environments. MedHallBench and HaluEval reveal the insufficiency of existing validation mechanisms.
- **Static Benchmarks are Insufficient**: Fixed Q&A benchmarks like MedQA fail to capture the dynamic and interactive nature of clinical decision-making, necessitating workflow-level evaluation.
- **Bias Issues Are Severe**: BiasMedQA evaluates seven types of bias and finds that the accuracy of SOTA medical LLMs drops to as low as 50% in certain biased scenarios.
- **Inspiration from DeepSeek-R1**: Reinforcement learning combined with long-chain reasoning could be an important direction for improving autonomous reasoning in medical agents.

## Highlights & Insights

- The **four-layer architectural model** clearly deconstructs the capability components of medical agents: Profile defines role boundaries, Clinical Planning manages task decomposition, Medical Reasoning ensures reasoning quality, and External Capacity Enhancement expands knowledge sources.
- The progressive relationship among the **four agent paradigms** holds practical value, providing a guide for architectural selection across medical scenarios of varying complexities.
- The **"Baymax" metaphor** effectively anchors long-term goals, highlighting a massive gap between current systems and the idealized omnipotent medical AI assistant, with key bottlenecks in hallucination management, multimodal integration, and trustworthy reasoning.
- The **iterative evolution concept of Agent Hospital** is highly inspiring: enabling autonomous evolution through large-scale repeated training with simulated patients, bypassing the bottleneck of acquiring real-world clinical data.

## Limitations & Future Work

- The coverage is limited to papers from 2022 to 2024 and predominantly in English, potentially omitting important works in other languages.
- The survey is primarily narrative and lacks quantitative comparisons of different agent systems on a unified benchmark.
- The discussion on privacy protection is somewhat superficial, only mentioning differential privacy and anonymization, without deeply exploring the privacy risks of agent collaboration under federated learning scenarios.
- Discussions regarding integration with physical systems (e.g., surgical robots, caregiving robots) remain at a visionary level, lacking concrete technical path analyses.
- The cost-benefit aspect of medical agents is not discussed in detail—development and deployment costs might prevent small and medium-sized healthcare institutions from benefiting.

## Related Work & Insights

- General agent surveys, such as Xi 2023, cover the overall methodology of LLM agents, whereas this paper provides a specialized analysis focusing on unique medical requirements (safety, traceability, multimodal integration).
- Medical AI surveys, such as Topol 2019, cover traditional AI medical applications, while this paper focuses on the transformative paradigm shifts of agents in the LLM era.
- **Insights**: (1) The core challenge of medical agents lies not in single-task performance but in system-level safety and reliability; (2) Iterative self-evolution frameworks (e.g., Agent Hospital) might be a key pathway to break through clinical data bottlenecks; (3) The department-based organizational model of multi-agent collaboration is worth adopting in other complex domains requiring interdisciplinary collaboration.

## Rating

⭐⭐⭐ This review provides a valuable organizational framework and development roadmap for the medical agent field through its clear four-layer architecture and four paradigms. However, as a narrative review, it lacks quantitative comparisons and practical guidance, and discussions on solutions for core challenges (hallucination, bias, privacy) remain at a conceptual level.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Large Language Models for Predictive Analysis: How Far Are They?](large_language_models_for_predictive_analysis_how_far_are_they.md)
- [\[ACL 2025\] From Selection to Generation: A Survey of LLM-based Active Learning](from_selection_to_generation_a_survey.md)
- [\[ACL 2025\] PlanGenLLMs: A Modern Survey of LLM Planning Capabilities](plangenllms_planning_survey.md)
- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] How to Enable Effective Cooperation Between Humans and NLP Models: A Survey of Principles, Formalizations, and Beyond](human_nlp_cooperation_survey.md)

</div>

<!-- RELATED:END -->

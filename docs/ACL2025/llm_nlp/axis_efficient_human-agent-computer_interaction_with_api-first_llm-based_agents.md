---
title: >-
  [Paper Note] AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents
description: >-
  [ACL 2025][LLM (Other)][API-first] This paper proposes the AXIS framework, which enables LLM Agents to complete application tasks by prioritizing API calls over simulating human UI actions. In Microsoft Word experiments, AXIS reduces task completion time by 65-70% and cognitive load by 38-53%, while maintaining an accuracy rate of 97-98%.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "API-first"
  - "UI Agent"
  - "skill exploration"
  - "cognitive load"
  - "Agent OS"
  - "Human-Agent-Computer Interaction"
date: 2026-05-08
content_hash: 7b2e5f0c713ebda2
---

# AXIS: Efficient Human-Agent-Computer Interaction with API-First LLM-Based Agents

**Conference**: ACL 2025  
**arXiv**: [2409.17140](https://arxiv.org/abs/2409.17140)  
**Code**: Not publicly available  
**Area**: LLM/NLP  
**Keywords**: API-first, UI Agent, skill exploration, cognitive load, Agent OS, Human-Agent-Computer Interaction

## TL;DR

This paper proposes the AXIS framework, which enables LLM Agents to complete application tasks by prioritizing API calls over simulating human UI actions. In Microsoft Word experiments, AXIS reduces task completion time by 65-70% and cognitive load by 38-53%, while maintaining an accuracy rate of 97-98%.

## Background & Motivation

**Background**: UI Agents based on multimodal LLMs (such as UFO) that can directly operate application interfaces to complete user tasks have become a research hotspot. However, the interfaces of existing applications are designed for **humans** and are not suitable for efficient operations by Agents.

**Limitations of Prior Work**:
   - **High Latency**: Each UI interaction step requires one LLM inference, and multi-step operations accumulate severe latency.
   - **Low Reliability**: In long-chain UI interactions, LLMs are prone to hallucinations, leading to error propagation.
   - **Difficulty in Generalization**: LLMs struggle to correctly interact with UI controls that were unseen during the pre-training phase.

**Key Challenge**: Existing UIs are products of the HCI (Human-Computer Interaction) paradigm, which is highly inefficient for the HACI (Human-Agent-Computer Interaction) paradigm. Analogous to factory transformation from the steam age to the electric age—simply replacing the power source is insufficient; the entire process must be redesigned.

**Goal**: To enable LLM Agents to complete application operation tasks efficiently and reliably.

**Key Insight**: API calls are more efficient than UI operations—a single API call can replace multiple UI interaction steps (e.g., changing "insert a 2×2 table" from a three-step UI flow of "Insert→Table→2×2" into a single API call).

**Core Idea**: Agents should **prioritize** calling APIs and only fallback to UI operations when APIs are unavailable. The framework should automatically explore applications and construct new APIs.

## Method

### Overall Architecture

The AXIS system is divided into three stages: (1) Trajectory Collection—the Agent explores the application and records interaction trajectories; (2) Skill Generation—extracting skills from trajectories and translating them into API code; (3) Skill Verification—ensuring skill reliability through static and dynamic testing.

### Key Designs

1. **Skill Definition**: Each skill includes a description, code, and usage examples. Based on code components, skills are categorized into 5 types: atomic UI skills, atomic API skills, composite UI skills, composite API skills, and hybrid API-UI skills. Skills support nested invocation, forming a hierarchical structure.

2. **Trajectory Collection (Stage I)**: 

    - **Follower Mode**: The Agent executes tasks step-by-step according to the application help documentation, strictly following guidance.
    - **Explorer Mode**: The Agent autonomously explores application functions using the brainstorming capabilities of the LLM. To increase state diversity, randomized initial states, a grid exploration strategy (deep vertical sub-menu navigation and horizontal ribbon switching), and three skill levels (corresponding to Microsoft Office Specialist certification courses) are adopted.

3. **Skill Generation (Stage II)**: Collaboration of three Agents:

    - **Monitor**: Reviews the skill library, extracts meaningful segments from the trajectories, and consolidates them into natural language skill insights.
    - **Generator**: Translates skill insights into executable code (which originally still contains many UI operations).
    - **Translator**: Connects to the RAG module to translate UI operation code into API calls ("UI-to-API translation") by referencing application documentation and the existing skill library.

4. **Skill Verification (Stage III)**:

    - **Static Verification**: Checks code structural compatibility (arguments, method calls, and dependent skills).
    - **Dynamic Verification**: The Validator generates various test inputs, and the Evaluator checks the final states to ensure the capability to generalize in real environments.

### API-First Strategy

During task execution, the Agent prioritizes searching for available API skills within the skill library. If a skill can be implemented via either API or UI, **only the API version is retained**. The system falls back to UI interaction only when the corresponding API is unavailable.

## Key Experimental Results

### Feasibility Study (Table 1-2)

Through exploration on Microsoft Word, 73 skills were obtained (44 at Level-1, 24 at Level-2, 5 at Level-3/4). They were subsequently evaluated on 50 Word tasks:

| Metric | UI Agent (UFO) | AXIS |
|------|----------------|------|
| Average Time (s) | 59.5 | **29.9** |
| Success Rate (%) | 52.0 | **84.0** |
| Average Steps | 3.2 | **2.0** |
| Average Cost ($) | 0.4 | **0.2** |

- API invocation rate: AXIS 55.7% vs UI Agent 8.1%
- Advanced API usage rate (Level $\ge 2$): AXIS 23.1%

### User Study (Tables 3-5)

20 participants compared manual operation, UI Agent, and AXIS on L1 (low difficulty) and L2 (high difficulty) tasks:

**Efficiency**:

| Metric | Manual | UI Agent | AXIS |
|------|------|----------|------|
| L1 Time (s) | 61.8 | 104.6 | **18.2** |
| L2 Time (s) | 167.6 | 155.5 | **57.1** |
| L1 Success Rate (%) | 100 | 75 | **98.3** |
| L2 Success Rate (%) | 97.5 | 45 | **95** |

**Cognitive Load (NASA-TLX)**:

| Metric | Manual (L2) | Agent (L2) |
|------|-----------|------------|
| Mental Demand | 70.0 | **7.5** |
| Physical Demand | 57.5 | **6.3** |
| Frustration | 62.5 | **10.0** |

- AXIS outperforms the UI Agent across all subjective preference dimensions (fluency, reliability, and perceived speed).
- AXIS exhibits higher alignment with human decision-making in complex tasks.

## Highlights & Insights

1. **Insight on Paradigm Shift**: The shift from HCI to HACI should not merely involve "adding an Agent to the UI." Instead, it requires redesigning the interaction patterns—where API-first is key.
2. **Automatic Skill Discovery**: The Agent can autonomously explore application functions and build a reusable skill library without requiring manually defined APIs.
3. **UI-to-API Translation**: UI operations are automatically "upgraded" to API calls through RAG retrieval of application documentation, which is both ingenious and practical.
4. **Hierarchical Skill Nesting**: The hierarchical structure design, which constructs complex skills from atomic operations, resembles function abstraction in programming.
5. **Comprehensive User Study**: In addition to technical metrics, the cognitive load is evaluated using standardized scales like NASA-TLX, indicating a highly application-oriented perspective.

## Limitations & Future Work

1. Currently, the framework relies heavily on Python APIs, making it difficult to support applications without native Python interfaces.
2. The stability and efficiency of the exploration process still need to be optimized.
3. Verified only on Microsoft Word; generalization to other applications (e.g., Photoshop, Excel) remains untested.
4. Maintenance and updates of the skill library (such as API changes after application updates) have not been discussed.
5. Insufficient security considerations—direct execution of API operations by the Agent may introduce permission and security risks.

## Related Work & Insights

- **UI Agent**: AppAgent, UFO, CogAgent, etc., which utilize MLLMs to operate application UIs.
- **Agent OS**: Apple Intelligence, Microsoft Copilot, and the concept of Agent OS.
- **UI Design**: MUD leverages LLMs to mine UI data, and SimUser simulates user feedback.

## Rating

⭐⭐⭐⭐ — Deep insights (API-first paradigm), solid experiments (including a comprehensive user study), and high practical value. The limitations lie in the evaluation being restricted to Word and the high complexity of the framework (three-stage multi-agent collaboration). This work provides valuable inspiration for the development of Agent OS.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Game Development as Human-LLM Interaction](game_development_as_human-llm_interaction.md)
- [\[ACL 2025\] AgentDropout: Dynamic Agent Elimination for Token-Efficient and High-Performance LLM-Based Multi-Agent Collaboration](agentdropout_dynamic_agent_elimination_for_token-efficient_and_high-performance_.md)
- [\[ACL 2025\] Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction](enhancing_conversational_agents_with_theory_of_mind_aligning_beliefs_desires_and.md)
- [\[ACL 2025\] X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents](xturing_enhanced_turing_test.md)
- [\[ACL 2025\] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](sample-efficient_human_evaluation_of_large_language_models_via_maximum_discrepan.md)

</div>

<!-- RELATED:END -->

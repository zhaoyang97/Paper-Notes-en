---
title: >-
  [Paper Note] Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems
description: >-
  [ACL 2025][Multi-Agent][multi-agent collaboration] This paper systematically decomposes multi-agent collaboration into four dimensions (governance mode, participation control, interaction pattern, context management). Through extensive experiments on two context-dependent tasks, it demonstrates that the combination of centralized governance + instructor-controlled participation + ordered interaction + instructor summarization is optimal, reducing token consumption by up to 93…
tags:
  - "ACL 2025"
  - "Multi-Agent"
  - "multi-agent collaboration"
  - "governance"
  - "interaction patterns"
  - "context management"
  - "LLM agents"
date: 2026-05-08
content_hash: cd7bf29fbe18cd29
---

# Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems

**Conference**: ACL 2025  
**arXiv**: [2505.12467](https://arxiv.org/abs/2505.12467)  
**Code**: None  
**Area**: Other  
**Keywords**: multi-agent collaboration, governance, interaction patterns, context management, LLM agents

## TL;DR
This paper systematically decomposes multi-agent collaboration into four dimensions (governance mode, participation control, interaction pattern, context management). Through extensive experiments on two context-dependent tasks, it demonstrates that the combination of centralized governance + instructor-controlled participation + ordered interaction + instructor summarization is optimal, reducing token consumption by up to 93% while maintaining or even improving accuracy.

## Background & Motivation

**Background**: Multi-agent LLM systems are increasingly used for complex tasks (medical diagnosis, scientific discovery, software development, etc.), but research primarily focuses on high-level architectural frameworks (such as CAMEL, MetaGPT, etc.) and role assignment.

**Limitations of Prior Work**:
   - Existing frameworks often adopt fixed interaction patterns, lacking analysis of fine-grained collaboration mechanisms such as "who speaks, when to speak, to whom, and using what context."
   - Most systems assume sequential pipeline operations, ignoring iterative discussions and consensus-building processes in real teams.
   - There is a lack of quantitative trade-off analysis of collaboration strategies on performance and computational efficiency.

**Key Challenge**: The actual effectiveness of multi-agent systems highly depends on the choice of interaction strategies, yet the community lacks systematic strategic analysis and guidance on optimal combinations.

**Goal**: To formalize the four dimensions of multi-agent collaboration and quantify the impact of each strategy through controlled experiments.

**Key Insight**: Instead of proposing a new framework, this work deeply analyzes the "mechanism design" inside frameworks, decomposing collaboration strategies into independently controllable dimensions.

**Core Idea**: The performance of multi-agent systems depends more on "how to collaborate" than "what framework to use"—centralized governance + selective participation + ordered interaction + summary management is the optimal combination.

## Method

### Overall Architecture
Define 4 collaboration dimensions × 2–4 strategies per dimension $\rightarrow$ combine to form all valid configurations $\rightarrow$ evaluate on 2 tasks $\rightarrow$ measure the efficiency-accuracy trade-off using TAR (Token-Accuracy Ratio).

### Key Designs

1. **Governance (Governance Mode)**:

    - **G1 Decentralized**: Agents self-organize, autonomously deciding when to participate and how to interact, ultimately reaching decisions through majority voting or consensus.
    - **G2 Centralized**: An instructor agent coordinates the entire process—deciding who speaks, when to speak, managing context, and determining when to terminate.
    - **Design Motivation**: This is the most fundamental dimension, determining the available strategy space for the other three dimensions.

2. **Participation (Participation Control)**:

    - **P1 All Participate** (G1): All agents speak in every round—high diversity but high redundancy.
    - **P2 Selective Participation** (G1): Agents determine by themselves whether to speak—highly efficient but potentially missing crucial information.
    - **P3 Instructor-Controlled Participation** (G2): The instructor decides who speaks in each round—most precise but highly dependent on the instructor's judgment.

3. **Interaction Patterns (Interaction Patterns)**:

    - **I1 Synchronous**: All agents generate responses simultaneously and broadcast them to everyone—parallel but prone to conflicts.
    - **I2 Ordered Turn-Take**: Speak sequentially in a predefined order—subsequent speakers can view prior outputs for progressive improvement.
    - **I3 Random Turn-Take**: Speak in a random order—avoiding ordering bias.
    - **I4 Selective Peer-to-Peer**: Agents autonomously choose who to speak to—high relevance but fragmented context.

4. **Context Management (Context Management)**:

    - **C1 Full Log of Previous Rounds**: Retain the complete conversation history—rich context but causing a token explosion.
    - **C2 Self-Summarization**: Each agent summarizes the previous rounds on their own—distributed compression.
    - **C3 Instructor Summarization**: The instructor provides a unified summary—high consistency but has an information bottleneck.

5. **Token-Accuracy Ratio (TAR)**: A newly proposed evaluation metric that considers both accuracy and token consumption simultaneously, where $\text{TAR} = \text{Accuracy} / \text{Total Tokens}$, used to compare the efficiency-quality trade-off of different configurations.

### Task Design
- **DEI (Distributed Evidence Integration)**: Patient discharge prediction on the MIMIC-III dataset. Five agents each hold different types of clinical information (history of present illness, procedures, lab results, medications, social history) and must collaborate to integrate this info and make a judgment.
- **SES (Structured Evidence Synthesis)**: Fact-checking on the AMBIFC dataset. Only a few out of multiple agents hold relevant evidence; agents with crucial evidence must convince others.

## Key Experimental Results

### DEI Task (Patient Discharge Prediction)

| Configuration | Acc | Input Token | Output Token | Rounds |
|------|-----|-------------|-------------|--------|
| Best Single Agent (BHC) | 60.8 | 541 | 109 | 1 |
| G2-P3-I2-C3 (Centralized+Ordered+Instructor Summary) | **67.5** | **~2K** | ~500 | ~3 |
| G1-P1-I1-C1 (Decentralized+All+Synchronous+Full Log) | 62.3 | ~28K | ~3K | ~5 |
| Token Savings | | **~93%** | | |

### SES Task (Fact-Checking)

| Configuration | Acc | Rounds |
|------|-----|--------|
| Theoretical Upper Bound (A_consistent alone) | 88.7 | 1 |
| G2-P3-I2-C3 | **85.2** | ~3 |
| G1-P1-I1-C1 | 78.5 | ~5 |

### Key Findings
- **Centralized governance uniformly outperforms decentralized**: On both tasks, G2 configurations achieved higher or comparable accuracy with significantly lower token consumption.
- **Ordered interaction (I2) outperforms synchronous (I1)**: Subsequent agents can see prior outputs, avoiding duplication and enabling progressive improvement.
- **Instructor summarization (C3) is the key to efficiency**: Compared to retaining full logs in C1, C3 reduces tokens by up to 93% without losing accuracy.
- **Selective participation is crucial on SES**: When most agents hold irrelevant information, letting the instructor filter speakers avoids noise.
- **The TAR metric reveals that the highest accuracy is not necessarily the optimal choice**: After considering computational costs, configurations with moderate accuracy but low token usage might be more practical.

## Highlights & Insights
- **Decomposing into four dimensions provides a solid framework for multi-agent collaboration research**: It operationalizes the vague concept of "collaboration style" into controllable variables, offering the community a standardized analytical framework. This can be transferred to the design and evaluation of any multi-agent system.
- **The finding that "centralization is actually more efficient" has practical guiding significance**: Contrary to the intuitive belief that "decentralization is more flexible," in LLM multi-agent scenarios, a capable coordinator/instructor is more effective than agent self-organization—since LLM agents have limited self-judgment and coordination capabilities.
- **The TAR metric fills the gap in efficiency evaluation**: Pure accuracy evaluation neglects API costs, making TAR highly valuable for practical deployment scenarios.

## Limitations & Future Work
- **Experiments conducted only on ChatGPT-4o**: Differences in the collaboration capabilities of different models might lead to different optimal strategies.
- **Only two tasks were tested**: DEI and SES represent two extreme paradigms, leaving intermediate hybrid scenarios uncovered.
- **Centralization relies on the quality of the instructor**: If the instructor agent family has poor judgment, overall performance can drop sharply (the paper acknowledges this as a single-point-of-failure risk).
- **The space of strategy combinations is not fully exhausted**: Certain combinations were excluded due to logical conflicts, but more flexible hybrid strategies may exist.

## Related Work & Insights
- **vs CAMEL (Li et al., 2023)**: CAMEL fixes an instructor-agent two-party dialogue paradigm, whereas this work analyzes more participants and interaction patterns.
- **vs MetaGPT (Qian et al., 2025)**: MetaGPT uses a pipeline architecture, while this work focuses on non-pipelined, discussion-style collaboration.
- **vs Debate (Du et al., 2024)**: Debate is a special case of the G1-P1-I1 configuration in this work, and they prove that this is not the optimal choice.

## Rating
- Novelty: ⭐⭐⭐⭐ The four-dimensional decomposition framework is systematic; although not an entirely new concept, it is highly formalized.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablation experiments across various strategy combinations, though limited to GPT-4o and two tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear structure, intuitive diagrams, and complete taxonomy.
- Value: ⭐⭐⭐⭐ Provides direct guiding significance for the design of multi-agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](../../ACL2026/multi_agent/scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[ACL 2025\] Multi-Agent Collaboration via Cross-Team Orchestration](multi-agent_collaboration_via_cross-team_orchestration.md)
- [\[ACL 2025\] Preventing Rogue Agents Improves Multi-Agent Collaboration](preventing_rogue_agents_improves_multi-agent_collaboration.md)
- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](../../NeurIPS2025/multi_agent/multi-agent_collaboration_via_evolving_orchestration.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](../../ACL2026/multi_agent/llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)

</div>

<!-- RELATED:END -->

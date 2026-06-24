---
title: >-
  [Paper Note] GA-S3: Comprehensive Social Network Simulation with Group Agents
description: >-
  [ACL 2025][Social Network Simulation] This paper proposes GA-S3, a social network simulation system based on "Group Agents". It aggregates individuals with similar behaviors into group agents, achieving efficient and accurate simulation of large-scale social networks through hierarchical generation, Markov network reasoning, and behavior modules.
tags:
  - "ACL 2025"
  - "Social Network Simulation"
  - "Group Agent"
  - "LLM Agent"
  - "Network Traffic Prediction"
  - "Markov Network"
date: 2026-05-08
content_hash: fe9ecd88585196a6
---

# GA-S3: Comprehensive Social Network Simulation with Group Agents

**Conference**: ACL 2025  
**arXiv**: [2506.03532](https://arxiv.org/abs/2506.03532)  
**Code**: [AI4SS/GAS-3](https://github.com/AI4SS/GAS-3)  
**Area**: Other  
**Keywords**: Social Network Simulation, Group Agent, LLM Agent, Network Traffic Prediction, Markov Network

## TL;DR

This paper proposes GA-S3, a social network simulation system based on "Group Agents". It aggregates individuals with similar behaviors into group agents, achieving efficient and accurate simulation of large-scale social networks through hierarchical generation, Markov network reasoning, and behavior modules.

## Background & Motivation

Social network simulation aims to represent virtual social networks, modeling user behaviors, relationships, and information flows to analyze and predict the outcomes of social interactions. This has significant value for studying the emergence of group behaviors, policy optimization, and business strategy formulation.

Core challenges faced by existing methods:

**Computational Infeasibility of Individual-Level Simulation**: Social networks contain billions of users, making it impossible to create an LLM agent for each user. Even the recent S3 system can only simulate using 1,000 fixed agents.

**Oversimplified Traditional Methods**: Discrete event simulation and system dynamics approaches tend to predict variables rather than reveal causal mechanisms, ignoring the heterogeneity of social behaviors.

**Poor Scalability**: Existing LLM-based social simulation systems are typically designed for specific events and fail to generalize to different types of network events.

The core innovation of GA-S3 is the introduction of the concept of **Group Agents**: instead of simulating every single individual, it aggregates people with similar behaviors into a group, performing reasoning and action at the group level. This significantly reduces computational costs while maintaining simulation authenticity.

## Method

### Overall Architecture

The GA-S3 system consists of three core modules, corresponding to the three stages of the agent's lifecycle:

1. **Hierarchical Generation** → Existence
2. **Decision-Reasoning** → Decision  
3. **Action** → Behavior

### Key Designs

1. **Perception Embedding**: When a new network event emerges, agents first perceive the event content. An LLM determines its domain (e.g., education, politics, commerce) and country of origin. This information is stored in the agent's memory module to form a foundational perception of the environment and content.

2. **Hierarchical Multi-way Tree Generation**:

    - Employs a top-down approach to organize the population into a multi-way tree structure.
    - Each layer is a fine-grained division of the previous layer, generated using LLM and prompt engineering.
    - Utilizes RAG technology: given an event, the system uses the Kimi model to search online for demographic information related to the country and domain, saving it to a local knowledge graph.
    - If data already exists for the same country and domain, the system directly retrieves it from the knowledge graph using Breadth-First Search (BFS).
    - For example: L1 (Student/Teacher) → L2 (Vocational Student/Educator/...) → L3 (Undergraduate/Master's/Full-time Teacher/...), refinement proceeds layer-by-layer down to 16 group agents.

3. **Agent Attribute Design**:

    - **ID and Country**: Unique identification.
    - **Population Size**: Determines interaction frequency and intensity.
    - **Personality Traits**: Classified into "susceptible", "ordinary", and "calm", affecting the range of emotional fluctuations.
    - **Emotion**: Categorized into happy, sad, and angry, quantified with numerical values.
    - **Attitude**: Positive or negative.

4. **Markov Network Reasoning**:

    - State transition equation: $P(S_i^t | S_i^{t-1}, \mathcal{E}_i^t, M_i^{t-1}) = \alpha_1 P(S_i^{t-1}) + \alpha_2 P(\mathcal{E}_i^t) + \alpha_3 P(M_i^{t-1})$
    - Emotions are updated by the LLM based on perception and previous states: $P(\mathcal{E}_i^t | O_i^t, S_i^{t-1}) = \text{LLM}(O_i^t, S_i^{t-1})$
    - Decisions are generated via the LLM policy function $\pi$: $P(A_i^t | S_i^t) = \pi(S_i^t)$
    - Memory employs a queue update mechanism where new information replaces old, simulating the transient nature of human attention.

5. **Four Fine-grained Realism Factors**:

    - **Population Weight**: Based on real data, affecting group activity.
    - **Personality Traits**: Controlling emotional/attitude fluctuations (susceptible > ordinary > calm).
    - **Emotional Decay**: Emotions and attitudes fade naturally over time.
    - **Forgetting Probability**: Past perceptions and events in short-term memory gradually fade.

6. **Behavior Module**: Supports five online behaviors: browsing, liking, commenting, sharing, and predicting. Browsing is the primary behavior, while other behaviors occur much less frequently. Real user groups browse events and associated interaction information (views, likes, etc.) first, generate emotions and attitudes, and then perform further interactions.

### Loss & Training

GA-S3 does not require LLM fine-tuning. It directly employs the open-source LLaMA3-8B (temperature=0.1 to ensure reproducibility). Hierarchical generation uses the Kimi model (web search) + GPT-4 (data cleaning). The system coordinates the entire workflow through four managers: Event Manager, Memory Manager, State Manager, and Object Manager.

## Key Experimental Results

### Main Results: Comparison with Other Social Simulation Systems (Table)

| Method | t-test ↓ | MAPE ↓ | DTW Mean ↓ | DTW Std ↓ |
|------|----------|--------|------------|-----------|
| PSP (Model-based) | 1.310 | 69.12% | 3.40e+07 | 0.4207 |
| S3 (Agent-based) | 1.820 | 68.66% | 3.09e+07 | 0.4035 |
| **GA-S3 (Ours)** | **0.389** | **16.48%** | **1.30e+07** | **0.1890** |

*GA-S3 substantially outperforms both baselines across all metrics, with MAPE reduced from roughly 69% to 16.48%.*

### Ablation Study (Table)

| # | Level | Memory | State | t-test ↓ | MAPE ↓ | DTW Mean ↓ |
|---|------|------|------|----------|--------|------------|
| 1 | L1 | ✓ | ✓ | 0.829 | 68.78% | 3.38e+07 |
| 2 | L2 | ✓ | ✓ | 0.603 | 33.73% | 2.84e+07 |
| 3 | L3 | ✗ | ✗ | 2.212 | 2884% | 7.80e+08 |
| 4 | L3 | ✓ | ✗ | 2.189 | 1339% | 1.39e+08 |
| 5 | L3 | ✗ | ✓ | 1.986 | 401% | 8.78e+07 |
| **6** | **L3** | **✓** | **✓** | **0.389** | **16.48%** | **1.30e+07** |

*Deeper levels perform better (L3 >> L2 >> L1), and both memory and state modules are critical.*

### Key Findings

1. **The depth of hierarchical generation is critical**: The MAPE of L3 (16 groups) is 16.48%, whereas L1 (2 groups) is 68.78% — finer-grained group divisions capture behavioral disparities more effectively.

2. **Both memory and state are indispensable**: Removing memory increases the t-test from 0.389 to 1.986; removing state increases it from 0.389 to 2.189. Removing both (L3/None/None) results in a massive MAPE of 2,884%.

3. **Group agents exhibit behavioral diversity**: The same group of agents exhibits vastly different patterns across different events, with prediction curves tracking real trends closely.

4. **Emotions/attitudes align partially with traffic trends**: Displaying similar bimodal patterns but lacking full correlation — this matches the weak-coupling characteristics of emotions and behaviors in real-world scenarios.

5. **Excellent reproducibility**: Z-scores remain consistently below 1, indicating highly stable experimental outcomes.

6. **Ablation of personality traits yields intuitive effects**: Removing them results in an abnormal increase in attitude values for the "calm" group, which contradicts realistic scenarios.

## Highlights & Insights

- **The concept of Group Agents is highly creative**: Finding an elegant middle ground between individual agents and statistical models, maintaining the LLM's reasoning capabilities while controlling computational overhead.
- **Adaptive Generation**: Dynamically constructs groups based on the event's domain and country instead of relying on manual configurations, greatly enhancing scalability.
- **Four Fine-grained Factors** (population weight, personality traits, emotional decay, and forgetting probability) make simulations much closer to the real world.
- **Self-constructed Benchmark** SNB fills a significant void in existing social network simulation datasets that lack fine-grained information on traffic changes.

## Limitations & Future Work

1. **Constrained Reasoning Capacity**: Decisions are currently output directly by the LLM, lacking deep reasoning techniques like Chain-of-Thought.
2. **Limited Benchmark Diversity**: Consists of only 30 events; although covering 10 domains and multiple countries, the scale remains relatively small.
3. **Absence of Explicit Network Structure**: Groups form implicit structures through domain and geographical boundaries, but there is no genuine social network topology.
4. **Inflexible Group Generation**: Relies on fixed hierarchical structures; future explorations can delve into dynamic level adaptation.
5. **Privacy and Ethics**: Even though anonymized, social network simulation inherently poses risks of being used for public opinion manipulation.

## Related Work & Insights

- **Generative Agents** (Park et al., 2023): Pioneering work in simulating individual behavior, but unable to scale to large populations.
- **S3** (Gao et al., 2023): Builds virtual social networks using LLMs with 1,000 fixed agents.
- **PSP** (Kong et al., 2018): Identifies epoch patterns of social media popularity, utilizing pattern matching for prediction.
- **Schelling Model** (1971): The first agent-based virtual social simulation.
- GA-S3's core variance: Employing **groups rather than individuals** as the fundamental simulation unit, supporting **adaptive generation** and **fine-grained factors**.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The concept of group agents is a valuable innovation in social simulation, and the combination of hierarchical generation and Markov reasoning is elegantly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensively ablated (covering level, memory, state, and fine-grained factors), evaluated across multiple events and dimensions, with Z-scores validating reproducibility.
- **Writing Quality**: ⭐⭐⭐ The overall framework is described clearly, but there are numerous formulas and some symbols are insufficiently explained in the text.
- **Value**: ⭐⭐⭐⭐ Provides a feasible path for large-scale social network simulation, with open-source code and benchmark datasets benefiting subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents](sotopia-ensuremathomega_dynamic_strategy_injection_learning_and_social_instructi.md)
- [\[ACL 2025\] S3 - Semantic Signal Separation](s3_-_semantic_signal_separation.md)
- [\[ACL 2025\] Consistent Client Simulation for Motivational Interviewing-based Counseling](consistent_client_simulation_for_motivational_interviewing-based_counseling.md)
- [\[ACL 2025\] Tuna: Comprehensive Fine-grained Temporal Understanding Evaluation on Dense Dynamic Videos](tuna_temporal_understanding.md)
- [\[ACL 2025\] Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights](graphically_speaking_unmasking_abuse_in_social_media_with_conversation_insights.md)

</div>

<!-- RELATED:END -->

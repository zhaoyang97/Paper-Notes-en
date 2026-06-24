---
title: >-
  [Paper Note] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs
description: >-
  [ACL 2025][Conversational Agents] A directed acyclic graph (DAG) based workflow framework is proposed. By decomposing the complex business constraints of an LLM agent into different state nodes in the graph and combining this with a response masking fine-tuning strategy, a production-grade e-commerce conversational agent is built. It significantly outperforms the GPT-4o baseline in both task accuracy and format adherence.
tags:
  - "ACL 2025"
  - "Conversational Agents"
  - "Workflow Graphs"
  - "DAG Framework"
  - "E-commerce Agent"
  - "Fine-Tuning Strategy"
date: 2026-05-08
content_hash: 78bebc18ec0099e2
---

# A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs

**Conference**: ACL 2025  
**arXiv**: [2505.23006](https://arxiv.org/abs/2505.23006)  
**Code**: None  
**Area**: Other  
**Keywords**: Conversational Agents, Workflow Graphs, DAG Framework, E-commerce Agent, Fine-Tuning Strategy

## TL;DR

A directed acyclic graph (DAG) based workflow framework is proposed. By decomposing the complex business constraints of an LLM agent into different state nodes in the graph and combining this with a response masking fine-tuning strategy, a production-grade e-commerce conversational agent is built. It significantly outperforms the GPT-4o baseline in both task accuracy and format adherence.

## Background & Motivation

**Background**: LLMs have demonstrated strong capabilities in service scenarios such as search, recommendation, and chat, with tool-calling-based conversational agents becoming increasingly popular.

**Limitations of Prior Work**: Deploying LLMs to real-world industrial scenarios faces three major challenges: (1) The probabilistic nature of generative LLMs makes it impossible to strictly comply with business rules (e.g., prohibiting recommendations of alcohol and tobacco to minors); (2) Mobile scenarios require specific output formats (length limits, emojis, etc.), which LLMs struggle to follow consistently; (3) Piling all requirements into a single system prompt leads to excessively long prompts, reducing latency and accuracy.

**Key Challenge**: The agent must maintain flexible conversation capabilities while strictly adhering to service-specific constraints. However, these two requirements inherently conflict due to the probabilistic nature of LLMs.

**Goal**: To build a controllable, reliable, and scalable conversational agent in real e-commerce scenarios that satisfies both the need for flexible dialogue and strict compliance.

**Key Insight**: Adopting a hybrid architecture that structures agent behavior using a DAG workflow graph, where each node possesses an independent system prompt, tools, and constraint rules, thereby avoiding single-long-prompt issues. Additionally, data collection and fine-tuning methods tailored to the graph structure are designed.

**Core Idea**: Dispersing complex business constraints across various nodes using a DAG workflow graph, combined with response masking fine-tuning, allowing smaller models to outperform GPT-4o within a structured agent architecture.

## Method

### Overall Architecture

The system models the e-commerce conversational agent as a Directed Acyclic Graph (DAG) consisting of LLM-calling nodes (green) and tool-calling nodes (pink). Starting from the initial chat node, the LLM routes to the corresponding task node (such as product recommendation or purchase messaging) based on user intent. Each node independently manages its own constraints and formatting requirements. The overall pipeline includes: (1) designing the workflow graph $\rightarrow$ (2) building a prototype agent with GPT-4o to collect data $\rightarrow$ (3) fine-tuning a proprietary model.

### Key Designs

1. **Multi-State DAG Framework**
    - **Function**: Disperses business constraints across various state nodes in the graph.
    - **Mechanism**: Each LLM-calling node possesses an independent system prompt (containing only constraints for that state) and a custom dialogue history manipulation function (modify_history). Tool nodes have defined input/output schemas.
    - **Design Motivation**: Avoids stacking all constraints into a single system prompt, which dilutes attention and degrades adherence.

2. **Dialogue History Manipulation (modify_history)**
    - **Function**: Crops the input dialogue history according to node requirements.
    - **Mechanism**: For instance, a purchase message node (purchase_message) removes all history, keeping only the final round of purchase information.
    - **Design Motivation**: Reduces hallucinations by restricting access to irrelevant prior context.

3. **Prototype Agent Assisted Data Collection**
    - **Function**: A three-step process to efficiently collect high-quality labeled data.
    - **Mechanism**: First, build a prototype agent using GPT-4o $\rightarrow$ human annotators interact with this prototype, recording all graph traversal paths $\rightarrow$ annotators review and correct erroneous responses, aided by automated checkers (e.g., JSON static type checking).
    - **Design Motivation**: Annotators struggle to generate complex responses involving multi-step reasoning and tool calls independently, necessitating assistance from a prototype agent.

4. **Response Masking Fine-Tuning (Response Masking)**
    - **Function**: Resolves the issue of prompt conflicts in multi-turn training within graph structures.
    - **Mechanism**: During training, loss masking is applied to responses from other nodes, calculating loss only for responses generated by the current node.
    - **Design Motivation**: Different responses within the same dialogue history may originate from different nodes (with different system prompts); standard multi-turn training would introduce conflicting supervision signals.

### Loss & Training

Adopt chatbot-style sequence formatting, where each node independently constructs training samples $(s_v, x_1, o_1, ..., x_n, o_n)$, with $s_v$ representing the system prompt of that node. The key is to mask out responses not generated by the current node in the loss calculation, preventing gradient interference among instructions from different nodes. Tool call outputs use constrained decoding to ensure correct formatting.

## Key Experimental Results

### Main Results

| Model | Architecture | Accuracy | Format Adherence | Response Validity |
|------|------|----------|-----------------|-------------------|
| Qwen 2.5 (32B) | Basic | 0.578 | 0.734 | 2.816 |
| Qwen 2.5 (32B) | WG | 0.616 | 0.813 | 2.831 |
| Qwen 2.5 (32B) | WG-FT | 0.884 | **0.969** | 2.880 |
| Gemma 3 (27B) | WG-FT | 0.887 | 0.966 | **2.911** |
| Internal (27-32B) | WG-FT | **0.890** | 0.987 | 2.953 |
| GPT-4o | WG | 0.888 | 0.964 | 2.882 |

### Ablation Study

| Evaluation Dimension | Internal ≥ GPT-4o Ratio |
|---------|----------------------|
| Casual Chat | 42.42% |
| Safety | 60.53% |
| Product Recommendation | 82.42% |
| Messenger Features | 60.61% |
| Overall | 63.29% |

### Key Findings

- The WG framework improves format adherence by up to 45% (from 0.655 to 0.951 on the internal model).
- Accuracy improves by up to 14%.
- WG-FT enables open-source models (27-32B) to reach or exceed GPT-4o performance across all metrics.
- In human evaluation, the internal model outperforms GPT-4o in all categories except casual chat, with language fluency identified as the key factor affecting user preferences in chat-oriented interactions.

## Highlights & Insights

- Accurately identifies the core challenge of industrial-grade agents as the tension between "flexibility vs. compliance" and elegantly resolves it via a DAG structure.
- Response masking fine-tuning serves as a crucial technique for training graph-structured agents, simply yet effectively resolving multi-node prompt conflicts.
- The approach of using a prototype agent to assist data collection is highly practical, lowering the barrier to collecting labeled data for complex agents.
- Demonstrates that small-to-medium open-source models can outperform GPT-4o in specific domains through proper architectural design and fine-tuning.

## Limitations & Future Work

- Data collection is highly dependent on human annotation, which is costly and subject to demographic biases (due to the limited gender and age distribution of annotators).
- LLM-as-a-Judge evaluations may not fully capture human preferences (such as the impact of language fluency).
- The potential of using LLMs to simulate users for automated data collection was not explored.
- Validated only in e-commerce scenarios; cross-domain generalizability remains to be verified.

## Related Work & Insights

- Differing from the positioning of other graph-structured agent frameworks like LangGraph, this work focuses on achieving production-grade response quality within a graph structure.
- Insight: For any agent scenario requiring strict compliance with multiple constraints, dispersing constraints into independent nodes is a superior engineering practice compared to piling up prompts.
- Frameworks like MARCO rely on independent guardrail components for validation and retry, which increases latency; in contrast, this work reduces violations at the source through structured design.

## Rating

- **Novelty**: ⭐⭐⭐ (The DAG concept is not new, but response masking fine-tuning and the overall industrial solution offer practical innovations)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Multi-model comparison + human evaluation + real-world deployment, though it lacks cross-domain validation)
- **Writing Quality**: ⭐⭐⭐⭐ (Industrial-paper style, with clear presentation and detailed case studies)
- **Value**: ⭐⭐⭐⭐ (Highly valuable reference for building production-grade agents)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Zero-Shot Conversational Stance Detection: Dataset and Approaches](zero-shot_conversational_stance_detection_dataset_and_approaches.md)
- [\[ACL 2025\] CONFETTI: Conversational Function-Calling Evaluation Through Turn-Level Interactions](confetti_conversational_function-calling_evaluation_through_turn-level_interacti.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[NeurIPS 2025\] Adjusted Count Quantification Learning on Graphs](../../NeurIPS2025/others/adjusted_count_quantification_learning_on_graphs.md)
- [\[ACL 2025\] Contextual Experience Replay for Self-Improvement of Language Agents](contextual_experience_replay_for_self-improvement_of_language_agents.md)

</div>

<!-- RELATED:END -->

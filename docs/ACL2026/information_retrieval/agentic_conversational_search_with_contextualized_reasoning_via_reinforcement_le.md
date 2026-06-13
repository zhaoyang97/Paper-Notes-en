---
title: >-
  [Paper Note] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning
description: >-
  [ACL 2026][Information Retrieval & RAG][Conversational Search] ConvAgent is proposed, which trains a conversational search agent to alternate between searching and reasoning across multi-turn interactions by decomposing…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Conversational Search"
  - "Reinforcement Learning"
  - "Contextualized Reasoning"
  - "Mixed-Initiative Behavior"
  - "Information Gain Reward"
date: 2026-05-08
content_hash: 1bb3b0bfaa796717
---

# Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning

**Conference**: ACL 2026 Findings  
**arXiv**: [2601.13115](https://arxiv.org/abs/2601.13115)  
**Code**: None  
**Area**: Conversational Search / LLM Agent  
**Keywords**: Conversational Search, Reinforcement Learning, Contextualized Reasoning, Mixed-Initiative Behavior, Information Gain Reward

## TL;DR

ConvAgent is proposed, which trains a conversational search agent to alternate between searching and reasoning across multi-turn interactions by decomposing RL training rewards into three complementary components: outcome reward, information gain reward, and mixed-initiative action reward.

## Background & Motivation

**Background**: LLMs are becoming the primary interface for human-computer interaction, but in multi-turn conversational searches, user intent evolves throughout the dialogue, requiring dynamic coordination of retrieval and generation.

**Limitations of Prior Work**: (1) Traditional methods utilize static "rewrite → retrieve → generate" pipelines where each module is optimized independently, preventing joint optimization; (2) Emerging deep search agents (e.g., Search-R1), while capable of jointly optimizing retrieval and generation, target only single-turn scenarios and lack multi-turn capabilities; (3) Existing methods ignore mixed-initiative behaviors (e.g., asking clarification questions at the appropriate time).

**Key Challenge**: Multi-turn conversational search requires simultaneous contextual understanding (decontextualization), search optimization (retrieval quality), and behavioral decision-making (when to answer/clarify/refuse). Existing methods cannot optimize these three dimensions concurrently.

**Goal**: Optimize multiple aspects simultaneously through contextualized reasoning within a single-agent framework.

**Key Insight**: Decompose the total reward into three complementary components and use the GRPO algorithm to train the agent to execute search and reasoning alternately across multiple turns.

**Core Idea**: Intermediate process rewards (information gain + mixed-initiative behavior) compensate for the sparse supervision of outcome rewards alone, enabling the model to learn more strategic search and interaction behaviors.

## Method

### Overall Architecture

In each dialogue turn, ConvAgent receives the dialogue history $\mathcal{H}_n$ and the current query $q_n$. It generates search queries, calls the retriever, analyzes results, decides on an action (answer/clarify/refuse), and finally produces a response through reasoning. The entire trajectory is optimized via GRPO.

### Key Designs

1.  **Information Gain Reward**:

    - **Function**: Optimizes search query quality and the utilization of retrieval results.
    - **Mechanism**: Measures the information overlap between retrieval results and ground-truth answers. F1-score is used for long answers, while substring matching accuracy is used for short answers: $\mathcal{R}_{IG} = \mathcal{S}_{Info}(\{P_n\}_1^k, a_n^*)$.
    - **Design Motivation**: Rewards based solely on the final answer are too sparse; intermediate retrieval quality signals help the model learn better query rewriting strategies.

2.  **Mixed-Initiative Action Reward**:

    - **Function**: Trains the model to take the appropriate action (answer/clarify/refuse) at the right time.
    - **Mechanism**: Models action decision-making as a classification task, providing rewards or penalties based on whether the generated sequence contains the correct action tags (e.g., `<clarify>`, `<noanswer>`): correct +1, incorrect -0.5.
    - **Design Motivation**: In dialogue scenarios, not every turn requires an answer—sometimes the user query is ambiguous and requires clarification, or the evidence is insufficient and should result in a refusal to answer.

3.  **Clarification Result Utilization Mechanism**:

    - **Function**: Utilizes model-generated clarification questions to improve downstream tasks.
    - **Mechanism**: The clarification question $q_n^c$ is concatenated to the rewritten query $q_n'$ for retrieval and also replaces the original query for final answer generation.
    - **Design Motivation**: Clarification should not merely be evaluated as a binary action; its downstream utility should also be measured.

### Loss & Training

The total reward is defined as $\mathcal{R}(\tau) = \mathcal{R}_{outcome} + 0.5 \times (\mathcal{R}_{IG} + \mathcal{R}_{MIA})$. Optimization is performed using the GRPO (Group Relative Policy Optimization) algorithm, which requires no explicit reward or value models. PPO was also experimented with as an alternative.

## Key Experimental Results

### Main Results

| Method | TopiOCQA F1 | INSCIT F1 | QReCC F1 | CORAL F1 |
| :--- | :--- | :--- | :--- | :--- |
| SFT-3b | 18.2 | 23.7 | 17.0 | 15.2 |
| Search-R1-3b | 26.1 | 5.8 | 5.9 | 3.9 |
| ConvAgent-3b | 25.2 | 23.5 | 24.1 | 22.4 |
| SFT-7b | 23.6 | 24.5 | 19.1 | 18.8 |
| Search-R1-7b | 37.0 | 9.1 | 8.6 | 3.8 |
| ConvAgent-7b | - | - | - | - |

### Ablation Study

| Configuration | Key Metric | Description |
| :--- | :--- | :--- |
| Remove IG reward | F1 decreases | Search optimization signals are critical for retrieval quality |
| Remove MIA reward | Mixed-initiative behavior degrades | Behavioral adaptation is important for dialogue quality |
| PPO vs GRPO | GRPO more stable | GRPO is simpler as it requires no additional reward model |

### Key Findings
- Search-R1 performs inconsistently in dialogue scenarios—while strong on TopiOCQA, it collapses on the other three datasets, indicating that single-turn agents do not adapt well to multi-turn tasks.
- ConvAgent demonstrates balanced performance across all four datasets, proving the importance of intermediate rewards.
- Information Gain rewards effectively improve query rewriting quality, even without using ground-truth rewritten queries as supervision.

## Highlights & Insights
- The reward decomposition strategy elegantly addresses the sparse reward problem in RL training without requiring human-annotated intermediate step supervision.
- The design of the Information Gain reward is clever, using the overlap between retrieval results and answers as a proxy signal for query quality.
- The introduction of mixed-initiative behavior brings the dialogue agent closer to a real-world user experience—knowing when to ask and when to answer.

## Limitations & Future Work
- The current method has only been validated on 3B and 7B models; the performance on larger models remains to be tested.
- Mixed-initiative behavior currently includes only three types, whereas real-world dialogue behaviors are much richer.
- The quality of user simulation may affect the training effectiveness.
- Future work could extend to multi-modal conversational search and more complex interaction patterns.

## Related Work & Insights
- **vs Search-R1**: Extends single-turn deep search to multi-turn dialogues, addressing multi-turn challenges through historical conditioning and intermediate rewards.
- **vs ChatR1**: ChatR1 relies on ground-truth rewritten queries as training signals, whereas ConvAgent's IG reward does not.
- **vs Traditional Conversational Search**: Unifies separate rewriting, retrieval, and generation modules into a single agent via end-to-end RL optimization.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of reward decomposition and mixed-initiative behavior is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across 4 datasets, comparison with multiple baselines, and detailed ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and a systematic description of the methodology.
- Value: ⭐⭐⭐⭐ Provides practical guidance for the development of conversational AI assistants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChatR1: Reinforcement Learning for Conversational Reasoning and Retrieval Augmented Question Answering](chatr1_reinforcement_learning_for_conversational_reasoning_and_retrieval_augment.md)
- [\[ACL 2026\] Learning to Extract Rational Evidence via Reinforcement Learning for Retrieval-Augmented Generation](learning_to_extract_rational_evidence_via_reinforcement_learning_for_retrieval-a.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ICML 2026\] Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning](../../ICML2026/information_retrieval/graph-r1_towards_agentic_graphrag_framework_via_end-to-end_reinforcement_learnin.md)
- [\[ACL 2026\] End-to-End Optimization of LLM-Driven Multi-Agent Search Systems via Heterogeneous-Group-Based Reinforcement Learning](end-to-end_optimization_of_llm-driven_multi-agent_search_systems_via_heterogeneo.md)

</div>

<!-- RELATED:END -->

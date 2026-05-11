---
title: >-
  [Paper Note] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning
description: >-
  [ACL 2026][Dialogue Systems][conversational search] This paper proposes ConvAgent, which trains a conversational search agent to alternate between retrieval and reasoning across multi-turn interactions by decomposing the…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "conversational search"
  - "reinforcement learning"
  - "contextualized reasoning"
  - "mixed-initiative behavior"
  - "information gain reward"
date: 2026-05-08
content_hash: f8918c731fd6604a
---

# Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning

**Conference**: ACL 2026
**arXiv**: [2601.13115](https://arxiv.org/abs/2601.13115)
**Code**: None
**Area**: Conversational Search / LLM Agent
**Keywords**: conversational search, reinforcement learning, contextualized reasoning, mixed-initiative behavior, information gain reward

## TL;DR

This paper proposes ConvAgent, which trains a conversational search agent to alternate between retrieval and reasoning across multi-turn interactions by decomposing the RL training reward into three complementary components: outcome reward, information gain reward, and mixed-initiative action reward.

## Background & Motivation

**Background**: LLMs are becoming the primary interface for human-computer interaction; however, in multi-turn conversational search, user intent evolves across turns, requiring dynamic coordination between retrieval and generation.

**Limitations of Prior Work**: (1) Traditional methods adopt static "rewrite→retrieve→generate" pipelines with independently optimized modules, precluding joint optimization. (2) Emerging deep search agents (e.g., Search-R1) enable joint optimization of retrieval and generation but target only single-turn scenarios, lacking multi-turn conversational capabilities. (3) Existing methods neglect mixed-initiative behaviors, such as posing clarification questions at appropriate moments.

**Key Challenge**: Multi-turn conversational search simultaneously demands contextual understanding (decontextualization), search optimization (retrieval quality), and action decision-making (when to answer, clarify, or abstain), yet no existing method jointly optimizes all three dimensions.

**Goal**: Simultaneously optimize multiple aspects within a unified agent framework through contextualized reasoning.

**Key Insight**: The total reward is decomposed into three complementary components, and the agent is trained via the GRPO algorithm to alternately perform retrieval and reasoning across turns.

**Core Idea**: Intermediate process rewards (information gain + mixed-initiative behavior) compensate for the sparse supervision of outcome-only rewards, enabling the model to learn more strategic search and interaction behaviors.

## Method

### Overall Architecture

At each conversational turn, ConvAgent receives the dialogue history $\mathcal{H}_n$ and the current query $q_n$, generates search queries through reasoning, invokes a retriever, analyzes the results, decides on an action (answer / clarify / abstain), and produces a response. The entire trajectory is optimized via GRPO.

### Key Designs

1. **Information Gain Reward**:

    - Function: Optimizes search query quality and the utilization of retrieved results.
    - Mechanism: Measures the information overlap between retrieved passages and the ground-truth answer. F1-score is used for long answers; substring-match accuracy is used for short answers: $\mathcal{R}_{IG} = \mathcal{S}_{Info}(\{P_n\}_1^k, a_n^*)$
    - Design Motivation: Final-answer-only rewards are too sparse; intermediate retrieval quality signals help the model learn better query rewriting strategies.

2. **Mixed-Initiative Action Reward**:

    - Function: Trains the model to adopt the appropriate action (answer / clarify / abstain) at the right moment.
    - Mechanism: Action decision-making is framed as a classification task. Rewards or penalties are assigned by detecting whether the generated sequence contains the correct action label (e.g., `<clarify>`, `<noanswer>`): +1 for correct, −0.5 for incorrect.
    - Design Motivation: In conversational settings, not every turn requires a direct answer—sometimes user queries are ambiguous and warrant clarification; other times, insufficient evidence justifies abstention.

3. **Utilization Mechanism for Clarification Outcomes**:

    - Function: Leverages model-generated clarification questions to improve downstream task performance.
    - Mechanism: The clarification question $q_n^c$ is concatenated with the rewritten query $q_n'$ for retrieval and also substitutes the original query for final answer generation.
    - Design Motivation: Clarification should not merely be evaluated on whether a question was asked, but also on whether asking it yields downstream benefit.

### Loss & Training

The total reward is $\mathcal{R}(\tau) = \mathcal{R}_{outcome} + 0.5 \times (\mathcal{R}_{IG} + \mathcal{R}_{MIA})$. Optimization is performed using the GRPO algorithm (Group Relative Policy Optimization), which requires neither an explicit reward model nor a value model. PPO is also explored as an alternative.

## Key Experimental Results

### Main Results

| Method | TopiOCQA F1 | INSCIT F1 | QReCC F1 | CORAL F1 |
|--------|-------------|-----------|----------|----------|
| SFT-3b | 18.2 | 23.7 | 17.0 | 15.2 |
| Search-R1-3b | 26.1 | 5.8 | 5.9 | 3.9 |
| ConvAgent-3b | 25.2 | 23.5 | 24.1 | 22.4 |
| SFT-7b | 23.6 | 24.5 | 19.1 | 18.8 |
| Search-R1-7b | 37.0 | 9.1 | 8.6 | 3.8 |
| ConvAgent-7b | - | - | - | - |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Remove IG reward | F1 drops | Retrieval optimization signal is critical for retrieval quality |
| Remove MIA reward | Mixed-initiative behavior degrades | Action adaptation is important for conversational quality |
| PPO vs. GRPO | GRPO more stable | GRPO is simpler without requiring an additional reward model |

### Key Findings
- Search-R1 exhibits unstable performance in conversational settings—strong on TopiOCQA but collapses on the other three datasets—demonstrating that single-turn agents do not generalize to multi-turn scenarios.
- ConvAgent achieves consistent performance across all four datasets, validating the importance of intermediate rewards.
- The information gain reward effectively improves query rewriting quality, even without ground-truth rewritten queries as supervision.

## Highlights & Insights
- The reward decomposition strategy elegantly addresses the sparse reward problem in RL training without requiring manually annotated intermediate-step supervision.
- The design of the information gain reward is elegant—retrieval-answer overlap serves as a proxy signal for query quality.
- The incorporation of mixed-initiative behavior brings conversational agents closer to real user experiences by enabling them to know when to ask and when to answer.

## Limitations & Future Work
- Validation is currently limited to 3B and 7B models; performance on larger models remains untested.
- Mixed-initiative behavior covers only three action types, whereas real-world dialogues involve a richer behavioral repertoire.
- The quality of user simulation may affect training outcomes.
- Future work could extend the framework to multimodal conversational search and more complex interaction patterns.

## Related Work & Insights
- **vs. Search-R1**: Extends single-turn deep search to multi-turn dialogue, addressing multi-turn challenges through history-conditioned queries and intermediate rewards.
- **vs. ChatR1**: ChatR1 relies on ground-truth rewritten queries as training signals, whereas ConvAgent's IG reward eliminates this requirement.
- **vs. Traditional Conversational Search**: Unifies separate rewriting, retrieval, and generation modules into a single agent optimized end-to-end via RL.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of reward decomposition and mixed-initiative behavior constitutes a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, multiple baselines, and ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear; method description is systematic.
- Value: ⭐⭐⭐⭐ Offers practical guidance for the development of conversational AI assistants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)
- [\[ICLR 2026\] ReIn: Conversational Error Recovery with Reasoning Inception](../../ICLR2026/dialogue/rein_conversational_error_recovery_with_reasoning_inception.md)
- [\[ACL 2026\] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)
- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[ACL 2026\] Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation](towards_proactive_information_probing_customer_service_chatbots_harvesting_value.md)

</div>

<!-- RELATED:END -->

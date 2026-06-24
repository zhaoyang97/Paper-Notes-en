---
title: >-
  [Paper Note] In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents
description: >-
  [ACL 2025][Long-term memory management] This paper proposes the Reflective Memory Management (RMM) mechanism. By combining prospective reflection (multi-granularity memory summarization) and retrospective reflection (reinforcement learning-driven online retrieval optimization), it builds an efficient memory management framework for long-term personalized dialogue systems, achieving an accuracy improvement of over 10% on LongMemEval.
tags:
  - "ACL 2025"
  - "Long-term memory management"
  - "personalized dialogue"
  - "prospective reflection"
  - "retrospective reflection"
  - "reinforcement learning retrieval"
date: 2026-05-08
content_hash: 9b0251a83ef880ff
---

# In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents

**Conference**: ACL 2025  
**arXiv**: [2503.08026](https://arxiv.org/abs/2503.08026)  
**Code**: None  
**Area**: Others  
**Keywords**: Long-term memory management, personalized dialogue, prospective reflection, retrospective reflection, reinforcement learning retrieval

## TL;DR

This paper proposes the Reflective Memory Management (RMM) mechanism. By combining prospective reflection (multi-granularity memory summarization) and retrospective reflection (reinforcement learning-driven online retrieval optimization), it builds an efficient memory management framework for long-term personalized dialogue systems, achieving an accuracy improvement of over 10% on LongMemEval.

## Background & Motivation

**Background**: Large language models have made significant progress in open-ended dialogues. However, due to the context window limits, they struggle to retain memories of user preferences and historical information during long-term interactions. External memory mechanisms have been proposed to augment the long-term memory capacity of LLMs, enabling dialogue agents to maintain session continuity.

**Limitations of Prior Work**: Existing approaches face two key challenges. First, **rigid memory granularity**—most methods store memories in fixed windows or turns, failing to capture the natural semantic structure of dialogues, which leads to fragmented and incomplete memories. A meaningful dialogue segment may span multiple turns, and fixed slicing disrupts semantic integrity. Second, **fixed retrieval mechanisms**—existing methods use static retrieval strategies (such as pure embedding similarity-based retrieval) that cannot adapt to the diverse dialogue contexts and user interaction patterns. Sometimes precise matching is needed, while other times topical relevance is required; a fixed strategy cannot balance both.

**Key Challenge**: Both "writing" and "reading" memories require flexibility—adaptive memory granularity is needed when writing based on dialogue content, and adaptive retrieval strategies are needed when reading based on the current query. Existing methods are too rigid in both directions.

**Goal**: To design a unified framework that simultaneously optimizes memory writing (storage) and reading (retrieval), allowing long-term dialogue agents to efficiently manage and utilize historical interaction information.

**Key Insight**: The authors draw inspiration from two reflective mechanisms of human memory: "prospective" and "retrospective" reflections. Humans actively organize and summarize experiences that are about to pass (prospective), and repeatedly refine search strategies when recall is needed (retrospective).

**Core Idea**: To construct a bidirectional reflection mechanism: prospective reflection dynamically summarizes dialogues into multi-granularity memory entries (solving the writing problem), and retrospective reflection iteratively optimizes retrieval strategies through reinforcement learning (solving the reading problem).

## Method

### Overall Architecture

The RMM system consists of two major modules. During the dialogue process, the Prospective Reflection module summarizes each interaction into memory entries of different granularities and stores them in the memory bank. When a response needs to be generated, the Retrospective Reflection module retrieves relevant information from the memory bank and continuously optimizes retrieval strategies via online RL. Finally, the retrieved memory entries are fed into the LLM along with the current dialogue context to generate a personalized response.

### Key Designs

1. **Prospective Reflection**:

    - **Function**: Dynamically organizes dialogue interactions into multi-granularity structured memory entries.
    - **Mechanism**: Extracts memories at three granularities—(a) **Utterance-level**: directly preserves single important dialogue utterances; (b) **Turn-level**: merges and summarizes multiple messages within a dialogue turn into a topical description; (c) **Session-level**: summarizes the entire session into high-level topic tags and key preference information after the session ends. LLMs are utilized to determine natural semantic boundaries of the dialogue and generate summaries.
    - **Design Motivation**: Different queries require different memory granularities—detailed queries ("What was the name of the restaurant recommended last time?") need precise utterance-level information, while topical queries ("My dietary preferences") require abstract session-level information. Multi-granularity storage ensures retrieval flexibility.

2. **Retrospective Reflection**:

    - **Function**: Iteratively optimizes the retrieval strategy via online RL based on the cited evidence of LLM.
    - **Mechanism**: After the LLM generates a response, it analyzes which retrieved memory entries were cited in the response. Cited entries receive a positive reward, while uncited ones receive a negative reward. These reward signals are used to update the policy parameters of the retrieval model. Specifically, policy gradient updates are applied to the retrieval model's scoring function, making useful memory entries rank higher for similar future queries.
    - **Design Motivation**: Unlike offline training of retrieval models, online RL can continuously adapt to user interaction patterns based on feedback in actual dialogues. This "learning-while-using" strategy is particularly suitable for long-term dialogue scenarios.

3. **Memory Bank Structure and Indexing**:

    - **Function**: Efficiently organizes and indexes a large number of memory entries.
    - **Mechanism**: The memory bank adopts a dual-layer indexing structure—the upper layer is an inverted index based on session topics, and the lower layer is embedding-based vector retrieval. During querying, the scope is first narrowed down through the topic index, and then accurate matching is performed via vector retrieval. Each memory entry contains metadata (timestamp, granularity level, source session ID) and semantic embeddings.
    - **Design Motivation**: As long-term dialogues accumulate, the memory bank grows continuously. The dual-layer index maintains recall rate while ensuring retrieval speed.

### Loss & Training

The RL training for retrospective reflection uses the REINFORCE algorithm. The reward function is defined as: $r=+1$ if the retrieved memory entry is cited by the LLM in the response, and $r=-1$ otherwise. The policy gradient updates the scoring parameters $\theta$ of the retrieval model: $\nabla_\theta J = \mathbb{E}[r \cdot \nabla_\theta \log \pi_\theta(m|q)]$, where $m$ is the memory entry and $q$ is the retrieval query.

## Key Experimental Results

### Main Results

Results on LongMemEval and MSC (Multi-Session Chat) benchmarks:

| Method | LongMemEval Acc | MSC-F1 | Retrieval Recall@5 |
|------|----------------|--------|-------------|
| No memory management | 42.3% | 18.7 | - |
| RAG (fixed retrieval) | 48.6% | 22.4 | 38.2% |
| MemoryBank | 50.1% | 24.1 | 41.7% |
| ReadAgent | 51.8% | 25.3 | 43.5% |
| RMM (Ours) | 53.2% | 27.8 | 49.1% |

RMM improves by more than 10% (42.3% → 53.2%) on LongMemEval compared to the baseline without memory, and by 1.4% compared to the strongest baseline ReadAgent.

### Ablation Study

| Configuration | LongMemEval Acc | Description |
|------|----------------|------|
| RMM Full | 53.2% | Full model |
| w/o Prospective Reflection | 49.5% | Uses fixed granularity, drops by 3.7% |
| w/o Retrospective Reflection | 50.8% | Uses fixed retrieval, drops by 2.4% |
| Utterance-level memory only | 48.1% | Finest granularity is insufficient |
| Session-level memory only | 47.3% | Too coarse-grained, loses information |
| Retrospective Reflection (supervised learning) | 51.4% | SL is worse than RL |

### Key Findings

- **Prospective reflection contributes the most** (3.7% improvement), indicating multi-granularity memory organization is key to long-term dialogue.
- Online RL for retrospective reflection outperforms offline supervised learning, validating the value of the "learning-while-using" strategy in long-term dialogues.
- Single-granularity memory (whether finest or coarsest) is inferior to multi-granularity, showing that different types of queries indeed require information at different granularities.
- As the number of dialogue turns increases, the advantage of RMM over fixed methods becomes more pronounced, indicating that memory management grows increasingly important in long dialogues.

## Highlights & Insights

- **The design philosophy of the bidirectional reflection mechanism** is elegant: prospective reflection addresses "how to store" and retrospective reflection addresses "how to retrieve", complementing each other to form a complete memory management loop. This approach can be generalized to any AI system requiring long-term information management.
- **Based on LLM citation as implicit reward** is an ingenious design—it does not require additional human annotation to evaluate retrieval quality. It leverages the LLM's own behavior (whether it uses the entry in its response) as a reward signal, achieving self-supervised retrieval optimization.
- The multi-granularity memory concept can be transferred to RAG systems: indexing documents by paragraph, section, and full text, and dynamically selecting retrieval results at the most appropriate granularity.

## Limitations & Future Work

- No forgetting or compression mechanism is designed for the expanding memory bank—the memory bank may become massive after long-term use, degrading retrieval efficiency.
- The reward signal based on LLM citations may be biased—LLMs might tend to cite longer or more prominent memory entries rather than the most relevant ones.
- The experimental scenarios are relatively simple (information-querying dialogues), and the effectiveness on dialogues requiring complex reasoning (such as psychological counseling or tutoring) remains unknown.
- Jointly optimizing prospective reflection and retrospective reflection (currently running independently) can be explored to let the storage strategy learn from retrieval feedback.

## Related Work & Insights

- **vs MemoryBank**: MemoryBank uses fixed-granularity storage and similarity-based retrieval, whereas RMM achieves more flexible memory management through multi-granularity and RL-based retrieval.
- **vs ReadAgent**: ReadAgent focuses on memory management in reading comprehension with a fixed paragraph-level granularity; RMM's three-level granularity design is better suited for the diversity of dialogue scenarios.
- **vs Retrieval-Augmented Generation**: The retrieval strategy of standard RAG is fixed after training, while RMM's retrospective reflection allows the retrieval strategy to continue optimizing after deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ The framework design of incorporating both prospective and retrospective reflections is novel, and the RL retrieval optimization is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive ablation studies and validation on multiple benchmarks, though lacking long-term tests with real users.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-elaborated motivation.
- Value: ⭐⭐⭐⭐ Provides a systematic solution for memory management of long-term dialogue agents with strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Effortless Active Labeling for Long-Term Test-Time Adaptation](../../CVPR2025/others/effortless_active_labeling_for_long-term_test-time_adaptation.md)
- [\[ACL 2025\] If Attention Serves as a Cognitive Model of Human Memory Retrieval, What is the Plausible Memory Representation?](if_attention_serves_as_a_cognitive_model_of_human_memory_retrieval_what_is_the_p.md)
- [\[CVPR 2026\] Learning Long-term Motion Embeddings for Efficient Kinematics Generation](../../CVPR2026/others/learning_long-term_motion_embeddings_for_efficient_kinematics_generation.md)
- [\[ACL 2025\] Hierarchical Memory Organization for Wikipedia Generation](hierarchical_memory_wikipedia_gen.md)
- [\[ACL 2025\] PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings](pvp_an_image_dataset_for_personalized.md)

</div>

<!-- RELATED:END -->

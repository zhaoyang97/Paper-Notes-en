---
title: >-
  [Paper Note] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA
description: >-
  [ACL 2025][Information Retrieval & RAG][Multi-hop QA] This paper identifies the "lost-in-retrieval" problem in RAG multi-hop QA—where subsequent sub-questions suffer a drastic drop in retrieval performance due to the lack of key entities after sub-question decomposition. To address this, the ChainRAG framework is proposed, which constructs a sentence graph, performs progressive retrieval, and rewrites sub-questions (to complete missing entities) to form a coherent reasoning c…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "Multi-hop QA"
  - "RAG"
  - "Sub-question rewriting"
  - "Sentence graph"
  - "Entity completion"
date: 2026-05-08
content_hash: 7d0ce924ac57921a
---

# Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA

**Conference**: ACL 2025  
**arXiv**: [2502.14245](https://arxiv.org/abs/2502.14245)  
**Code**: [GitHub](https://github.com/nju-websoft/ChainRAG)  
**Area**: Information Retrieval  
**Keywords**: Multi-hop QA, RAG, Sub-question rewriting, Sentence graph, Entity completion

## TL;DR

This paper identifies the "lost-in-retrieval" problem in RAG multi-hop QA—where subsequent sub-questions suffer a drastic drop in retrieval performance due to the lack of key entities after sub-question decomposition. To address this, the ChainRAG framework is proposed, which constructs a sentence graph, performs progressive retrieval, and rewrites sub-questions (to complete missing entities) to form a coherent reasoning chain, consistently outperforming baselines across three datasets: MuSiQue, 2Wiki, and HotpotQA.

## Background & Motivation

**Key Findings: "Lost-in-Retrieval"**: RAG multi-hop QA typically decomposes complex questions into sub-questions and retrieves sequentially. However, **the second sub-question often lacks explicit entities** (replaced by pronouns such as "this author"), leading to a sharp decline in retrieval performance. Empirical analysis indicates that the Recall@2 of the second sub-question is on average 18.29% lower than that of the first sub-question across three datasets.

**Specific Case**: Question "Where is the hometown of the author of *The Worries of the City*?" $\to$ Sub-question 1 "Who is the author of *The Worries of the City*?" $\to$ Sub-question 2 "Where is the hometown of this author?". In Sub-question 2, "this author" lacks a specific entity, leading to the retrieval of irrelevant text and ultimately an incorrect answer.

**Limitations of Prior Work**: Iterative retrieval methods (such as Iter-RetGen, IRCoT) utilize previously retrieved contexts to construct subsequent queries, but fail to explicitly resolve the missing entity issue. Other methods like GraphRAG require LLMs to extract entities and relations to construct knowledge graphs, which is computationally expensive.

## Method

### Overall Architecture

ChainRAG consists of four stages:

1. **Sentence Graph Construction**: Extract named entities from texts to construct a sentence graph where sentences are nodes and entity co-occurrences are edge labels.
2. **Question Decomposition**: Decompose multi-hop questions into sub-questions using an LLM.
3. **Iterative Processing**: Execute a loop of Retrieval $\to$ Answering $\to$ Rewriting the next sub-question for each sub-question.
4. **Answer Integration**: Aggregate all retrieved sentences and sub-answers to generate the final response.

### Key Designs

**1. Sentence Graph with Entity Indexing**

The sentence graph is the core data structure of ChainRAG, used to organize knowledge scattered across different texts and support entity completion.

- **Nodes**: Each sentence $s_i$ is a node.
- **Edges**: There are three types of edges:
    - **Entity Co-occurrence (EC)**: Connected if two sentences share key entities. To reduce redundancy, only entities in the top-$\alpha\%$ ($\alpha=60$) of BM25 importance are retained as key entities $\mathcal{K}_i \subseteq \mathcal{E}_i$.
    - **Semantic Similarity (SS)**: Sentence embeddings are computed using OpenAI's `text-embedding-3-small`. Each sentence maintains a set $\mathcal{R}_i$ of its top-$m$ ($m=10$) most similar sentences, and edges are added if they are mutually in each other's top-$m$.
    - **Structural Adjacency (SA)**: Edges are added between sentences with a distance $\le 3$ in the original text, helping reconstruct the overall document structure for broader retrieval.

- **Entity Index**: Stores the mapping from each entity to all sentences containing it, facilitating subsequent entity completion and retrieval expansion.

**2. Seed Sentence Retrieval + Graph Expansion**

Retrieval for each sub-question is conducted in two steps:

**Step A: Seed Sentence Retrieval**
- Compute similarity between the sub-question embedding and all sentences (completed quickly via matrix multiplication).
- Filter out low-similarity sentences to narrow down the candidate set.
- Use a cross-encoder (BGE-Reranker) to evaluate the relevance between candidate sentences and the sub-question.
- Select top-$k$ ($k=3$) sentences as seeds.

**Step B: Retrieval Expansion on the Sentence Graph**
- Start from the seed sentences and iteratively expand to neighbor nodes along the graph edges.
- After each expansion, use the LLM to judge whether sufficient information has been obtained to answer the sub-question, and stop if so.
- Optimization Mechanism: The initial expansion retrieves all one-hop neighbors at once (reducing LLM calls); a maximum total length constraint is set (to prevent excessively long contexts).

**3. Sub-question Rewriting**

This is the core step to address the "lost-in-retrieval" problem:

- **Trigger Condition**: Check whether the sub-question contains pronouns ("this", "it", "they", etc.).
- **Rewriting Process**: Feed the current sub-question, along with all previous sub-questions and their answers, into the LLM to generate a new sub-question containing concrete entities.
- **Example**: Sub-question 2 "In which region of S-Fone is this place located?" $\to$ Utilizes Sub-question 1 answer "Da Nang, Vietnam" $\to$ Rewritten as "In which region of S-Fone is Da Nang, Vietnam located?"
- **Special Case**: If a previous sub-question was not answered, summarize its corresponding context and integrate it into the context of the current sub-question.

**Dual Effects**: (1) Restores retrieval performance; (2) Preserves key information from preceding sub-questions to support current reasoning.

**4. Two Strategies for Answer Integration**

- **Sub-answer Integration (AnsInt)**: Derives the final answer using only the sub-questions and their answers. Advantage: Processes text for only one sub-question at a time, requiring no strong long-context capability. Disadvantage: An incorrect answer to any sub-question affects the entire outcome.
- **Sub-context Integration (CxtInt)**: Merges all retrieved sentences, de-duplicates, and re-ranks them to generate the answer. Advantage: Alleviates the impact of incorrect sub-question answers. Disadvantage: Requires the LLM to process longer contexts, introducing more noise.

### Loss & Training

ChainRAG is a **train-free method**, with all components utilizing off-the-shelf models:
- Embedding Model: OpenAI `text-embedding-3-small`
- Cross-encoder: BGE-Reranker
- Entity Recognition: spaCy
- LLM: GPT-4o-mini / Qwen2.5-72B / GLM-4-Plus

## Key Experimental Results

### Main Results: Three Multi-Hop QA Datasets

| Method | MuSiQue F1 | MuSiQue EM | 2Wiki F1 | 2Wiki EM | HotpotQA F1 | HotpotQA EM |
|------|-----------|-----------|---------|---------|------------|------------|
| NaiveRAG | 29.82 | 19.00 | 50.61 | 42.50 | 56.92 | 42.00 |
| NaiveRAG w/ QD | 37.49 | 26.00 | 56.88 | 38.50 | 60.00 | 43.50 |
| Iter-RetGen | 38.41 | 33.00 | 58.43 | 50.50 | 57.77 | 42.00 |
| LongRAG | 44.88 | 32.00 | 62.39 | 49.00 | 64.74 | 51.00 |
| HippoRAG w/ IRCoT | 46.50 | 28.50 | 62.38 | 48.00 | 56.12 | 40.00 |
| **ChainRAG (AnsInt)** | **50.54** | 37.00 | 62.55 | 52.00 | 60.73 | 46.00 |
| **ChainRAG (CxtInt)** | 47.87 | **38.50** | 56.54 | 50.50 | **64.59** | **50.00** |

(The above are results for GPT-4o-mini; on Qwen2.5-72B, CxtInt achieves an average F1 of 59.92, outperforming the runner-up method HippoRAG's 54.68 by 9.6%.)

### Ablation Study

| Ablation Item | MuSiQue F1 | 2Wiki F1 | HotpotQA F1 |
|--------|-----------|---------|------------|
| ChainRAG (Full) | 50.54 | 62.55 | 60.73 |
| w/o Sub-question Rewriting | ~35 (-30%) | ~55 | ~55 |
| w/o EC Edge | Slightly decreased | Significantly decreased | Slightly decreased |
| w/o SS Edge | Slightly decreased | Slightly decreased | Slightly decreased |
| w/o SA Edge | Slightly decreased | Slightly decreased | Significantly decreased |
| w/o Sentence Graph (chunk mode) | Decreased | Decreased | EM significantly decreased |

Removing sub-question rewriting leads to a drop of approximately 30% in both F1 and EM on MuSiQue, confirming the severity of the "lost-in-retrieval" problem.

### Retrieval Performance Verification

| Sub-question | MuSiQue Recall@2 | 2Wiki Recall@2 | HotpotQA Recall@2 |
|--------|-----------------|---------------|-------------------|
| Sub-question 1 | 55.52 | 57.50 | 54.67 |
| Sub-question 2 (Original) | 40.91 | 49.87 | 49.17 |
| Sub-question 2 (Rewritten) | **58.81** | **54.32** | **61.83** |

After rewriting, the Recall of Sub-question 2 even surpasses that of Sub-question 1 (by leveraging information from Sub-question 1 to assist retrieval).

### Key Findings

- **AnsInt vs CxtInt Depends on LLM Capabilities**: GPT-4o-mini (strong reasoning) is suitable for AnsInt; Qwen2.5-72B and GLM-4-Plus (strong long-context capability) are suitable for CxtInt.
- **Decomposition Sometimes Offers Limited Help or Is Even Harmful**: Applying question decomposition to NaiveRAG actually degrades GLM-4-Plus performance on MuSiQue, directly validating the lost-in-retrieval problem.
- **Entity Recovery Errors Are More Fatal Than Decomposition Errors**: An average F1 of 55.43 is observed for incorrect decomposition, compared to 51.34 for incorrect entity recovery, indicating that the latter impacts performance more severely.
- **spaCy Entity Recognition is Sufficient**: Replacing spaCy with Qwen2.5-72B for entity recognition yields only a ~1.5 F1 improvement, showing that spaCy achieves the best balance between efficiency and effectiveness.

## Highlights & Insights

1. **Precise Problem Definition**: "Lost-in-retrieval" is clearly quantified through empirical data (an 18.29% drop in Recall), providing an actionable diagnostic framework for retrieval failures in multi-hop QA.
2. **Ingenious Sentence Graph Design**: Three edge types (entity co-occurrence, semantic similarity, and structural adjacency) organize information from different dimensions. Graph expansion achieves finer-grained and more comprehensive knowledge acquisition than traditional chunk-based retrieval.
3. **Rewriting > Constructing KGs**: Compared to high-cost solutions like GraphRAG/HippoRAG that require LLMs to extract entity relations for constructing knowledge graphs, sub-question rewriting offers a much lighter alternative.
4. **Efficiency Advantage**: Reduces LLM call counts by an average of 17.3% compared to LongRAG, and is several times fewer than HippoRAG.

## Limitations & Future Work

1. **Additional Overhead in the Iterative Process**: Although much more efficient than HippoRAG, it still requires more LLM calls compared to NaiveRAG and Iter-RetGen.
2. **Unverified Domain Adaptability**: The three datasets used are all from the general domain (Wikipedia); its effectiveness in highly specialized scenarios (e.g., legal or medical domains) remains unknown.
3. **Error Propagation in Entity Recognition**: Incorrect entity recognition and recovery represent the largest performance bottleneck (with a 79.3% recovery accuracy), and errors can propagate through the reasoning chain.
4. **Granularity of Sentences May Be Too Fine**: In certain long-document scenarios, sentence-level indexing might lead to excessive information fragmentation.

## Related Work & Insights

- **Iterative Retrieval in RAG**: Iter-RetGen uses previously generated text as the next query; Self-RAG includes a self-reflection mechanism. The uniqueness of ChainRAG lies in explicitly completing missing entities rather than simply reusing contexts.
- **Graph-structured RAG**: RAPTOR (tree-structured), GraphRAG/HippoRAG (LLM-constructed KGs) incur high costs. ChainRAG's sentence graph only requires spaCy + an embedding model, making it much lighter.
- **Insights**: (1) "Lost-in-retrieval" is a universal issue in multi-hop QA, and any RAG system utilizing question decomposition should consider entity completion; (2) The three-edge design of the sentence graph can be generalized to other fine-grained retrieval tasks.

## Rating

| Dimension | Score (1-10) | Description |
|------|------------|------|
| Novelty | 8 | The "lost-in-retrieval" problem definition is valuable, and the sentence graph + rewriting solution is simple and effective. |
| Technical Depth | 7 | The sentence graph is well-designed, and the two integration strategies accommodate different LLM features. |
| Experimental Thoroughness | 8 | 3 datasets $\times$ 3 LLMs + full ablation + retrieval analysis + efficiency analysis + error analysis. |
| Writing Quality | 8 | The problem motivation is established through empirical analysis, with clear and persuasive cases. |
| Value | 8 | Train-free, directly integrable; the combination of a sentence graph and rewriting is a low-cost, general-purpose improvement. |
| Total Score | 7.8 | Precise problem definition, highly practical solution, and comprehensive experiments. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] MergePRAG: Orthogonal Merging of Passage-experts for Multi-hop Parametric RAG](../../ICLR2026/information_retrieval/mergeprag_orthogonal_merging_of_passage-experts_for_multi-hop_parametric_rag.md)
- [\[NeurIPS 2025\] Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG](../../NeurIPS2025/information_retrieval/think_straight_stop_smart_structured_reasoning_for_efficient_multi-hop_rag.md)
- [\[ACL 2025\] MAIN-RAG: Multi-Agent Filtering Retrieval-Augmented Generation](main-rag_multi-agent_filtering_retrieval-augmented_generation.md)
- [\[ACL 2025\] Micro-Act: Mitigate Knowledge Conflict in QA via Actionable Self-Reasoning](micro_act_knowledge_conflict_reasoning.md)
- [\[ACL 2026\] BRIEF-Pro: Universal Context Compression with Short-to-Long Synthesis for Fast and Accurate Multi-Hop Reasoning](../../ACL2026/information_retrieval/brief-pro_universal_context_compression_with_short-to-long_synthesis_for_fast_an.md)

</div>

<!-- RELATED:END -->

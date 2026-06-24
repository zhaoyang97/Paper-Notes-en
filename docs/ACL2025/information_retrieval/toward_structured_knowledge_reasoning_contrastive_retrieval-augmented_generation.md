---
title: >-
  [Paper Note] Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience
description: >-
  [ACL 2025 Findings][Information Retrieval & RAG][Structured Knowledge Reasoning] This paper proposes the CoRE framework, which constructs an experience memory repository containing both successful and failed reasoning trajectories via Monte Carlo Tree Search (MCTS), and retrieves positive and negative exemplars during inference via Contrastive In-Context Learning (Contrastive ICL) to enhance the structural data (tables, databases) reasoning capabilities of LLMs…
tags:
  - "ACL 2025 Findings"
  - "Information Retrieval & RAG"
  - "Structured Knowledge Reasoning"
  - "Contrastive Retrieval-Augmented Generation"
  - "Experience Memory"
  - "Monte Carlo Tree Search"
  - "Text-to-SQL"
date: 2026-05-08
content_hash: df9e9bbc98ce133f
---

# Toward Structured Knowledge Reasoning: Contrastive Retrieval-Augmented Generation on Experience

**Conference**: ACL 2025 Findings  
**arXiv**: [2506.00842](https://arxiv.org/abs/2506.00842)  
**Code**: [GitHub](https://github.com/Kuvvius/CoRE)  
**Area**: Information Retrieval  
**Keywords**: Structured Knowledge Reasoning, Contrastive Retrieval-Augmented Generation, Experience Memory, Monte Carlo Tree Search, Text-to-SQL

## TL;DR

This paper proposes the CoRE framework, which constructs an experience memory repository containing both successful and failed reasoning trajectories via Monte Carlo Tree Search (MCTS), and retrieves positive and negative exemplars during inference via Contrastive In-Context Learning (Contrastive ICL) to enhance the structural data (tables, databases) reasoning capabilities of LLMs, achieving average improvements of 3.44% and 4.24% on Text-to-SQL and TableQA, respectively.

## Background & Motivation

**Background**: LLMs perform exceptionally well on textual tasks but degrade significantly when dealing with structured data (such as database tables, knowledge graphs). Mainstream approaches are divided into two categories: (1) Agent-based multi-step decomposition methods (e.g., MAC-SQL, DIN-SQL) use fixed few-shot exemplars to guide SQL generation; (2) RAG-based methods retrieve auxiliary information from external knowledge bases.

**Limitations of Prior Work**: (1) Existing RAG methods provide limited help for structured knowledge reasoning because the retrieved general documents lack alignment with the target structured formats; (2) Agent-based methods rely on fixed few-shot exemplars and lack dynamic adaptation—fixed exemplars may have low relevance to new questions; (3) Existing methods lack systematic positive and negative contrastive signals—they cannot distinguish "right paths" from "wrong paths" and fail to learn from errors.

**Key Challenge**: Structured data accounts for an extremely low proportion of LLM pre-training data, making it difficult for models to understand the implicit relations within tables (such as foreign key associations and column constraints). In contrast, human experts build mental models for structured reasoning through "deliberate practice"—exposure to a large number of successful and failed cases—which LLMs lack.

**Goal**: (1) Construct a rich reasoning experience memory repository (containing both successful and failed trajectories); (2) Design dynamic retrieval and contrastive learning mechanisms to utilize these experiences; (3) Provide a plug-and-play, training-free solution.

**Key Insight**: Inspired by the theory of "deliberate practice" in cognitive science—where human experts establish highly efficient mental representations through repeated exposure to diverse success and failure cases—CoRE mimics this process to build "experience memory" for LLMs.

**Core Idea**: Use MCTS to simulate human trial-and-error processes to automatically generate a large number of successful/failed reasoning trajectories, thereby establishing an experience memory repository; during inference, retrieve related positive and negative cases and prompt the LLM via contrastive ICL to "learn from successful experiences and avoid repeating failed mistakes."

## Method

### Overall Architecture

CoRE consists of three modules: (1) **Experience Memory Builder**—explores reasoning paths using MCTS to collect experience trajectories with reward labels; (2) **Retriever**—retrieves related positive (high reward) and negative (low reward) exemplars for new questions; (3) **Contrastive Thinker**—organizes positive and negative exemplars into contrastive prompts to guide LLM answer generation. The entire framework is training-free and does not modify LLM parameters.

### Key Designs

1. **Experience Memory Builder（基于 MCTS 的经验记忆构建）**:

    - **Function**: Automatically generate a large number of diverse reasoning trajectories, containing reward labels for intermediate steps.
    - **Mechanism**: Decompose the structured reasoning problem into a sequence of sub-questions $\tau = \{(q_1, a_1), (q_2, a_2), ..., (q_n, a_n)\}$, and search for the optimal trajectory using MCTS. Four stages: **Selection**—balance exploration and exploitation using the UCT formula $q_k^* = \arg\max [Q_{value} + w\sqrt{\ln N(s)/N(c)}]$; **Expansion**—generate $d$ candidate sub-questions using LLM; **Simulation**—simulated forward to the terminal state, evaluating with a hybrid reward function $f_r = r_1^\alpha \cdot r_2^{1-\alpha}$ (where $r_1$ is consistency reward, and $r_2$ is self-evaluation reward); **Back-propagation**—propagate back along the path to update Q-values. The final generated experience memory contains records of $(q, a, r)$ consisting of the question, answer, and reward label.
    - **Design Motivation**: Compared to directly generating few-shot exemplars with LLMs, MCTS can systematically explore the reasoning space and generate diverse success and failure cases. The original training set is expanded by 8-9 times, significantly increasing coverage.

2. **Retriever（双排序检索器）**:

    - **Function**: Retrieve the most relevant positive and negative exemplars for a new question.
    - **Mechanism**: Adopt a two-stage ranking strategy. First, retrieve the top-$k$ experiences using semantic similarity $\text{Sim}(Q_{current}, Q_{e_i})$. Then, perform secondary ranking based on reward labels: positive cases are ranked by a linear combination of $\text{rank}_{sim}$ and $\text{rank}_{reward}$ (preferring high rewards); negative cases are ranked by combining $\text{rank}_{sim}$ with the reverse of $\text{rank}_{reward}$ (preferring low rewards). SQL queries are appended with natural language descriptions of ASTs to improve retrieval accuracy.
    - **Design Motivation**: Negative exemplars retrieved solely by semantic similarity might not be "relevant enough" or "typical enough." Secondary ranking via reward labels ensures that negative exemplars are both relevant to the new question and genuinely erroneous cases.

3. **Contrastive Thinker（对比式推理器）**:

    - **Function**: Use retrieved positive and negative exemplars to guide the LLM to generate the correct answer.
    - **Mechanism**: Build a contrastive prompt template: "Reference successful experiences and avoid repeating failed mistakes." Positive exemplars serve as patterns to learn, while negative exemplars (with error analysis) serve as patterns to avoid. It supports two contrastive modes: (a) Single-round contrast—display both positive and negative exemplars in one prompt; (b) Multi-round contrast—first provide positive exemplars for the model to generate an initial answer, and then provide negative exemplars for the model to correct. Both methods achieve similar performance, and the multi-round approach can accommodate token constraints.
    - **Design Motivation**: Providing only positive exemplars may not yield enough information, while providing only negative exemplars might "mislead" the model (as LLMs tend to copy erroneous cases unintentionally). Contrastive learning displays "contrasting pairs" simultaneously, allowing the model to more accurately comprehend the correct reasoning patterns.

### Loss & Training

CoRE is a training-free framework and does not involve model training. The reward function during the MCTS process is a weighted geometric mean of the consistency reward (answering frequency) and the self-evaluation reward.

## Key Experimental Results

### Main Results

Text-to-SQL (Bird dataset, LLaMA-3-70b or GPT-4):

| Method | EX (%) | +CoRE | Gain |
|------|--------|-------|------|
| DIN-SQL | 30.5 | 34.0 | +3.5 |
| DAIL-SQL | 31.6 | 35.2 | +3.6 |
| MAC-SQL | 34.9 | 40.8 | +5.9 |
| MAC-SQL (GPT-4) | 46.6 | 51.6 | +5.0 |
| **Average Gain** | — | — | **+3.44** |

TableQA (WikiTQ and FinQA, GPT-3.5):

| Dataset | Method | Accuracy | +CoRE | Gain |
|--------|------|----------|-------|------|
| WikiTQ | StructGPT | 64.4 | 66.1 | +1.7 |
| WikiTQ | Dater | 58.4 | 63.5 | +5.1 |
| FinQA | StructGPT | 51.2 | 53.1 | +1.9 |
| FinQA | Dater | 52.4 | 59.0 | +6.6 |

### Ablation Study

GPT-4 contrastive mode analysis under 2-shot setting (MAC-SQL + Bird):

| Configuration | EX (%) | Description |
|------|--------|------|
| Fixed Exemplars (baseline) | 56.40 | Original fixed few-shot |
| Positive + Positive | 58.92 | Dynamic retrieval of positive exemplars only |
| Negative + Negative | 45.20 | Negative exemplars only, performance degradation |
| Positive + Negative (Single-round) | 58.92 | Contrastive ICL single-round approach |
| Positive $\to$ Negative (Multi-round) | 58.08 | Contrastive ICL multi-round approach |
| Full CoRE | **58.92** | Optimal configuration of Contrastive ICL |

### Key Findings

- **Contrastive ICL is significantly superior to using only positive exemplars**: Retrieving only positive exemplars (+positive+positive) yields an improvement of approximately 2.52%, but introducing negative contrast achieves a much larger boost on difficult questions (up to 17.2%), indicating that "learning from mistakes" is critical for complex reasoning.
- **Using only negative exemplars leads to performance degradation**: The negative+negative configuration degrades performance to 45.2%, worse than not using RAG at all. This validates the phenomenon that LLMs unconsciously mimic incorrect templates in the context.
- **The volume of experience memory generated by MCTS is substantially increased**: The training set for the Bird dataset in CoRE is expanded from ~9,400 to 85,956 records (by 8-9x), and WikiTQ is expanded from 11,321 to 98,586 records, greatly enhancing retrieval coverage.
- **Most significant improvements on hard questions**: On the challenging subset of Bird, CoRE improves the EX of MAC-SQL from approximately 20% to 40% (nearly doubled), while the improvement on the simple subset is relatively minor.

## Highlights & Insights

- **Computational implementation of "deliberate practice"**: Transferring learning theory from cognitive science to LLM reasoning, where MCTS plays the role of "practice" and experience memory serves as the "mental model." This analogy is elegant and effective. The same concept can be applied to other tasks requiring systematic trial-and-error, such as code generation and mathematical reasoning.
- **Training-free, plug-and-play design**: CoRE does not modify any model parameters and improves performance solely by altering exemplars in the prompt, rendering it highly compatible. The one-time investment in MCTS can be reused across multiple downstream tasks.
- **The insight of "harmful negative exemplars but effective comparison"**: Providing only negative exemplars harms model performance, but pairing negative exemplars with positive ones to form contrast brings the largest improvement. This insight is useful for all methods utilizing ICL—what to put in the context is more important than how much to put.

## Limitations & Future Work

- **Experience memory needs to be reconstructed for new tasks**: For each new database or domain, the experience memory must be regenerated using MCTS. This offline cost, though one-time, is non-negligible (requiring extensive LLM API calls).
- **Highly dependent on the quality of the grounding stage**: CoRE only improves the reasoning stage. If the preceding schema linking and table pruning (grounding) fail, CoRE cannot compensate. This is supported by the massive performance leap when using the golden schema in experiments.
- **Incompatible with Self-Consistency strategies**: The performance of CoRE + DAIL-SQL (SC) drops by 0.2%, suggesting a conflict between Sampler Contrastive Context information and high-temperature voting.
- **Future Directions**: Update experience memory online (continuously accumulating experience as new questions arise), or unify both grounding and reasoning stages into the contrastive framework.

## Related Work & Insights

- **vs DAIL-SQL**: DAIL-SQL retrieves few-shot exemplars from the training set based on structural similarity, but relies only on positive exemplars from limited sources. CoRE's experience memory scales up the data size by 8-9x using MCTS and introduces positive-negative contrastive signals.
- **vs ExpeL**: ExpeL also learns from experiences using LLMs, but relies on human-annotated datasets to summarize experience. CoRE automatically generates experiences and reward labels using MCTS, achieving a higher level of automation.
- **vs Self-RAG / Corrective RAG**: These works focus on "when to retrieve" and "retrieval quality assessment," whereas CoRE focuses on "how to utilize retrieved results" by using a contrastive mechanism to improve the utilization efficiency of retrieved content.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of MCTS-based experience memory construction and contrastive ICL is novel, with an elegant cognitive science analogy.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across multiple datasets and baselines, with detailed ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ The framework description is clear, though it contains many symbols requiring careful reading.
- **Value**: ⭐⭐⭐⭐ A plug-and-play practical framework with direct application value for structured data reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] HyperGraphRAG: Retrieval-Augmented Generation via Hypergraph-Structured Knowledge Representation](../../NeurIPS2025/information_retrieval/hypergraphrag_retrieval-augmented_generation_via_hypergraph-structured_knowledge.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)
- [\[ACL 2025\] SeaKR: Self-aware Knowledge Retrieval for Adaptive Retrieval Augmented Generation](seakr_self-aware_knowledge_retrieval_for_adaptive_retrieval_augmented_generation.md)
- [\[ACL 2025\] HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases](hybgrag_hybrid_rag_skb.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](../../ICLR2026/information_retrieval/g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)

</div>

<!-- RELATED:END -->

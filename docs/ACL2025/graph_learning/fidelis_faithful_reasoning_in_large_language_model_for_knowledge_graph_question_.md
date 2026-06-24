---
title: >-
  [Paper Note] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering
description: >-
  [ACL 2025][Graph Learning][Knowledge Graph Question Answering] Proposes the FiDeLiS framework, which narrows down the search space via a Path-RAG-preselected candidate set, and progressively constructs and validates reasoning paths using Deductive-Verification Beam Search (DVBS). This improves LLM accuracy and interpretability in knowledge graph question answering without requiring training.
tags:
  - "ACL 2025"
  - "Graph Learning"
  - "Knowledge Graph Question Answering"
  - "LLM Reasoning"
  - "Beam Search"
  - "Deductive Verification"
  - "Path-RAG"
date: 2026-05-08
content_hash: 1b46bba145336a6e
---

# FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering

**Conference**: ACL 2025  
**arXiv**: [2405.13873](https://arxiv.org/abs/2405.13873)  
**Code**: [https://github.com/Y-Sui/FiDeLiS](https://github.com/Y-Sui/FiDeLiS)  
**Area**: Graph Learning  
**Keywords**: Knowledge Graph Question Answering, LLM Reasoning, Beam Search, Deductive Verification, Path-RAG

## TL;DR

Proposes the FiDeLiS framework, which narrows down the search space via a Path-RAG-preselected candidate set, and progressively constructs and validates reasoning paths using Deductive-Verification Beam Search (DVBS). This improves LLM accuracy and interpretability in knowledge graph question answering without requiring training.

## Background & Motivation

**Background**: LLMs exhibit outstanding performance in complex reasoning tasks, but are highly prone to hallucination or factually inconsistent outputs. Utilizing Knowledge Graphs (KGs) as external knowledge sources is a viable solution to alleviate this issue, as the structured format of KGs supports explicit and traceable reasoning.

**Limitations of Prior Work**:
   - **Retrieval methods** (e.g., RoG): Only 67% of the generated reasoning steps are valid; 33% contain formatting errors or cite non-existent KG facts.
   - **Agent-based methods** (e.g., ToG): Although enhancing reasoning accuracy, they suffer from high computational costs, high latency, and poor scalability.
   - Core dilemma: How to balance task faithfulness and efficiency?

**Key Challenge**: Retrieval methods are fast but lack accuracy (unable to guarantee valid reasoning paths), while Agent-based methods are accurate but too slow (requiring multiple rounds of LLM-KG interaction).

**Goal**: To design a unified framework that guarantees the verifiability of every step in the reasoning path (faithfulness) while reducing computational costs by narrowing down the search space (efficiency).

**Key Insight**: Combining the strengths of retrieval and exploration—where Path-RAG handles efficient pre-selection of candidates (the advantage of retrieval), and DVBS progressively verifies reasoning paths (the advantage of Agent-based methods).

**Core Idea**: Narrowing down the search space with RAG + ensuring logical validity of each reasoning step with deductive verification, achieving both faithful and efficient KG reasoning without any training.

## Method

### Overall Architecture

FiDeLiS consists of two core components:
1. **Path-RAG**: Pre-selects a small-scale candidate set to narrow down the search space for KG exploration.
2. **DVBS (Deductive-Verification Beam Search)**: Progressively constructs reasoning paths, ensuring logical consistency at each step through deductive verification.

Mathematical Formalization:
$$P(a|q,\mathcal{G}) = \sum_{\mathcal{P}} P_\theta(a|q,\mathcal{P}) \prod_{k=1}^n P_\phi(t_k|q,t_{<k},\mathcal{G})$$

### Key Designs

#### 1. Path-RAG: Reasoning Path Retrieval-Augmented Generation

- **Function**: Pre-selects high-quality candidate reasoning steps for each step of the beam search.
- **Three-step pipeline**:
    - **Initialization**: Encodes all entities $e_i$ and relations $r_i$ in the KG into dense vectors $z(e_i) = \text{LM}(e_i) \in \mathbb{R}^d$ using a pre-trained language model, then stores them in a nearest-neighbor index.
    - **Keyword-driven retrieval**: The LLM extracts keywords from the user question, encodes them, and retrieves the top-$m$ similar entities and relations.
    - **Candidate reasoning step construction**: Defines a scoring function that combines semantic similarity and graph connectivity.
- **Scoring Function** (Core Formula):
  $$S((r,e)) = S_0((r,e)) + \alpha \max_{\forall(r_j,e_j) \in N(e)} S_0((r_j,e_j))$$
  where $S_0((r,e)) = S_{\text{rel}}(r) + S_{\text{ent}}(e)$ is the basic semantic score, and the second term accounts for the maximum score of next-hop candidates.
- **The role of $\alpha$**: Balances immediate semantic relevance and future connectivity potential; a higher $\alpha$ favors long-term gains.
- **Design Motivation**: Replaces the multi-round LLM-KG interaction of Agent-based methods with a one-time retrieval to acquire high-quality candidates.

#### 2. DVBS: Deductive Verification Beam Search

- **Function**: Progressively builds reasoning paths, validates logical consistency at each step, and terminates the search once the question becomes derivable.
- **Three-step pipeline**:
    - **Plan generation**: The LLM first generates planning steps $w$ to answer the question, acting as an extra hint for downstream decisions.
    - **Beam Search**: At time step $t$, the LLM selects reasoning steps from the candidate set $\mathcal{S}^t$:
    $$\mathcal{H}_t = \text{Top}_k \{h \oplus \text{LM}(s^t | q, s^{1:t-1}, w, \mathcal{S}^t) : h \in \mathcal{H}_{t-1}\}$$
    - **Deductive verification**: A two-layer verification ensures reasoning quality.
- **Two-layer verification**:
    - **Global verification** $C_{\text{global}}$: Checks if $(s^t \land s^{1:t-1}) \models q'$, verifying whether the current reasoning path is sufficient to derive the answer to the question.
    - **Local verification** $C_{\text{local}}$: Checks whether $s^t$ logically follows from $s^{1:t-1}$ to ensure step-to-step consistency.
- **Termination condition**: Explores until both global and local verifications are satisfied.
- **Design Motivation**: Addresses the issues of premature termination and over-extension, ensuring that the reasoning path is complete and valid.

### Loss & Training

**FiDeLiS is a training-free framework** that relies entirely on the zero/few-shot capabilities of LLMs and the KG structure. 5-shot prompts are used by default.

## Key Experimental Results

### Main Results

Comparison across three benchmark datasets (using gpt-3.5-turbo / gpt-4-turbo):

| Method | Backend | WebQSP Hits@1 | CWQ Hits@1 | CR-LT Acc |
|------|------|--------------|------------|-----------|
| Zero-shot CoT | gpt-3.5 | 57.42 | 43.21 | 37.42 |
| RoG (Finetuning) | - | 83.15 | 61.39 | 60.32 |
| ToG | gpt-3.5 | 75.13 | 57.59 | 62.48 |
| **FiDeLiS** | gpt-3.5 | **79.32** | **63.12** | **67.34** |
| ToG | gpt-4 | 81.84 | 68.51 | 67.24 |
| **FiDeLiS** | gpt-4 | **84.39** | **71.47** | **72.12** |

FiDeLiS (gpt-4) achieves the best performance across all datasets, and as a training-free method, even outperforms fine-tuned methods like RoG and DeCAF.

### Ablation Study

Ablation of each component (gpt-3.5-turbo):

| Ablation Setting | WebQSP | CWQ | CR-LT |
|---------|--------|-----|-------|
| Full FiDeLiS | 79.32 | 63.12 | 67.34 |
| Replace Path-RAG with vanilla retriever | 72.35 | 57.11 | 59.78 |
| Replace Path-RAG with ToG | 75.11 | 59.47 | 63.47 |
| Remove beam search | 60.35 | 49.78 | 61.87 |
| Remove deductive verification | 74.13 | 57.23 | 63.89 |
| Remove planning | 76.23 | 60.14 | 64.13 |

### Key Findings

1. **Beam search is the most critical component**: Upon removal, accuracy on WebQSP plummets from 79.32% to 60.35% (-18.97%).
2. **Path-RAG outperforms all alternative solutions**: Exceeding vanilla retriever by ~7% and ToG as a retriever by ~4%.
3. **Reasoning path lengths are closer to the ground truth**: The average depth for FiDeLiS is 2.4 (on WebQSP), compared to 3.1 for ToG and 2.3 for ground truth.
4. **Significant efficiency advantage**: Reduces running time by approximately 1.7x compared to ToG (43.83s vs 74.26s per question on WebQSP).
5. **Path-RAG coverage**: Coverage rate (CR) is 72.61% at depth 1, 69.38% at depth 2, and 62.78% at depth >3, all greatly exceeding the vanilla retriever.
6. **Deductive verification > adequacy verification > logit scoring**: 79.32 vs 74.13 vs 73.47 on WebQSP.
7. **While only 67% of RoG's reasoning paths are valid**, FiDeLiS ensures 100% validity through progressive verification.

## Highlights & Insights

1. **Bridging the gap between retrieval and Agent-based methods**: Path-RAG inherits the efficiency of retrieval, while DVBS inherits the accuracy of Agent-based approaches, bringing the best of both worlds.
2. **Two-layer design in deductive verification**: Global verification checks adequacy, and local verification checks validity, resolving both the premature termination and over-extension problems simultaneously.
3. **Training-free**: Relies entirely on the in-context learning capabilities of LLMs, offering high versatility and easy transferring to new KGs.
4. **Engineering value of plan generation**: Though an engineering trick, it unlocks the high-level reasoning capabilities of LLMs and brings consistent gains.
5. **Compelling case study**: The Iranian political system example clearly demonstrates how FiDeLiS provides more comprehensive and accurate answers than CoT, RoG, and ToG.

## Limitations & Future Work

1. Depends heavily on the quality and completeness of external KGs; incomplete or outdated KGs will negatively impact performance.
2. Multi-round LLM calls still incur non-trivial overhead; although better than ToG, a gap remains for real-time applications.
3. Has not been fully validated in open domains or on ultra-large-scale KGs.
4. Deductive verification relies on the LLM's logical reasoning ability; any reasoning errors inherent of the LLM may propagate.
5. The embedding quality of Path-RAG is limited by the representation capacity of the pre-trained LM.

## Related Work & Insights

- **ToG** (Sun et al.): Representative of Agent-based methods, whereas FiDeLiS boosts efficiency by 1.7x.
- **RoG** (Luo et al.): Representative of retrieval-based methods, yet 33% of its reasoning steps are invalid.
- **KD-CoT** (Wang et al.): Validates sub-reasoning steps using an external KG, sharing a similar concept with the verification mechanism in FiDeLiS.
- Insight: Introducing deductive reasoning as a "calibration tool" into the search process is an effective strategy for boosting the faithfulness of LLM reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of Path-RAG and DVBS is cleverly designed, and the introduction of deductive verification is highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Very comprehensive evaluation spanning three datasets, multiple LLM backends, ablation studies, robustness tests, efficiency analyses, and case studies.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured with standardized descriptions of equations and algorithms.
- **Value**: ⭐⭐⭐⭐⭐ — Outperforms fine-tuning-based methods without requiring training, offering high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Graph-constrained Reasoning: Faithful Reasoning on Knowledge Graphs with Large Language Models](../../ICML2025/graph_learning/graph-constrained_reasoning_faithful_reasoning_on_knowledge_graphs_with_large_la.md)
- [\[ACL 2025\] Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering](ontology-guided_reverse_thinking_makes_large_language_models_stronger_on_knowled.md)
- [\[ACL 2025\] The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering](the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)
- [\[ACL 2025\] Agent Steerable Search for Knowledge Graph Question Answering](agent_steerable_search_for_knowledge_graph_question_answering.md)
- [\[ACL 2025\] Can Knowledge Graphs Make Large Language Models More Trustworthy? An Empirical Study Over Open-ended Question Answering](kg_llm_trustworthy_qa.md)

</div>

<!-- RELATED:END -->

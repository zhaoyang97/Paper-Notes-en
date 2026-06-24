---
title: >-
  [Paper Note] iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering
description: >-
  [ACL 2025][NLP Understanding][Knowledge Base Question Answering] iQUEST proposes an iterative sub-question guided framework that dynamically generates answerable sub-questions at each reasoning step to maintain reasoning direction. Combined with GNNs to aggregate semantic information from two-hop neighbors for "look-ahead" entity exploration, it achieves SOTA or near-SOTA performance on four benchmarks (CWQ, WebQSP, WebQuestions, and GrailQA) without the need to fine-tune the…
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Knowledge Base Question Answering"
  - "Multi-hop Reasoning"
  - "Sub-question Guidance"
  - "GNN"
  - "Entity Exploration"
date: 2026-05-08
content_hash: ed1d69696b2e66c5
---

# iQUEST: An Iterative Question-Guided Framework for Knowledge Base Question Answering

**Conference**: ACL 2025  
**arXiv**: [2506.01784](https://arxiv.org/abs/2506.01784)  
**Code**: [GitHub](https://github.com/Wangshuaiia/iQUEST)  
**Area**: NLP Understanding  
**Keywords**: Knowledge Base Question Answering, Multi-hop Reasoning, Sub-question Guidance, GNN, Entity Exploration

## TL;DR

iQUEST proposes an iterative sub-question guided framework that dynamically generates answerable sub-questions at each reasoning step to maintain reasoning direction. Combined with GNNs to aggregate semantic information from two-hop neighbors for "look-ahead" entity exploration, it achieves SOTA or near-SOTA performance on four benchmarks (CWQ, WebQSP, WebQuestions, and GrailQA) without the need to fine-tune the LLM.

## Background & Motivation

**Key Challenge of Multi-Hop KBQA**: Complex questions require multi-hop reasoning over the Knowledge Graph (KG) (e.g., "What is the official flower of the area impacted by Tropical Storm Fabio in the North Pacific region?"), but face two key difficulties:
   - **(1) Difficulty in maintaining reasoning path coherence**: As the number of reasoning steps increases, the model easily gets "lost", and ambiguous entities (e.g., Mexico) introduce noise that further interferes with reasoning.
   - **(2) Premature discarding of critical multi-hop connections**: Existing methods mostly rely on local relevance scoring of one-hop neighbors, potentially discarding crucial paths that seem irrelevant but are highly matched in their two-hop neighbors (e.g., "John Williams", a two-hop neighbor of "Harry Potter Films", is the actual key component).

**Limitations of Prior Work**: Agent-based methods like ToG and Interactive-KBQA are effective but lack continuous reasoning guidance; question decomposition methods are mostly one-time static decompositions, failing to adaptively adjust based on reasoning progress.

**Human Cognitive Inspiration**: Studies show that humans maintain higher levels of attention by "posing and solving sub-problems." iQUEST applies this insight to guide LLM reasoning.

## Method

### Overall Architecture

iQUEST consists of three core modules that work collaboratively through an iterative loop:

1. **Iterative Question Guidance (IQG)**: Generates a sub-question at each step that can be answered by the current context.
2. **Two-Hop Entity Exploration**: Uses a GNN to aggregate two-hop neighbor information to select the most relevant entities.
3. **Answer Extraction (AE)**: Answers the sub-question based on retrieved evidence and determines if the information is sufficient.

Iteration workflow: Generate sub-question $\rightarrow$ Retrieve one-hop neighbors $\rightarrow$ GNN evaluation (including two-hop) $\rightarrow$ Answer sub-question $\rightarrow$ Update context $\rightarrow$ Determine sufficiency.

### Key Designs

**1. Iterative Question Guidance**

Key difference from traditional question decomposition: Instead of performing a one-time static split of the original question, it **dynamically generates new sub-questions based on the current reasoning state** at each step:

$$Q_{\text{sub}}^{(n)} = \text{IQG-LLM}(Q, \mathcal{C})$$

where the context $\mathcal{C} = [Q_{\text{sub}}^{(1)}, A_{\text{sub}}^{(1)}, Q_{\text{sub}}^{(2)}, \dots, A_{\text{sub}}^{(n-1)}]$ contains all answered sub-questions and answers. At each step, the LLM simultaneously decides whether further decomposition is needed; if not, the current sub-question is directly used for KG exploration.

**Example**: Original question "What is the official flower of the area impacted by Tropical Storm Fabio in the North Pacific region?" $\rightarrow$ Step 1 sub-question: "Which area in the North Pacific region was impacted by Tropical Storm Fabio?" $\rightarrow$ Upon obtaining the answer $\rightarrow$ Step 2 sub-question: "What is the official flower of that area?"

**2. Two-Hop Entity Exploration with GNN**

**Step A: Neighbor Retrieval** — Retrieve all one-hop neighbors of the current entity via SPARQL templates.

**Step B: GNN Aggregation of Two-Hop Information** — Execute another SPARQL query for each one-hop neighbor to obtain its two-hop neighbors, and then update the one-hop neighbor representation using a GraphSAGE-style aggregation:

$$\hat{\mathbf{h}}_{1h} = \sigma\left(\mathbf{W} \cdot [\mathbf{h}_{1h} \| \text{AGG}\{\mathbf{h}_{2h} \mid e_{2h} \in \mathcal{N}(e_{1h})\}]\right)$$

- $\mathbf{h}_{1h}$/$\mathbf{h}_{2h}$: BERT-encoded representations of one-hop/two-hop neighbors.
- AGG: Mean-pooling aggregation (applicable to an arbitrary number of neighbors).
- The concatenation operation preserves the original information of the center node, preventing identity dilution.

**Step C: Relevance Classification** — Concatenate the updated representation $\hat{\mathbf{h}}_{1h}$ with the sub-question representation and perform binary classification via a two-layer MLP:

$$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{W}_2 \sigma(\mathbf{W}_1 \mathbf{h} + \mathbf{b}_1) + \mathbf{b}_2)$$

The probability of the "relevant" class is taken as the score, and the top-$k$ ($k=3$) entities are selected as supporting evidence.

**3. Answer Extraction LLM (AE-LLM)**

Based on the selected top-$k$ entities and the sub-question, the LLM generates an intermediate answer, which is added to the context $\mathcal{C}$. The LLM then determines whether sufficient information exists to directly answer the original question. If sufficient, it integrates all sub-questions and answers to generate the final response.

### Loss & Training

- **GNN Training**: Cross-entropy loss $L = -\sum_{i=1}^{2} y_i \log(\hat{y}_i)$, with training data derived from single-hop reasoning samples and negative samples generated via random sampling.
- **Encoder**: `bert-base-uncased` (hidden dimension of 768), GNN hidden dimension of 128.
- **No LLM Fine-Tuning**: Both IQG-LLM and AE-LLM utilize off-the-shelf APIs (GPT-4o / DeepSeek-R1 / LLaMA).

## Key Experimental Results

### Main Results: Four KBQA Benchmarks (Hit@1)

| Method | CWQ | WebQSP | WebQuestions | GrailQA |
|------|-----|--------|-------------|---------|
| Interactive-KBQA | 49.07 | 71.20 | - | - |
| ToG (GPT-4) | 69.50 | 82.60 | 57.90 | 81.40 |
| KG-CoT | 62.30 | 84.90 | 68.00 | - |
| Chain-of-Question | 78.80 | 78.10 | - | - |
| **iQUEST (GPT-4o)** | **73.85** | **88.93** | **81.23** | 73.52 |

SOTA on WebQSP (88.93) and WebQuestions (81.23), and ranked second on CWQ and GrailQA.

### Ablation Study: Consistent Improvement from GNN

| Model Combination | CWQ | WebQSP | WebQuestions | GrailQA |
|----------|-----|--------|-------------|---------|
| LLaMA 3B + GPT-4o (IQG) | 20.14 | 40.42 | 48.65 | 32.87 |
| + GNN | 23.66 (+3.52) | 43.73 (+3.31) | 50.11 (+1.46) | 34.19 (+1.32) |
| GPT-4o + GPT-4o (IQG) | 68.42 | 88.10 | 80.20 | 69.30 |
| + GNN | **73.85 (+5.43)** | 88.93 (+0.83) | 81.23 (+1.03) | **73.52 (+4.22)** |

GNN brings consistent improvements across all model combinations and datasets.

### AE-LLM Internal Knowledge vs. Reasoning Ability

| AE-LLM (IQG=GPT-4o, +GNN) | CWQ | WebQSP |
|---------------------------|-----|--------|
| DeepSeek-R1 (Strong Reasoning) | 55.64 | 83.21 |
| LLaMA 70B (Weak Reasoning) | 50.30 (-5.34) | 84.36 (+1.15) |
| GPT-4o (Strong Knowledge) | **73.85 (+18.21)** | **88.93 (+5.72)** |

AE-LLM is more dependent on internal knowledge (GPT-4o achieves +18%) rather than reasoning capability.

### Key Findings

- **Diminishing Returns in IQG**: Once the reasoning capability of IQG-LLM surpasses a certain threshold, the marginal gain diminishes rapidly.
- **KG Can Be Harmful to Small Models**: LLaMA 3B + KG results in performance degradation, as small models struggle to effectively integrate external knowledge.
- **WebQuestions +23% vs ToG**: Benefiting from continuous guidance of the original question, which effectively handles query ambiguity.

## Highlights & Insights

1. **Paradigm Shift of "Guidance over Decomposition"**: Dynamically generating sub-questions at each step and adaptively adjusting based on the reasoning state aligns closer with human cognitive patterns.
2. **"Look-Ahead" Reasoning via GNN**: Aggregating two-hop neighbor info enables the model to consider potential subsequent paths, avoiding short-sighted decisions.
3. **Exquisite Ablation Design**: Distinguishing the contributions of reasoning ability vs. internal knowledge based on hypotheses, leading to the clear conclusion that "IQG shares the reasoning load, while AE relies on knowledge."
4. **Lightweight Training**: Only the GNN is trained, while the LLM is completely fine-tuning-free.

## Limitations & Future Work

1. **Computational Overhead of Dual LLMs**: Two LLM calls (IQG + AE) increase latency and costs.
2. **GNN Limited to Two Hops**: This may be insufficient for domains requiring deeper multi-hop reasoning.
3. **Only Validated on Freebase**: Adaptation to other KGs like Wikidata remains untested.
4. **Small Models Fail to Utilise KG Effectively**: This restricts applications in low-resource scenarios.

## Related Work & Insights

- **KBQA Methods**: iQUEST combines IR-based entity exploration with SP-based SPARQL querying.
- **Question Decomposition**: Methods like CoQ focus on static decomposition, whereas iQUEST shifts towards dynamic guidance.
- **Insights**: (1) Iterative sub-question generation can be generalized to multi-step reasoning NLP tasks; (2) The GNN look-ahead mechanism can be applied to graph-based retrieval-augmented generation (RAG) scenarios.

## Rating

| Dimension | Score (1-10) | Description |
|------|------------|------|
| Novelty | 8 | Creative combination design of iterative guidance and GNN look-ahead. |
| Technical Depth | 7 | Reasonable GNN design; in-depth and convincing ablation studies. |
| Experimental Thoroughness | 8 | 4 datasets x 4 LLMs, comprehensive ablation studies. |
| Writing Quality | 7 | Clear structure, professional hypothesis-validation-styled ablation. |
| Value | 7 | No fine-tuning of LLM required, directly integrable. |
| Overall Score | 7.5 | Novel and effective method, with an ablation design worth learning from. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2025\] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] Beyond Prompting: An Efficient Embedding Framework for Open-Domain Question Answering](embqa_embedding_odqa.md)
- [\[ACL 2025\] A Comprehensive Graph Framework for Question Answering with Mode-Seeking Preference Alignment](a_comprehensive_graph_framework_for_question_answering_with_mode-seeking_prefere.md)
- [\[ACL 2025\] Active LLMs for Multi-hop Question Answering](active_llms_for_multi-hop_question_answering.md)

</div>

<!-- RELATED:END -->

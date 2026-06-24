---
title: >-
  [Paper Note] How Retrieved Context Shapes Internal Representations in RAG
description: >-
  [ACL 2026 Findings][Information Retrieval & RAG][Retrieval-Augmented Generation] This paper systematically analyzes how retrieved documents in RAG influence the internal states of LLMs from the perspective of hidden representations. It identifies five key patterns: random documents induce large representation drifts and trigger refusal behaviors; relevant documents primarily confirm rather than alter parametric knowledge; a single relevant document can anchor representations…
tags:
  - "ACL 2026 Findings"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Hidden Representations"
  - "Representation Drift"
  - "Knowledge Conflict"
  - "Interpretability"
date: 2026-05-08
content_hash: 6d967d1f598b3b70
---

# How Retrieved Context Shapes Internal Representations in RAG

**Conference**: ACL 2026 Findings  
**arXiv**: [2602.20091](https://arxiv.org/abs/2602.20091)  
**Code**: None  
**Area**: Information Retrieval / RAG  
**Keywords**: Retrieval-Augmented Generation, Hidden Representations, Representation Drift, Knowledge Conflict, Interpretability

## TL;DR

This paper systematically analyzes how retrieved documents in RAG influence the internal states of LLMs from the perspective of hidden representations. It identifies five key patterns: random documents induce large representation drifts and trigger refusal behaviors; relevant documents primarily confirm rather than alter parametric knowledge; a single relevant document can anchor representations in multi-document scenarios; later layers progressively emphasize parametric knowledge, thereby limiting the impact of retrieved evidence; and LLMs can distinguish random documents in early layers but remain unable to reliably differentiate between distractor and relevant documents even in the final layers.

## Background & Motivation

**Background**: RAG has become the mainstream approach for enhancing LLMs, improving factual accuracy by introducing external documents during generation. However, retrieved document sets typically contain a mix of varying relevance and utility.

**Limitations of Prior Work**: (1) Existing RAG research primarily analyzes output behavior (accuracy, hallucination rates), failing to distinguish whether performance changes stem from effective evidence integration, suppression of parametric knowledge, or uncertainty responses from the model; (2) It remains unclear how retrieved documents are processed internally—whether they are integrated into reasoning or ignored; (3) There is a lack of systematic research on how different types of documents (relevant/distractor/random) affect internal representations.

**Key Challenge**: Observing RAG behavior only at the output level is like looking into a black box—identical erroneous outputs might result from completely different internal processing mechanisms. Understanding RAG requires a deep dive into the internal representation level.

**Goal**: To systematically analyze how different types of retrieved documents affect LLM hidden states and how internal representation changes correlate with downstream generation behavior.

**Key Insight**: Using controlled experiments—fixing the RAG pipeline while systematically varying the relevance types (relevant/distractor/random) and quantity combinations of documents to compare hidden representation differences with and without context.

**Core Idea**: The influence of retrieved documents on LLM internal representations is far more subtle than what is observed at the output level—relevant documents barely change representations (merely confirming existing knowledge), while random documents trigger the largest representation drifts (triggering refusal modes).

## Method

### Overall Architecture

This paper is a mechanistic analysis that does not train models but opens the RAG black box at the level of hidden representations using controlled experiments. After fixing the RAG pipeline, the relevance types (relevant / distractor / random) and quantity combinations of documents are systematically varied for each query. The hidden states $h \in \mathbb{R}^{L \times D}$ of the "last prompt token" are extracted across all layers to compare the representation offset between context-inclusive and context-free scenarios. The workflow proceeds from query + retrieved documents through a retrieval layer (MassiveDS library + Contriever for top-20, classified by GPT-5), followed by PCA visualization, cosine similarity, and representation separability to output "which types of documents change internal states at which layers." Experiments span 4 QA datasets (TriviaQA, NQ, PopQA, StrategyQA) $\times$ 3 LLMs (Gemma3-27B, Llama4-17B, Qwen3-Next-80B).

### Key Designs

**1. Controlled Document Taxonomy: Deconstructing the "Retrieval Mixture" into Three Isolable Independent Variables**

Documents retrieved in real-world scenarios are a mixture of relevant, distractor, and random types. Without deconstruction, representation changes cannot be attributed to specific document types. This paper strictly defines three categories: relevant documents contain the ground-truth answer or direct supporting information; distractor documents are semantically similar to the query but lack supporting information and may mislead the model; random documents have low semantic similarity and are completely useless. Classification is performed by GPT-5 and manually verified, forming the prerequisite for all subsequent mechanistic conclusions.

**2. Query Difficulty Stratification: Distinguishing Between "Known" and "Retrieval-Dependent" Scenarios**

The processing mechanisms for retrieved documents may differ significantly when a model possesses parametric knowledge versus when it lacks it. Aggregating these would obscure such differences. This paper first tests each query using a query-only (no retrieval) setup: those answered correctly are labeled "easy," otherwise "hard." This allows for separate investigation of how retrieved documents affect "known knowledge" versus "unknown knowledge"—later revealing that random documents can push "easy" queries toward refusal, a distinction crucial for observing this phenomenon.

**3. Layer-wise Representation Analysis: Observing Information Integration or Overriding Through Transformer Layers**

Different Transformer layers serve different functions: shallow layers handle lexical processing, middle layers perform semantic integration, and deep layers handle decision-making. Output analysis alone cannot reveal where retrieved evidence takes effect. This paper extracts the hidden state of the last prompt token at every layer, using PCA to visualize representation distributions across context types and tracking evolution trajectories from shallow to deep. This analysis reveals key observations: "early layers (around L23) can separate random documents, but relevant and distractor documents remain mixed until the final layer," and "later layers progressively pull relevant document representations back toward context-free representations, indicating that deeper layers emphasize parametric knowledge."

### Loss & Training

Ours is an analytical work and does not involve model training. It utilizes pre-trained LLMs (both instruction-tuned and base versions) for inference and representation extraction.

## Key Experimental Results

### Main Results

**Impact of Different Context Types on Response Behavior (Gemma3-27B, TriviaQA)**

| Context Type | Easy Accuracy | Hard Accuracy | Easy Refusal Rate | Hard Refusal Rate |
| :--- | :--- | :--- | :--- | :--- |
| No Context | ~90% | ~0% | ~3% | ~12% |
| Relevant Docs | 90.4% | 65.2% | 3.1% | 7.0% |
| Distractor Docs | 8.5% | 0.7% | 61.8% | 74.2% |
| Random Docs | 1.7% | 0% | 97.6% | 98.1% |

**Performance in Multi-Document Scenarios**

| Context Combination | Easy Accuracy | Hard Accuracy |
| :--- | :--- | :--- |
| Relevant Only | 90.4% | 65.2% |
| 1 Rel. + 3 Dist. | 82.6% | 57.1% |
| 1 Rel. + 3 Rand. | 87.7% | 60.2% |
| Distractor Only | 8.5% | 0.7% |
| Random Only | 1.7% | 0% |

### Ablation Study

| Observation | Finding | Practical Implication |
| :--- | :--- | :--- |
| Base vs. IT | Base models show no representation drift, refusal rate <20% | Refusal behavior is a byproduct of instruction tuning |
| 20 Docs No Filter | Accuracy close to relevant docs only | LLMs can autonomously suppress noise |
| Layer-wise Analysis | No diff at L12 $\rightarrow$ Random separable at L23 $\rightarrow$ Rel/Dist still mixed at L35 | Semantic differentiation processes from coarse to fine |

### Key Findings

- **Observation 1**: Random documents induce the largest representation drift (counter-intuitive), which strongly correlates with frequent refusal behaviors. Instruction tuning amplifies this effect.
- **Observation 2**: Relevant documents barely change representations—they primarily act as confirmation signals to increase confidence (significant increase in log-likelihood) rather than introducing new information.
- **Observation 3**: In multi-document scenarios, a single relevant document can anchor the representation, suppressing the influence of additional noise.
- **Observation 4**: Early layers first distinguish random documents (L23), while relevant and distractor documents remain difficult to separate fully even in the final layers.
- **Observation 5**: Later layers progressively pull relevant document representations toward context-free representations, suggesting that deeper layers place more weight on parametric knowledge.

## Highlights & Insights

- Explains the RAG mechanism from a representation perspective—relevant documents act as "confirmers" rather than "information injectors," shifting the understanding of how RAG functions.
- Identifies that the refusal behavior introduced by instruction tuning is a double-edged sword: it protects against random documents but also causes the model to refuse "easy" queries that it could otherwise answer due to random context.
- Practical insight: Increasing retrieval breadth is safe—as long as one relevant document is present, the model can suppress noise, eliminating the need for aggressive document filtering.

## Limitations & Future Work

- The analysis is primarily based on QA tasks; its applicability to tasks like long-text generation has not been verified.
- Using GPT-5 for document classification might introduce systematic bias.
- The study did not analyze how the attention mechanism implements the anchoring effect for representations.
- Future research could investigate how to use representation drift signals for automatic detection of retrieval quality.

## Related Work & Insights

- **vs. Wadhwa et al. (2024)**: They probed internal representation biases when relevant context is present but did not study real-world scenarios with mixed document types.
- **vs. Shi et al. (2023)**: They analyzed the interference effect of noisy documents via output behavior; this work provides a mechanistic explanation from the representation level.
- **vs. Liu et al. (2024)**: They found that LLMs are sensitive to document order; this work finds that such sensitivity can be offset by a single relevant document.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic analysis of RAG from the perspective of hidden representations; all five observations are new findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered four datasets, three models, and single/multi-document settings, but lacks non-QA tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear analytical logic with well-articulated practical implications of the findings.
- Value: ⭐⭐⭐⭐⭐ Provides a theoretical foundation at the representation level and practical guidance for RAG system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)
- [\[ICLR 2026\] The Topology of Reasoning: Augmenting Generation with Retrieved Cell Complexes for Text-Graph QA](../../ICLR2026/information_retrieval/topology_of_reasoning_retrieved_cell_complex-augmented_generation_for_textual_gr.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)
- [\[ICML 2026\] How can embedding models bind concepts?](../../ICML2026/information_retrieval/how_can_embedding_models_bind_concepts.md)

</div>

<!-- RELATED:END -->

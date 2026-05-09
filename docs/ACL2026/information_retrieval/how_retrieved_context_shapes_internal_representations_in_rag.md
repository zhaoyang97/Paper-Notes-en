---
title: >-
  [Paper Note] How Retrieved Context Shapes Internal Representations in RAG
description: >-
  [ACL 2026][Retrieval-Augmented Generation] This paper systematically analyzes how retrieved documents influence the internal states of LLMs in RAG from the perspective of hidden representations, identifying five key patterns: random documents induce large representation drift and trigger refusal behavior; relevant documents primarily confirm rather than alter parametric knowledge; a single relevant document can anchor representations in multi-document settings; later layers progressively emphasize parametric knowledge, thereby limiting the influence of retrieved evidence; and LLMs can distinguish random documents in early layers but fail to reliably separate distractor documents from relevant ones even at the final layer.
tags:
  - ACL 2026
  - Retrieval-Augmented Generation
  - hidden representations
  - representation drift
  - knowledge conflict
  - interpretability
date: 2026-05-08
content_hash: 079b6c44b472be07
---

# How Retrieved Context Shapes Internal Representations in RAG

**Conference**: ACL 2026
**arXiv**: [2602.20091](https://arxiv.org/abs/2602.20091)
**Code**: None
**Area**: Information Retrieval / RAG
**Keywords**: Retrieval-Augmented Generation, hidden representations, representation drift, knowledge conflict, interpretability

## TL;DR

This paper systematically analyzes how retrieved documents influence the internal states of LLMs in RAG from the perspective of hidden representations, identifying five key patterns: random documents induce large representation drift and trigger refusal behavior; relevant documents primarily confirm rather than alter parametric knowledge; a single relevant document can anchor representations in multi-document settings; later layers progressively emphasize parametric knowledge, thereby limiting the influence of retrieved evidence; and LLMs can distinguish random documents in early layers but fail to reliably separate distractor documents from relevant ones even at the final layer.

## Background & Motivation

**Background**: RAG has become a mainstream approach for augmenting LLMs, improving factual accuracy by incorporating external documents at generation time. However, retrieved document sets typically contain a mixture of content with varying relevance and utility.

**Limitations of Prior Work**: (1) Existing RAG research primarily analyzes output behavior (accuracy, hallucination rate), making it impossible to distinguish whether performance changes stem from effective evidence integration, suppression of parametric knowledge, or the model's uncertainty response. (2) It remains unclear how retrieved documents are processed internally within LLMs—whether they are integrated into reasoning or ignored. (3) There is no systematic study of how different document types (relevant/distractor/random) affect internal representations.

**Key Challenge**: Observing RAG behavior solely at the output level is akin to treating the model as a black box—identical erroneous outputs may arise from entirely different internal processing mechanisms. Understanding RAG requires examination at the level of internal representations.

**Goal**: To systematically analyze how different types of retrieved documents influence LLM hidden states, and how changes in internal representations relate to downstream generation behavior.

**Key Insight**: Controlled experiments are employed—fixing the RAG pipeline while systematically varying document relevance types (relevant/distractor/random) and quantity combinations, and comparing hidden representations with and without context.

**Core Idea**: The influence of retrieved documents on LLM internal representations is far more subtle than what is observable at the output level—relevant documents barely alter representations (merely confirming existing knowledge), whereas random documents induce the largest representation drift (triggering refusal patterns).

## Method

### Overall Architecture

The analytical framework comprises: (1) Data layer—four QA datasets (TriviaQA, NQ, PopQA, StrategyQA) and three LLMs (Gemma3-27B, Llama4-17B, Qwen3-Next-80B); (2) Retrieval layer—MassiveDS corpus (1.4 trillion tokens) with a Contriever retriever, retrieving top-20 documents per query and classifying them as relevant/distractor/random using GPT-5; (3) Analysis layer—extracting layer-wise hidden states of the last prompt token $h \in \mathbb{R}^{L \times D}$, with PCA visualization, cosine similarity, and representational separability used for analysis.

### Key Designs

1. **Controlled Document Classification Scheme**:

    - Function: Partitions retrieved documents into three categories to isolate the effects of different context types.
    - Mechanism: Relevant documents (containing the ground-truth answer or information directly supporting it), distractor documents (semantically similar to the query but lacking information that supports deriving the answer, potentially misleading the model), and random documents (low semantic similarity to the query, containing no useful information). Classification is performed by GPT-5 and validated by human annotators.
    - Design Motivation: All three document types co-occur in real retrieval scenarios. Isolating the effect of each type is a prerequisite for understanding RAG mechanisms.

2. **Query Difficulty Stratification**:

    - Function: Distinguishes easy queries (answerable without retrieval) from hard queries (requiring retrieval).
    - Mechanism: Each query is first tested with the model using only the query (no retrieval); queries answered correctly are labeled easy, others hard. This enables analysis of how retrieved documents differentially affect "known" versus "unknown" knowledge.
    - Design Motivation: The processing mechanism for retrieved documents may differ fundamentally depending on whether the model already possesses the relevant parametric knowledge.

3. **Layer-wise Representation Analysis**:

    - Function: Reveals how different layers process retrieved context.
    - Mechanism: Hidden states of the last prompt token are extracted at each layer, and PCA is used to visualize the representational distribution across different context types. The evolution of representations from shallow to deep layers is tracked.
    - Design Motivation: Different Transformer layers serve distinct functions—early layers handle lexical processing, middle layers perform semantic integration, and later layers make decisions. Layer-wise analysis reveals where retrieved information is integrated or overridden.

### Loss & Training

This is an analytical study and involves no model training. Pre-trained LLMs (both instruction-tuned and base versions) are used for inference and representation extraction.

## Key Experimental Results

### Main Results

**Effect of Different Context Types on Response Behavior (Gemma3-27B, TriviaQA)**

| Context Type | Easy Accuracy | Hard Accuracy | Easy Refusal Rate | Hard Refusal Rate |
|---|---|---|---|---|
| No context | ~90% | ~0% | ~3% | ~12% |
| Relevant document | 90.4% | 65.2% | 3.1% | 7.0% |
| Distractor document | 8.5% | 0.7% | 61.8% | 74.2% |
| Random document | 1.7% | 0% | 97.6% | 98.1% |

**Performance in Multi-Document Settings**

| Context Combination | Easy Accuracy | Hard Accuracy |
|---|---|---|
| Relevant only | 90.4% | 65.2% |
| 1 relevant + 3 distractors | 82.6% | 57.1% |
| 1 relevant + 3 random | 87.7% | 60.2% |
| Distractor only | 8.5% | 0.7% |
| Random only | 1.7% | 0% |

### Ablation Study

| Observation | Finding | Practical Implication |
|---|---|---|
| Base vs. instruction-tuned | Base model shows no representation drift; refusal rate <20% | Refusal behavior is a product of instruction tuning |
| 20 documents without filtering | Accuracy approaches relevant-document-only setting | LLMs can autonomously suppress noisy documents |
| Layer-wise analysis | L12: no difference → L23: random separable → L35: relevant/distractor still mixed | Semantic discrimination proceeds from coarse to fine |

### Key Findings

- **Observation 1**: Random documents induce the largest representation drift (counterintuitively), strongly correlating with high model refusal rates. Instruction tuning amplifies this effect.
- **Observation 2**: Relevant documents barely alter representations—they primarily function as confirmation signals that increase confidence (log-likelihood improves significantly) rather than introducing new information.
- **Observation 3**: In multi-document settings, a single relevant document suffices to anchor representations and suppress the influence of additional noise.
- **Observation 4**: Early layers first discriminate random documents (L23), while relevant and distractor documents remain difficult to fully separate even at the final layer.
- **Observation 5**: Later layers progressively pull relevant-document representations toward the no-document representation, indicating that deep layers place greater emphasis on parametric knowledge.

## Highlights & Insights

- The paper provides a representation-level explanation of how RAG operates—relevant documents act as "confirmers" rather than "information injectors," which revises the prevailing understanding of how RAG functions.
- The refusal behavior introduced by instruction tuning is shown to be a double-edged sword: it offers protection against random documents but also causes the model to refuse answering easy queries that it could otherwise answer correctly when random context is present.
- Practical implication: increasing retrieval breadth is safe—as long as one relevant document is present, the model can suppress noise without requiring aggressive document filtering.

## Limitations & Future Work

- The analysis is primarily conducted on QA tasks; applicability to other tasks such as long-form generation remains unvalidated.
- Using GPT-5 for document classification may introduce systematic bias.
- The attention mechanisms underlying the representational anchoring effect are not analyzed.
- Future work could investigate how representation drift signals can be used to automatically assess retrieval quality.

## Related Work & Insights

- **vs. Wadhwa et al. (2024)**: That work probes internal representational biases in the presence of relevant context but does not study real-world scenarios with mixed document types.
- **vs. Shi et al. (2023)**: That work analyzes the interference effects of noisy documents from the perspective of output behavior; this paper provides a mechanistic explanation at the representation level.
- **vs. Liu et al. (2024)**: That work finds LLMs are sensitive to document ordering; this paper shows that such sensitivity can be offset by a single relevant document.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First systematic analysis of RAG from the perspective of hidden representations; all five observations are original findings.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four datasets, three models, and single/multi-document settings, but lacks non-QA tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Analytical logic is clear, and the practical implications of the findings are thoroughly articulated.
- Value: ⭐⭐⭐⭐⭐ Provides a representation-level theoretical foundation and practical guidance for RAG system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] RepoShapley: Shapley-Enhanced Context Filtering for Repository-Level Code Completion](reposhapley_shapley-enhanced_context_filtering_for_repository-level_code_complet.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[ICLR 2026\] Q-RAG: Long Context Multi-Step Retrieval via Value-Based Embedder Training](../../ICLR2026/information_retrieval/q_rag_long_context_multi_step_retrieval.md)
- [\[ICLR 2026\] Beyond RAG vs. Long-Context: Learning Distraction-Aware Retrieval for Efficient Knowledge Grounding](../../ICLR2026/information_retrieval/beyond_rag_vs_long-context_learning_distraction-aware_retrieval_for_efficient_kn.md)

</div>

<!-- RELATED:END -->

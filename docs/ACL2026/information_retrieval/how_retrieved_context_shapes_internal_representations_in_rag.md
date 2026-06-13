---
title: >-
  [Paper Note] How Retrieved Context Shapes Internal Representations in RAG
description: >-
  [ACL 2026][Information Retrieval & RAG][Retrieval-Augmented Generation] This paper systematically analyzes how retrieved documents affect the internal states of LLMs from the perspective of hidden representations. It ide…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "Retrieval-Augmented Generation"
  - "Hidden Representations"
  - "Representation Drift"
  - "Knowledge Conflict"
  - "Interpretability"
date: 2026-05-08
content_hash: cb29b91a4bc1aa91
---

# How Retrieved Context Shapes Internal Representations in RAG

**Conference**: ACL 2026 Findings  
**arXiv**: [2602.20091](https://arxiv.org/abs/2602.20091)  
**Code**: None  
**Area**: Information Retrieval / RAG  
**Keywords**: Retrieval-Augmented Generation, Hidden Representations, Representation Drift, Knowledge Conflict, Interpretability

## TL;DR

This paper systematically analyzes how retrieved documents affect the internal states of LLMs from the perspective of hidden representations. It identifies five key patterns: random documents cause significant representation drift and trigger refusal behavior; relevant documents primarily confirm rather than alter parametric knowledge; a single relevant document can anchor the representation in multi-document scenarios; later layers gradually emphasize parametric knowledge, thereby limiting the influence of retrieved evidence; and LLMs can distinguish random documents in early layers but remain unable to reliably differentiate between distractor and relevant documents even in the final layers.

## Background & Motivation

**Background**: RAG has become a mainstream method for enhancing LLMs, improving factual accuracy by introducing external documents during generation. However, retrieved document sets typically contain a mixture of content with varying relevance and utility.

**Limitations of Prior Work**: (1) Existing RAG research primarily analyzes output behavior (accuracy, hallucination rates), which fails to distinguish whether performance changes stem from effective evidence integration, suppression of parametric knowledge, or uncertainty-based responses; (2) it is unclear how retrieved documents are processed internally—whether they are integrated into reasoning or ignored; (3) there is a lack of systematic research on how different types of documents (relevant/distractor/random) affect internal representations.

**Key Challenge**: Observing RAG behavior only at the output level is like looking at a black box—identical erroneous outputs might arise from completely different internal processing mechanisms. Understanding RAG requires a deep dive into the internal representation level.

**Goal**: Systematically analyze how different types of retrieved documents influence LLM hidden states and how changes in internal representations correlate with downstream generation behavior.

**Key Insight**: Using controlled experiments—fixing the RAG pipeline while systematically varying the relevance types (relevant/distractor/random) and quantity combinations of documents—to compare the differences in hidden representations with and without context.

**Core Idea**: The influence of retrieved documents on internal LLM representations is far more subtle than observed at the output level—relevant documents barely change the representation (merely confirming existing knowledge), whereas random documents instead cause the largest representation drift (triggering refusal modes).

## Method

### Overall Architecture

The analysis framework includes: (1) Data layer—four QA datasets (TriviaQA, NQ, PopQA, StrategyQA) and three LLMs (Gemma3-27B, Llama4-17B, Qwen3-Next-80B); (2) Retrieval layer—MassiveDS database (1.4 trillion tokens) + Contriever retriever, retrieving top-20 documents for each query and classifying them as relevant/distractor/random using GPT-5; (3) Analysis layer—extracting hidden states $h \in \mathbb{R}^{L \times D}$ for the last prompt token, utilizing PCA visualization, cosine similarity, and representation separability for analysis.

### Key Designs

1.  **Controlled Document Classification System**:
    *   Function: Categorizes retrieved documents into three classes to isolate the effects of different context types.
    *   Mechanism: Relevant documents (containing the ground truth answer or directly supporting the derivation of the answer), distractor documents (semantically similar to the query but lacking support for deriving the answer, potentially misleading the model), and random documents (low semantic similarity to the query, providing no useful information). Classification was performed by GPT-5 and verified manually.
    *   Design Motivation: Real-world retrieval scenarios involve a mixture of these three types. Separately studying the impact of each type is a prerequisite for understanding RAG mechanisms.

2.  **Query Difficulty Stratification**:
    *   Function: Distinguishes between easy queries (the model can answer without retrieval) and hard queries (retrieval is required).
    *   Mechanism: For each query, the model is first tested with the query alone (no retrieval); those answered correctly are marked as easy, otherwise they are hard. This allows for analysis of the different effects of retrieved documents on "known knowledge" versus "unknown knowledge."
    *   Design Motivation: The mechanism by which a model processes retrieved documents may differ significantly depending on whether it already possesses parametric knowledge.

3.  **Hierarchical Representation Analysis**:
    *   Function: Reveals how different layers process retrieval context.
    *   Mechanism: Hidden states of the last prompt token are extracted at every layer, and the representation distribution of different context types is visualized using PCA. The evolution of representation patterns is tracked from shallow to deep layers.
    *   Design Motivation: Different Transformer layers serve different functions—shallow layers handle lexical processing, middle layers perform semantic integration, and deep layers handle decision-making. Hierarchical analysis reveals where retrieved information is integrated or overridden.

### Loss & Training

This is an analytical work and does not involve model training. Pre-trained LLMs (both instruction-tuned and base versions) were used for inference and representation extraction.

## Key Experimental Results

### Main Results

**Impact of Different Context Types on Response Behavior (Gemma3-27B, TriviaQA)**

| Context Type | Easy Accuracy | Hard Accuracy | Easy Refusal Rate | Hard Refusal Rate |
| :--- | :--- | :--- | :--- | :--- |
| No Context | ~90% | ~0% | ~3% | ~12% |
| Relevant Docs | 90.4% | 65.2% | 3.1% | 7.0% |
| Distractor Docs | 8.5% | 0.7% | 61.8% | 74.2% |
| Random Docs | 1.7% | 0% | 97.6% | 98.1% |

**Performance in Multi-document Scenarios**

| Context Combination | Easy Accuracy | Hard Accuracy |
| :--- | :--- | :--- |
| Relevant Only | 90.4% | 65.2% |
| 1 Relevant + 3 Distractor | 82.6% | 57.1% |
| 1 Relevant + 3 Random | 87.7% | 60.2% |
| Distractor Only | 8.5% | 0.7% |
| Random Only | 1.7% | 0% |

### Ablation Study

| Observation | Finding | Practical Implication |
| :--- | :--- | :--- |
| Base vs. Instruct | Base models show no representation drift; refusal rate <20% | Refusal behavior is a product of instruction tuning |
| 20 Docs No Filtering | Accuracy close to relevant docs only | LLMs can autonomously suppress noise |
| Hierarchical Analysis | No difference at L12 $\rightarrow$ Random separable at L23 $\rightarrow$ Relevant/Distractor still mixed at L35 | Semantic differentiation moves from coarse to fine |

### Key Findings

*   **Observation 1**: Random documents trigger the largest representation drift (counter-intuitive), which is strongly correlated with the model's frequent refusal to answer. Instruction tuning amplifies this effect.
*   **Observation 2**: Relevant documents barely change the representation—they serve primarily as confirmation signals that increase confidence (significant boost in log-likelihood) rather than introducing entirely new information.
*   **Observation 3**: In multi-document scenarios, a single relevant document can anchor the representation, suppressing the influence of additional noise.
*   **Observation 4**: Early layers first distinguish random documents (L23), while relevant and distractor documents remain difficult to separate completely even in the final layers.
*   **Observation 5**: Later layers gradually pull the representation of relevant documents toward the no-context representation, indicating that deeper layers place more emphasis on parametric knowledge.

## Highlights & Insights

*   Explains the working mechanism of RAG from a representation perspective—relevant documents act as "confirmers" rather than "information injectors," shifting the understanding of how RAG functions.
*   Identifies that refusal behavior introduced by instruction tuning is a double-edged sword: it provides protection against random documents but also causes the model to refuse easy queries it could otherwise answer due to random context.
*   Practical revelation: Increasing retrieval breadth is safe—as long as one relevant document is present, the model can suppress noise, reducing the need for aggressive document filtering.

## Limitations & Future Work

*   The analysis is primarily based on QA tasks; the applicability to tasks such as long-form text generation has not been verified.
*   Using GPT-5 for document classification may introduce systematic bias.
*   No analysis was performed on how the attention mechanism achieves the representation anchoring effect.
*   Future research could explore using representation drift signals to automatically detect retrieval quality.

## Related Work & Insights

*   **vs Wadhwa et al. (2024)**: They probed internal representation bias when relevant context is present but did not study real-world scenarios with mixed document types.
*   **vs Shi et al. (2023)**: They analyzed the interference effects of noisy documents from output behavior; this paper provides a mechanistic explanation at the representation level.
*   **vs Liu et al. (2024)**: They found that LLMs are sensitive to document order; this paper finds that such sensitivity can be offset by a single relevant document.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First systematic analysis of RAG from the perspective of hidden representations; all five observations are new findings.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets, three models, single/multi-document settings, but lacks non-QA tasks.
*   Writing Quality: ⭐⭐⭐⭐⭐ Clear analytical logic; practical implications of the findings are well-articulated.
*   Value: ⭐⭐⭐⭐⭐ Provides a theoretical foundation and practical guidance at the representation level for RAG system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Large Language Models Balance Internal Knowledge with User and Document Assertions](how_large_language_models_balance_internal_knowledge_with_user_and_document_asse.md)
- [\[ACL 2026\] Is Agentic RAG Worth It? An Experimental Comparison of RAG Approaches](is_agentic_rag_worth_it_an_experimental_comparison_of_rag_approaches.md)
- [\[ACL 2026\] Context Attribution with Multi-Armed Bandit Optimization](context_attribution_with_multi-armed_bandit_optimization.md)
- [\[ACL 2026\] When Retrieval is Ineffective in Biomedical RAG: A Large-Scale Empirical Study](when_retrieval_doesnt_help_a_large-scale_study_of_biomedical_rag.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] The Distracting Effect: Understanding Irrelevant Passages in RAG
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG] This paper proposes a formal metric for the Distracting Effect (DE) of passages and develops multiple techniques to acquire highly distracting passages (including answer-skewed retrieval and categorized generation). It demonstrates the robustness of this metric across different LLMs, and finally improves QA accuracy by up to 7.5% through fine-tuning LLMs with highly distracting training samples.
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG"
  - "distracting patches"
  - "Retrieval-Augmented Generation"
  - "data augmentation"
  - "robust fine-tuning"
date: 2026-05-08
content_hash: 6a0d7c4456784f1a
---

# The Distracting Effect: Understanding Irrelevant Passages in RAG

**Conference**: ACL 2025  
**arXiv**: [2505.06914](https://arxiv.org/abs/2505.06914)  
**Code**: None  
**Area**: NLP Understanding / Retrieval-Augmented Generation  
**Keywords**: RAG, distracting patches, Retrieval-Augmented Generation, data augmentation, robust fine-tuning

## TL;DR

This paper proposes a formal metric for the Distracting Effect (DE) of passages and develops multiple techniques to acquire highly distracting passages (including answer-skewed retrieval and categorized generation). It demonstrates the robustness of this metric across different LLMs, and finally improves QA accuracy by up to 7.5% through fine-tuning LLMs with highly distracting training samples.

## Background & Motivation

**Background**: Retrieval-Augmented Generation (RAG) is a crucial approach for enabling LLMs to solve knowledge-intensive tasks. Appending retrieved passages to the prompt can effectively reduce hallucinations. However, retrieval is not always successful, and retrieved results often contain distracting passages—those that are semantically relevant to the query but do not contain the correct answer, potentially misleading the LLM.

**Limitations of Prior Work**: (1) The understanding of distracting passages remains at a simple binary classification level (fully irrelevant vs. distracting), lacking a quantitative metric; (2) Existing methods for obtaining distracting passages are limited to the top results of standard retrieval, which may fail to find sufficient distracting passages in small corpora or for specific queries; (3) As retrievers become stronger, the irrelevant results they return actually become more distracting—a problem that intensifies over time.

**Key Challenge**: Stronger retrievers are supposed to bring better RAG performance, but after undergoing stricter filtering by the retriever, irrelevant results become more confusing to LLMs. Meanwhile, there is a lack of a systematic method to quantify and utilize this distracting effect.

**Goal**: (1) How to formally measure the distracting effect of a passage on a specific query; (2) How to systematically acquire highly distracting passages; (3) How to utilize highly distracting passages to improve the robustness of RAG systems.

**Key Insight**: Defining the distracting effect as the probability that the LLM does not choose to abstain (i.e., outputting "NO-RESPONSE") when given only the query and the passage, which is both a simple and effective quantitative indicator.

**Core Idea**: By quantifying the distracting effect score of passages and combining multiple acquisition methods (standard retrieval + answer-skewed retrieval + categorized generation), a highly distracting training set is constructed to fine-tune LLMs for enhanced RAG robustness.

## Method

### Overall Architecture

The framework consists of three parts: (1) defining and computing the distracting effect metric; (2) acquiring highly distracting passages through retrieval and generation methods; (3) constructing training sets with the acquired highly distracting passages to fine-tune LLMs for enhanced QA robustness.

### Key Designs

1. **Distracting Effect (DE) Metric**:

    - **Function**: Quantifies the extent to which an irrelevant passage distracts an LLM with respect to a specific query.
    - **Mechanism**: A prompt is constructed to instruct the LLM to answer query $q$ based on passage $p$, outputting "NO-RESPONSE" if the passage does not contain the answer. The distracting effect is defined as $DE_q(p) = 1 - p^{LLM}(\text{NO-RESPONSE}|q,p)$, which is the probability of the LLM not choosing to abstain. The score ranges from 0 to 1, where higher scores indicate the passage is more likely to trick the LLM. Generating the full answer is unnecessary; only the probability of the first token is checked, keeping computational cost low.
    - **Design Motivation**: This metric leverages the LLM's own capability to identify relevant information, does not rely on an external reference model, does not require assuming the parameterized memory of the model, and is applicable to any RAG task beyond question answering.

2. **Answer-Skewed Retrieval**:

    - **Function**: Retrieves passages that are relevant to the query but irrelevant to the answer.
    - **Mechanism**: Modifies the query embedding of a dense retriever by subtracting the answer's information. Two variants are proposed: subtraction $E^{sub}(q,a) = E_Q(q) - \lambda E_D(a)$ directly subtracts the answer embedding; projection $E^{proj}(q,a) = E_Q(q) - \lambda \frac{\langle E_Q(q), E_D(a) \rangle E_D(a)}{\|E_D(a)\|^2}$ projects out the component in the direction of the answer. The hyperparameter $\lambda$ controls the strength of excluding the answer information. Retrieved results are further filtered by an NLI model to exclude passages containing the correct answer.
    - **Design Motivation**: Top results of standard retrieval may contain correct answers or highly relevant passages. Answer-skewed retrieval actively searches for passages topically relevant to the query but devoid of the answer, increasing diversity in acquiring distracting passages.

3. **Categorized Generation**:

    - **Function**: Generates different types of distracting passages using LLMs, covering scenarios that retrieval cannot reach.
    - **Mechanism**: Defines four types of distracting passages, each guided by few-shot prompts using Claude 3.5 Sonnet for generation: (1) **Related Topic ($G^{rel}$)**: Discusses a highly related topic but does not contain the answer (e.g., asking for Lincoln's birthday $\rightarrow$ giving his son Robert's birthday); (2) **Hypothetical ($G^{hypo}$)**: Gives a different answer in a hypothetical setting (e.g., "In ancient Roman times..."); (3) **Negation ($G^{neg}$)**: Provides incorrect answers in a negative form (e.g., "A common misconception is that..."); (4) **Modal Statement ($G^{modal}$)**: Provides incorrect answers in an uncertain tone (e.g., "The pyramid might have been...").
    - **Design Motivation**: In small corpora or specific topic queries, retrieval may fail to find distracting passages. Generation methods can synthesize distracting passages for any query, and different types cover various distracting mechanisms.

### Loss & Training

Standard instruction-tuning loss is used to fine-tune Llama-3.2-3B and Llama-3.1-8B. Training set construction strategy "Hard": 50% of the samples contain 1 relevant passage + 4 of the most distracting passages, and 50% contain 5 highly distracting passages (no relevant passage). The five passages are randomly shuffled. Baseline comparisons "Retrieve" and "Rerank" use the top-5 results of standard retrieval.

## Key Experimental Results

### Main Results

| Dataset | Fine-Tuning Strategy | Llama-3.2-3B acc | Llama-3.1-8B acc |
|-------|---------|-----------------|-----------------|
| NQ | None | 37.9 | 40.3 |
| NQ | Retrieve | 40.7 | 46.9 |
| NQ | Rerank | 39.7 | 47.0 |
| NQ | **Hard** | **42.8** | **49.4** |
| TriviaQA | None | 67.8 | 73.5 |
| TriviaQA | Retrieve | 67.6 | 78.7 |
| TriviaQA | **Hard** | **74.5** | **82.0** |
| WebQA | None | 41.9 | 40.6 |
| WebQA | Retrieve | 42.1 | 48.0 |
| WebQA | **Hard** | **49.7** | **51.0** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Gold Passage + Weak Distracting (DE<0.2) | Accuracy drops by 0.5-4.4% | Weakly distracting passages have limited influence |
| Gold Passage + Strong Distracting (DE>0.8) | Accuracy drops by 6-11% | Strongly distracting passages significantly degrade performance |
| Cross-LLM DE Correlation | High Spearman correlation coefficient | Distracting effect is an intrinsic attribute of passages |
| Optimal proportion of each method | R+st 52%, Gmodal ~15% | Joint use covers more queries |
| Ungrounded instances Hard vs Retrieve | +5.3-16.1% (3B) | Highly distracting training shows huge improvement in scenarios without gold passages |

### Key Findings

- **Distracting effect is highly consistent across LLMs**: DE scores from different LLMs (3B to 70B parameters) exhibit a strong Spearman correlation, indicating that the distracting effect depends primarily on the passage itself rather than the model.
- **Irrelevant passages returned by stronger retrievers + rerankers are more distracting**: The top-1 irrelevant passage from standard retrieval + reranking (R+st) exhibits the highest distracting effect on LLMs.
- **Joint usage of all methods (retrieval + skewed + synthesized) finds more distracting passages** than standard retrieval for approximately 48% of the queries.
- **Modal-type synthesized passages ($G^{modal}$)** exhibit the highest average distracting effect, while the **Related Topic ($G^{rel}$)** class has the lowest.
- **Hard fine-tuning yields the most significant improvement in ungrounded scenarios (no gold passages)** (5.3-16.1% improvement for the 3B model), because the model relies entirely on parametric memory in these cases and is easily misled by distracting passages.

## Highlights & Insights

- Elevates the concept of distracting effects from a binary classification to a continuous metric, offering a more fine-grained tool for understanding.
- Reveals the counterintuitive phenomenon where "stronger retrievers lead to more distracting irrelevant results", providing key insights for RAG system design.
- Clever design of answer-skewed retrieval: subtracts the answer direction in the embedding space, using vector operations to capture the semantic of "relevant to the query but irrelevant to the answer".
- The categorization of four distracting passage types (Related/Hypothetical/Negation/Modal) provides a valuable perspective on understanding LLM vulnerabilities.

## Limitations & Future Work

- The four synthesized passage types may not cover all forms of distraction, and the categorization remains to be expanded.
- Only verified on QA tasks, making its applicability to other RAG scenarios like summarization and dialogue unknown.
- Only evaluated on English benchmarks; multilingual generalization remains unverified.
- The training data is constructed using only 800 NQ queries, which is relatively small in scale.
- The hyperparameter $\lambda$ in answer-skewed retrieval requires tuning, and the optimal values may differ across queries.
- Synthesizing passages with Claude 3.5 Sonnet incurs high computational costs.

## Related Work & Insights

- Cuconasu et al. (2024) first distinguished between random and distracting passages; our work quantifies distraction into continuous values.
- Jin et al. (2024) observed that irrelevant results from strong retrievers are more distracting; this work offers deeper analysis and solutions.
- Yoran et al. (2024) and Lin et al. (2024) explored fine-tuning with retrieved results to enhance RAG robustness; this study demonstrates the added value of highly distracting passages.
- The self-reflection method to judge passage relevance in Self-RAG (Asai et al., 2024) shares philosophical similarities with the DE metric proposed here.
- This approach provides valuable references for the joint optimization of retrievers and generators in RAG systems.

## Rating

- **Novelty**: 8/10 — The formal definition of distracting effect and the design of answer-skewed retrieval are both innovative.
- **Technical Depth**: 8/10 — The metric design is supported by theory, and the multi-method framework is complete.
- **Experimental Thoroughness**: 8/10 — Evaluated on 4 datasets and 7 LLMs, with deep analysis and rigorous statistical testing.
- **Writing Quality**: 9/10 — Highly logical arguments, intuitive examples, and a clear structure.
- **Value**: 9/10 — Holds direct and practical value for improving the robustness of RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] From Ambiguity to Accuracy: The Transformative Effect of Coreference Resolution on RAG Systems](from_ambiguity_to_accuracy_the_transformative_effect_of_coreference_resolution_o.md)
- [\[ACL 2025\] PersonaBench: Evaluating AI Models on Understanding Personal Information through Accessing (Synthetic) Private User Data](personabench_evaluating_ai_models_on_understanding_personal_information_through_.md)
- [\[ACL 2025\] Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models](pandora_box_rag_noise.md)
- [\[ACL 2025\] Contradiction Detection in RAG-Based Chatbots](contradiction_detection_in_rag-based_chatbots.md)
- [\[ACL 2026\] VideoStir: Understanding Long Videos via Spatio-Temporally Structured and Intent-Aware RAG](../../ACL2026/information_retrieval/videostir_understanding_long_videos_via_spatio-temporally_structured_and_intent-.md)

</div>

<!-- RELATED:END -->

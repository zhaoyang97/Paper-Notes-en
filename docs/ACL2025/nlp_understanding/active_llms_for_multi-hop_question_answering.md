---
title: >-
  [Paper Note] Active LLMs for Multi-hop Question Answering
description: >-
  [ACL 2025][NLP Understanding][Multi-hop QA] This paper proposes an active large language model framework that enables the LLM to actively decide when external information retrieval is required and when direct reasoning can be performed, thereby achieving a more efficient and accurate reasoning process in multi-hop question answering tasks.
tags:
  - "ACL 2025"
  - "NLP Understanding"
  - "Multi-hop QA"
  - "Active Learning"
  - "Large Language Models"
  - "Retrieval-Augmented Generation"
  - "Chain-of-Thought"
date: 2026-05-08
content_hash: 77c2a1de6f1764c4
---

# Active LLMs for Multi-hop Question Answering

**Conference**: ACL 2025  
**Area**: NLP Understanding  
**Keywords**: Multi-hop QA, Active Learning, Large Language Models, Retrieval-Augmented Generation, Chain-of-Thought

## TL;DR
This paper proposes an active large language model framework that enables the LLM to actively decide when external information retrieval is required and when direct reasoning can be performed, thereby achieving a more efficient and accurate reasoning process in multi-hop question answering tasks.

## Background & Motivation

**Background**: Multi-hop question answering (Multi-hop QA) requires models to perform reasoning across multiple documents or pieces of knowledge. Currently, mainstream approaches include Retrieval-Augmented Generation (RAG) and Chain-of-Thought (CoT) reasoning. These methods typically employ a static retrieval-reasoning pipeline, executing the same processing steps for every sub-question.

**Limitations of Prior Work**: Existing multi-hop QA systems suffer from two core limitations: first, "over-retrieval", which involves retrieving information for all sub-questions even when some can be answered directly using the model's parametric knowledge; second, "under-retrieval", which occurs when external evidence is not fetched in time when needed, leading to errors in intermediate reasoning steps that cascade and amplify.

**Key Challenge**: Static pipelines cannot dynamically adjust strategies based on the difficulty of each specific sub-question and the model's knowledge coverage. Simple questions waste retrieval resources, while complex questions may introduce noise due to poor retrieval quality.

**Goal**: To design a framework that allows the LLM to "actively" perceive its own knowledge boundaries and dynamically decide whether external retrieval is needed, thereby achieving a balance between efficiency and accuracy in multi-hop reasoning.

**Key Insight**: Inspired by active learning, the authors observe that LLMs exhibit significant differences in confidence across different sub-questions, and this confidence signal can be utilized to guide retrieval decisions.

**Core Idea**: Replace the static retrieval pipeline with an active decision-making mechanism, allowing the LLM to autonomously judge whether to retrieve and what to retrieve at each step of multi-hop reasoning, thereby achieving adaptive multi-hop QA.

## Method

### Overall Architecture
The system takes a multi-hop question as input, which is first decomposed into a sequence of sub-questions by the decomposition module. When processing each sub-question, the active decision module evaluates the LLM's answer confidence for the current sub-question: if the confidence exceeds a threshold, the parametric knowledge is directly used to generate the answer; otherwise, the retrieval module is triggered to fetch relevant documents from an external knowledge base. Finally, the answers to each sub-question are integrated to generate the final response.

### Key Designs

1. **Active Decision Module**:

    - **Function**: To determine whether external retrieval is required at each reasoning step.
    - **Mechanism**: Utilize the LLM's output probability distribution as a confidence signal. Specifically, for each sub-question, the LLM is first prompted to attempt a direct answer, and the average log-probability of the generated token sequence is computed as the confidence score $c = \frac{1}{T}\sum_{t=1}^{T}\log p(y_t|y_{<t}, q)$. When $c$ falls below a predefined threshold $\tau$, a retrieval operation is triggered.
    - **Design Motivation**: To avoid retrieving information for all sub-questions, thereby reducing unnecessary latency overhead and noise introduction.

2. **Adaptive Decomposition**:

    - **Function**: To dynamically decompose complex multi-hop questions into manageable sub-questions.
    - **Mechanism**: Use an iterative decomposition strategy that generates only the next sub-question at each step, allowing subsequent sub-questions to be dynamically adjusted based on previous answers. Unlike static decomposition that generates all sub-questions at once, this approach is better suited for handling dependencies between questions.
    - **Design Motivation**: Logical dependencies often exist between sub-questions in multi-hop QA, where the phrasing or focus of a subsequent sub-question may depend on the answer to the preceding one.

3. **Confidence-Calibrated Retrieval**:

    - **Function**: To execute high-quality document retrieval and integration when retrieval is triggered.
    - **Mechanism**: Instead of retrieving documents relevant only to the current sub-question, the context of the preceding reasoning chain is also included as part of the query to improve retrieval targeting. The retrieved results are reranked and fed into the LLM along with the original query to regenerate the answer.
    - **Design Motivation**: Subsequent sub-questions in multi-hop QA often require integration with preceding reasoning results to achieve accurate retrieval.

### Loss & Training
A two-stage training approach is adopted: the first stage trains the optimal value of the confidence threshold on annotated data, and the second stage fine-tunes the active decision policy via reinforcement learning, where the reward signal is a trade-off between final answer correctness and the number of retrievals.

## Key Experimental Results

### Main Results

| Dataset | Metric(EM) | Active LLMs | Standard RAG | CoT | IRCoT | Gain |
|--------|----------|-------------|---------|-----|-------|------|
| HotpotQA | EM | 72.8 | 67.3 | 63.1 | 69.5 | +3.3 |
| 2WikiMQA | EM | 68.4 | 62.7 | 58.9 | 65.1 | +3.3 |
| MuSiQue | EM | 45.2 | 39.6 | 35.8 | 42.1 | +3.1 |
| Bamboogle | EM | 76.5 | 71.2 | 68.4 | 73.8 | +2.7 |

### Ablation Study

| Configuration | HotpotQA(EM) | Avg. Retrievals | Description |
|------|-------------|-------------|------|
| Full model | 72.8 | 1.8 | Full model |
| Always retrieve | 70.1 | 3.2 | Retrieve at every step, introducing noise |
| Never retrieve | 63.1 | 0 | Pure parametric knowledge |
| Fixed threshold | 70.9 | 2.1 | Non-adaptive threshold |
| Static decomposition | 69.4 | 1.9 | Decompose all sub-questions at once |

### Key Findings
- The active decision module contributes the most; removing it degrades the framework to standard RAG, dropping EM by approximately 3-5 points.
- On average, only 1.8 retrievals are needed per question, a 44% reduction compared to the "always retrieve" setting (3.2 times), while delivering higher accuracy.
- The advantage is more pronounced on difficult questions requiring more than 3 reasoning hops, indicating that active decision-making is more critical in long reasoning chains.

## Highlights & Insights
- Using the LLM's output probability as a self-confidence signal to guide retrieval decisions is simple, effective, and requires no auxiliary classifiers. This "introspection" mechanism can be transferred to other scenarios demanding dynamic decision-making.
- Iterative sub-question decomposition prevents error accumulation, allowing subsequent sub-questions to adaptively adjust based on preceding answers. This design concept offers valuable insights for all tasks involving multi-step reasoning.

## Limitations & Future Work
- Although the confidence threshold can be adjusted adaptively, it relies heavily on the calibration quality of the LLM's probability distribution, which might fail for poorly calibrated models.
- For sub-questions requiring mathematical calculations or logical deduction, retrieving external documents is not necessarily helpful and needs to be combined with tool-use mechanisms.
- Evaluation is only conducted on English datasets; performance in cross-lingual multi-hop question answering scenarios remains to be verified.
- Retrieval quality remains a bottleneck; when the coverage of external knowledge bases is insufficient, even correctly triggered retrieval may fail to obtain valid information.
- The performance of the current method in open-domain QA scenarios has not been verified, where confidence estimation for open-domain questions is more challenging.
- Future work can combine the active decision mechanism with stronger reasoning models (such as the o1 series) to further enhance long-chain reasoning capabilities.

## Related Work & Insights
- **vs IRCoT**: While IRCoT performs retrieval at every step, the proposed method reduces unnecessary retrieval via active decision-making, improving accuracy while lowering latency.
- **vs Self-RAG**: Self-RAG also considers adaptive retrieval but primarily controls it through special tokens. This work directly leverages output probabilities, which is more end-to-end.
- **vs ReAct**: ReAct allows LLMs to autonomously decide actions but lacks an explicit confidence mechanism, making decisions more reliant on prompt design.
- The active retrieval strategy presented in this paper can also be integrated with knowledge graphs to dynamically select information sources between structured knowledge and unstructured documents.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Introducing active learning concepts to multi-hop QA is innovative, though the concept of adaptive retrieval is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across multiple datasets with complete ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and fluent methodology presentation.
- **Value**: ⭐⭐⭐⭐ Highly relevant for practical RAG system deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] RISE: Reasoning Enhancement via Iterative Self-Exploration in Multi-hop Question Answering](rise_reasoning_enhancement_via_iterative_self-exploration_in_multi-hop_question_.md)
- [\[ACL 2025\] Multi-Hop Reasoning for Question Answering with Hyperbolic Representations](multi-hop_reasoning_for_question_answering_with_hyperbolic_representations.md)
- [\[ACL 2025\] BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering](belle_a_bi-level_multi-agent_reasoning_framework_for_multi-hop_question_answerin.md)
- [\[ACL 2025\] Self-Critique Guided Iterative Reasoning for Multi-hop Question Answering](self-critique_guided_iterative_reasoning_for_multi-hop_question_answering.md)
- [\[ACL 2025\] ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision](rescore_multihop_qa.md)

</div>

<!-- RELATED:END -->

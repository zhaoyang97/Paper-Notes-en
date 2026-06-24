---
title: >-
  [Paper Note] Do not Abstain! Identify and Solve the Uncertainty
description: >-
  [ACL 2025][Uncertainty Identification] This paper proposes the ConfuseBench benchmark and a method for identifying uncertainty sources based on the uniqueness of the inquiry answer. It introduces InteractDPO to dynamically generate preference pairs during policy training to enhance inquiry quality, enabling LLMs to proactively identify and resolve uncertainty rather than simply abstaining.
tags:
  - "ACL 2025"
  - "Uncertainty Identification"
  - "ConfuseBench"
  - "InteractDPO"
  - "Uncertainty Classification"
  - "Retrieval-augmented Generation"
date: 2026-05-08
content_hash: 9fb1966f32bcc15a
---

# Do not Abstain! Identify and Solve the Uncertainty

**Conference**: ACL 2025  
**arXiv**: [2506.00780](https://arxiv.org/abs/2506.00780)  
**Code**: [GitHub](https://github.com/somebodyhh1/ConfuseBench)  
**Area**: Others  
**Keywords**: Uncertainty Identification, ConfuseBench, InteractDPO, Uncertainty Classification, Retrieval-augmented Generation  

## TL;DR

This paper proposes the ConfuseBench benchmark and a method for identifying uncertainty sources based on the uniqueness of the inquiry answer. It introduces InteractDPO to dynamically generate preference pairs during policy training to enhance inquiry quality, enabling LLMs to proactively identify and resolve uncertainty rather than simply abstaining.

## Background & Motivation

**The Overconfidence Problem of LLMs**: Large language models often exhibit overconfidence in uncertain scenarios, generating hallucinatory responses. Existing solutions primarily adopt conservative strategies, simply outputting "I don't know" when encountering uncertainty.

**Limitations of Abstention/Avoidance**: For inherently unknowable questions (e.g., "the weather in 2050"), abstaining is the correct approach. However, for answerable questions that contain uncertainty, simply abstaining misses opportunities to resolve uncertainty through retrieval, chains of thought, or clarification. For instance, when a model lacks confidence regarding "the impact of quantum computing on climate modeling", it should proactively identify the sources of uncertainty and adopt corresponding strategies.

**Three Sources of Uncertainty**:
   - **Document Scarcity**: The model lacks the factual information required to answer, which can be supplemented through retrieval.
   - **Limited Capacity**: The query is too complex for the model, which can be resolved via Chain-of-Thought (CoT) or a stronger model.
   - **Query Ambiguity**: The query itself is unclear and requires further clarification from the user.

**Limitations of Prior Work**: Prior studies have focused on single sources of uncertainty (e.g., only conducting iterative retrieval or only performing clarification), failing to comprehensively consider different sources and implement respective remedies.

## Method

### Overall Architecture

The proposed method consists of two core phases: (1) inquiry-answer-based uncertainty source identification; (2) enhancing inquiry generation quality through InteractDPO.

### Benchmark Construction: ConfuseBench

| Dataset | Document Scarcity | Query Ambiguity | Limited Capacity |
|--------|----------|----------|----------|
| HotpotQA | 859 | 702 | 141 |
| AmbigQA | 543 | 537 | 167 |
| ExpertQA | 442 | 397 | 141 |
| TechQA | 470 | 683 | 140 |
| ToolBench | 479 | 590 | 144 |

- Covers three major LLM application scenarios: QA, assistant interactions, and tool usage.
- Utilizes GPT-4o combined with Abstract Meaning Representation (AMR) graphs to systematically introduce ambiguity.
- The benchmark evaluation set contains 650 cases (50 + 50 + 30 per dataset).
- Evaluation metrics: Answer Quality (AQ), Uncertainty Classification Accuracy (UCA), Inquiry Quality (IQ).

### Key Designs 1: Inquiry-Based Uncertainty Identification

**Core Idea**: Instead of directly determining the source of uncertainty, the model is first prompted to generate a follow-up inquiry. The type of uncertainty is then identified based on the characteristics of the response to this inquiry.

**Theoretical Foundation** (Theorem 5.2): The uncertainty carried by the inquiry is positively correlated with the original query, i.e., $|U_k(q) - U_k(x)| \leq -\log p(q^*|x,c,\theta)$, where $U_k$ represents epistemic uncertainty.

**Decision Rules**:
- The inquiry response points to a **unique objective fact** $\rightarrow$ the source of uncertainty is **document scarcity**, requiring retrieval.
- The inquiry response can have **multiple plausible answers** $\rightarrow$ the source of uncertainty is **query ambiguity**, requiring clarification.
- The inquiry is simply a **plain paraphrase of the original query** or is logically incoherent $\rightarrow$ the source of uncertainty is **limited capacity**, requiring CoT.

**Verification Method**: The model is provided with a preset answer. If it can only repeat this answer, it indicates a unique answer (an objective fact). If it can easily generate alternative answers, it indicates an open-ended question (clarification required).

### Key Designs 2: InteractDPO

Traditional DPO utilizes offline preference data, whereas InteractDPO dynamically generates preference pairs during training:

1. The model generates an inquiry based on the prompt.
2. It obtains documents or clarifications through **real-time interaction** with a retrieval system or simulated user.
3. The model generates a final response based on the interaction results.
4. If the original query is successfully resolved $\rightarrow$ the inquiry is labeled as "chosen".
5. If it fails to resolve $\rightarrow$ the inquiry is labeled as "rejected".

**Difference from OnlineDPO**: OnlineDPO relies on the LLM to judge which inquiry is superior (lacking real-world supervision signals), whereas InteractDPO acquires authentic feedback through actual environment interactions.

## Key Experimental Results

### Preliminary Test Results (UCA Accuracy)

| Model | HotpotQA | AmbigQA | TechQA | ExpertQA | ToolBench | Average |
|------|----------|---------|--------|----------|-----------|------|
| GPT-4o | 0.531 | 0.377 | 0.477 | 0.400 | 0.685 | 0.494 |
| DeepSeek-V3 | 0.462 | 0.431 | 0.400 | 0.438 | 0.562 | 0.459 |
| Qwen2.5-72B | 0.631 | 0.592 | 0.431 | 0.408 | 0.700 | 0.552 |
| Qwen2.5-7B | 0.431 | 0.454 | 0.415 | 0.408 | 0.415 | 0.425 |

Even the best-performing model achieves only about 55% classification accuracy, indicating that identifying the sources of uncertainty is highly challenging.

### Performance Comparison (Average UCA Accuracy)

| Method | GPT-4o | DeepSeek-V3 | Qwen2.5-72B | Llama-3-70B | Qwen2.5-7B | Mistral-7B |
|------|--------|-------------|-------------|-------------|------------|------------|
| Direct prompting | 0.494 | 0.459 | 0.552 | 0.408 | 0.425 | 0.454 |
| Inquiry-based identification | 0.569 | 0.537 | 0.577 | 0.537 | 0.477 | 0.529 |
| Answer-based identification | **0.606** | **0.554** | **0.603** | **0.548** | **0.500** | **0.548** |

All models exhibit substantial performance improvements when employing answer-based identification, with an average increase of approximately 10 percentage points.

### Performance of InteractDPO

| Training Method | Average UCA |
|----------|---------|
| GPT-4o (Upper Bound) | 0.606 |
| No training (vanilla) | 0.543 |
| SFT | 0.574 |
| DPO | 0.585 |
| OnlineDPO | 0.592 |
| **InteractDPO** | **0.606** |

InteractDPO enables Qwen2.5-7B to reach the performance level of GPT-4o.

### Key Findings

1. **Model bias attributed to ambiguity**: All models (especially weaker ones) tend to attribute uncertainty to query ambiguity, with an ambiguity recall rate as high as 85-97%, whereas the document recall rate is only 10-19%.
2. **Reluctance of models to admit limited capacity**: Analogous to overconfidence, when encountering uncertainty, models prefer to blame external factors rather than admit their own limited reasoning capacity.
3. **Interference of noisy documents**: When presented with a clear query and noisy documents, models can get distracted by the noisy documents, attempting to ask the user to revise the query to align with the noisy documents.

## Highlights & Insights

1. **Paradigm Shift**: Shifts the perspective from "whether LLMs should answer" to "how LLMs should resolve uncertainty," presenting a more proactive paradigm for managing uncertainty.
2. **Theoretical Contribution**: Proves the positive correlation between the uncertainty of inquiries and the original queries, providing a theoretical foundation for indirect identification.
3. **Practical Identification Workflow**: Cleverly utilizes the uniqueness of the inquiry answer to distinguish between retrieval and clarification needs, which is both intuitive and reasonable.
4. **Online Feedback in InteractDPO**: Acquires training signals through actual interaction, which is more reliable than depending on LLM self-judgment.

## Limitations & Future Work

1. **Limited uncertainty types**: Only three of the most common categories are considered. In practice, there are finer-grained classifications (e.g., lack of factual knowledge vs. lack of background knowledge, ambiguity vs. factual errors vs. temporal out-of-domain issues, etc.).
2. **Simplistic solution for capacity limitation**: Only CoT is utilized, whereas more advanced reasoning methods like Tree of Thoughts or MCTS may be necessary.
3. **Dependence on GPT-4o**: The construction and evaluation of the benchmark heavily rely on GPT-4o, potentially introducing systematic bias.

## Related Work & Insights

- **Uncertainty Identification**: Amayuelas et al. (2024) suggest that models should understand what they do not know; Deng et al. (2024) train models to provide explanations for unanswerable queries.
- **Uncertainty Resolution**: Trivedi et al. (2022) resolve multi-hop queries through iterative reasoning and retrieval; Qian et al. (2024) create the IN3 benchmark to evaluate the generation of clarifying questions.
- **Uncertainty Decomposition**: Yadkori et al. (2024) decompose uncertainty via distribution shifts; the answer validation method in this paper is highly inspired by their work.

## Rating

⭐⭐⭐⭐ — This work introduces a valuable new problem and a systematic solution. ConfuseBench fills the gap in evaluating uncertainty source identification, and InteractDPO serves as an effective approach for online preference learning. The taxonomies of uncertainty categories could be further enriched.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Age Estimation Models Do Not Process Biometric Data](../../ICML2026/others/position_age_estimation_models_do_not_process_biometric_data.md)
- [\[ACL 2025\] All That Glitters is Not Novel: Plagiarism in AI Generated Research](plagiarism_ai_generated_research.md)
- [\[ACL 2025\] Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments](limited_generalizability_in_argument_mining_state-of-the-art_models_learn_datase.md)
- [\[ACL 2025\] Adaptive Retrieval without Self-Knowledge? Bringing Uncertainty Back Home](adaptive_retrieval_without_self-knowledge_bringing_uncertainty_back_home.md)
- [\[ICML 2025\] Access Controls Will Solve the Dual-Use Dilemma](../../ICML2025/others/access_controls_will_solve_the_dual-use_dilemma.md)

</div>

<!-- RELATED:END -->

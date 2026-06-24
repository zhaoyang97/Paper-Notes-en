---
title: >-
  [Paper Note] DRS: Deep Question Reformulation With Structured Output
description: >-
  [ACL 2025][Question Reformulation] Proposes DRS (Deep Question Reformulation with Structured Output), a zero-shot method that improves the question reformulation accuracy of GPT-3.5 from 23.03% to 70.42% through entity-driven DFS search and structured output constraints. This enables LLMs to effectively help users convert unanswerable questions into answerable counterparts.
tags:
  - "ACL 2025"
  - "Question Reformulation"
  - "Unanswerable Questions"
  - "DFS Search"
  - "Structured Output"
  - "Zero-Shot Methods"
date: 2026-05-08
content_hash: 74241c160260d513
---

# DRS: Deep Question Reformulation With Structured Output

**Conference**: ACL 2025  
**arXiv**: [2411.17993](https://arxiv.org/abs/2411.17993)  
**Code**: [Yes](https://github.com/Lizhecheng02/DRS)  
**Area**: Others  
**Keywords**: Question Reformulation, Unanswerable Questions, DFS Search, Structured Output, Zero-Shot Methods

## TL;DR

Proposes DRS (Deep Question Reformulation with Structured Output), a zero-shot method that improves the question reformulation accuracy of GPT-3.5 from 23.03% to 70.42% through entity-driven DFS search and structured output constraints. This enables LLMs to effectively help users convert unanswerable questions into answerable counterparts.

## Background & Motivation

When users interact with documents in unfamiliar domains, they often ask questions that cannot be answered by the given document. For instance, if a text describes the nutritional facts of mustard, a user might ask, "How many calories does mustard have?", while the actual text only contains information regarding carbohydrates, fats, and proteins.

Existing works focus on three directions:

**Detection Methods**: Identifying unanswerable questions (e.g., SQuAD 2.0).

**Clarification Methods**: Asking the user for clarification.

**Reformulation Methods**: Reformulating unanswerable questions into answerable forms.

The **key challenge** is simultaneously satisfying two criteria:
- The reformulated question must be **answerable from the given text** (answerability).
- It must **preserve the core entities and intent of the original question** (intent preservation).

Even GPT-3.5 achieves only a 23.03% accuracy under zero-shot CoT, exposing the insufficiency of LLMs in balancing these two objectives. Existing methods (few-shot, CoT) either focus excessively on preserving original question entities (leaving the question still unanswerable) or prioritize answerability at the cost of deviating from user intent.

## Method

### Overall Architecture

DRS consists of three steps: (1) Entity extraction and filtering $\to$ (2) DFS combinatorial search + structured question generation $\to$ (3) Candidate question re-evaluation.

### Key Designs

1. **Entity Extraction and Filtering**:

    - Step 1: Use zero-shot prompting to let the LLM extract all important entities in the question (minimizing omissions).
    - Step 2: Prompt the LLM to classify entities into five categories: subject, object, predicate, attribute, and others.
    - Only preserve subject, object, and attribute categories (as they carry the core intent), while discarding predicates, etc.
    - Design Motivation: Directly extracting entities with LLMs often introduces verb phrases (e.g., "step down"), which escalates the error rate in subsequent generation steps.

2. **DFS Combinatorial Search + Structured Generation**:

    - Use a DFS algorithm to systematically explore entity combinations (only entering target generation when the number of entities in a combination exceeds half of the filtered total).
    - For each qualified combination, execute the following sequentially:
        - Prompt the LLM to generate a **structured statement** based on the document and the selected entities.
        - Generate a **structured question** containing all selected entities based on that statement.
        - Verify if the question contains all necessary entities.
        - Verify if the question is answerable from the document.
    - If the conditions are met, store the candidate; stop searching once a certain threshold is reached.
    - **Key Insight**: Generating statements before generating questions yields a significantly higher success rate than directly prompting LLMs to reformulate the question.
    - Control search depth and iteration steps to balance efficiency and accuracy.

3. **Candidate Question Re-evaluation**:

    - Evaluate the answerability of all candidate questions (verified by LLM).
    - Calculate the entity overlap score: $\text{Score} = \frac{|\text{Entities}_{cand} \cap \text{Entities}_{orig}|}{|\text{Entities}_{orig}|}$
    - Select the candidate with the highest entity overlap among the answerable candidates as the final output.

### Evaluation Framework Improvements

- Discovered that the previous Llama2-7B evaluator generalizes poorly (average accuracy of only 52.78%, close to random).
- Proposed using GPT-4o-mini as the evaluator, achieving an average accuracy of 90.70% with high consistency across datasets.
- Evaluation criteria: (1) The reformulated question must be answerable from the text, and (2) entity overlap must be $\ge 50\%$. Success is achieved only when both conditions are met.

## Key Experimental Results

### Main Results — Four Models & Six Datasets (Table)

| Model | Method | QA2 | BanditQA | BBC | Reddit | Yelp | SQuADv2 | Avg |
|------|------|-----|---------|-----|--------|------|---------|------|
| GPT-3.5 | Zero-Shot CoT | 29.15 | 15.63 | 13.56 | 20.35 | 15.69 | 43.79 | 23.03 |
| GPT-3.5 | Few-Shot CoT | 44.94 | 48.78 | 16.95 | 18.58 | 25.49 | 41.22 | 32.66 |
| GPT-3.5 | **DRS (ours)** | **81.80** | **73.20** | **62.71** | **75.22** | **66.67** | **62.90** | **70.42** |
| GPT-4o-mini | Zero-Shot CoT | 49.39 | 37.50 | 42.37 | 17.70 | 35.29 | 50.10 | 38.73 |
| GPT-4o-mini | **DRS (ours)** | **88.26** | **80.16** | **79.66** | **83.19** | **78.43** | **78.30** | **81.33** |
| Gemma2-9B | Zero-Shot | 36.03 | 21.33 | 20.34 | 18.58 | 19.61 | 42.21 | 26.35 |
| Gemma2-9B | **DRS (ours)** | **59.92** | **60.73** | **55.93** | **59.29** | **49.02** | **55.62** | **56.75** |

DRS significantly outperforms all baselines across all models and datasets, achieving an approximately 3-fold improvement on GPT-3.5.

### Parameter Sensitivity — Number of Candidate Questions (Summary Table)

| # Candidates | GPT-3.5 | GPT-4o-mini | Gemma2-9B |
|--------|---------|-------------|-----------|
| 1 | 63.57 | 78.56 | ~52 |
| 2 | ~69 | ~81 | ~56 |
| 3 | **70.42** | **81.33** | **56.75** |
| 4 | ~69 | ~80 | ~53 |
| 5 | ~68 | ~80 | ~55 |

Optimal performance is reached with 2-3 candidates, and even a single candidate significantly outperforms all baselines.

### Key Findings

- DRS under the zero-shot setting outperforms few-shot CoT (70.42% vs. 32.66% on GPT-3.5, more than doubling performance).
- DRS is highly robust to temperature changes (performance variance of $\le 3$ percentage points across different temperatures).
- Human evaluation confirms that almost 100% of the generated reformulated questions are meaningful and relevant to the document (only 1 meaningless case out of 6 datasets).
- The increase in inference time is limited: on GPT-3.5, DRS (2 candidates) takes 10.07s compared to 6.80s for CoT, but doubles the accuracy.
- Additional experiments on GPT-4 (avg 70.28%) and Llama3.1-70B (avg 68.48%) validate its strong generalizability.

## Highlights & Insights

- **Structured output constraint** is the key innovation: the two-step statement-then-question generation guarantees output quality.
- DFS search successfully converts a combinatorial explosion problem into a manageable search space with effective pruning strategies.
- Entity classification (five-category filtering) effectively resolves the issues brought by verb phrases being mixed into LLM-extracted entities.
- The upgrade of the evaluator (from Llama2-7B to GPT-4o-mini) is itself a significant contribution.

## Limitations & Future Work

- DFS search requires multiple document passes, inducing higher computational costs than single-pass methods.
- The six datasets do not cover highly specialized domain-specific documents.
- Entity classification depends heavily on LLM accuracy and may struggle in complex, niche domains.
- Future work can explore highly efficient methods that enable the LLM to perform reformulation in a single pass.

## Related Work & Insights

- A lineage of studies based on unanswerable question datasets such as SQuAD 2.0 and CouldAsk.
- Complementary to Chain-of-Thought (CoT): while CoT enhances reasoning without restricting the output structure, DRS stabilizes and improves generation quality through structured constraints.
- The entity-driven approach can be extended to other text rewriting tasks requiring strong intent preservation.

## Rating

- **Novelty**: 7/10 — The combination of DFS and structured output is effective but not a fundamental breakthrough.
- **Experimental Thoroughness**: 9/10 — Thorough evaluations across 4 models (and additional tests), 6 datasets, parameter sensitivity, human evaluation, and inference time analysis.
- **Writing Quality**: 7/10 — The case study is clear, but the method description is somewhat verbose.
- **Value**: 7/10 — The question reformulation scenario is highly practical and brings a significant performance gain (3x), but its scope of application remains relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Graph-Structured Trajectory Extraction from Travelogues](graph-structured_trajectory_extraction_from_travelogues.md)
- [\[ACL 2025\] Multi-Hop Question Generation via Dual-Perspective Keyword Guidance](multi-hop_question_generation_via_dual-perspective_keyword_guidance.md)
- [\[ACL 2025\] Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus](mapping_the_podcast_ecosystem_with_the_structured_podcast_research_corpus.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)
- [\[ACL 2025\] STRICTA: Structured Reasoning in Critical Text Assessment for Peer Review and Beyond](stricta_structured_reasoning_in_critical_text_assessment_for_peer_review_and_bey.md)

</div>

<!-- RELATED:END -->

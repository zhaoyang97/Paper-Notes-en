---
title: >-
  [Paper Note] Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG
description: >-
  [ACL 2025][Information Retrieval & RAG][RAG knowledge boundary] DTA proposes dividing RAG queries into four quadrants based on parametric knowledge boundaries and retrieval knowledge boundaries. For queries where "both are unknown," DTA constructs preference data and applies DPO to train the model to answer "I don't know." This addresses the issue of RAFT models generating answers even when retrieval is entirely noisy, achieving an effective balance between accuracy and appro…
tags:
  - "ACL 2025"
  - "Information Retrieval & RAG"
  - "RAG knowledge boundary"
  - "honest alignment"
  - "DPO"
  - "abstention"
  - "RAFT"
date: 2026-05-08
content_hash: 317a7c67ef6acbda
---

# Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG

**Conference**: ACL 2025  
**arXiv**: [2505.20871](https://arxiv.org/abs/2505.20871)  
**Code**: [https://github.com/Divide-Then-Align](https://github.com/Divide-Then-Align)  
**Area**: LLM Agent / RAG / Alignment  
**Keywords**: RAG knowledge boundary, honest alignment, DPO, abstention, RAFT

## TL;DR
DTA proposes dividing RAG queries into four quadrants based on parametric knowledge boundaries and retrieval knowledge boundaries. For queries where "both are unknown," DTA constructs preference data and applies DPO to train the model to answer "I don't know." This addresses the issue of RAFT models generating answers even when retrieval is entirely noisy, achieving an effective balance between accuracy and appropriate abstention.

## Background & Motivation

### Background
**Background**: RAG fine-tuning (RAFT) enhances robustness by training LLMs in contexts with noisy retrieval, representing a significant advancement in the RAG field.

**Limitations of Prior Work**: 

### Limitations of Prior Work
**Limitations of Prior Work**: RAFT-conditioned models generate answers under **any** circumstances, even when retrieval is entirely noisy and the model's parametric knowledge does not contain the answer.

### Key Challenge
**Key Challenge**: For high-risk domains (medical, legal, financial), the inability to say "I don't know" when lacking knowledge is a fatal flaw.

### Proposed Solution
**Proposed Solution**: Even when explicitly instructed in the prompt to answer "I don't know", the RAFT-trained model still tends to hallucinate answers.

**Key Challenge**: Robustness training in RAFT helps the model learn to extract information from noise (good), but also forces it to generate answers even when there is no information (bad).

**Goal**: Enable RAG models to honestly answer "I don't know" when queries exceed their knowledge boundaries.

**Key Insight**: Categorize queries into four quadrants based on combinations of the two knowledge boundaries, and construct different preference strategies for each quadrant.

**Core Idea**: Four-quadrant division (parametric knowledge ✓/✗ × retrieval knowledge ✓/✗) + quadrant-specific preference data + DPO alignment = no drop in accuracy + ability to say "I don't know".

## Method

### Overall Architecture
**Divide**: For each query $q$ in the training data, determine (1) $q \in KB_{param}$? (Can the model answer correctly without retrieval?) (2) $q \in KB_r$? (Does the retrieved document contain the answer?) $\rightarrow$ Divide into four quadrants. **Align**: Construct preference pairs for each quadrant and train using DPO.

### Key Designs

1. **Four-Quadrant Division**:

    - **✓✓**: Parametric knowledge known + retrieval found the answer $\rightarrow$ Preference: Correct answer > Incorrect answer
    - **✓✗**: Parametric knowledge known + retrieval failed to find the answer $\rightarrow$ Preference: Correct answer > "I don't know" > Incorrect answer
    - **✗✓**: Parametric knowledge unknown + retrieval found the answer $\rightarrow$ Preference: Correct answer > "I don't know" > Incorrect answer
    - **✗✗**: Neither knows $\rightarrow$ Preference: **"I don't know" > Any incorrect answer**
    - **Design Motivation**: Different quadrants require different optimal strategies between "responding vs. abstaining".

2. **Knowledge Boundary Determination**:

    - Parametric knowledge boundary $KB_{param}$: Sample $N$ non-retrieval responses for the same query; if the accuracy exceeds a threshold $\delta$, then ✓.
    - Retrieval knowledge boundary $KB_r$: Check if the retrieved documents contain the correct answer via lexical matching.
    - **Design Motivation**: An actionable judgment method is required to automatically divide quadrants.

3. **Preference Data Construction + DPO**:

    - Construct different chosen/rejected pairs for each quadrant.
    - Construct a total of 10,000 preference samples.
    - Continue training on top of the RAFT model using DPO.
    - **Design Motivation**: DPO is simpler and more efficient than RLHF, and preference data can be constructed automatically without manual annotation.

## Key Experimental Results

### Main Results

| Method | NQ Acc | TriviaQA Acc | WebQ Acc | Abstention Rate (✗✗) |
|------|--------|-------------|----------|----------|
| RAFT | High | High | High | ~0% (No abstention) |
| RAFT + prompt "IDK" | Slight drop | Slight drop | Slight drop | <5% |
| **DTA** | **Comparable** | **Comparable** | **Comparable** | **~80%** |

### Key Findings
- **RAFT models rarely abstain**: Even when the prompt explicitly requests it, RAFT models still have a >95% probability of fabricating answers in the ✗✗ quadrant.
- **DTA enables the model to abstain in ~80% of cases in the ✗✗ quadrant**: Meanwhile, accuracy in other quadrants is largely uncompromised.
- **10K preference samples are sufficient**: DPO training is highly efficient, and the model can learn abstention behavior with a small amount of data.
- **The four-quadrant evaluation framework is more comprehensive than single accuracy**: 9 metrics cover multi-dimensional trade-offs between accuracy and abstention.

## Highlights & Insights
- **The conceptualization of the "four quadrants of knowledge boundary" is highly clear**: Formalizing the vague "when should RAG say I don't know" into an actionable four-quadrant judgment provides a standard framework for RAG honesty research.
- **Finding the "overconfidence" issue in RAFT carries great practical significance**: This serves as an important complement to the popular RAFT approach—robustness and honesty are two distinct objectives.
- **DPO as a plug-and-play post-training step**: It can be directly applied to any existing RAFT model without retraining from scratch.

## Limitations & Future Work
- **The abstention threshold $\delta$ needs to be set manually**: Different scenarios may require different inclinations to abstain.
- **Errors exist in knowledge boundary determination**: Using accuracy over $N$ samples as a proxy metric is not entirely precise.
- **Evaluation is limited to QA tasks**: Abstention behavior in open-ended generation tasks is more complex.
- **Potential for over-abstention**: An 80% abstention rate might be too conservative for certain scenarios.

## Related Work & Insights
- **vs RAFT (Yoran et al., 2024)**: RAFT focuses on noise robustness, while DTA focuses on honesty—the two are complementary.
- **vs Astute RAG**: Astute RAG resolves conflicts through knowledge integration, while DTA resolves knowledge gaps through abstention—differing in methodology.
- **vs Self-RAG**: Self-RAG uses reflection tokens to determine if retrieval is needed, while DTA determines if it should abstain—positioned further downstream.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Both the conceptualization of the four quadrants and the identification of the overconfidence issue in RAFT are highly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across 3 datasets and 9 metrics, but limited to QA.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Precise formal definitions, intuitive diagrams and tables, and a clear logical flow.
- **Value**: ⭐⭐⭐⭐ Significant importance for the reliable and secure deployment of RAG systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Astute RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models](astute_rag_knowledge_conflicts.md)
- [\[ACL 2025\] KnowShiftQA: How Robust are RAG Systems when Textbook Knowledge Shifts in K-12 Education?](knowshiftqa_rag_knowledge_shifts.md)
- [\[ICML 2026\] REAL: Resolving Knowledge Conflicts in Knowledge-Intensive Visual Question Answering via Reasoning-Pivot Alignment](../../ICML2026/information_retrieval/real_resolving_knowledge_conflicts_in_knowledge-intensive_visual_question_answer.md)
- [\[ACL 2025\] GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis](gainrag_preference_alignment.md)
- [\[ACL 2025\] ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search](arise_risk_adaptive_search.md)

</div>

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Feedback Adaptation for Retrieval-Augmented Generation
description: >-
  [ACL 2026][Information Retrieval & RAG][RAG] This paper proposes "Feedback Adaptation" as a new problem setting for RAG systems—studying how quickly and effectively corrective feedback propagates to future queries. It de…
tags:
  - "ACL 2026"
  - "Information Retrieval & RAG"
  - "RAG"
  - "Feedback Adaptation"
  - "Correction Latency"
  - "PatchRAG"
  - "Online Learning"
date: 2026-05-08
content_hash: 0397b4d54a1df4af
---

# Feedback Adaptation for Retrieval-Augmented Generation

**Conference**: ACL 2026  
**arXiv**: [2604.06647](https://arxiv.org/abs/2604.06647)  
**Code**: None  
**Area**: Information Retrieval / RAG  
**Keywords**: RAG, Feedback Adaptation, Correction Latency, PatchRAG, Online Learning

## TL;DR
This paper proposes "Feedback Adaptation" as a new problem setting for RAG systems—studying how quickly and effectively corrective feedback propagates to future queries. It defines two evaluation axes: Correction Latency and Post-feedback Performance, and introduces PatchRAG as a training-free inference-time feedback integration scheme that achieves immediate correction and strong generalization.

## Background & Motivation

**Background**: RAG has become the dominant paradigm for grounding LLMs to external knowledge. However, existing research assumes that knowledge and system behavior remain static after deployment. In real-world deployments, RAG systems are frequently corrected by users or experts—providing feedback when outputs are outdated, incorrect, or undesirable.

**Limitations of Prior Work**: (1) Existing methods handle feedback through retraining or fine-tuning, introducing inherent latency between feedback provision and behavior change; (2) Existing evaluation protocols focus only on overall accuracy, failing to capture the adaptation speed and quality of the system after feedback; (3) Current benchmarks conflate correctness with adaptability, obscuring critical dimensions of system behavior in interactive scenarios.

**Key Challenge**: Training-based methods can achieve strong performance but suffer from delay (high Correction Latency), while inference-time methods can react immediately but may lack generalization (low Post-feedback Performance). This trade-off is entirely invisible under existing evaluation frameworks.

**Goal**: (1) Formalize the feedback adaptation problem; (2) Define evaluation metrics that capture adaptation dynamics; (3) Provide a proof-of-concept instance.

**Key Insight**: Elevate "adapting to feedback" from a training/maintenance concern to a first-class research problem. Feedback adaptation is not about improving average accuracy, but about characterizing the dynamics of knowledge updates under interactive conditions.

**Core Idea**: Define two orthogonal evaluation axes—Correction Latency (how fast feedback takes effect) and Post-feedback Performance (generalization to semantically related queries)—and demonstrate that immediate adaptation is possible using PatchRAG.

## Method

### Overall Architecture
The framework consists of three layers: (1) Problem Definition—formalizing feedback adaptation and its evaluation axes; (2) PatchRAG—an inference-time scheme for storing and retrieving feedback patches; (3) Snapshot Evaluation Protocol—conducting comparative evaluations before and after feedback injection to isolate the marginal effect of feedback.

### Key Designs

1.  **Correction Latency Evaluation Axis**:
    - **Function**: Measures the time elapsed from feedback provision to a consistent change in system behavior.
    - **Mechanism**: Given feedback $f_t$ at time $t$, correction latency is defined as the time elapsed before the system consistently produces corrected outputs for semantically consistent queries. Any method relying on retraining necessarily has significant latency, regardless of final accuracy.
    - **Design Motivation**: Two systems might have the same final accuracy, but the time they continue to produce erroneous outputs after feedback can differ vastly. Correction latency captures this variance invisible in standard evaluations.

2.  **Post-feedback Performance Evaluation Axis**:
    - **Function**: Measures the quality of system adaptation to queries semantically consistent with the feedback.
    - **Mechanism**: Unlike standard test accuracy, this metric is explicitly conditioned on the presence of feedback, focusing on generalization to queries with consistent intent but different phrasing. Systems that only memorize feedback instances without generalizing to related queries will perform poorly.
    - **Design Motivation**: Complementary to Correction Latency—one measures "when to adapt," the other measures "how well to adapt." Together, they reveal behaviors unseen in standard accuracy evaluations.

3.  **PatchRAG: Inference-time Feedback Integration**:
    - **Function**: Immediately integrates feedback at inference time by storing and retrieving feedback patches without retraining.
    - **Mechanism**: Each feedback item is stored as a tuple $f_i = (q_i, a_i, c_i)$ (original query, corrected answer, supporting evidence). For a new query $q$, a hybrid intent-context retrieval is used: $S_i(q) = \lambda \cdot \text{sim}(q, q_i) + (1-\lambda) \cdot \text{sim}(q, c_i)$, balancing intent matching and content grounding. Top-k feedback items are integrated into generation as context via ICL.
    - **Design Motivation**: PatchRAG is intentionally kept minimal—no architecture modification, no parameter training, only storage and retrieval. The goal is to prove that immediate adaptation is possible rather than providing an ultimate solution. The hybrid intent-context retrieval addresses the need for generalization across varying surface forms with consistent intent.

### Loss & Training
PatchRAG involves no training. Evaluation uses three datasets: NQ, TriviaQA, and HotpotQA, compared against baselines such as Standard RAG, Self-RAG, Auto-RAG, and ChatQA-1.5.

## Key Experimental Results

### Main Results

| Method | NQ | TriviaQA | HotpotQA | Correction Latency |
| :--- | :--- | :--- | :--- | :--- |
| Standard RAG | 28.7 | 67.1 | 28.5 | High (requires retraining) |
| Auto-RAG | 37.9 | 60.9 | 44.9 | High |
| PatchRAG | **Competitive** | **Competitive** | **Competitive** | **Immediate (Zero Latency)** |

### Ablation Study

| Evaluation Axis | Training-based Methods | PatchRAG | Description |
| :--- | :--- | :--- | :--- |
| Correction Latency | High (requires retraining time) | **Zero** | Immediately reflects feedback |
| Post-feedback Performance | High (but after latency) | **High** | Intent-aware retrieval supports generalization |
| Overall Accuracy | High | Competitive | Standard evaluation fails to distinguish |

### Key Findings
- Training-based methods exhibit a structural latency-performance trade-off—this is completely invisible in standard accuracy evaluations.
- PatchRAG achieves strong post-feedback performance with zero correction latency, proving that immediate adaptation is feasible.
- Hybrid intent-context retrieval is more effective than pure intent or pure content retrieval, as it handles both surface form variations and content relevance.
- Stress tests under imperfect feedback conditions show that PatchRAG possesses reasonable robustness.

## Highlights & Insights
- **Feedback Adaptation as a First-class Citizen**: Elevating "how to respond to corrections after deployment" from an operational issue to a core research problem, with a clear evaluation framework. This has broad implications for all interactive AI systems.
- **Concept of Correction Latency**: Analogous to "Time to Fix" in software engineering, correction latency quantifies the real gap between user correction and system change. This metric can be generalized to evaluate any system requiring rapid adaptation.
- **Power of Minimal Design**: PatchRAG proves the concept through a minimalist design, demonstrating that "storage + retrieval + ICL" can achieve immediate adaptation, providing a baseline for future, more complex solutions.

## Limitations & Future Work
- PatchRAG is a proof-of-concept rather than a final solution; retrieval efficiency and conflict management after large-scale feedback accumulation are not explored.
- Only factuality corrections were evaluated; feedback regarding preferences or style was not addressed.
- Evaluation is based on a snapshot protocol rather than true online streaming evaluation.
- Feedback quality is assumed to be perfect or near-perfect; in real deployment, feedback may be noisy or contradictory.

## Related Work & Insights
- **vs. Continual Learning**: Continual learning focuses on not forgetting old knowledge, while feedback adaptation focuses on rapidly integrating new corrections.
- **vs. Model Editing**: Model editing updates via parameter modification, whereas PatchRAG does not modify parameters.
- **vs. Online Learning**: Online learning optimizes aggregate performance, while feedback adaptation focuses on temporal dynamics after correction.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The proposal and formalization of feedback adaptation as an independent research problem is a significant contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ Covers three datasets, but the novel evaluation protocol requires wider validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Extremely clear problem definition and elegantly designed evaluation framework.
- **Value**: ⭐⭐⭐⭐⭐ Opens a new dimension for RAG evaluation and provides direct guidance for deployment practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation](disco-rag_discourse-aware_retrieval-augmented_generation.md)
- [\[ACL 2026\] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)
- [\[ACL 2026\] Language-Coupled Reinforcement Learning for Multilingual Retrieval-Augmented Generation](language-coupled_reinforcement_learning_for_multilingual_retrieval-augmented_gen.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->

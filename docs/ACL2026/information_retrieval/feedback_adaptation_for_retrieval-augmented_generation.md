---
title: >-
  [Paper Note] Feedback Adaptation for Retrieval-Augmented Generation
description: >-
  [ACL 2026][RAG] This paper proposes *feedback adaptation* as a new problem setting for RAG systems—investigating how quickly and effectively corrective feedback propagates to future queries. It defines two evaluation axes, correction latency and post-feedback performance, and introduces PatchRAG as a training-free, inference-time feedback integration approach that achieves immediate correction and strong generalization.
tags:
  - ACL 2026
  - RAG
  - feedback adaptation
  - correction latency
  - PatchRAG
  - online learning
date: 2026-05-08
content_hash: b49ae359081392c8
---

# Feedback Adaptation for Retrieval-Augmented Generation

**Conference**: ACL 2026
**arXiv**: [2604.06647](https://arxiv.org/abs/2604.06647)
**Code**: None
**Area**: Information Retrieval / RAG
**Keywords**: RAG, feedback adaptation, correction latency, PatchRAG, online learning

## TL;DR
This paper proposes *feedback adaptation* as a new problem setting for RAG systems—investigating how quickly and effectively corrective feedback propagates to future queries. It defines two evaluation axes, correction latency and post-feedback performance, and introduces PatchRAG as a training-free, inference-time feedback integration approach that achieves immediate correction and strong generalization.

## Background & Motivation

**Background**: RAG has become the dominant paradigm for grounding LLMs in external knowledge. However, existing research assumes that knowledge and system behavior remain static after deployment. In practice, RAG systems are frequently corrected by users or domain experts—who provide feedback when outputs are outdated, erroneous, or otherwise unsatisfactory.

**Limitations of Prior Work**: (1) Existing methods handle feedback via retraining or fine-tuning, introducing an inherent delay between feedback provision and behavioral change. (2) Existing evaluation protocols focus solely on overall accuracy, failing to capture how quickly and how well a system adapts after receiving feedback. (3) Current benchmarks conflate correctness with adaptability, obscuring a critical dimension of system behavior in interactive scenarios.

**Key Challenge**: Training-based methods can achieve strong performance but incur latency (high correction latency), whereas inference-time methods can react immediately but may generalize poorly (low post-feedback performance). This trade-off is entirely invisible under existing evaluation frameworks.

**Goal**: (1) Formalize the feedback adaptation problem; (2) Define evaluation metrics that capture adaptation dynamics; (3) Provide a proof-of-concept instantiation.

**Key Insight**: Elevating "feedback adaptation" from a training/maintenance concern to a first-class research problem. Feedback adaptation is not about improving average accuracy, but about characterizing the dynamics of knowledge updates under interactive conditions.

**Core Idea**: Define two orthogonal evaluation axes—correction latency (how quickly feedback takes effect) and post-feedback performance (generalization to semantically related queries)—and use PatchRAG to demonstrate that immediate adaptation is achievable.

## Method

### Overall Architecture
The framework consists of three layers: (1) problem formalization—defining feedback adaptation and its evaluation axes; (2) PatchRAG—an inference-time scheme for storing and retrieving feedback patches; (3) snapshot evaluation protocol—comparing system behavior before and after feedback injection to isolate the marginal effect of feedback.

### Key Designs

1. **Correction Latency Evaluation Axis**:

    - Function: Measures the time elapsed between feedback provision and consistent behavioral change.
    - Mechanism: Given feedback $f_t$ at time $t$, correction latency is defined as the elapsed time before the system begins consistently producing corrected outputs for semantically aligned queries. Any method relying on retraining necessarily incurs substantial latency, regardless of its final accuracy.
    - Design Motivation: Two systems may achieve identical final accuracy yet differ dramatically in how long they continue producing erroneous outputs after feedback. Correction latency captures this difference, which is invisible under standard evaluation.

2. **Post-Feedback Performance Evaluation Axis**:

    - Function: Measures the quality of adaptation to queries that are semantically consistent with the feedback.
    - Mechanism: Unlike standard test accuracy, this metric explicitly conditions on the presence of feedback, focusing on generalization to queries that share the same intent but differ in surface form. Systems that merely memorize feedback instances without generalizing to related queries will perform poorly.
    - Design Motivation: Complementary to correction latency—one measures *when* adaptation occurs, the other measures *how well* adaptation occurs. Together, they reveal behavioral dimensions invisible to standard accuracy evaluation.

3. **PatchRAG: Inference-Time Feedback Integration**:

    - Function: Immediately integrates feedback at inference time by storing and retrieving feedback patches, without any retraining.
    - Mechanism: Each feedback item is stored as a tuple $f_i = (q_i, a_i, c_i)$ (original query, corrected answer, supporting evidence). For a new query $q$, intent–context hybrid retrieval is applied: $S_i(q) = \lambda \cdot \text{sim}(q, q_i) + (1-\lambda) \cdot \text{sim}(q, c_i)$, balancing intent matching and content grounding. The top-$k$ feedback items are incorporated into generation via in-context learning (ICL).
    - Design Motivation: PatchRAG is intentionally minimal—no architectural modifications, no parameter updates, only storage and retrieval. The goal is to demonstrate that immediate adaptation is possible, not to provide a definitive solution. Intent–context hybrid retrieval addresses the generalization requirement for queries that share intent but differ in surface form.

### Loss & Training
PatchRAG involves no training. Evaluation is conducted on NQ, TriviaQA, and HotpotQA, with comparisons against Standard RAG, Self-RAG, Auto-RAG, and ChatQA-1.5.

## Key Experimental Results

### Main Results

| Method | NQ | TriviaQA | HotpotQA | Correction Latency |
|------|-----|---------|---------|---------|
| Standard RAG | 28.7 | 67.1 | 28.5 | High (requires retraining) |
| Auto-RAG | 37.9 | 60.9 | 44.9 | High |
| PatchRAG | **Competitive** | **Competitive** | **Competitive** | **Immediate (zero latency)** |

### Ablation Study

| Evaluation Axis | Training-Based Methods | PatchRAG | Notes |
|--------|-----------|---------|------|
| Correction Latency | High (retraining time required) | **Zero** | Feedback reflected immediately |
| Post-Feedback Performance | High (but after delay) | **High** | Intent-aware retrieval supports generalization |
| Overall Accuracy | High | Competitive | Indistinguishable under standard evaluation |

### Key Findings
- Training-based methods exhibit a structural latency–performance trade-off that is entirely invisible under standard accuracy evaluation.
- PatchRAG achieves strong post-feedback performance with zero correction latency, demonstrating that immediate adaptation is feasible.
- Intent–context hybrid retrieval outperforms retrieval based solely on intent or content, as it simultaneously handles surface-form variation and content relevance.
- Stress testing under imperfect feedback conditions shows that PatchRAG exhibits reasonable robustness.

## Highlights & Insights
- **Feedback Adaptation as a First-Class Problem**: Elevating "how to respond to corrections after deployment" from an operational concern to a core research problem, with a clearly defined evaluation framework. This has broad implications for all interactive AI systems.
- **The Concept of Correction Latency**: Analogous to "time-to-fix" in software engineering, correction latency quantifies the real gap between user correction and system behavioral change. This metric generalizes naturally to any system requiring rapid adaptation.
- **The Power of Minimal Design**: PatchRAG demonstrates the concept through extreme simplicity, showing that "store + retrieve + ICL" suffices for immediate adaptation and establishes a baseline for more sophisticated future approaches.

## Limitations & Future Work
- PatchRAG is a proof of concept rather than a final solution; retrieval efficiency and conflict resolution under large-scale feedback accumulation remain unexplored.
- Only factual corrections are evaluated; preference- or style-level feedback is not addressed.
- Evaluation relies on a snapshot protocol rather than true online streaming evaluation.
- Feedback quality is assumed to be perfect or near-perfect; in real deployments, feedback may be noisy or contradictory.

## Related Work & Insights
- **vs. Continual Learning**: Continual learning focuses on retaining old knowledge without forgetting; feedback adaptation focuses on rapidly integrating new corrections.
- **vs. Model Editing**: Model editing achieves updates via parameter modification; PatchRAG requires no parameter changes.
- **vs. Online Learning**: Online learning optimizes aggregate performance; feedback adaptation focuses on temporal dynamics following individual corrections.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Proposing and formalizing feedback adaptation as an independent research problem is a significant contribution.
- Experimental Thoroughness: ⭐⭐⭐ — Three datasets are used, but the novel evaluation protocol requires broader validation.
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem definition is exceptionally clear; the evaluation framework is elegantly designed.
- Value: ⭐⭐⭐⭐⭐ — Opens a new dimension in RAG evaluation with direct implications for deployment practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Domain-Specific Data Generation Framework for RAG Adaptation](domain-specific_data_generation_framework_for_rag_adaptation.md)
- [\[ACL 2026\] CodePromptZip: Code-specific Prompt Compression for Retrieval-Augmented Generation in Coding Tasks with LMs](codepromptzip_code-specific_prompt_compression_for_retrieval-augmented_generatio.md)
- [\[ACL 2026\] MASS-RAG: Multi-Agent Synthesis Retrieval-Augmented Generation](mass-rag_multi-agent_synthesis_retrieval-augmented_generation.md)
- [\[ACL 2026\] Stable-RAG: Mitigating Retrieval-Permutation-Induced Hallucinations in Retrieval-Augmented Generation](stable-rag_mitigating_retrieval-permutation-induced_hallucinations_in_retrieval-.md)
- [\[ACL 2026\] Beyond Explicit Refusals: Soft-Failure Attacks on Retrieval-Augmented Generation](beyond_explicit_refusals_soft-failure_attacks_on_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->

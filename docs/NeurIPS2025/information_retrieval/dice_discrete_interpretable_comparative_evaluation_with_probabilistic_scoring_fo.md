---
title: >-
  [Paper Note] DICE: Discrete Interpretable Comparative Evaluation with Probabilistic Scoring for RAG
description: >-
  [NeurIPS 2025 (ResponsibleFM Workshop)][Information Retrieval & RAG][RAG evaluation] This paper proposes the DICE framework, which achieves interpretable, robust…
tags:
  - "NeurIPS 2025 (ResponsibleFM Workshop)"
  - "Information Retrieval & RAG"
  - "RAG evaluation"
  - "LLM-as-judge"
  - "pairwise comparison"
  - "Swiss tournament"
  - "probabilistic scoring"
date: 2026-05-08
content_hash: c537654ad75d4fba
---

# DICE: Discrete Interpretable Comparative Evaluation with Probabilistic Scoring for RAG

**Conference**: NeurIPS 2025 (ResponsibleFM Workshop)
**arXiv**: [2512.22629](https://arxiv.org/abs/2512.22629)  
**Code**: [GitHub](https://github.com/shiyan-liu/DICE)  
**Area**: LLM Efficiency / RAG Evaluation
**Keywords**: RAG evaluation, LLM-as-judge, pairwise comparison, Swiss tournament, probabilistic scoring

## TL;DR
This paper proposes the DICE framework, which achieves interpretable, robust, and efficient evaluation of RAG systems through a two-stage assessment pipeline (evidence-coupled deep analysis + probabilistic {A, B, Tie} scoring) combined with a Swiss-system tournament. On a Chinese financial QA dataset, DICE attains 85.7% agreement with human experts, substantially outperforming RAGAS (45.7%).

## Background & Motivation
**Background**: RAG system evaluation predominantly relies on scalar metrics (BLEU, ROUGE, BERTScore) and RAG-specific frameworks (e.g., RAGAS with faithfulness/relevancy/context tri-dimensional scoring), yet these approaches suffer from poor interpretability and the absence of uncertainty quantification.

**Limitations of Prior Work**: (1) Scalar scores obscure meaningful differences between systems and cannot explain *why* A is better than B; (2) LLM-as-judge methods exhibit systematic biases (position bias, verbosity bias) and lack confidence estimation; (3) Exhaustive pairwise comparison among multiple systems incurs $O(N^2)$ complexity, which does not scale.

**Key Challenge**: How can interpretability (explaining why one system is better), robustness (confidence-awareness), and efficiency (scalable multi-system comparison) be achieved simultaneously?

**Goal**: Design an evidence-coupled, probabilistic, and efficient RAG evaluation framework.

**Key Insight**: Decompose evaluation into two stages — first perform deep analysis to generate reasoning traces, then extract a probability distribution over {A, B, Tie} from logits for confidence-aware scoring.

**Core Idea**: Two-stage evidence-coupled evaluation (deep analysis → probabilistic scoring) combined with a Swiss-system tournament for efficient ranking.

## Method

### Overall Architecture
The DICE framework comprises: (1) Stage I evidence-coupled deep analysis — an LLM judge performs multi-dimensional analysis on each pair of system outputs, producing structured reasoning traces and a single-token {A, B, Tie} judgment; (2) Stage II probabilistic confidence scoring — logits are extracted to compute a softmax probability distribution, and hard or soft scoring is applied based on a confidence threshold; (3) Swiss-system tournament — dynamic pairing with Elo scoring enables $O(N\log N)$ multi-system ranking.

### Key Designs

1. **Two-Stage Evaluation Protocol**:

    - Function: Stage I produces reasoning traces and preliminary judgments; Stage II quantifies those judgments as probability scores.
    - Mechanism: Analysis and decision-making are decoupled — each evidence–answer pair is first comprehensively analyzed (factual accuracy, completeness, evidence integration quality), after which $P = \text{softmax}([l_A, l_B, l_{Tie}])$ is extracted from logits.
    - Design Motivation: Separating deep analysis from numerical scoring ensures both transparency (traceable reasoning) and quantified confidence.

2. **Confidence-Aware Scoring**:

    - Function: Distinguish high- from low-confidence judgments via the probability margin $\Delta P = P_{max} - P_{second}$.
    - Mechanism: High-confidence cases ($\Delta P \geq 0.1$) use hard scoring (1,0)/(0,1)/(0.5,0.5); low-confidence cases use soft scoring, redistributing the Tie probability proportionally between A and B.
    - Design Motivation: Avoid forcing borderline cases into binary decisions; retaining probability information under low confidence yields more robust comparisons.

3. **Swiss-System Tournament**:

    - Function: Reduce exhaustive $O(N^2)$ pairwise comparisons to an $O(N\log N)$ dynamic-pairing tournament.
    - Mechanism: Systems are initialized at Elo 1500; each round pairs systems with similar current Elo scores (without repetition), and Elo updates are weighted with an upset bonus to accelerate convergence.
    - Design Motivation: For 8 systems, only 16 comparisons are required (vs. 28 for exhaustive evaluation), saving 42.9% of computation while producing rankings identical to the exhaustive approach.

### Evaluation Judgment Hierarchy
System responses are classified into four tiers: fully correct > partially correct > insufficient information > completely incorrect; pairwise outcomes are determined according to this hierarchy.

## Key Experimental Results

### Human Expert Validation (70 Chinese Financial QA Items)

| System | Accuracy | Cohen's κ |
|--------|----------|-----------|
| **DICE** | **85.7%** | **0.742** |
| RAGAS | 45.7% | 0.096 |

### Swiss-System vs. Exhaustive Comparison

| Metric | Swiss-System | Exhaustive |
|--------|-------------|------------|
| Number of comparisons | 16 | 28 |
| Savings | 42.9% | — |
| Ranking consistency | 100% identical | Baseline |
| Elo variance | Lower | Higher |

### Key Findings
- In all disagreement cases, DICE is more "conservative" than human experts, preferring Tie judgments over decisive wins.
- Generator model capacity is the dominant factor: Qwen2.5-7B-based systems consistently outperform Qwen2.5-0.5B-based systems.
- The Swiss-system tournament not only reduces computation but also yields more stable Elo scores than exhaustive evaluation, owing to information-gain-optimal dynamic pairing.

## Highlights & Insights
- **Two-Stage Separation**: Disentangling "why" (reasoning traces) from "how much" (probability scores) yields greater interpretability than direct LLM scoring.
- **Probability Margin Threshold**: Using the margin of the logit-derived probability distribution to distinguish high- from low-confidence judgments is a simple yet effective way to quantify evaluation uncertainty.
- **Swiss-System Tournament for AI Evaluation**: A tournament algorithm borrowed from chess adapts seamlessly to multi-system RAG evaluation scenarios.

## Limitations & Future Work
- **Small dataset scale**: Only 70 QA pairs, limited to the Chinese financial domain.
- **Reliance on a single judge model** (DeepSeek-R1): Inter-judge consistency across different models is not evaluated.
- **Workshop paper**: Experimental scale and system diversity are limited.

## Related Work & Insights
- **vs. RAGAS**: RAGAS employs scalar tri-dimensional scoring (faithfulness + relevancy + context) and lacks a comparative dimension; DICE performs direct pairwise comparison and outputs reasoning traces.
- **vs. Chatbot Arena**: Arena uses crowdsourced preferences with Elo but lacks evidence coupling and confidence quantification; DICE integrates evidence analysis into the evaluation pipeline.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of two-stage probabilistic evaluation and Swiss-system tournament is creative.
- Experimental Thoroughness: ⭐⭐⭐ Workshop scale; dataset is relatively small.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear and notation is rigorous.
- Value: ⭐⭐⭐⭐ Directly actionable for RAG evaluation practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RAG-IGBench: Innovative Evaluation for RAG-based Interleaved Generation in Open-domain Question Answering](rag-igbench_innovative_evaluation_for_rag-based_interleaved_generation_in_open-d.md)
- [\[ICLR 2026\] Hierarchical Concept-based Interpretable Models](../../ICLR2026/information_retrieval/hierarchical_concept-based_interpretable_models.md)
- [\[AAAI 2026\] REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](../../AAAI2026/information_retrieval/reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)
- [\[ICLR 2026\] Summaries as Centroids for Interpretable and Scalable Text Clustering](../../ICLR2026/information_retrieval/summaries_as_centroids_for_interpretable_and_scalable_text_clustering.md)
- [\[ACL 2026\] Bayesian Active Learning with Gaussian Processes Guided by LLM Relevance Scoring](../../ACL2026/information_retrieval/bayesian_active_learning_with_gaussian_processes_guided_by_llm_relevance_scoring.md)

</div>

<!-- RELATED:END -->

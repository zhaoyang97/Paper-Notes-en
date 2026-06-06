---
title: >-
  [Paper Note] De-Anonymization at Scale via Tournament-Style Attribution
description: >-
  [ACL 2026][LLM Safety][Authorship Attribution] This paper proposes DAS (De-Anonymization at Scale), an LLM-based method for large-scale authorship de-anonymization. By employing a tournament-style elimination strategy co…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Authorship Attribution"
  - "De-anonymization"
  - "LLM Privacy Threats"
  - "Tournament-Style Matching"
  - "Peer Review"
date: 2026-05-08
content_hash: 1873ce612e01aa1e
---

# De-Anonymization at Scale via Tournament-Style Attribution

**Conference**: ACL 2026  
**arXiv**: [2601.12407](https://arxiv.org/abs/2601.12407)  
**Code**: None  
**Area**: AI Safety / Privacy  
**Keywords**: Authorship Attribution, De-anonymization, LLM Privacy Threats, Tournament-Style Matching, Peer Review

## TL;DR

This paper proposes DAS (De-Anonymization at Scale), an LLM-based method for large-scale authorship de-anonymization. By employing a tournament-style elimination strategy combined with dense retrieval pre-filtering and multi-round voting aggregation, it enables author matching across tens of thousands of candidate texts, revealing the privacy threats LLMs pose to anonymous platforms such as double-blind reviews.

## Background & Motivation

**Background**: Traditional authorship attribution (AA) is typically studied in small-scale closed-set scenarios—given a few candidate authors and labeled samples, a classifier is trained for attribution. However, real-world anonymous systems (e.g., academic peer reviews) may involve tens of thousands of candidates without labeled data.

**Limitations of Prior Work**: (1) Traditional methods are infeasible in large-scale scenarios as they require building author profiles for every candidate; (2) Recent work using GPT-3/4 for authorship attribution remains limited to small-scale candidate sets; (3) The text analysis capabilities of LLMs may turn large-scale de-anonymization into a realistic threat.

**Key Challenge**: Anonymous systems (e.g., double-blind reviews, whistleblower forums) rely on identity concealment to protect fairness and safety, but LLMs might identify anonymous authors by analyzing signals like writing patterns and domain expertise.

**Goal**: To develop a practical LLM author matching method that can operate within a pool of tens of thousands of candidate texts and to evaluate the extent of the threat it poses to anonymous systems.

**Key Insight**: Model large-scale author matching as a tournament-style elimination—randomly grouping candidates, having the LLM select the most likely match in each group, and advancing winners to the next round to produce a final ranking.

**Core Idea**: Progressive elimination + Dense retrieval pre-filtering + Multi-round voting aggregation = Large-scale de-anonymization within a constrained token budget.

## Method

### Overall Architecture

DAS consists of three components: (1) Dense retrieval pre-filtering—using embedding retrieval to narrow the candidate pool from the $10^5$ level to the $10^3$ level; (2) Tournament-style elimination—partitioning candidates into fixed-size groups where the LLM selects the most likely match, with winners regrouped and compared repeatedly until a top-k ranking is generated; (3) Multi-round voting aggregation—conducting multiple independent runs (with different random groupings) and scoring candidates based on win counts to produce a final aggregated ranking.

### Key Designs

1.  **Tournament-Style Progressive Elimination**:
    - **Function**: Decomposes one-to-many matching into multiple rounds of small-scale comparisons.
    - **Mechanism**: Randomly divides candidates into fixed-size groups (e.g., 5 per group). The LLM compares the query text with all candidates in the group to select the most likely match. Winners advance to the next round for regrouping, repeating until converging to the top-k.
    - **Design Motivation**: LLMs have limited context windows and cannot compare tens of thousands of candidates simultaneously; group comparisons reduce complexity to a logarithmic scale.

2.  **Dense Retrieval Pre-filtering**:
    - **Function**: Reduces the search space to a scale manageable by LLMs.
    - **Mechanism**: Employs an embedding model to encode the query and all candidates, retrieving the top-$N$ (e.g., 1000) via vector similarity as input for the subsequent tournament.
    - **Design Motivation**: Reducing the search space from $10^5$ to $10^3$ makes subsequent LLM comparisons feasible within a token budget.

3.  **Multi-round Voting Aggregation**:
    - **Function**: Improves ranking stability and precision.
    - **Mechanism**: Runs the tournament multiple times independently (with different random groupings), assigning scores to winning candidates in each run. Aggregating scores across all rounds produces the final ranking. Candidates who consistently win across different groupings receive higher rankings.
    - **Design Motivation**: A single random grouping might introduce bias due to unfair competition within a group; multi-round aggregation increases robustness.

### Loss & Training

DAS is a training-free inference-time method that leverages the existing text analysis capabilities of LLMs. The core computation is derived from LLM pairwise (or groupwise) comparison prompts.

## Key Experimental Results

### Main Results

**De-anonymization Performance on Anonymous Review Data**

| Scenario | Candidate Pool Size | DAS Accuracy | Random Baseline |
| :--- | :--- | :--- | :--- |
| Peer Review | Thousands | Far higher than random | ~0.01% |
| Enron Emails | Standard Benchmark | Outperforms prior methods | - |
| Blog Posts | Large Scale | Outperforms prior methods | - |

### Ablation Study

| Component | Effect after Removal | Description |
| :--- | :--- | :--- |
| Dense Retrieval Pre-filtering | Infeasible to run | Candidate pool too large |
| Multi-round Voting | Accuracy decreases | Single round is unstable |
| Tournament Elimination | Accuracy decreases | Progressive comparison is necessary |

### Key Findings

- DAS successfully identifies same-author texts in anonymous review data with thousands of candidates, achieving accuracy far exceeding the random baseline.
- It outperforms previous direct LLM prompting methods on standard benchmarks (Enron, Blogs).
- Multi-round voting significantly enhances ranking precision and stability.
- Dense retrieval pre-filtering is not only an efficiency measure but also improves the quality of subsequent matching by narrowing the candidate pool.

## Highlights & Insights

- Reveals a serious privacy threat—LLMs make large-scale de-anonymization practically feasible.
- The tournament-style design elegantly addresses the computational bottleneck of large-scale one-to-many matching.
- The methodology is generalizable—it can be applied to any text attribution scenario requiring a match from a large candidate pool.

## Limitations & Future Work

- While accuracy is higher than random, it remains limited and may not constitute a practical threat in all specific scenarios.
- The recall quality of dense retrieval may limit the final accuracy.
- As a potential privacy attack tool, it necessitates corresponding defensive measures and ethical discussions.
- The ability to distinguish between authors with similar styles (e.g., members of the same laboratory) may be limited.

## Related Work & Insights

- **vs Huang et al. (2024a)**: Prior work used GPT for small-scale attribution; DAS scales this to the tens-of-thousands level.
- **vs Traditional AA**: Traditional methods require labeled data and small candidate sets; DAS is entirely zero-shot and large-scale.
- **vs Stylometry**: DAS utilizes the implicit stylistic analysis capabilities of LLMs without requiring explicit feature engineering.

## Rating

- Novelty: ⭐⭐⭐⭐ The tournament-style large-scale attribution design is novel, and the privacy threat perspective is significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes real review data and standard benchmarks, though the scale of anonymous review experiments could be even larger.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and systematic description of the method.
- Value: ⭐⭐⭐⭐ Provides practical significance for the security assessment of anonymous systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents](../../ICML2026/llm_safety/from_weak_cues_to_real_identities_evaluating_inference-driven_de-anonymization_i.md)
- [\[ACL 2026\] ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images](generating_attribution_reports_for_manipulated_facial_images_a_dataset_and_basel.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)

</div>

<!-- RELATED:END -->

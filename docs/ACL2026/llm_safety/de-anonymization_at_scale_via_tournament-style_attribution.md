---
title: >-
  [Paper Note] De-Anonymization at Scale via Tournament-Style Attribution
description: >-
  [ACL 2026][LLM Safety][Authorship Attribution] This paper proposes DAS (De-Anonymization at Scale), an LLM-based large-scale authorship de-anonymization method that combines tournament-style elimination…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Authorship Attribution"
  - "De-Anonymization"
  - "LLM Privacy Threat"
  - "Tournament-Style Matching"
  - "Peer Review"
date: 2026-05-08
content_hash: e68ccd7224042a83
---

# De-Anonymization at Scale via Tournament-Style Attribution

**Conference**: ACL 2026
**arXiv**: [2601.12407](https://arxiv.org/abs/2601.12407)  
**Code**: None  
**Area**: AI Safety / Privacy
**Keywords**: Authorship Attribution, De-Anonymization, LLM Privacy Threat, Tournament-Style Matching, Peer Review

## TL;DR

This paper proposes DAS (De-Anonymization at Scale), an LLM-based large-scale authorship de-anonymization method that combines tournament-style elimination, dense retrieval pre-filtering, and multi-round voting aggregation to perform author matching across tens of thousands of candidate texts, revealing the privacy threat that LLMs pose to anonymous platforms such as double-blind peer review.

## Background & Motivation

**Background**: Traditional authorship attribution (AA) has been studied in closed-set, small-scale settings—given a small number of candidate authors and labeled samples, a classifier is trained for attribution. However, real-world anonymous systems (e.g., academic peer review) may involve tens of thousands of candidates with no labeled data.

**Limitations of Prior Work**: (1) Traditional methods are infeasible at large scale, as they require constructing an author profile for each candidate; (2) recent work employing GPT-3/4 for authorship attribution remains limited to small candidate sets; (3) the text analysis capabilities of LLMs may render large-scale de-anonymization a realistic threat.

**Key Challenge**: Anonymous systems (e.g., double-blind peer review, whistleblower forums) rely on identity concealment to protect fairness and safety, yet LLMs may identify anonymous authors by analyzing writing style, domain expertise, and other signals.

**Goal**: To develop an LLM-based author matching method that operates practically over candidate pools of tens of thousands of texts, and to assess the degree of threat it poses to anonymous systems.

**Key Insight**: Large-scale author matching is framed as a tournament-style elimination competition—candidates are randomly grouped, and the LLM selects the most probable match within each group; winners advance to the next round, ultimately producing a ranked list.

**Core Idea**: Progressive elimination + dense retrieval pre-filtering + multi-round voting aggregation = large-scale de-anonymization within a constrained token budget.

## Method

### Overall Architecture

DAS comprises three components: (1) dense retrieval pre-filtering—embedding-based retrieval reduces a candidate pool of $10^5$ to $10^3$; (2) tournament-style elimination—candidates are divided into fixed-size groups, the LLM selects the most probable match within each group, and winners are re-grouped and iteratively compared until a top-$k$ ranking is produced; (3) multi-round voting aggregation—multiple independent runs (with different random groupings) accumulate scores for each candidate's wins, and the aggregated scores yield the final ranking.

### Key Designs

1. **Tournament-Style Progressive Elimination**:

    - Function: Decomposes one-to-many matching into multiple rounds of small-scale comparisons.
    - Mechanism: Candidates are randomly divided into fixed-size groups (e.g., groups of 5); the LLM compares the query text against all candidates within a group and selects the most probable match. Winners are re-grouped in the next round, and this process repeats until convergence to top-$k$.
    - Design Motivation: LLMs have limited context windows and cannot simultaneously compare tens of thousands of candidates; group-wise comparison reduces complexity to logarithmic scale.

2. **Dense Retrieval Pre-Filtering**:

    - Function: Reduces the search space to a scale manageable by LLMs.
    - Mechanism: An embedding model encodes the query and all candidates; vector similarity retrieval selects the top-$N$ (e.g., 1,000) candidates as input to the subsequent tournament.
    - Design Motivation: Reduces the search space from $10^5$ to $10^3$, making subsequent LLM comparisons feasible within the token budget.

3. **Multi-Round Voting Aggregation**:

    - Function: Improves ranking stability and precision.
    - Mechanism: The tournament is run multiple times independently (with different random groupings); each run assigns scores to winning candidates, and scores across all rounds are aggregated to produce the final ranking. Candidates that consistently win across different groupings receive higher rankings.
    - Design Motivation: A single random grouping may introduce bias due to uneven within-group competition; multi-round aggregation increases robustness.

### Loss & Training

DAS is a training-free inference-time method that relies solely on the text analysis capabilities of LLMs. The core computation consists of pairwise comparison prompts issued to the LLM.

## Key Experimental Results

### Main Results

**De-Anonymization Performance on Anonymous Review Data**

| Setting | Candidate Pool Size | DAS Accuracy | Random Baseline |
|---------|-------------------|-------------|----------------|
| Peer Review | Thousands | Far above random | ~0.01% |
| Enron Email | Standard benchmark | Outperforms prior methods | - |
| Blog Posts | Large scale | Outperforms prior methods | - |

### Ablation Study

| Component | Effect of Removal | Notes |
|-----------|------------------|-------|
| Dense retrieval pre-filtering | System becomes infeasible | Candidate pool too large |
| Multi-round voting | Accuracy decreases | Single round is unstable |
| Tournament elimination | Accuracy decreases | Progressive comparison is necessary |

### Key Findings

- DAS successfully identifies co-authored texts among thousands of candidates in anonymous review data, with accuracy far exceeding random chance.
- DAS outperforms prior direct LLM prompting methods on standard benchmarks (Enron, blogs).
- Multi-round voting substantially improves ranking precision and stability.
- Dense retrieval pre-filtering serves not only as an efficiency measure but also improves subsequent matching quality by narrowing the candidate pool.

## Highlights & Insights

- The paper reveals a serious privacy threat—LLMs make large-scale de-anonymization practically feasible.
- The tournament-style design elegantly resolves the computational bottleneck of large-scale one-to-many matching.
- The methodology is generalizable and can be applied to any text attribution scenario requiring matching from a large candidate pool.

## Limitations & Future Work

- Although accuracy is well above random, it remains limited and may be insufficient to constitute a practical threat in certain settings.
- The recall quality of dense retrieval may constrain final accuracy.
- As a potential privacy attack tool, corresponding defense mechanisms and ethical discussions are warranted.
- The method may have limited discriminative power among stylistically similar authors (e.g., members of the same research group).

## Related Work & Insights

- **vs. Huang et al. (2024a)**: Prior work uses GPT for small-scale attribution; DAS scales to tens of thousands of candidates.
- **vs. Traditional AA**: Traditional methods require labeled data and small candidate sets; DAS is fully zero-shot and large-scale.
- **vs. Stylometry**: DAS leverages the implicit stylistic analysis capabilities of LLMs without requiring explicit feature engineering.

## Rating

- Novelty: ⭐⭐⭐⭐ The tournament-style large-scale attribution design is novel, and the privacy threat perspective is important.
- Experimental Thoroughness: ⭐⭐⭐⭐ Real review data and standard benchmarks are used, though the scale of the anonymous review experiments could be larger.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clear and the method is described systematically.
- Value: ⭐⭐⭐⭐ Practically significant for evaluating the security of anonymous systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images](generating_attribution_reports_for_manipulated_facial_images_a_dataset_and_basel.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[ACL 2026\] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG](tpa_next_token_probability_attribution_for_detecting_hallucinations_in_rag.md)
- [\[ACL 2026\] Adaptive Text Anonymization: Learning Privacy-Utility Trade-offs via Prompt Optimization](adaptive_text_anonymization_learning_privacy-utility_trade-offs_via_prompt_optim.md)
- [\[ACL 2026\] Look Twice before You Leap: A Rational Framework for Localized Adversarial Anonymization](look_twice_before_you_leap_a_rational_framework_for_localized_adversarial_anonym.md)

</div>

<!-- RELATED:END -->

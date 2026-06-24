---
title: >-
  [Paper Note] Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning
description: >-
  [ACL 2025][LLM Safety][machine unlearning] This paper systematically studies the selection of the retain set in entity unlearning, proposes the Syntactically Similar Neighbor Set, and reveals that syntactic similarity (rather than domain/entity similarity) is the primary driver of knowledge degradation during unlearning. Regularization with a syntactically similar retain set optimally protects all types of neighbor knowledge simultaneously.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "machine unlearning"
  - "retain set"
  - "entity unlearning"
  - "syntactic similarity"
  - "neighbor set"
  - "regularization"
date: 2026-05-08
content_hash: 6bf0996c9b994bd5
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning

**Conference**: ACL 2025  
**arXiv**: [2502.11441](https://arxiv.org/abs/2502.11441)  
**Code**: Unreleased  
**Area**: LLM Safety  
**Keywords**: machine unlearning, retain set, entity unlearning, syntactic similarity, neighbor set, regularization

## TL;DR

This paper systematically studies the selection of the retain set in entity unlearning, proposes the Syntactically Similar Neighbor Set, and reveals that syntactic similarity (rather than domain/entity similarity) is the primary driver of knowledge degradation during unlearning. Regularization with a syntactically similar retain set optimally protects all types of neighbor knowledge simultaneously.

## Background & Motivation

**Background**: LLM unlearning research concentrates on unlearning method design (GA, DPO, NPO, etc.), lacking in-depth analysis on the composition and selection of the retain set.

**Limitations of Prior Work**: Existing neighbor sets (Domain Neighbor, Entity Neighbor) are constructed based on domains or entity relations, but whether these methods truly capture the most vulnerable knowledge regions during unlearning remains unverified.

**Key Challenge**: The unlearning process collaterally damages knowledge in the retain set. Which specific knowledge is most vulnerable? Are existing assumptions (domain/entity similarity being the most critical) correct?

**Goal**: Answer two research questions: (RQ1) How does unlearning affect the performance of different neighbor sets? (RQ2) Which type of neighbor set achieves the best regularization effect?

**Key Insight**: Re-examine the propagation patterns of unlearning from the perspective of syntactic structure rather than semantics or domain.

**Core Idea**: Unlearning primarily propagates along syntactic patterns—questions with similar structures are most easily forgotten collaterally, and are also the most suitable for retention regularization.

## Method

### Overall Architecture
1. Define three types of neighbor sets: Domain Neighbor (same domain), Entity Neighbor (entity-related), Syntactically Similar Neighbor (syntactically similar).
2. Apply 4 unlearning methods (GA/DPO/NPO/IDK) in both Real-world and TOFU scenarios.
3. Analyze the Relative Utility Drop (RUD) of each neighbor set.
4. Design a $3 \times 3$ experimental matrix to test the regularization performance of different retain set configurations.

### Key Designs

1. **Syntactically Similar Neighbor Set Construction**

    - Step 1: GPT-4o entity masking (masking names, dates, and organization names) to focus on syntactic structure.
    - Step 2: Compute Levenshtein similarity of masked questions and perform clustering (threshold $\theta_{high}$, cluster size $\ge 3$).
    - Step 3: Select entities from the retain set that do not belong to other neighbor sets, and generate new QA pairs according to the clustered syntactic patterns.
    - Step 4: Model probing verification (retains only QA pairs that the model can answer correctly).

2. **Evaluation Metrics**

    - Model Utility (MU): Arithmetic mean of ROUGE, BERT Cosine Sim, Probability, and Entailment.
    - Forget Efficacy (FE): Aggregation of the same metric set on the forget set.
    - Relative Utility Drop (RUD): $(MU_{after} - MU_{before}) / MU_{before} \times 100$
    - The Forget Efficacy of all unlearning methods is standardized to the $0.65\text{--}0.75$ range (for fair comparison).

3. **Regularization Experimental Design**

    - Two regularization losses: GD (standard gradient descent retention) and KL (KL divergence retention).
    - $3 \times 3$ matrix: train retain set (Domain / Entity / Syntactically Similar) $\times$ test retain set.
    - Each cell takes the average RUD of the 4 unlearning methods.

## Key Experimental Results

### Table 1: Relative Utility Drop (RUD) on different neighbor sets during unlearning (Real-world, Llama-3-8B-Instruct)

| Neighbor Set Type | GA | NPO | IDK | DPO | Trend |
|-----------|-----|-----|-----|-----|------|
| Domain Neighbor | Slight Drop | Slight Drop | Slight Drop | Slight Drop | Least affected |
| Entity Neighbor | Slight Drop | Slight Drop | Slight Drop | Slight Drop | Comparable to Domain |
| **Syntactically Similar** | **Significant Drop** | **Significant Drop** | **Significant Drop** | **Significant Drop** | **Most affected** |
| Syn Sim & Domain (Intersection) | Greater Drop | Greater Drop | Greater Drop | Greater Drop | Overlap exacerbates forgetting |

### Table 2: Regularization performance heatmap (GD regularization, average RUD across GA/DPO/NPO/IDK)

| Train Retain ↓ \ Test Retain → | Domain | Entity | Syn. Similar |
|-------------------------------|--------|--------|-------------|
| Domain | Medium | Medium | Low |
| Entity | Medium | Medium | Low |
| **Syntactically Similar** | **Medium-High** | **Medium-High** | **High (+14.7pp)** |

Key Finding: Training with the Syntactically Similar retain set achieves the optimal or near-optimal retention performance across all test retain types.

## Highlights & Insights

- **Challenging Traditional Assumptions**: Previously, domain/entity similarity was widely believed to govern unlearning propagation. This paper systematically proves that syntactic similarity is the dominant factor.
- **Paraphrasing for Robustness Verification**: Even after paraphrasing, syntactically similar neighbors still undergo more forgetting, showing that this is not merely a superficial syntactic matching effect.
- **Gradient Analysis for Mechanistic Explanation**: The gradient norm (Frobenius norm) of syntactically similar instances increases faster during unlearning, quantitatively explaining the propagation mechanism.
- **Practical Guidance**: Utilizing the Syntactically Similar Neighbor Set for regularization is the optimal strategy, with GD showing a more significant difference than KL (14.7pp vs 7.35pp).
- **Consistency Across Scenarios**: Conclusions hold consistently across both Real-world and TOFU scenarios.

## Limitations & Future Work

- Only focuses on entity unlearning; harmful knowledge unlearning and copyright content unlearning might exhibit different patterns.
- Experiments are limited to 7B/8B models (LLaMA-2-7B-Chat, LLaMA-3-8B-Instruct); behaviors in larger models might differ.
- The construction of the syntactically similar set depends on GPT-4o for entity masking, introducing extra cost and uncertainty.
- Does not explore the effect of mixing multiple neighbor sets as the retain set.
- No improvements are made to the unlearning methods themselves; the work only analyzes data selection for the retain set.

## Related Work & Insights

| Dimension | Ours | Opt-Out (Choi et al.) | TOFU (Maini et al.) | RWKU (Jin et al.) |
|------|------|----------------------|---------------------|-------------------|
| Research Focus | retain set selection | Unlearning method (OT regularization) | Fictitious unlearning benchmark | Real-world entity benchmark |
| Neighbor Set Types | **Domain + Entity + Syntactic** | Domain + Entity | Domain | Entity |
| Key Findings | Syntactic > Domain/Entity | Wasserstein > L2 | — | — |
| Method Contribution | Data perspective (retain set selection) | Algorithmic perspective (regularization) | Dataset | Dataset + Evaluation |
| Regularization Analysis | GD vs KL $\times$ 3 retain types | Wasserstein vs various distances | None | None |

## Rating

- Novelty: ⭐⭐⭐⭐ (Novel perspective from syntactic similarity, challenging traditional assumptions of domain and entity similarity)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (4 methods $\times$ 3 neighbor types $\times$ 2 scenarios $\times$ 2 regularizations + paraphrase + gradient analysis)
- Writing Quality: ⭐⭐⭐⭐ (Clarified structure driven by RQs, intuitive tables and figures)
- Value: ⭐⭐⭐⭐ (Provides practical guidance for retain set selection in unlearning methods, complementary to Opt-Out)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport](opt-out_investigating_entity-level_unlearning_for_large_language_models_via_opti.md)
- [\[ACL 2025\] Language Models Can Subtly Deceive Without Lying: A Case Study on Strategic Phrasing](language_models_can_subtly_deceive_without_lying_a_case_study_on_strategic_phras.md)
- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ACL 2026\] Forget What Matters, Keep the Rest: Selective Unlearning of Informative Tokens](../../ACL2026/llm_safety/forget_what_matters_keep_the_rest_selective_unlearning_of_informative_tokens.md)

</div>

<!-- RELATED:END -->

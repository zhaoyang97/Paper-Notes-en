---
title: >-
  [Paper Note] Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport
description: >-
  [ACL 2025][LLM Safety][machine unlearning] This paper proposes Opt-Out, an entity-level LLM unlearning method based on optimal transport theory. By utilizing Sliced Wasserstein Distance to regularize parameter shifts, it achieves fine-grained unlearning. Concurrently, the authors construct ELUDe, the first entity-level unlearning dataset (containing 20 target entities, 144 neighbor entities, 15K+ forget QA pairs, and 90K+ retain QA pairs). Opt-Out comprehensively outperforms…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "machine unlearning"
  - "entity-level"
  - "optimal transport"
  - "Wasserstein distance"
  - "privacy"
  - "GDPR"
date: 2026-05-08
content_hash: 9336d22de7dd873f
---

# Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport

**Conference**: ACL 2025  
**arXiv**: [2406.12329](https://arxiv.org/abs/2406.12329)  
**Code**: [https://github.com/brightjade/Opt-Out](https://github.com/brightjade/Opt-Out)  
**Area**: LLM Security  
**Keywords**: machine unlearning, entity-level, optimal transport, Wasserstein distance, privacy, GDPR

## TL;DR

This paper proposes Opt-Out, an entity-level LLM unlearning method based on optimal transport theory. By utilizing Sliced Wasserstein Distance to regularize parameter shifts, it achieves fine-grained unlearning. Concurrently, the authors construct ELUDe, the first entity-level unlearning dataset (containing 20 target entities, 144 neighbor entities, 15K+ forget QA pairs, and 90K+ retain QA pairs). Opt-Out comprehensively outperforms existing methods on Llama-3.1-8B and Phi-3.5.

## Background & Motivation

**Background**: The "right to be forgotten" in GDPR requires LLMs to delete personal data upon user request. However, fully retraining the model from scratch is prohibitively expensive, which places approximate unlearning methods at the center of research attention.

**Limitations of Prior Work**: Existing approaches (e.g., GA, NPO, DPO) are primarily evaluated on small-scale, random instance-level subsets, failing to address real-world scenarios where all data pertaining to a specific individual must be completely deleted.

**Key Challenge**: While gradient ascent (GA) is capable of unlearning, it easily triggers catastrophic model collapse (where retain quality drops to 0). Even with the inclusion of training on a retain set, the balance between unlearning and retention remains suboptimal.

**Goal**: Define the entity-level unlearning task, build a large-scale evaluation dataset, and propose a fine-grained unlearning method based on optimal transport.

**Key Insight**: By using Wasserstein distance to measure the "transportation cost" between current parameters and initial parameters, parameters critical to unlearning can shift significantly, while parameters essential for retention remain stable.

**Core Idea**: Parameter distribution distance regularization under the optimal transport framework allows for more fine-grained parameter-level unlearning control compared to L2 or Cosine distance.

## Method

### Overall Architecture
1. **Forget Set Construction**: Select 20 high-popularity target entities from Wikipedia (using page view counts as a proxy metric), generate QA pairs paragraph-by-paragraph using GPT-4o, and deduplicate utilizing BERT embeddings, resulting in ~647 QA pairs per entity.
2. **Retain Set Construction**: For each target entity, select 10 neighbor entities (based on bidirectional links, high page views, and person-type entities, inspired by hard negatives) and generate QA pairs similarly. Additionally, include 50K Alpaca-GPT4 instruction data to serve as the world set.
3. **Optimal Transport Unlearning**: Unlearn using NPO loss + Retain loss + Wasserstein regularization.

### Key Designs

1. **ELUDe Dataset**

    - 20 target entities + 144 unique neighbor entities (with overlaps)
    - 15,651 forget QA pairs + 90,954 retain QA pairs
    - Neighbor selection criteria: bidirectional Wikipedia links + top-10 page views over the past 3 years + person-type entities
    - Larger data volume compared to TOFU/RWKU (covering complete knowledge of each entity)

2. **NPO Unlearning Loss**

    - More stable than GA: simplifies to GA in the high-temperature limit, but remains bounded, significantly delaying model collapse
    - Equation: $\mathcal{L}_{\text{NPO}} = -\mathbb{E}_{\mathcal{D}_f}[\log\sigma(-\eta\log\frac{\phi_\theta(y|x)}{\phi_{\text{ref}}(y|x)})]$

3. **Sliced Wasserstein Regularization**

    - Directly computing Wasserstein distance has a complexity of $O(n^3\log n)$, making it computationally infeasible.
    - Instead, Sliced Wasserstein Distance (SWD) is adopted, which projects distributions randomly to low-dimensional spaces and computes the 1D Wasserstein distance.
    - Total loss: $\mathcal{L} = \mathcal{L}_{\text{NPO}} + \mathcal{L}_{\text{RT}} + \lambda \cdot SW_p(\theta, \theta_0)$
    - Key advantage: Incorporates structural information of the parameter distribution, yielding more fine-grained control than pointwise distances such as L2 (Euclidean) and Cosine.

## Key Experimental Results

### Main Results (Llama-3.1-8B-Instruct, average of 5 entities)

| Method | FQ ↑ | RQ ↑ | MMLU | ARC-C | 8 Benchmark Avg ↑ |
|------|------|------|------|-------|-------------------|
| Original | 45.5 | 51.2 | 68.1 | 51.8 | 64.7 |
| GA* (collapse) | 70.9 | 0.0 | 33.9 | 23.6 | 33.8 |
| NPO* (collapse) | 89.7 | 0.0 | 36.3 | 24.7 | 37.5 |
| NPO+RT | 82.6 | 46.6 | 62.5 | 50.1 | 62.8 |
| IDK+RT | 71.9 | 46.1 | 63.2 | 49.4 | 62.8 |
| **Opt-Out** | **87.8** | **46.6** | **63.2** | 49.8 | **63.3** |

### Ablation Study (Llama-3.1-8B-Instruct)

| Regularization Distance | FQ ↑ | RQ ↑ | Benchmark Avg ↑ |
|-----------|------|------|-----------------|
| **Wasserstein** | **87.8** | **46.6** | **63.3** |
| Euclidean | 81.5 | 46.2 | 63.0 |
| Cosine | 81.6 | 45.8 | 62.8 |
| Chebyshev | 86.3 | 45.4 | 62.2 |
| Manhattan | 47.0 | 50.9 | 64.6 (regularization too strong, barely unlearned) |

## Highlights & Insights

- **First large-scale entity-level unlearning dataset ELUDe**: Compiles 20 entities + 144 neighbors, with total QA size greatly exceeding TOFU/RWKU.
- **Novel optimal transport perspective**: SWD regularization leverages parameter distribution structural information, achieving more fine-grained control than Euclidean distance.
- **Comprehensive evaluation**: Demonstrates optimal performance across MIA defense (Opt-Out 48.6% $\approx$ ideal 50%) and 9 types of adversarial attacks.
- **Hard positive effects of neighbor entity data**: Removing neighbor data significantly drops RQ, validating the design intuition similar to hard negatives in contrastive learning.

## Limitations & Future Work

- The dataset is based on Wikipedia entities, which might differ from real-world user privacy data.
- The model might still generate gibberish after unlearning; the user experience issue is not fully resolved.
- Constrained by computational resources, the method was not validated on 70B+ scale models.
- Although SWD reduces complexity, it still introduces extra overhead; the paper does not report detailed training time comparisons.

## Related Work & Insights

| Dimension | Opt-Out (Ours) | TOFU (Maini et al.) | RWKU (Jin et al.) |
|------|---------------|---------------------|-------------------|
| Unlearning Granularity | Entity-level (complete knowledge) | Instance-level (fictional authors) | Entity-level (real celebrities) |
| Data Scale | 15K forget + 90K retain | 20 QA/author × 200 | 2,879 QA |
| Neighbor Entities | 144 entities (hard negatives) | No specific design | Yes, but small-scale |
| Regularization Method | Wasserstein (SWD) | None | None |
| Evaluation Dimensions | FQ + RQ + MIA + adversarial attacks | FQ + general benchmark | FQ + neighbors + attacks |

## Rating

- Novelty: ⭐⭐⭐⭐ (Optimal transport for unlearning regularization is a fresh perspective, and the ELUDe dataset is valuable)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Evaluated across 2 models × multiple baselines × MIA × adversarial attacks × ablations)
- Writing Quality: ⭐⭐⭐⭐ (Well-structured with rich tables and figures)
- Value: ⭐⭐⭐⭐ (Entity-level unlearning is a key requirement for GDPR compliance, making both the method and the dataset highly practical)

| No Reg | High | Low |
| L2 Reg | Medium | Medium |
| **Wasserstein** | **High** | **High** |

### Key Findings
- Opt-Out achieves the best balance between unlearning and retention.
- Wasserstein distance outperforms L2 regularization.
- Neighbor entities are a crucial test: evaluation without neighbor retention testing is incomplete.
- General capabilities remain largely unaffected.

## Highlights & Insights
- Entity-level unlearning is closer to real-world demands than instance-level unlearning.
- Wasserstein distance regularization is theoretically elegant.
- The concept of neighbor entities is inspired by contrastive learning.

## Limitations & Future Work
- Based only on Wikipedia, not real private data.
- Future directions: large-scale entity unlearning, integration with RAG.

## Related Work & Insights
- **vs Jang et al.**: Gradient ascent is prone to collapse.
- **vs TOFU**: TOFU is instance-level, while Opt-Out is entity-level.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to define entity-level unlearning + Wasserstein regularization
- Experimental Thoroughness: ⭐⭐⭐⭐ 20 entities + 144 neighbors
- Writing Quality: ⭐⭐⭐⭐ Clear
- Value: ⭐⭐⭐⭐⭐ Direct value for LLM privacy compliance

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)
- [\[ACL 2025\] Which Retain Set Matters for LLM Unlearning? A Case Study on Entity Unlearning](which_retain_set_matters_for_llm_unlearning_a_case_study_on_entity_unlearning.md)
- [\[ACL 2025\] Modality-Aware Neuron Pruning for Unlearning in Multimodal Large Language Models](manu_modality_aware_unlearning.md)
- [\[ACL 2025\] MMUnlearner: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models](mmunlearner_reformulating_multimodal_machine_unlearning_in_the_era_of_multimodal.md)
- [\[ACL 2025\] REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space](revs_unlearning_sensitive_information_in_language_models_via_rank_editing_in_the.md)

</div>

<!-- RELATED:END -->

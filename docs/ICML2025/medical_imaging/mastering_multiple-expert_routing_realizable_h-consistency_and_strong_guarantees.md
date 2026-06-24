---
title: >-
  [Paper Note] Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees
description: >-
  [ICML 2025][Medical Imaging][Learning to Defer] This paper proposes new surrogate loss functions and efficient algorithms for the multiple-expert routing (learning to defer) problem, establishing theoretical guarantees for realizable H-consistency, H-consistency bounds, and Bayes consistency across both single-stage and two-stage learning scenarios.
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "Learning to Defer"
  - "Multiple-Expert Routing"
  - "H-Consistency"
  - "Surrogate Loss"
  - "Bayes Consistency"
date: 2026-05-08
content_hash: 5d5efe7dca0a8b2c
---

# Mastering Multiple-Expert Routing: Realizable H-Consistency and Strong Guarantees

**Conference**: ICML 2025  
**arXiv**: [2506.20650](https://arxiv.org/abs/2506.20650)  
**Code**: None  
**Area**: Medical Imaging (Learning to Defer/Expert Routing)  
**Keywords**: Learning to Defer, Multiple-Expert Routing, H-Consistency, Surrogate Loss, Bayes Consistency

## TL;DR
This paper proposes new surrogate loss functions and efficient algorithms for the multiple-expert routing (learning to defer) problem, establishing theoretical guarantees for realizable H-consistency, H-consistency bounds, and Bayes consistency across both single-stage and two-stage learning scenarios.

## Background & Motivation

**Background**: Multiple-expert routing (or learning to defer) is a decision-making problem that optimally allocates input instances to different experts (including AI models and human experts), which is increasingly important in NLP generation, image processing, and medical diagnosis. For example: simple problems are handled by small models, while complex ones are referred to larger models or human experts.

**Limitations of Prior Work**: Recent studies have proposed various surrogate loss functions to optimize routing decisions, but there remain unresolved issues regarding consistency guarantees. In particular: (1) the realizable H-consistency of existing surrogate losses has not been proven; (2) whether H-consistency bounds exist; (3) how Bayes consistency is guaranteed in multiple-expert scenarios.

**Key Challenge**: Practical deployment requires guarantees over a finite hypothesis space $H$ (rather than infinite capacity), but most existing theories remain at the level of Bayes consistency (infinite capacity). The 0-1 loss of routing is non-optimizable and requires a surrogate loss, but the consequences of using surrogate losses need to be quantified.

**Goal**: To provide a comprehensive theoretical guarantee framework for both single-stage and two-stage multiple-expert routing.

**Key Insight**: Starting from the design of surrogate losses, establish consistency guarantees at all levels through rigorous mathematical analysis.

**Core Idea**: Design a new family of surrogate losses that simultaneously satisfy realizable H-consistency and H-consistency bounds, providing the strongest theoretical guarantees for multiple-expert routing.

## Method

### Overall Architecture

- **Problem Definition**: Given an input $x$, select from $K+1$ options ($K$ experts + the predictor itself), aiming to minimize the total error and computational cost of allocation.
- **Single-stage**: Jointly learn the predictor and the routing function.
- **Two-stage**: Fix the experts and only learn the routing function.

### Key Designs

1. **Single-Stage Realizable H-Consistent Surrogate Loss Family**:

    - Propose a new family of surrogate loss functions that satisfy realizable H-consistency.
    - Meaning of realizable H-consistency: when the surrogate loss achieves the optimum in $H$, the original 0-1 loss also achieves the optimum in $H$.
    - Further prove that a specific member within this family satisfies H-consistency bounds (quantifying the relationship: surrogate loss gap $\rightarrow$ original loss gap).
    - **Why it matters**: This is the strongest consistency guarantee, whereas previously only Bayes consistency was available (which requires an infinite capacity assumption).

2. **Multi-Level Theoretical Guarantees for the Two-Stage Setting**:

    - For the two-expert scenario: derive new surrogate losses that simultaneously satisfy realizable H-consistency, H-consistency bounds, and Bayes consistency.
    - For the multiple-expert scenario: obtain similar guarantees under natural assumptions (bounded expert errors).
    - **Why it matters**: The two-stage setting is more practical (as experts are typically already deployed), but its theoretical analysis is more challenging.

3. **Enhanced Guarantees under Low-Noise Conditions**:

    - Provide tighter bounds under low-noise assumptions (i.e., when most inputs have a clear optimal choice).
    - Features include faster convergence rates and smaller constants.
    - **Why it matters**: In real-world scenarios, most inputs indeed have a clear optimal route.

### Loss & Training

- **Single-stage surrogate loss**: A parameterized family of losses $\Phi_\alpha$ satisfying convexity + consistency.
- **Two-stage surrogate loss**: A new loss designed specifically for the fixed-expert scenario.
- Training method: Standard gradient descent/SGD (surrogate loss guarantees optimizability).
- Hyperparameter $\alpha$ controls the trade-off of different consistency properties.

## Key Experimental Results

### Main Results

| Dataset | Setting | Ours | Prev. SOTA | Description |
|--------|------|---------|---------|------|
| CIFAR-10 + expert | Two-stage / 2-expert | Better | L2D series | Acc improvement |
| CIFAR-100 + expert | Two-stage / multi-expert | Better | CE-based | Acc improvement |
| HAM10000 (skin) | Medical routing | Competitive | Existing methods | Safety-critical scenario |
| NLP routing | Multi-expert | Better | mixture baselines | NLG tasks |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Different $\alpha$ values | Monotonic changes | Verifies the controllability of the parameterized family |
| Bayes consistency only | Poor generalization | Finite capacity requires H-consistency |
| Single-stage vs Two-stage | Respective advantages | Depends on whether experts can be retrained |
| Low-noise vs High-noise | Low-noise is better | Verifies the enhanced guarantees |

### Key Findings

- The new surrogate losses consistently outperform or match existing methods in experiments.
- Theoretical guarantees indeed translate to better performance in finite-capacity (practical network) scenarios.
- Performance improvement under low-noise conditions is more pronounced.
- Safety guarantees in medical routing scenarios hold practical significance.

## Highlights & Insights

1. **Theoretical Completeness**: Provides the first complete theory of realizable H-consistency + H-consistency bounds + Bayes consistency for multiple-expert routing.
2. **Practicality**: The theory is not just on paper; the new losses indeed perform better in experiments.
3. **Comprehensive Coverage**: Covers single-stage + two-stage, dual-expert + multi-expert, and general conditions + low-noise conditions.
4. **Answering Open Questions**: Resolves several open theoretical questions from previous literature.

## Limitations & Future Work

1. Full guarantees for multiple experts require natural assumptions (bounded expert errors) and are not completely unconditional.
2. Experimental scale is relatively small, with insufficient validation on large-scale LLM routing.
3. The cost model for computation is relatively simplified (assuming the cost of each expert is known).
4. Dynamic/non-stationary expert scenarios are not considered.

## Related Work & Insights

- The L2D series of Madras et al., Mozannar & Sontag are the primary precursor works.
- The H-consistency theoretical framework of Mohri et al. provides the methodological foundation.
- Insight: The H-consistency analysis method can be generalized to other decision-making problems involving multi-system collaboration.

## Rating
- Novelty: ⭐⭐⭐⭐ Solid theoretical contribution
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated in multiple scenarios
- Writing Quality: ⭐⭐⭐⭐ Rigorously written
- Value: ⭐⭐⭐⭐ Provides a solid theoretical foundation for multiple-expert routing

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] CLoE: Expert Consistency Learning for Missing Modality Segmentation](../../CVPR2025/medical_imaging/cloe_expert_consistency_learning_for_missing_modality_segmentation.md)
- [\[ICML 2025\] MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding](medxpertqa_benchmarking_expert-level_medical_reasoning_and_understanding.md)
- [\[ICML 2025\] Do Multiple Instance Learning Models Transfer?](do_multiple_instance_learning_models_transfer.md)
- [\[ICCV 2025\] ViCTr: Vital Consistency Transfer for Pathology Aware Image Synthesis](../../ICCV2025/medical_imaging/victr_vital_consistency_transfer_for_pathology_aware_image_synthesis.md)
- [\[CVPR 2025\] EchoONE: Segmenting Multiple Echocardiography Planes in One Model](../../CVPR2025/medical_imaging/echoone_segmenting_multiple_echocardiography_planes_in_one_model.md)

</div>

<!-- RELATED:END -->

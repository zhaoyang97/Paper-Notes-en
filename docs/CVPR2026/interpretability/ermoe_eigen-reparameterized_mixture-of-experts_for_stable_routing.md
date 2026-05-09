---
title: >-
  [Paper Note] ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization
description: >-
  [CVPR 2026][Mixture of Experts] ERMoE proposes reparameterizing MoE expert weights within an orthogonal eigenbasis and replacing conventional routing logits with eigenbasis scores (cosine similarity), achieving stable routing and interpretable expert specialization without auxiliary load-balancing losses.
tags:
  - CVPR 2026
  - Mixture of Experts
  - eigen-reparameterization
  - routing stability
  - expert specialization
  - Vision Transformer
date: 2026-05-08
content_hash: f08bc36a12c3f624
---

# ERMoE: Eigen-Reparameterized Mixture-of-Experts for Stable Routing and Interpretable Specialization

**Conference**: CVPR 2026
**arXiv**: [2511.10971](https://arxiv.org/abs/2511.10971)
**Code**: None
**Area**: Interpretability
**Keywords**: Mixture of Experts, eigen-reparameterization, routing stability, expert specialization, Vision Transformer

## TL;DR
ERMoE proposes reparameterizing MoE expert weights within an orthogonal eigenbasis and replacing conventional routing logits with eigenbasis scores (cosine similarity), achieving stable routing and interpretable expert specialization without auxiliary load-balancing losses.

## Background & Motivation
1. **Background**: MoE architectures scale model capacity via sparse activation, yet misalignment between routing logits and expert representations causes routing instability and underutilization of experts, while load imbalance introduces computational bottlenecks.
2. **Limitations of Prior Work**: Auxiliary load-balancing losses (LBL) reduce imbalance but introduce interfering gradients that weaken expert specialization and downstream accuracy. The root cause is the decoupling between the router's and experts' representation spaces.
3. **Key Challenge**: Routers must accurately assign tokens to the most suitable experts, yet conventional learnable routing logits operate in an unconstrained parameter space with no intrinsic connection to the actual representational capacity of each expert.
4. **Goal**: Design a routing mechanism whose assignment decisions directly reflect each expert's intrinsic representational subspace, fundamentally resolving the router–expert misalignment problem.
5. **Key Insight**: Reparameterize expert weights via SVD-style eigen-decomposition so that routing is based on feature–basis alignment rather than learned logits.
6. **Core Idea**: Each expert's weight is factorized into an orthogonal eigenbasis $\mathbf{W}^{(e)} = \mathbf{U}^{(e)} \text{diag}(s^{(e)}) \mathbf{V}^{(e)\top}$, and routing scores are computed as the cosine similarity between token features and the expert basis.

## Method

### Overall Architecture
A ViT backbone extracts token embeddings. Within each ERMoE block, the router computes an eigenbasis score for each expert—the cosine similarity between the token feature and its attention-weighted contextual projection onto the expert basis—selects the top-$k$ experts exceeding a confidence threshold $T$, and aggregates their outputs with normalized score weighting.

### Key Designs

1. **Eigen-Reparameterized Experts**:
   - **Function**: Constrain expert weights within an orthogonal basis space.
   - **Mechanism**: Each expert weight is factorized as $\mathbf{W}^{(e)} = \mathbf{U}^{(e)} \text{diag}(s^{(e)}) \mathbf{V}^{(e)\top}$, where $\mathbf{U}, \mathbf{V}$ are orthogonal matrices and $s$ is a learnable scaling factor. Orthogonal constraints enforce separable expert directions, reducing feature redundancy and representation collapse.
   - **Design Motivation**: The parameter spaces of conventional MoE experts overlap heavily, causing different experts to learn similar representations. Orthogonal basis constraints mathematically guarantee the separability of expert subspaces.

2. **Eigenbasis Routing Scores**:
   - **Function**: Route based on content alignment rather than unconstrained logits.
   - **Mechanism**: For a given expert, the input token and its attention-weighted context are each projected onto that expert's eigenbasis; the routing score is the cosine similarity between the two projections. Only experts whose scores exceed confidence threshold $T$ are eligible for selection, from which top-$k$ are taken.
   - **Design Motivation**: Binding routing to each expert's actual representation space ensures that assignment decisions directly reflect feature–basis alignment, eliminating the need for LBL and its gradient interference.

3. **ERMoE-ba Variant for Brain Age Prediction**:
   - **Function**: Extend ERMoE to 3D medical imaging.
   - **Mechanism**: A 2D ViT is extended to a 3D ViT to process T1 MRI volumetric data; routing operates between regional experts and free experts, and the weighted output drives a brain age estimator. Expert routing patterns enable anatomically interpretable expert specialization.
   - **Design Motivation**: Validate the effectiveness of ERMoE beyond natural images and demonstrate the interpretability of the routing mechanism.

### Loss & Training
Standard classification/regression losses are used; no auxiliary load-balancing loss is required. Orthogonality constraints are maintained via Cayley parameterization or Gram–Schmidt orthogonalization.

## Key Experimental Results

### Main Results

| Dataset | Metric | ERMoE | V-MoE | Soft MoE | Gain |
|---------|--------|-------|-------|----------|------|
| ImageNet | Top-1 Acc | SOTA | 2nd | — | Clear advantage |
| COCO (retrieval) | R@1 | SOTA | — | 2nd | Improved |
| Flickr30K (retrieval) | R@1 | SOTA | — | — | Improved |
| Brain Age Prediction | MAE | >7% reduction | — | — | Significant gain |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Full ERMoE | Best | Orthogonal basis + eigenbasis routing |
| Standard routing logits | Degraded | Lacks content alignment |
| With LBL | Degraded | LBL introduces interfering gradients |
| Non-orthogonal experts | Degraded | Increased expert overlap |

### Key Findings
- ERMoE achieves a flatter expert load distribution without LBL, indicating that alignment-based routing naturally promotes load balance.
- The brain age variant reveals anatomically interpretable expert specialization, with different experts attending to distinct brain regions.
- The Gini coefficient decreases substantially compared to DINO's 0.97, confirming the mitigation of routing imbalance.

## Highlights & Insights
- **Fundamental resolution of router–expert misalignment**: Rather than patching symptoms (e.g., adding LBL), the method eliminates the problem at the representation level.
- **Interpretability as a natural byproduct**: Orthogonal bases render expert directions separable, naturally yielding interpretable specialization patterns.
- The methodology is transferable to MoE models in NLP.

## Limitations & Future Work
- Orthogonality constraints introduce additional training computational overhead.
- Validation is currently limited to ViT; applicability to larger-scale language MoE models has not been tested.
- The threshold $T$ affects performance and requires careful tuning.

## Related Work & Insights
- **vs. V-MoE**: V-MoE first introduced sparse experts into ViT but still relies on standard routing logits. ERMoE replaces these with eigenbasis scores for greater stability.
- **vs. Soft MoE**: Soft MoE replaces hard top-$k$ with soft assignments, but scoring still operates in an auxiliary space. ERMoE binds scoring to each expert's internal representations.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Eigen-reparameterization combined with alignment-based routing constitutes a fundamental innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-task validation and the brain age application demonstrate interpretability.
- **Writing Quality**: ⭐⭐⭐⭐ In-depth problem analysis with clear mathematical exposition.
- **Value**: ⭐⭐⭐⭐⭐ Establishes a new paradigm for MoE routing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Draft and Refine with Visual Experts](draft_and_refine_with_visual_experts.md)
- [\[AAAI 2026\] DR.Experts: Differential Refinement of Distortion-Aware Experts for Blind Image Quality Assessment](../../AAAI2026/interpretability/drexperts_differential_refinement_of_distortion-aware_experts_for_blind_image_qu.md)
- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)
- [\[ICLR 2026\] RADAR: Reasoning-Ability and Difficulty-Aware Routing for Reasoning LLMs](../../ICLR2026/interpretability/radar_reasoning-ability_and_difficulty-aware_routing_for_reasoning_llms.md)
- [\[NeurIPS 2025\] Towards Interpretability Without Sacrifice: Faithful Dense Layer Decomposition with Mixture of Decoders](../../NeurIPS2025/interpretability/towards_interpretability_without_sacrifice_faithful_dense_layer_decomposition_wi.md)

</div>

<!-- RELATED:END -->

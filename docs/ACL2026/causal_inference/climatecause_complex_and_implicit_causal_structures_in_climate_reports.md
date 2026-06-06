---
title: >-
  [Paper Note] ClimateCause: Complex and Implicit Causal Structures in Climate Reports
description: >-
  [ACL 2026][Causal Inference][Causal Discovery] ClimateCause constructs the first expert-annotated dataset for complex and implicit causal structures in climate reports (874 causal relations), supporting nested causality…
tags:
  - "ACL 2026"
  - "Causal Inference"
  - "Causal Discovery"
  - "Climate Change"
  - "Implicit Causality"
  - "Nested Causality"
  - "IPCC Report"
content_hash: 158d3a4bfe2d51f2
---

# ClimateCause: Complex and Implicit Causal Structures in Climate Reports

**Conference**: ACL 2026
**arXiv**: [2604.14856](https://arxiv.org/abs/2604.14856)  
**Code**: [GitHub](https://github.com/laallein/ClimateCause)  
**Area**: Causal Inference / Dataset
**Keywords**: Causal Discovery, Climate Change, Implicit Causality, Nested Causality, IPCC Report

## TL;DR
ClimateCause constructs the first expert-annotated dataset for complex and implicit causal structures in climate reports (874 causal relations), supporting nested causality, multi-event decomposition, correlation direction, and spatiotemporal context annotation. LLM benchmarking shows causal chain reasoning remains a major challenge.

## Method

### Key Designs

1. **Noun Phrase Reconstruction and Multi-Event Decomposition**: Standardizes causes/effects into comparable canonical forms, with Belongs_to and Combined fields distinguishing exemplification from joint action.

2. **Implicit and Nested Causality Annotation**: Captures causality expressed through semantics rather than explicit triggers (e.g., "anthropogenic greenhouse gas emissions" implicitly contains humans → greenhouse gas emissions).

3. **Causal Graph Semantic Complexity Readability Metric**: Five complexity dimensions with min-max normalization, measuring the cognitive complexity of causal reasoning rather than surface-level readability.

## Key Experimental Results

- 57.33% of statements contain semantically complex causal structures
- LLMs perform far worse on causal chain reasoning than correlation inference
- Statement length significantly correlates with causal complexity ($r=0.590$)

## Highlights & Insights
- Causal structure readability metric is novel and practically valuable — helps assess report comprehensibility for policymakers
- Nested causality concept transferable to other specialized domains (medical reports, legal documents)

## Rating

- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] I-CAM-UV: Integrating Causal Graphs over Non-Identical Variable Sets Using Causal Additive Models with Unobserved Variables](../../AAAI2026/causal_inference/i-cam-uv_integrating_causal_graphs_over_non-identical_variable_sets_using_causal.md)
- [\[ACL 2026\] Learning Invariant Modality Representation for Robust Multimodal Learning from a Causal Inference Perspective](learning_invariant_modality_representation_for_robust_multimodal_learning_from_a.md)
- [\[ACL 2026\] iTAG: Inverse Design for Natural Text Generation with Accurate Causal Graph Annotations](itag_inverse_design_for_natural_text_generation_with_accurate_causal_graph_annot.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](../../ICML2026/causal_inference/controllable_generative_sandbox_for_causal_inference.md)
- [\[AAAI 2026\] Sparse Additive Model Pruning for Order-Based Causal Structure Learning](../../AAAI2026/causal_inference/sparse_additive_model_pruning_for_order-based_causal_structure_learning.md)

</div>

<!-- RELATED:END -->

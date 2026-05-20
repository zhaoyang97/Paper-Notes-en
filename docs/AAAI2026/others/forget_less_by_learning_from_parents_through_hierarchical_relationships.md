---
title: >-
  [Paper Note] Forget Less by Learning from Parents Through Hierarchical Relationships
description: >-
  [AAAI 2026][Custom diffusion models] This paper proposes FLLP (Forget Less by Learning from Parents), a framework that mitigates catastrophic forgetting in custom diffusion models (CDMs) by establishing parent-child hier…
tags:
  - "AAAI 2026"
  - "Custom diffusion models"
  - "catastrophic forgetting"
  - "hyperbolic space"
  - "hierarchical relationships"
  - "concept learning"
date: 2026-05-08
content_hash: 5170455f8822e9f0
---

# Forget Less by Learning from Parents Through Hierarchical Relationships

**Conference**: AAAI 2026
**arXiv**: [2601.01892](https://arxiv.org/abs/2601.01892)  
**Code**: None  
**Area**: Continual Learning / Image Generation
**Keywords**: Custom diffusion models, catastrophic forgetting, hyperbolic space, hierarchical relationships, concept learning

## TL;DR

This paper proposes FLLP (Forget Less by Learning from Parents), a framework that mitigates catastrophic forgetting in custom diffusion models (CDMs) by establishing parent-child hierarchical relationships among concepts in hyperbolic space. It leverages the tree-structure modeling capability of the Lorentz manifold to preserve knowledge during new concept learning and enable continual concept integration.

## Background & Motivation

**Background**: Custom Diffusion Models (CDMs) such as DreamBooth and Textual Inversion can inject new concepts into pre-trained diffusion models using a small number of example images, enabling personalized image generation for user-specific objects such as pets or particular items.

**Limitations of Prior Work**: When sequentially learning multiple new concepts, CDMs suffer severely from catastrophic forgetting—learning a new concept overwrites the representations of previously learned ones. Existing methods primarily focus on minimizing inter-concept interference (e.g., orthogonalization, parameter isolation), while neglecting potentially positive interactions among concepts, where knowledge from related concepts could mutually reinforce one another.

**Key Challenge**: Prior methods frame multi-concept learning as a conflict management problem (how to prevent new concepts from overwriting old ones), rather than as a collaborative opportunity (how to leverage inter-concept relationships to facilitate learning and retention).

**Goal**: (1) Model hierarchical structural relationships among concepts; (2) leverage knowledge from "parent concepts" to guide the learning of "child concepts"; (3) preserve and enhance previously learned concepts while acquiring new ones.

**Key Insight**: Concepts are organized into a hierarchical tree structure, where more general concepts (e.g., "dog") serve as parent nodes of more specific ones (e.g., "my golden retriever"). These concepts are embedded in hyperbolic space (Lorentz manifold), exploiting hyperbolic geometry's natural suitability for modeling tree-structured hierarchies.

**Core Idea**: By defining parent-child relationships among concepts in hyperbolic space, previously learned "parent concepts" serve as anchors and guides for learning new "child concepts," achieving a dual benefit of knowledge retention and new concept adaptation.

## Method

### Overall Architecture

FLLP augments the standard CDM training pipeline with hierarchical modeling in hyperbolic space. The input is a sequence of concepts to be learned, each with a small number of example images, and the framework learns them sequentially. The core enhancement lies in embedding concepts onto the Lorentz manifold within the latent representation space, establishing parent-child relationships, and using parent concept embeddings to constrain and guide the learning of child concepts.

### Key Designs

1. **Lorentz Manifold Embedding**:

    - Function: Provides a geometric space suited for tree-structured hierarchical modeling of concept representations.
    - Mechanism: Text or visual embeddings of concepts are mapped onto the Lorentz manifold $\mathbb{H}^n$ (a model of hyperbolic space). In hyperbolic space, the distance metric naturally reflects hierarchical relationships—points closer to the origin are more "general," while those farther away are more "specific." The Lorentz distance $d_L(u, v) = \text{arccosh}(-\langle u, v \rangle_L)$ is used to measure semantic distances between concepts.
    - Design Motivation: Euclidean embeddings cannot effectively model hierarchical relationships, as tree structures suffer from severe distortion when embedded in Euclidean space. The exponential volume growth of hyperbolic space matches the branching properties of tree structures, enabling low-distortion embedding of hierarchies.

2. **Parent-Child Concept Relationship Mechanism**:

    - Function: Leverages knowledge from previously learned concepts to guide the learning of new ones.
    - Mechanism: When learning a new concept, the most semantically related previously learned concept is identified in hyperbolic space as its "parent concept." The new concept's initial embedding is derived from the parent concept via the exponential map with fine-tuning. A parent-child distance constraint is applied during training—the new concept must not deviate too far from its parent, yet must not fully coincide with it either.
    - Design Motivation: Initializing from the parent concept leverages prior knowledge (e.g., "golden retriever" inherits most characteristics of "dog"), reducing the burden of learning from scratch. The distance constraint simultaneously ensures that the parent concept is not overwritten (not too close) and that the new concept remains semantically related (not too far).

3. **Anti-Forgetting Regularization**:

    - Function: Constrains the stability of concept embeddings in hyperbolic space.
    - Mechanism: A displacement penalty is applied to embeddings of previously learned concepts, preventing their positions in hyperbolic space from shifting significantly due to new concept learning. Key parameters associated with old concepts in the diffusion model (e.g., key/value matrices in cross-attention layers) are also protected.
    - Design Motivation: Even with the protection of parent-child relationships, gradient updates may still indirectly affect old concept representations. Additional regularization provides a secondary safeguard.

### Loss & Training

The total loss comprises: (1) the standard CDM training loss (diffusion denoising loss); (2) a hyperbolic distance constraint loss (parent-child relationship preservation); and (3) a stability regularization term for old concept embeddings. Training proceeds sequentially over concepts, with each new concept's learning guided by its parent concept.

## Key Experimental Results

### Main Results

Evaluation is conducted on three public datasets and one synthetic benchmark.

| Dataset | Metric | FLLP | Prev. SOTA | Gain | Notes |
|--------|------|------|----------|------|------|
| Public Dataset 1 | Robustness | Best | -- | Consistent gain | Multi-concept retention |
| Public Dataset 2 | Generalization | Best | -- | Consistent gain | New concept generation quality |
| Public Dataset 3 | Overall metric | Best | -- | Consistent gain | Overall performance |
| Synthetic Benchmark | Forgetting rate | Lowest | -- | Significant reduction | Controlled experiment |

### Ablation Study

| Configuration | Performance | Notes |
|------|------|------|
| FLLP (Full) | Best | Hyperbolic embedding + parent-child relationship + regularization |
| w/o Hyperbolic Space (Euclidean) | Degraded | Euclidean space fails to model hierarchical structure effectively |
| w/o Parent-Child Relationship | Degraded | Absence of knowledge transfer mechanism |
| w/o Regularization | Old concept degradation | New concept learning interferes with old concepts |

### Key Findings

- Hyperbolic space embedding yields clear performance improvements over Euclidean embedding, validating the value of hierarchical geometric structure.
- The parent-child relationship mechanism not only prevents forgetting but also improves learning efficiency and generation quality for new concepts through knowledge transfer.
- FLLP consistently outperforms existing methods in both robustness and generalization, demonstrating that "collaborative learning" is more effective than "conflict isolation."
- Controlled experiments on the synthetic benchmark clearly quantify the contribution of each component.

## Highlights & Insights

- Modeling concept hierarchies in **hyperbolic space** introduces Riemannian geometry into continual concept learning, serving as an excellent example of cross-domain methodological transfer.
- The **parent-child learning paradigm** reframes the forgetting problem from "conflict management" to "collaborative learning," a perspective shift that can generalize to other continual learning scenarios.
- The method supports continual concept integration without requiring all concepts to be available simultaneously, making it well-suited for practical use cases where users incrementally add new concepts.

## Limitations & Future Work

- Computations in hyperbolic space (exponential maps, logarithmic maps, etc.) are more complex than their Euclidean counterparts, introducing additional computational overhead.
- Automatic determination of parent-child relationships may be inaccurate for semantically ambiguous concept pairs.
- The method assumes meaningful hierarchical relationships exist among concepts, and may offer limited benefit for entirely unrelated concepts.
- Exploring multi-parent tree structures rather than a simple single-parent hierarchy could better model complex inter-concept relationships.

## Related Work & Insights

- **vs DreamBooth**: DreamBooth targets single-concept customization; this work extends to continual multi-concept learning.
- **vs C-LoRA / Custom Diffusion**: These methods avoid interference through parameter isolation, while FLLP achieves collaboration via hierarchical relationships—the underlying philosophies differ but may be complementary.
- **vs Hyperbolic Embedding Methods (e.g., Poincaré Embeddings)**: This work transfers hyperbolic embeddings from NLP and recommender systems to continual learning in generative models, broadening the scope of application.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of hyperbolic space and parent-child concept relationships is distinctively novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Four datasets with comprehensive ablation study
- Writing Quality: ⭐⭐⭐⭐ Intuition is clear and geometric interpretations are elegant
- Value: ⭐⭐⭐⭐ Introduces a new methodological paradigm for continual concept learning

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CAE: Hierarchical Semantic Alignment for Image Clustering](hierarchical_semantic_alignment_for_image_clustering.md)
- [\[AAAI 2026\] LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval](leanrag_knowledge-graph-based_generation_with_semantic_aggregation_and_hierarchi.md)
- [\[AAAI 2026\] Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors](verification-guided_context_optimization_for_tool_calling_via_hierarchical_llms-.md)
- [\[AAAI 2026\] Bipartite Mode Matching for Vision Training Set Search from a Hierarchical Data Server](bipartite_mode_matching_for_vision_training_set_search_from_a_hierarchical_data_.md)
- [\[NeurIPS 2025\] Rethinking PCA Through Duality](../../NeurIPS2025/others/rethinking_pca_through_duality.md)

</div>

<!-- RELATED:END -->

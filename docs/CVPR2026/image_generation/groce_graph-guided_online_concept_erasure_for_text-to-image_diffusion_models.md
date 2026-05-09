---
title: >-
  [Paper Note] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models
description: >-
  [CVPR 2026][Image Generation][Concept Erasure] GrOCE proposes a training-free concept erasure framework based on dynamic semantic graphs, achieving precise, context-aware online removal of target concepts in text-to-image diffusion models through three cooperative components: semantic graph construction, adaptive clustering identification, and selective severing.
tags:
  - CVPR 2026
  - Image Generation
  - Concept Erasure
  - Diffusion Models
  - Semantic Graph
  - Training-Free
  - Online Inference
date: 2026-05-08
content_hash: 96c3c5d440f07a84
---

# GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models

**Conference**: CVPR 2026
**arXiv**: [2511.12968](https://arxiv.org/abs/2511.12968)
**Code**: Available
**Area**: Image Generation
**Keywords**: Concept Erasure, Diffusion Models, Semantic Graph, Training-Free, Online Inference

## TL;DR
GrOCE proposes a training-free concept erasure framework based on dynamic semantic graphs, achieving precise, context-aware online removal of target concepts in text-to-image diffusion models through three cooperative components: semantic graph construction, adaptive clustering identification, and selective severing.

## Background & Motivation
1. **State of the Field**: Text-to-image diffusion models frequently generate harmful, biased, or copyright-infringing content. Concept erasure aims to remove target content while preserving non-target semantics.
2. **Limitations of Prior Work**: (i) Fine-tuning-based methods are computationally expensive, suffer from catastrophic forgetting, and struggle to adapt to emerging risks; (ii) inference-time intervention methods rely on heuristic mappings and fail to capture deep semantic entanglement. Both categories treat concepts as isolated entities, ignoring the rich relational structure in the latent space.
3. **Root Cause**: Concepts in diffusion models are encoded as entangled manifolds with fuzzy boundaries and high-order dependencies. Removing one concept (e.g., "violence") may damage semantically adjacent concepts (e.g., "conflict," "action").
4. **Paper Goals**: Design a training-free online concept erasure method that understands and leverages semantic relationships between concepts, enabling precise target erasure without harming neighboring concepts.
5. **Starting Point**: Reformulate concept erasure as a graph-cutting problem — identifying and removing the minimal vertex subset connected to the target concept.
6. **Core Idea**: Construct a dynamic semantic graph over concepts, identify the target concept cluster via multi-hop traversal and diffusion scoring, then selectively sever the semantic components of that cluster from the prompt embedding.

## Method

### Overall Architecture
Given a text prompt and a target concept, three steps operate collaboratively: (1) **Construct**: build a dynamic semantic graph from contextualized embeddings; (2) **Identify**: locate the target concept cluster via multi-hop traversal and diffusion scoring; (3) **Sever**: remove the cluster's semantic components from the prompt embedding while preserving non-target semantics and global sentence structure.

### Key Designs

1. **Dynamic Semantic Graph Construction (Construct)**:
    - **Function**: Constructs a real-time semantic association network among lexical concepts.
    - **Mechanism**: Each node $v_i$ corresponds to a concept embedding $x_i$; an edge is added between two nodes if their cosine similarity exceeds a local threshold $\tau_i$, with edge weight $w_{ij} = \exp(-(τ_i - \langle x_i, x_j \rangle)/\sigma)$. The threshold adapts via local similarity variance: $\tau_i = \tau_0 + \lambda \cdot \text{std}$, accommodating regions of varying density in the embedding space.
    - **Design Motivation**: A static threshold cannot adapt to regions of varying density in the embedding space. The adaptive threshold enforces more conservative connectivity in dense regions and more permissive connectivity in sparse regions.

2. **Adaptive Clustering Identification (Identify)**:
    - **Function**: Identifies the set of concepts semantically entangled with the target concept.
    - **Mechanism**: Multi-hop traversal (with similarity decay) is performed starting from the target concept to delineate its semantic scope of influence. Diffusion scoring further quantifies the semantic influence of each neighbor. The compact target concept cluster is then extracted.
    - **Design Motivation**: Simple keyword filtering cannot handle implicit semantic associations (e.g., "bear" → "grizzly" → "polar bear"); multi-hop traversal captures higher-order dependencies.

3. **Selective Severing (Sever)**:
    - **Function**: Removes the semantic components of the target cluster from the prompt embedding while preserving non-target semantics.
    - **Mechanism**: Graph-guided soft projection is applied to eliminate semantic directions in the prompt embedding associated with the target cluster, while approximately preserving orthogonal semantic directions. The edited prompt is injected into the model prior to diffusion inference.
    - **Design Motivation**: Hard deletion may disrupt the global sentence structure; soft projection maintains embedding space coherence while removing target semantics.

### Loss & Training
Entirely training-free; operates solely at inference time. No gradient access or model retraining is required.

## Key Experimental Results

### Main Results

| Dataset / Task | Metric | GrOCE | ConAbl | AdaVD | Notes |
|---|---|---|---|---|---|
| Concept Erasure | CS↓ | SOTA | 2nd best | — | More thorough erasure |
| Non-target Fidelity | FID↓ | SOTA | — | 2nd best | Less damage to non-targets |
| Runtime | seconds | ~0.1 | ~tens of sec. | ~seconds | Order-of-magnitude speedup |

### Ablation Study

| Configuration | Key Metric | Notes |
|---|---|---|
| Full GrOCE | Best | All three components intact |
| w/o graph guidance | Degraded | Degenerates to simple keyword filtering |
| w/o multi-hop traversal | Degraded | Fails to capture higher-order associations |
| w/o adaptive threshold | Degraded | Global threshold insufficient for precision |

### Key Findings
- GrOCE simultaneously achieves SOTA on erasure accuracy and non-target fidelity, demonstrating the superiority of graph-guided reasoning over isolated concept treatment.
- Runtime is orders of magnitude faster than training-based methods, enabling genuinely online concept removal.
- The semantic graph reveals hierarchical relationships and co-occurrence patterns among concepts, providing interpretability.

## Highlights & Insights
- **Introducing a graph perspective** elevates concept erasure from "per-concept processing" to "structured reasoning," representing a methodological advancement.
- The **training-free and online** nature enables rapid adaptation to newly emerging harmful concepts, offering high practical deployment value.
- The semantic graph is inherently interpretable — it clarifies not only what was erased but also why.

## Limitations & Future Work
- Only handles concepts accessible via text; purely visual concepts (e.g., specific pose–lighting combinations) cannot be addressed.
- Assumes linear separability of concepts in the embedding space; may fail in non-convex concept regions.
- Threshold and decay parameters for cluster identification require tuning.

## Related Work & Insights
- **vs. ESD/CA**: Require fine-tuning model weights, which is computationally expensive and prone to forgetting. GrOCE is entirely training-free.
- **vs. AdaVD**: Assumes linear separability and fails in non-convex regions. GrOCE captures more complex relationships through graph structure.
- **vs. UCE**: Performs inference-time intervention but assumes stable activation patterns, which may break under prompt paraphrasing. GrOCE's graph structure is more robust.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Graph-guided concept erasure represents an entirely new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-task validation (cartoon concepts / artistic styles) with thorough efficiency comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Mathematical formalization is clear; problem definition is rigorous.
- **Value**: ⭐⭐⭐⭐⭐ An important contribution to AI safety with high practical deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)
- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[CVPR 2026\] EMMA: Concept Erasure Benchmark with Comprehensive Semantic Metrics and Diverse Categories](emma_concept_erasure_benchmark_with_comprehensive_semantic_metrics_and_diverse_c.md)

<!-- RELATED:END -->

---
title: >-
  [Paper Note] Hierarchical Concept-based Interpretable Models
description: >-
  [ICLR 2026][Information Retrieval & RAG][Concept Embedding Models] HiCEMs introduces a hierarchical concept embedding model that automatically discovers fine-grained sub-concepts within the embedding space of a pretraine…
tags:
  - "ICLR 2026"
  - "Information Retrieval & RAG"
  - "Concept Embedding Models"
  - "Hierarchical Concepts"
  - "Concept Splitting"
  - "Sub-concept Discovery"
  - "Concept Intervention"
date: 2026-05-08
content_hash: 153de17363e06e56
---

# Hierarchical Concept-based Interpretable Models

**Conference**: ICLR 2026
**arXiv**: [2602.23947](https://arxiv.org/abs/2602.23947)  
**Code**: None  
**Area**: Explainable AI / Concept Models
**Keywords**: Concept Embedding Models, Hierarchical Concepts, Concept Splitting, Sub-concept Discovery, Concept Intervention

## TL;DR
HiCEMs introduces a hierarchical concept embedding model that automatically discovers fine-grained sub-concepts within the embedding space of a pretrained CEM via Concept Splitting—without requiring additional annotations—thereby constructing a hierarchical concept structure that supports test-time concept interventions at multiple granularities to improve task performance.

## Background & Motivation
Modern deep neural networks are difficult to interpret due to the opacity of their latent representations, hindering model understanding, debugging, and debiasing. Concept Embedding Models (CEMs) address this by mapping inputs to human-interpretable concept representations. However, CEMs suffer from two fundamental limitations: (1) they cannot represent inter-concept relationships, treating all concepts as flat and independent while ignoring their natural hierarchical structure (e.g., "feather color" → "red breast" / "blue wing"); and (2) training hierarchical models requires concept annotations at multiple granularity levels, incurring prohibitive annotation costs. The root cause is that hierarchical concept structures are critical for deep understanding and precise intervention, yet acquiring multi-level annotated data is impractical. The core idea of this paper is to automatically discover sub-concepts within the embedding space of an existing CEM via Concept Splitting, constructing a hierarchical concept structure without any additional annotations.

## Method

### Overall Architecture
The HiCEMs pipeline consists of three steps: (1) train a standard CEM to obtain a reliable concept embedding space; (2) apply Concept Splitting to the trained CEM to discover sub-concepts without additional supervision; (3) train HiCEM on the discovered sub-concepts to obtain an interpretable model supporting hierarchical concept intervention. The input is an image; the concept bottleneck layer outputs hierarchical concept predictions (parent concepts + sub-concepts); and the final label prediction layer performs classification based on the entire hierarchical structure.

### Key Designs

1. **Concept Embedding Model (CEM) Foundation**: CEM learns each concept as a vector in a high-dimensional embedding space rather than a simple binary scalar. For input image $x$, the concept encoder $g$ produces concept embeddings $c_i = g_i(x) \in \mathbb{R}^d$, and concept presence is determined via similarity to positive/negative concept prototype vectors. Compared to the binary concept predictions of standard CBMs (Concept Bottleneck Models), CEM's continuous embedding space carries richer semantic information—it is precisely this information-rich embedding space that enables subsequent sub-concept discovery.

2. **Concept Splitting**: Given the embedding space of a concept $c$ in a trained CEM, Concept Splitting analyzes the distribution of that concept's activation patterns across the full training set. The core assumption is that if a coarse-grained concept actually encompasses multiple sub-concepts, its embedding vectors will form multiple separable clusters in the embedding space. The algorithm proceeds as follows: (a) collect all embedding vectors $\{c^{(j)}\}$ for concept $c$ over the training set; (b) perform cluster analysis (e.g., k-means or hierarchical clustering) to identify meaningful sub-clusters; (c) each sub-cluster corresponds to a sub-concept, with its cluster centroid serving as the sub-concept prototype vector. The design motivation is to exploit structural information already present but not explicitly utilized in the CEM embedding space—during concept prediction training, CEMs naturally capture sub-concept-level discriminability in the embedding space. Clustering makes these implicit sub-concept structures explicit.

3. **HiCEMs Architecture**: A hierarchical structure is introduced on top of the standard CEM. For each parent concept $c_i$ and its sub-concepts $\{c_{i,1}, c_{i,2}, \ldots, c_{i,K}\}$ discovered via Concept Splitting, HiCEM simultaneously predicts the presence of both parent and sub-concepts. Key architectural design choices include: **(a) Hierarchical Consistency**: sub-concept predictions are logically consistent with parent concepts—if the parent concept "wing color" is absent, its sub-concepts should not be activated; **(b) Hierarchical Aggregation**: the label prediction layer can selectively leverage concept information at different granularities—coarse-grained concepts suffice for some tasks, while fine-grained sub-concepts provide critical discriminative information for others; **(c) Multi-granularity Intervention Interface**: at test time, users can intervene at any hierarchical level—correcting a parent concept cascades to sub-concepts, while correcting a sub-concept affects only the local scope.

4. **PseudoKitchens Dataset**: To validate HiCEM, the paper proposes a new concept-based dataset. Images are generated using a 3D kitchen rendering engine, containing multi-level concepts of kitchenware and food items. The dataset features natural hierarchical relationships among concepts (e.g., "container" → "cup" / "bowl"), making it suitable for evaluating hierarchical concept models.

### Loss & Training
The HiCEM training loss comprises three components: (1) **Concept Prediction Loss**: binary cross-entropy computed separately for parent and sub-concepts, $\mathcal{L}_{concept} = \mathcal{L}_{parent} + \lambda \mathcal{L}_{sub}$; (2) **Task Prediction Loss**: classification cross-entropy based on the hierarchical concept representations; (3) **Hierarchical Consistency Regularization**: encourages logical consistency between sub-concept and parent concept predictions. The training pipeline proceeds as follows: train a standard CEM to convergence → run Concept Splitting to discover sub-concepts → train HiCEM on the discovered hierarchy. The number of sub-concepts $K$ is determined via validation—different values of $K$ are evaluated, and the configuration yielding the best concept separability on the validation set is selected.

## Key Experimental Results

### Main Results

| Dataset | Metric | HiCEM | Standard CEM | CBM | Notes |
|--------|------|------|----------|------|------|
| MNIST-ADD | Task Acc | ~High | Baseline | Lower | Digit addition task |
| SHAPES | Task Acc | ~High | Baseline | Lower | Shape attribute recognition |
| CUB-200 | Task Acc | Competitive | Baseline | Lower | Fine-grained bird classification |
| AwA2 | Task Acc | Competitive | Baseline | Lower | Animal attribute prediction |
| PseudoKitchens | Task Acc | Best | Baseline | Lower | Proposed 3D kitchen dataset |

Note: HiCEM maintains accuracy comparable to or better than CEM across all datasets while providing more fine-grained explanations.

### Concept Intervention Experiments

| Dataset | Intervention Setting | No Intervention | Coarse-grained | Fine-grained (HiCEM) | Notes |
|--------|---------|------|---------|---------|------|
| CUB-200 | Increasing # interventions | Baseline | Improvement | Greater improvement | Fine-grained intervention more effective |
| AwA2 | Increasing # interventions | Baseline | Improvement | Greater improvement | Cumulative effect of hierarchical intervention |
| SHAPES | Increasing # interventions | Baseline | Improvement | Greater improvement | Advantage especially pronounced at moderate intervention counts |

### User Study

| Evaluation Dimension | Result | Notes |
|---------|------|------|
| Sub-concept Interpretability | Users could assign meaningful names to most sub-concepts | Validates that sub-concepts discovered by Concept Splitting carry human-interpretable semantics |
| Explanation Usefulness | HiCEM's hierarchical explanations preferred over CEM's flat explanations | Hierarchical structure provides more intuitive error tracing |
| Intervention Efficiency | Fine-grained intervention requires fewer corrections | Precisely targeting the erroneous sub-concept is more efficient than correcting coarse-grained concepts |

### Key Findings
- Sub-concepts discovered by Concept Splitting exhibit high human interpretability—on the CUB dataset, "wing color" is split into sub-concepts such as "striped wing" and "solid-color wing," which users can intuitively understand.
- Fine-grained concept intervention is more effective than coarse-grained intervention: on CUB, intervening on 5 fine-grained sub-concepts outperforms intervening on 5 coarse-grained parent concepts.
- HiCEM provides richer explanations without sacrificing task accuracy, breaking the commonly observed interpretability–accuracy trade-off.
- Experiments on PseudoKitchens indicate that HiCEM's advantage is most pronounced in domains with naturally hierarchical concepts.
- Meaningful sub-concept structure does exist in CEM embedding spaces—validating that CEMs implicitly learn information beyond the granularity of their annotations during training.
- The optimal number of splits varies across concepts: some concepts naturally contain multiple sub-concepts, while others are "atomic" and require no further splitting.

## Highlights & Insights
- **Zero-annotation Sub-concept Discovery**: The paper's primary contribution lies in leveraging the structure naturally formed in the CEM embedding space during training to discover fine-grained sub-concepts without any new annotations.
- **Hierarchical Interpretability**: The paper advances explainable AI from "which concepts does the model use" to "which aspect of a concept does the model specifically rely on."
- **Refined Concept Intervention**: Test-time intervention evolves from "correcting a concept" to "correcting the right sub-concept at the right hierarchical level," substantially improving intervention efficiency.
- **New Dataset PseudoKitchens**: Provides a controlled experimental environment for concept hierarchy research (3D rendering enables precise control over concept combinations), filling a gap in the field.
- **Theoretical Insight**: The finding that CEM embedding spaces naturally encode richer information than their annotations suggests similar exploration for other representation learning methods.

## Limitations & Future Work
- The quality of Concept Splitting is highly dependent on the quality of the initial CEM embedding space—poorly trained CEMs may yield sub-concepts with no meaningful semantics.
- Only single-level splitting (parent → child) is currently supported; multi-level splitting is not addressed (explored in the companion workshop paper "Digging Deeper").
- The choice of clustering algorithm and hyperparameters (e.g., $K$) still requires manual tuning or validation.
- Scalability to large-scale datasets (e.g., ImageNet) has not been validated.
- Hierarchical consistency constraints may be overly strict—in practice, sub-concepts do not always strictly subsume under parent concepts.
- No systematic comparison with attention-based interpretability methods (e.g., GradCAM) or feature attribution methods (e.g., SHAP) is provided.

## Related Work & Insights
- **Concept Bottleneck Models (CBM)**: The foundational framework for interpretable AI, upon which HiCEM introduces hierarchical structure.
- **Concept Embedding Models (CEM)**: The direct predecessor of HiCEM, representing concepts via continuous embeddings rather than binary scalars.
- **Digging Deeper (ICLR 2026 Workshop)**: A follow-up work from the same group that extends Concept Splitting to multiple levels (MLCS) with the Deep-HiCEMs architecture.
- **Concept Activation Vectors (TCAV)**: An alternative concept discovery approach that does not construct hierarchical structures.
- **Inspiration**: The automatic discovery of concept hierarchies generalizes to: (1) fairness analysis—discovering subgroups of sensitive attributes; (2) model debugging—pinpointing the precise concept level at which model errors occur; (3) data augmentation—structured sampling based on concept hierarchies.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Digging Deeper: Learning Multi-Level Concept Hierarchies](digging_deeper_learning_multi-level_concept_hierarchies.md)
- [\[ICLR 2026\] Summaries as Centroids for Interpretable and Scalable Text Clustering](summaries_as_centroids_for_interpretable_and_scalable_text_clustering.md)
- [\[ICLR 2026\] TokMem: One-Token Procedural Memory for Large Language Models](tokmem_one-token_procedural_memory_for_large_language_models.md)
- [\[ICLR 2026\] G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge](g-reasoner_foundation_models_for_unified_reasoning_over_graph-structured_knowled.md)
- [\[ICML 2026\] Hierarchical Abstract Tree for Cross-Document Retrieval-Augmented Generation](../../ICML2026/information_retrieval/hierarchical_abstract_tree_for_cross-document_retrieval-augmented_generation.md)

</div>

<!-- RELATED:END -->

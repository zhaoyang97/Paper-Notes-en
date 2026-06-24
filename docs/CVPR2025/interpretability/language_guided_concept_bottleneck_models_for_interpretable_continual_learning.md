---
title: >-
  [Paper Note] Language Guided Concept Bottleneck Models for Interpretable Continual Learning
description: >-
  [CVPR 2025][Interpretability][Continual Learning] This paper introduces language-guided Concept Bottleneck Models (CBMs) into continual learning. It uses ChatGPT to generate human-interpretable concepts and the CLIP text encoder to encode concept embeddings, constructing a concept bottleneck layer. This provides transparent decision explanations while mitigating catastrophic forgetting, outperforming the SOTA on ImageNet-subset by 3.06%.
tags:
  - "CVPR 2025"
  - "Interpretability"
  - "Continual Learning"
  - "Concept Bottleneck Models"
  - "CLIP"
  - "Catastrophic Forgetting"
date: 2026-05-08
content_hash: cc12d2ac9b42c513
---

# Language Guided Concept Bottleneck Models for Interpretable Continual Learning

**Conference**: CVPR 2025  
**arXiv**: [2503.23283](https://arxiv.org/abs/2503.23283)  
**Code**: [https://github.com/FisherCats/CLG-CBM](https://github.com/FisherCats/CLG-CBM)  
**Area**: LLM Efficiency  
**Keywords**: Continual Learning, Concept Bottleneck Models, Interpretability, CLIP, Catastrophic Forgetting

## TL;DR
This paper introduces language-guided Concept Bottleneck Models (CBMs) into continual learning. It uses ChatGPT to generate human-interpretable concepts and the CLIP text encoder to encode concept embeddings, constructing a concept bottleneck layer. This provides transparent decision explanations while mitigating catastrophic forgetting, outperforming the SOTA on ImageNet-subset by 3.06%.

## Background & Motivation

**Background**: Continual learning requires models to continuously learn new tasks without forgetting old knowledge (catastrophic forgetting). Existing approaches are categorized into regularization, replay, and architecture expansion, but they are black-box models that lack interpretability.

**Limitations of Prior Work**: As models continually update their knowledge, understanding what they have learned and how they retain old information becomes crucial. ICICLE attempts to improve interpretability through prototype part networks, but this severely limits model plasticity.

**Key Challenge**: There is a trade-off between interpretability and mitigating catastrophic forgetting—imposing transparency constraints often limits the model's ability to adapt to new tasks.

**Goal**: Design a framework that simultaneously enhances both interpretability and continual learning performance.

**Key Insight**: Concept Bottleneck Models are inherently interpretable (with the intermediate layer corresponding to human concepts). Combining the zero-shot capability of CLIP and the concept generation of ChatGPT can provide semantic concepts that generalize across tasks for continual learning.

**Core Idea**: Use ChatGPT to generate concept words for each category, encode them into a Concept Bottleneck Layer using the CLIP text encoder, and achieve interpretable representations that generalize across tasks through semantic consistency alignment.

## Method

### Overall Architecture
When a new task arrives: (1) ChatGPT is used to generate human-interpretable concepts for the new categories; (2) the CLIP text encoder encodes these concepts into embedding vectors to construct the Concept Bottleneck Layer (CBL); (3) images pass through the CLIP visual encoder to extract features and compute a concept score matrix with the CBL; (4) the concept score vectors are used for final classification. Semantic knowledge-augmented prototypes are utilized to mitigate forgetting.

### Key Designs

1. **Language-Guided Concept Bottleneck Layer (Language-Guided CBL)**:

    - **Function**: Inserts a human-interpretable intermediate concept layer between feature extraction and classification.
    - **Mechanism**: ChatGPT is queried to generate descriptive concept words for each category. A concept selection module then selects the most informative and discriminative concepts from the candidates to build a task-specific concept pool $\mathcal{C}$. The concept activation matrix $E_{clip} = f_I(\mathcal{X}) \cdot f_T(\mathcal{C})^\top$ measures the alignment between images and each concept.
    - **Design Motivation**: Each neuron in the concept bottleneck corresponds to an interpretable concept, naturally providing explanations for decision-making.

2. **Semantic-Augmented Prototypes**:

    - **Function**: Enhances class prototypes using semantic knowledge to mitigate catastrophic forgetting.
    - **Mechanism**: Constructed class prototype representations using concept score vectors. When a new task arrives, new and old concepts are associated via semantic similarity to stabilize the decision boundaries of old classes.
    - **Design Motivation**: Traditional prototype methods rely solely on feature distance, whereas semantic augmentation provides more robust inter-class discrimination.

3. **Concept Visualization and Interpretation**:

    - **Function**: Provides human-interpretable explanations for model predictions.
    - **Mechanism**: For each prediction, the highly activated concepts and their scores are displayed to visually explain "why the model made this classification."
    - **Design Motivation**: Understanding model decisions is particularly important in continual learning scenarios.

### Loss & Training
Cross-entropy loss + Mahalanobis loss to guide semantic knowledge learning, used for concept selection.

## Key Experimental Results

### Main Results
Outperformed SOTA on 7 benchmark datasets, with a 3.06% improvement in final average accuracy on ImageNet-subset while maintaining interpretability throughout.

### Key Findings
- Language-guided concept bottlenecks not only improve interpretability but also unexpectedly enhance continual learning performance.
- The generalization capability of concepts across tasks is stronger than that of purely visual features.

## Highlights & Insights
- Systematically introduces the interpretability advantages of CBMs to continual learning for the first time.
- The ChatGPT + CLIP concept generation pipeline can be generalized to other scenarios requiring interpretability.

## Limitations & Future Work
- Concept quality depends on the accuracy of ChatGPT's generation.
- The dimension of the concept bottleneck layer continuously increases as the number of tasks grows.

## Related Work & Insights
- **vs ICICLE**: Prototype part networks are interpretable but restrict plasticity. The proposed CBM provides more flexible interpretability.
- **vs Standard CLIP-CBM**: Only handles static classification. This work extends it to continual learning scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of CBM and continual learning is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across 7 datasets.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured.
- Value: ⭐⭐⭐⭐ Practically advances explainable AI (XAI).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Attribute-formed Class-specific Concept Space: Endowing Language Bottleneck Model with Better Interpretability and Scalability](albm_attribute_concept_space.md)
- [\[ICLR 2026\] There Was Never a Bottleneck in Concept Bottleneck Models](../../ICLR2026/interpretability/there_was_never_a_bottleneck_in_concept_bottleneck_models.md)
- [\[AAAI 2026\] Partially Shared Concept Bottleneck Models](../../AAAI2026/interpretability/partially_shared_concept_bottleneck_models.md)
- [\[CVPR 2026\] Rethinking Concept Bottleneck Models: From Pitfalls to Solutions](../../CVPR2026/interpretability/rethinking_concept_bottleneck_models_from_pitfalls_to_solutions.md)
- [\[AAAI 2026\] Concepts from Representations: Post-hoc Concept Bottleneck Models via Sparse Decomposition of Visual Representations](../../AAAI2026/interpretability/concepts_from_representations_post-hoc_concept_bottleneck_models_via_sparse_deco.md)

</div>

<!-- RELATED:END -->

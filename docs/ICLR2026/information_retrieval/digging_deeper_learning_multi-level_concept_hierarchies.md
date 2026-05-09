---
title: >-
  [Paper Note] Digging Deeper: Learning Multi-Level Concept Hierarchies
description: >-
  [ICLR 2026 Workshop on Principled Design for Trustworthy AI][multi-level concept hierarchy] This paper proposes Multi-Level Concept Splitting (MLCS), which extends concept splitting from a single layer to a recursive multi-level process. Using only top-level concept annotations, MLCS automatically discovers concept hierarchy trees of arbitrary depth. The authors further introduce the Deep-HiCEMs architecture to represent and leverage these deep hierarchies, enabling test-time concept interventions at multiple levels of granularity.
tags:
  - ICLR 2026 Workshop on Principled Design for Trustworthy AI
  - multi-level concept hierarchy
  - concept embedding models
  - concept splitting
  - sub-concept discovery
  - test-time intervention
date: 2026-05-08
content_hash: db1d04518d5c6b28
---

# Digging Deeper: Learning Multi-Level Concept Hierarchies

**Conference**: ICLR 2026 Workshop on Principled Design for Trustworthy AI
**arXiv**: [2603.10084](https://arxiv.org/abs/2603.10084)
**Code**: None
**Area**: Explainable AI / Concept-based Models
**Keywords**: multi-level concept hierarchy, concept embedding models, concept splitting, sub-concept discovery, test-time intervention

## TL;DR
This paper proposes Multi-Level Concept Splitting (MLCS), which extends concept splitting from a single layer to a recursive multi-level process. Using only top-level concept annotations, MLCS automatically discovers concept hierarchy trees of arbitrary depth. The authors further introduce the Deep-HiCEMs architecture to represent and leverage these deep hierarchies, enabling test-time concept interventions at multiple levels of granularity.

## Background & Motivation

**Background**: Concept-based Models are among the central research directions in explainable AI. Concept Bottleneck Models (CBMs) and Concept Embedding Models (CEMs) provide structured explanation pathways by forcing models to first predict human-understandable intermediate concepts before inferring the final task label. CEMs specifically introduce a concept embedding space design, representing each concept as a high-dimensional vector rather than a simple binary prediction, improving accuracy while preserving interpretability.

**Limitations of Prior Work**: Standard CBMs and CEMs suffer from two critical issues:
(1) They treat concepts as flat and independent entities, completely ignoring hierarchical relations among concepts. For instance, "wing color" and "flight capability" are causally related in semantics, yet the model treats them as orthogonal dimensions.
(2) Obtaining fine-grained concept explanations requires dense multi-granularity annotations at training time (e.g., annotating not just "has wings" but also "wings are striped," "wings are pointed," etc.), which is prohibitively expensive.

**Key Challenge**: The same group's concurrent work HiCEMs (ICLR 2026 main conference) addresses the first issue—explicitly modeling concept relations with hierarchical structures and proposing Concept Splitting to automatically discover sub-concepts from a pretrained CEM's embedding space without additional annotation. However, both HiCEMs and Concept Splitting are restricted to **shallow hierarchies** (i.e., splitting a parent concept into only one layer of sub-concepts), and cannot capture deeper multi-level concept trees found in practice—for example, a four-level structure such as "animal → bird → waterbird → pelican" cannot be expressed.

**Goal**: Two sub-problems are addressed:
(1) How to extend single-level Concept Splitting into a recursive multi-level version that automatically constructs deep concept trees from top-level annotations alone?
(2) How to design a model architecture that represents and exploits multi-level concept hierarchies while supporting concept interventions at arbitrary levels of abstraction?

**Key Insight**: The authors observe that Concept Splitting is essentially a clustering operation in the CEM's concept embedding space. If the embedding vectors of a concept exhibit multiple natural cluster structures across different samples, each cluster corresponds to a meaningful sub-concept. This operation can be applied recursively: the embedding space of a sub-concept may itself contain further separable cluster structures.

**Core Idea**: Recursively apply splitting operations in the concept embedding space to construct multi-level concept trees (MLCS), then represent and exploit these deep hierarchies with the Deep-HiCEMs architecture.

## Method

### Overall Architecture
The entire method builds upon CEMs. An input image $x$ is first passed through a shared backbone (e.g., ResNet) for feature extraction, then through a concept encoder that maps features into a concept embedding space, where each concept $c_i$ corresponds to a high-dimensional embedding vector $\mathbf{e}_i$—encoding not only the presence or absence of the concept but also its fine-grained attributes. A task predictor then generates classification results from concept embeddings. MLCS and Deep-HiCEMs add two steps on top of this: (1) a post-processing stage where MLCS recursively discovers multi-level sub-concepts; and (2) the Deep-HiCEMs architecture encodes these hierarchical structures into the model, enabling multi-granularity interventions at inference time.

### Key Designs

1. **Multi-Level Concept Splitting (MLCS)**:

    - **Function**: Recursively discovers multi-level concept hierarchies from a pretrained CEM's embedding space without any additional annotation.
    - **Mechanism**: Given the set of embedding vectors $\{\mathbf{e}_c^{(1)}, \mathbf{e}_c^{(2)}, \ldots\}$ for a parent concept $c$ (collected from different training samples), MLCS performs clustering analysis (e.g., k-means or Gaussian mixture models) in this high-dimensional space to identify naturally formed clusters. Each cluster is defined as a sub-concept $c_i$. This process is then repeated for **each sub-concept**: embeddings of samples belonging to that sub-concept are collected and clustered again. The recursion continues until the sub-concept embeddings at a given level no longer exhibit meaningful multi-cluster structure, as determined by a split quality criterion. The final output is a concept hierarchy tree, where different concepts may have different tree depths. A key constraint is that all splitting operations are performed within the same pretrained CEM's embedding space, requiring neither model retraining nor new annotations.
    - **Design Motivation**: Single-level Concept Splitting can only produce direct child concepts from a parent, missing deeper semantic structures. The recursive design of MLCS enables the system to automatically capture concept hierarchies of arbitrary depth, with different concepts allowed to have uneven depths—simple concepts may yield only one layer of sub-concepts, while complex concepts may produce three or four layers.

2. **Deep-HiCEMs Architecture**:

    - **Function**: Explicitly represents and leverages the multi-level concept hierarchies discovered by MLCS within the model, supporting multi-granularity reasoning and intervention.
    - **Mechanism**: In contrast to the original HiCEMs, which supports only two levels (parent concepts + sub-concepts), Deep-HiCEMs is extended to represent concept hierarchies of arbitrary depth. Each layer of concepts has its own independent embedding representation and prediction head. Layers are connected via parent–child consistency constraints: the prediction of a child concept must be consistent with its parent (e.g., if the parent concept "has wings" is false, the child concept "wings are striped" must also be false). The task predictor can leverage concept embeddings from all layers for final classification, simultaneously exploiting both coarse-grained and fine-grained information.
    - **Design Motivation**: Shallow HiCEMs can only support interventions at a single granularity. Deep-HiCEMs allows users to select an appropriate granularity level for intervention based on their domain expertise—domain experts can intervene at very fine sub-concepts (e.g., "stripe density of wing feathers"), while non-specialists can intervene at coarser levels (e.g., "has wings").

3. **Adaptive Splitting Termination Mechanism**:

    - **Function**: Automatically determines the hierarchical depth for each concept without requiring manual specification.
    - **Mechanism**: At each step of recursive splitting, the system evaluates split quality using metrics including the separability of sub-concept clusters (e.g., silhouette score) and semantic coherence (whether sub-concepts correspond to meaningful visual attributes). Splitting stops when quality falls below a threshold. This allows different concepts to have different tree depths, reflecting the varying semantic complexity of individual concepts.
    - **Design Motivation**: Forcing all concepts to have the same hierarchical depth is unreasonable—"color" may need only two sub-concepts ("warm/cool"), while "shape" may require four levels ("geometric shape → polygon → regular polygon → regular hexagon"). The adaptive mechanism aligns the concept tree structure with the semantic structure inherent in the data.

### Loss & Training
Training proceeds in two independent stages. **Stage 1**: A standard CEM is trained normally using a weighted sum of concept prediction loss (binary cross-entropy) and task prediction loss (cross-entropy), yielding a high-quality concept embedding space. **Stage 2**: The CEM parameters are frozen, and MLCS is applied recursively to the embedding space to obtain multi-level concept trees. Deep-HiCEMs is then trained using these hierarchical structures, with objectives including: (1) prediction accuracy at each concept level; (2) final task prediction accuracy; and (3) hierarchy consistency constraints ensuring that child concept activations are logically consistent with parent concept activations.

## Key Experimental Results

### Main Results: Deep-HiCEMs vs. Standard CEM and HiCEMs

| Model | Concept Hierarchy Depth | Task Accuracy | Concept Interpretability | Intervention Granularity |
|-------|------------------------|--------------|--------------------------|--------------------------|
| Standard CEM | No hierarchy (flat) | Baseline accuracy | Single-level concept explanation | Single-granularity intervention |
| HiCEMs (single-level splitting) | 2 levels | ≈ CEM accuracy | Parent + child two-level explanation | Two-level intervention |
| **Deep-HiCEMs (MLCS)** | **Multi-level (≥3 levels)** | **Maintains high accuracy** | **Multi-level fine-grained explanation** | **Arbitrary-granularity intervention** |
| Sparse Autoencoder | No hierarchy | Depends on sparsity | Sparse activation explanation | No intervention support |

### Intervention Effect Comparison

| Intervention Strategy | Granularity | Task Accuracy Gain | Notes |
|-----------------------|-------------|-------------------|-------|
| No intervention | — | Baseline | Original model prediction |
| Top-level concept intervention | Coarse | Moderate gain | Correcting parent concepts (e.g., "has wings") |
| Single-level sub-concept intervention | Medium | Substantial gain | Correcting first-level sub-concepts |
| Multi-level deep intervention (MLCS) | Fine | **Maximum gain** | Correcting at the most relevant fine-grained level |
| Random-level intervention | Mixed | Unstable | Demonstrates that granularity selection affects intervention effectiveness |

### Key Findings
- **MLCS-discovered sub-concepts are human-interpretable**: Human evaluation experiments confirm that automatically discovered multi-level sub-concepts can be assigned meaningful semantic labels by human raters. For example, the concept "wing color" is automatically split into "dark wings" and "light wings," with "dark wings" further split into "black" and "dark brown."
- **Deep-HiCEMs maintains high task accuracy**: Increasing concept hierarchy depth does not significantly sacrifice predictive performance, demonstrating that hierarchical structure complements rather than replaces the original representational capacity.
- **Multi-level intervention outperforms single-level intervention**: At test time, intervening at finer concept levels is more targeted than coarse-grained intervention, because fine-grained sub-concepts carry more precise semantics and corrections have a smaller, more controllable scope of influence.
- **Different datasets exhibit different natural hierarchy depths**: Concept trees on the CUB (fine-grained bird classification) dataset are generally deeper than those on MNIST-ADD (digit addition), reflecting the greater semantic complexity of avian visual attributes.
- **Connection to Sparse Autoencoders**: The paper draws an analogy between concept splitting and sparse autoencoders (SAEs)—both aim to discover finer-grained features, but concept splitting produces a tree-structured hierarchy while SAEs produce a flat sparse feature dictionary.

## Highlights & Insights
- **Deep concept discovery with zero additional annotation**: This is the paper's most central contribution. Generalizing Concept Splitting from single-level to recursive multi-level operation requires no additional annotations whatsoever—all sub-concepts are "mined" from the CEM's embedding space. The elegance of this design lies in exploiting the fact that a well-trained CEM embedding space already encodes rich semantic structure; appropriate clustering algorithms suffice to make this latent structure explicit.
- **Practical value of multi-granularity intervention**: Traditional CBM concept intervention is one-size-fits-all—all concepts are operated at the same granularity. Deep-HiCEMs allows users to select an appropriate intervention granularity based on their expertise and the specific scenario, which is highly valuable for real-world deployment. For example, in medical imaging, radiologists can intervene at very fine sub-concepts (e.g., "degree of nodule margin spiculation"), while general practitioners can intervene at coarser concepts (e.g., "presence of nodule").
- **Bridging concept hierarchies and SAEs**: By connecting Concept Splitting with Sparse Autoencoders, the paper suggests that "feature decomposition" and "concept hierarchy" in explainable AI may be two perspectives on the same problem, providing a roadmap for methodological integration between the two communities.

## Limitations & Future Work
- **Limited experimental scale as a workshop paper**: As a 4–6 page workshop paper, experiments are conducted primarily on small-scale datasets (MNIST-ADD, CUB, etc.), lacking large-scale validation at the ImageNet level, making it difficult to assess scalability in realistic complex settings.
- **Split quality depends on the base CEM embedding quality**: If the initial CEM's concept embedding space is of low quality (e.g., severe concept entanglement), recursive splitting by MLCS may produce meaningless sub-concepts. As a post-processing method, its upper bound is constrained by the base model.
- **Computational cost grows with depth**: Each level of recursive splitting requires clustering over all sample embeddings, and costs increase with depth. For large-scale datasets and very deep hierarchy trees, this may become a bottleneck.
- **No cross-task transfer of discovered hierarchies**: The current method discovers concept hierarchies independently for each task; whether discovered hierarchies can transfer to related but different tasks is unexplored.
- **Lack of comparison with attention-based explainability methods**: The paper's comparisons are primarily within the concept-based model paradigm, without systematic evaluation against mainstream explainability methods such as attention mechanisms, Grad-CAM, or SHAP.

## Related Work & Insights
- **vs. HiCEMs (ICLR 2026 main conference)**: A concurrent work from the same group. HiCEMs is the foundation of this paper, introducing hierarchical concept embeddings and the core idea of single-level Concept Splitting. This paper's contribution is extending that work from shallow to multi-level hierarchies—incremental in appearance but technically requiring solutions to two key challenges: recursive split quality control and Deep-HiCEMs architecture design.
- **vs. Concept Bottleneck Models (CBMs)**: CBMs are the seminal work in concept-based models, using concepts as a bottleneck layer but treating them as flat and independent. This paper represents a qualitative advance over CBMs in expressive power—from a flat list to a multi-level tree.
- **vs. Concept Embedding Models (CEMs)**: CEMs propose representing concepts with high-dimensional embeddings rather than binary scalars, providing the semantically rich embedding space necessary for Concept Splitting. This paper builds directly on the CEM embedding space.
- **vs. Sparse Autoencoders (SAEs)**: SAEs, recently prominent in LLM interpretability research, also perform "feature decomposition" but produce a flat feature dictionary. The concept hierarchy in this paper provides a structured organizational scheme; the two approaches are complementary.
- **Insights**: The idea of automatic concept hierarchy discovery is transferable to other modalities—for instance, recursively splitting token-level concepts into sub-token semantic features in NLP, or constructing cross-modal concept alignment hierarchies in multimodal models.

## Rating
- Novelty: ⭐⭐⭐⭐ The core idea (recursive multi-level splitting) is a natural and important extension of HiCEMs—incremental but valuable
- Experimental Thoroughness: ⭐⭐⭐ Workshop paper length constraints limit experimental scale; quantitative ablations and large-scale validation are lacking
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, method description is concise, and the relationship to the main-conference paper HiCEMs is well articulated
- Value: ⭐⭐⭐⭐ Multi-level concept hierarchies represent a critical step from "usable" to "practically useful" concept models, with meaningful implications for real-world deployment

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Hierarchical Concept-based Interpretable Models](hierarchical_concept-based_interpretable_models.md)
- [\[CVPR 2026\] MuCo: Multi-turn Contrastive Learning for Multimodal Embedding Model](../../CVPR2026/information_retrieval/muco_multi-turn_contrastive_learning_for_multimodal_embedding_model.md)
- [\[ICLR 2026\] Query-Level Uncertainty in Large Language Models](query-level_uncertainty_in_large_language_models.md)
- [\[NeurIPS 2025\] Deep Research Brings Deeper Harm](../../NeurIPS2025/information_retrieval/deep_research_brings_deeper_harm.md)
- [\[ICLR 2026\] Token-Guard: Towards Token-Level Hallucination Control via Self-Checking Decoding](token-guard_towards_token-level_hallucination_control_via_self-checking_decoding.md)

<!-- RELATED:END -->

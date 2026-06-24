---
title: >-
  [Paper Note] Universal Scene Graph Generation
description: >-
  [CVPR 2025][Graph Learning][Universal Scene Graph] This paper proposes the Universal Scene Graph (USG) representation and its parser USG-Par, which generates a unified scene graph from arbitrary combinations of modalities (images, text, video, 3D) using a cross-modal object associator and text-centric scene contrastive learning, capturing both modality-invariant and modality-specific scene semantics.
tags:
  - "CVPR 2025"
  - "Graph Learning"
  - "Universal Scene Graph"
  - "Cross-Modal Alignment"
  - "Multimodal Scene Understanding"
  - "Scene Graph Parsing"
  - "Text-Centric Contrastive Learning"
date: 2026-05-08
content_hash: d5999504449d9399
---

# Universal Scene Graph Generation

**Conference**: CVPR 2025  
**arXiv**: [2503.15005](https://arxiv.org/abs/2503.15005)  
**Code**: None  
**Area**: Graph Learning / Scene Graph Generation  
**Keywords**: Universal Scene Graph, Cross-Modal Alignment, Multimodal Scene Understanding, Scene Graph Parsing, Text-Centric Contrastive Learning

## TL;DR
This paper proposes the Universal Scene Graph (USG) representation and its parser USG-Par, which generates a unified scene graph from arbitrary combinations of modalities (images, text, video, 3D) using a cross-modal object associator and text-centric scene contrastive learning, capturing both modality-invariant and modality-specific scene semantics.

## Background & Motivation

**Background**: Scene Graph (SG) is an efficient structured representation for describing scene semantics, where nodes represent objects and edges represent relationships between objects. Existing Scene Graph Generation (SGG) research has carried out extensive work in single modality domains such as image SGG, video SGG, 3D SGG, and text SGG, each achieving notable progress.

**Limitations of Prior Work**: In the real world, multiple modalities (images, text, video, 3D data) often coexist, with each modality expressing distinct scene characteristics. However, current SGG research is almost entirely restricted to single-modality scene modeling, failing to leverage the complementary advantages of different modality SG representations to describe complete scene semantics. For instance, image SGs excel at capturing spatial relationships, while text SGs excel at abstract semantic relationships, but the two cannot be fused and used together.

**Key Challenge**: A natural modality gap exists between object representations from different modalities, making cross-modal object alignment difficult. Meanwhile, scene graph datasets of different modalities exhibit severe domain imbalance, which biases unified training toward modalities with larger data volumes.

**Goal**: To design a representation and method capable of generating a unified scene graph from any combination of modality inputs, where the scene graph must concurrently contain cross-modal shared scene semantics (modality-invariant) and modality-specific details.

**Key Insight**: It is observed that text SGs are easier to acquire and standardize compared to SGs of other modalities, and text serves as a natural cross-modal bridge. Therefore, text is utilized as an anchor to align object and relationship representations across other modalities.

**Core Idea**: To introduce the Universal SG (USG) representation along with a modular USG-Par parser, employing an object associator to alleviate cross-modal alignment difficulties and leveraging text-centric contrastive learning to address the domain imbalance problem.

## Method

### Overall Architecture
USG-Par is a modular end-to-end architecture that takes arbitrary combinations of modalities (e.g., image+text, video+3D) as input and outputs a unified USG. The overall pipeline consists of three steps: 1) object and relationship feature extraction for each modality individually; 2) cross-modal object alignment via an object associator; and 3) scene graph fusion module execution to generate a USG containing both modality-invariant and modality-specific scenes.

### Key Designs

1. **Universal Scene Graph (USG) Representation**:

    - **Function**: Defines a new type of scene graph representation capable of fusing multimodal inputs to form a complete semantic description.
    - **Mechanism**: USG divides a scene into two components: the modality-invariant scene and the modality-specific scene. The modality-invariant portion describes the core semantics shared across all modalities (e.g., "person next to table"), while the modality-specific portion preserves details unique to each modality (e.g., spatial localization in images, abstract relationships in text). Both nodes and edges in the USG are tagged with modality annotations, allowing downstream tasks to retrieve information as needed.
    - **Design Motivation**: Information in a single modality SG is one-sided; USG achieves a stronger scene representation capability than any single-modality SG by fusing complementary information.

2. **Object Associator**:

    - **Function**: Establishes object-level correspondences across different modalities.
    - **Mechanism**: Objects are first detected and their features extracted within each modality, and then a cross-modal attention mechanism is applied to compute similarity matrices between objects from different modalities. Semantic alignment is achieved using the Hungarian algorithm or soft-alignment strategies to pair identical objects. Paired objects can share information to enrich each other's representations. The core formula calculates cross-modal similarity between objects as $s_{ij} = \text{sim}(f_i^{m_1}, f_j^{m_2})$, where $f$ represents the object feature.
    - **Design Motivation**: Cross-modal object alignment is a critical bottleneck in USG generation. Representations of the same object in different modalities vary drastically (e.g., visual patches in images vs. word vectors in text), requiring a dedicated alignment module to bridge this modality gap.

3. **Text-Centric Scene Contrasting**:

    - **Function**: Alleviates the domain imbalance problem across datasets of different modalities.
    - **Mechanism**: Using text SG as a unified semantic anchor, object and relationship representations from other modalities are contrastively learned against corresponding elements in the text SG. Specifically, objects and relationships detected in image/video/3D modalities are projected onto the text space, and contrastive loss is optimized to bring semantically matched cross-modal pairs closer while pushing mismatched pairs apart. The benefit of centering on text is that it serves as a shared reference space for all other modalities.
    - **Design Motivation**: Performing direct contrastive learning across all modalities would submerge training signals of certain modalities due to unbalanced data volume. The text-centric strategy simplifies the $n$-modality $O(n^2)$ contrastive relationships to $O(n)$, reducing complexity and improving stability.

### Loss & Training
An overall multi-task loss is utilized for training, including: classification losses for scene graph generation (cross-entropy for object and relationship classification), pairwise loss for object association, and text-centric contrastive learning loss. SGG branches of each modality can be independently supervised, while cross-modal alignment is trained jointly via contrastive loss.

## Key Experimental Results

### Main Results

| Setting | Dataset / Metric | USG-Par | Single-Modality SOTA | Gain |
|------|-------------|---------|-----------|------|
| Image SGG | Visual Genome / R@50 | 38.2 | 36.5 (Motifs) | +1.7 |
| Text SGG | NYT / F1 | 72.1 | 70.8 | +1.3 |
| Video SGG | Action Genome / R@50 | 34.5 | 32.9 | +1.6 |
| Cross-Modal USG | Image+Text Joint / Scene F1 | 58.7 | N/A (No Precedent) | — |

### Ablation Study

| Configuration | Scene Semantic Score | Description |
|------|------------|------|
| Full USG-Par | 58.7 | Full Model |
| w/o Object Associator | 53.2 | Cross-modal alignment failure leads to -5.5 |
| w/o Text-Centric Contrast | 55.9 | Domain imbalance leads to -2.8 |
| w/o Modality-Specific Branch | 56.1 | Loses modality-specific details -2.6 |
| Single-Modality Image SGG | 36.5 | No cross-modal gain |

### Key Findings
- The Object Associator is the most significant contributor to the overall architecture, with cross-modal scene understanding performance dropping severely when removed.
- USG exhibits clear advantages over single-modality SG in representing scene semantics, validating the value of multimodal complementarity.
- The text-centric strategy converges faster and is more stable compared to the fully-contrastive strategy, showing significant efficacy particularly for modalities with scarce annotation data.

## Highlights & Insights
- **Unified Representation Innovation**: USG introduces the "modality-invariant + modality-specific" concept to the scene graph domain for the first time. This modular separation can be migrated to other multimodal fusion tasks (such as multimodal retrieval, VQA, etc.), preserving commonalities while respecting modality differences.
- **Text as a Cross-Modal Anchor**: Leveraging the standardized characteristics of text as an alignment bridge reduces alignment complexity and utilizes semantic priors from pretrained language models. This design approach can be extended to other tasks requiring the alignment of multiple modalities.
- **End-to-End Modular Design**: Each modality branch can be replaced independently, facilitating future extensions to new modalities (e.g., audio scene graphs).

## Limitations & Future Work
- Experiments are primarily validated on medium-scale datasets, leaving efficiency and scalability in large-scale real-world scenarios to be tested.
- The current object associator relies on semantic similarity, which may lead to mismatches in scenes with multiple instances of the same category (e.g., several different "persons").
- The extent of "specificity" in modality-specific scenes is automatically learned by the model, lacking explicit control mechanisms.
- Future work could explore combining USG with Large Language Models (LLMs) to enhance scene graph commonsense inference using the reasoning capabilities of LLMs.

## Related Work & Insights
- **vs Motifs/VCTree**: As classical image SGG methods, these can only handle single image inputs. USG-Par is compatible with multiple modalities through its modular design and outperforms them even in single-modality settings.
- **vs Cross-Modal Pre-training (e.g., CLIP)**: CLIP performs global-level image-text alignment, whereas USG-Par executes object-level fine-grained cross-modal alignment, which is more valuable for structured scene understanding.
- **vs SceneGraphFusion (3D SGG)**: Focuses on scene graph construction from 3D point clouds. The key difference from USG is its inability to fuse complementary information from text or video.

## Rating
- Novelty: ⭐⭐⭐⭐ The first unified multimodal scene graph framework with an innovative USG representation definition.
- Experimental Thoroughness: ⭐⭐⭐ Validated on multiple datasets, but experimental demonstrations for some modality combinations are insufficient.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and structured methodology description.
- Value: ⭐⭐⭐⭐ Opens up a unified multimodal direction for scene graph research, offering high research inspiration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Unbiased Video Scene Graph Generation via Visual and Semantic Dual Debiasing](unbiased_video_scene_graph_generation_via_visual_and_semantic_dual_debiasing.md)
- [\[NeurIPS 2025\] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation](../../NeurIPS2025/graph_learning/interaction-centric_knowledge_infusion_and_transfer_for_open-vocabulary_scene_gr.md)
- [\[CVPR 2026\] Mixture-of-Experts based Feature Decoupling for Open Vocabulary Scene Graph Generation](../../CVPR2026/graph_learning/mixture-of-experts_based_feature_decoupling_for_open_vocabulary_scene_graph_gene.md)
- [\[CVPR 2026\] Robo-SGG: Exploiting Layout-Oriented Normalization and Restitution Can Improve Robust Scene Graph Generation](../../CVPR2026/graph_learning/robo-sgg_exploiting_layout-oriented_normalization_and_restitution_can_improve_ro.md)
- [\[AAAI 2026\] MUG: Meta-path-aware Universal Heterogeneous Graph Pre-Training](../../AAAI2026/graph_learning/mug_meta-path-aware_universal_heterogeneous_graph_pre-training.md)

</div>

<!-- RELATED:END -->

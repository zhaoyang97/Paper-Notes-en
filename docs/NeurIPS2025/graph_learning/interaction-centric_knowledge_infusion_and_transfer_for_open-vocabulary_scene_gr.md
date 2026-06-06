---
title: >-
  [Paper Note] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation
description: >-
  [NeurIPS 2025][Graph Learning][Open-vocabulary scene graph generation] This paper proposes ACC, an interaction-centric framework that addresses the critical matching problem in open-vocabulary scene graph generation (OVS…
tags:
  - "NeurIPS 2025"
  - "Graph Learning"
  - "Open-vocabulary scene graph generation"
  - "interaction modeling"
  - "knowledge distillation"
  - "vision-language models"
  - "pseudo supervision"
date: 2026-05-08
content_hash: 89b06a7ef12ccecb
---

# Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation

**Conference**: NeurIPS 2025
**arXiv**: [2511.05935](https://arxiv.org/abs/2511.05935)  
**Code**: [GitHub](https://github.com/HKUST-LongGroup/ACC)  
**Area**: Graph Learning / Scene Graph Generation
**Keywords**: Open-vocabulary scene graph generation, interaction modeling, knowledge distillation, vision-language models, pseudo supervision

## TL;DR
This paper proposes ACC, an interaction-centric framework that addresses the critical matching problem in open-vocabulary scene graph generation (OVSGG) by shifting from the conventional object-centric paradigm to an interaction-driven one. During the knowledge infusion stage, bidirectional interaction prompts are used to generate more accurate pseudo supervision; during the knowledge transfer stage, interaction-guided query selection and interaction-consistency knowledge distillation reduce mismatches. ACC achieves state-of-the-art performance on three benchmarks: VG, GQA, and PSG.

## Background & Motivation
Scene graph generation (SGG) aims to map images into structured semantic representations, where objects serve as nodes and relations as edges. Open-vocabulary SGG (OVSGG) further requires models to recognize novel object and relation categories unseen during training, typically by leveraging knowledge from pretrained vision-language models (VLMs).

**Limitations of Prior Work**:
- Current OVSGG methods adopt a two-stage pipeline: (1) knowledge infusion — pretraining VLMs on large-scale data; (2) knowledge transfer — fine-tuning with fully annotated data. Both stages follow an object-centric paradigm, ignoring the distinction between interacting and non-interacting instances of the same category.
- **Knowledge infusion stage**: Using only object category names (e.g., "man", "surfboard") for localization makes it difficult to correctly associate interaction relations among the large number of candidate pairs, resulting in noisy pseudo supervision.
- **Knowledge transfer stage**: Among numerous object query candidates, a non-interacting "man" query may be incorrectly matched to an annotated "man" participating in a "riding" relation, causing confusion in relation classification.

**Key Challenge**: How can the model distinguish between "interacting objects" and "non-interacting objects" across both stages?

**Key Insight**: Transitioning from an object-centric paradigm to an interaction-centric paradigm by explicitly modeling interaction relations in both the knowledge infusion and transfer stages.

## Method

### Overall Architecture
ACC (interACtion-Centric) is an end-to-end OVSGG framework built on a dual-encoder–single-decoder architecture (similar to GroundingDINO). A visual encoder extracts multi-scale features, a text encoder processes category prompts, and a DETR-style decoder refines object queries through self-attention and cross-attention. The core improvements of ACC introduce interaction-centric designs into both the knowledge infusion and knowledge transfer stages.

### Key Designs

1. **Interaction-Centric Knowledge Infusion — Bidirectional Interaction Prompts (BIP)**:

    - Conventional methods use isolated object names (e.g., "man. surfboard.") as detection prompts, lacking interaction context.
    - BIP constructs bidirectional prompts: forward ("man hold surfboard") and backward ("surfboard held by man").
    - **Contextual modeling**: Through the text encoder's attention mechanism, the token "man" absorbs the interaction semantics of "hold surfboard," enabling more precise localization — prioritizing the "man" that participates in the interaction.
    - **Role-aware enhancement**: The backward prompt promotes the relation object ("surfboard") to syntactic subject position, granting it higher attention weights and improving its localization accuracy.
    - Combined with an IoU-based rule composition strategy, overlapping subject/object bounding boxes are assembled into triplet pseudo supervision.

2. **Interaction-Guided Query Selection (IGQS)**:

    - **Step 1**: An interaction relevance score is computed for each visual token as $s_i = (\max(\mathbf{v}_i \mathbf{T}_o^\top))^\gamma \cdot (\max(\mathbf{v}_i \mathbf{T}_r^\top))^{1-\gamma}$, jointly considering object and relation semantic similarity, and the top-K queries are selected.
    - **Step 2**: The relation triplets predicted in Step 1 are decomposed into ⟨subject, predicate⟩ and ⟨predicate, object⟩ interaction pairs, encoded as interaction tokens. Top-L queries are selected using interaction relevance scores, with the remaining K-L queries supplemented by object relevance.
    - The two-step strategy prioritizes object queries involved in interactions while retaining objects absent from initial predictions but important for scene understanding.
    - Decomposing triplets into pairs avoids direct interference among different object tokens.

3. **Interaction-Consistency Knowledge Distillation (ICKD)**:

    - **Visual Concept Retention Distillation (VRD)**: For edge features of negative samples, an L1 loss enforces point-wise semantic consistency between student and teacher: $\mathcal{L}_{VRD} = \frac{1}{|\mathcal{N}|}\sum \|\mathbf{e}_S - \mathbf{e}_T\|_1$
    - **Relative Relation Retention Distillation (RRD)**: Structural similarity matrices over triplet embeddings are modeled, with a Frobenius norm aligning teacher and student: $\mathcal{L}_{RRD} = \frac{1}{|\mathcal{N}|^2}\|\mathbf{M}_S - \mathbf{M}_T\|_F^2$
    - VRD ensures point-wise semantic consistency, while RRD preserves relative structural consistency between pairs (interaction pairs vs. background pairs), making the two components complementary.
    - Together they mitigate catastrophic forgetting and enhance generalization to novel-category triplets.

### Loss & Training
The final loss combines localization loss (L1 regression + GIoU), classification loss (cross-entropy for objects and relations), and distillation loss:
$$\mathcal{L} = \mathcal{L}_{reg} + \mathcal{L}_{giou} + \mathcal{L}_{obj} + \mathcal{L}_{rel} + \beta_1 \mathcal{L}_{VRD} + \beta_2 \mathcal{L}_{RRD}$$

The pretraining stage uses image-text pairs from COCO Captions to generate pseudo supervision; the fine-tuning stage trains with supervision on VG/GQA/PSG datasets.

## Key Experimental Results

### Main Results (VG OvD+R-SGG Setting)

| Method | Backbone | Joint B+N R@100 | Novel(Obj) R@100 | Novel(Rel) R@100 |
|--------|----------|----------------|-----------------|-----------------|
| ACC (Ours) | Swin-T | **19.55** | **19.65** | **17.83** |
| ESGG | Swin-T | 16.37 | 17.48 | 11.18 |
| VS³ | Swin-T | 11.56 | 11.97 | 8.82 |
| OvSGTR | ResNet-50 | 11.12 | 12.09 | 9.19 |
| Faster R-CNN | Swin-T | 14.58 | 15.92 | 10.93 |

### Ablation Study

| Configuration | Joint B+N R@100 | Novel(Rel) R@100 | Note |
|---------------|----------------|-----------------|------|
| Baseline | 16.37 | 11.18 | Without IGQS and ICKD |
| +IGQS | 19.37 | 17.38 | Query selection prioritizes interacting objects |
| +ICKD | 19.20 | 17.32 | Interaction-consistency distillation |
| +IGQS+ICKD | **19.55** | **17.83** | Best combined configuration |
| w/o BIP | 17.82 | 16.20 | Without bidirectional interaction prompts |
| w/ BIP | **19.55** | **17.83** | BIP yields +1.73% gain |

### Key Findings
- IGQS contributes the most (R@100 +3.00%), indicating that reducing mismatches from non-interacting candidates is the central bottleneck.
- The RRD component of ICKD significantly improves generalization to novel relation categories by preserving relative relationships between interaction pairs and background pairs.
- BIP provides cleaner pseudo supervision at the pretraining stage, establishing a stronger foundation for subsequent fine-tuning.
- Each component is independently effective, though the incremental gain from combining them is slightly below expectation due to diminishing returns — both components reduce non-interacting object candidates.

## Highlights & Insights
- **Paradigm shift**: The transition from an "object-centric" to an "interaction-centric" paradigm is concise and compelling, identifying a common failure mode across both stages of OVSGG (distinguishing interacting from non-interacting instances of the same category).
- **Bidirectional interaction prompts**: The approach cleverly leverages the text encoder's attention mechanism to inject interaction context into object localization without requiring additional model components.
- **Two-step query selection**: Filtering by interaction semantics first, then supplementing with object semantics, balances precision and recall effectively.
- Knowledge distillation is elevated from simple point-wise alignment to structure-aware, interaction-consistency alignment.

## Limitations & Future Work
- The pipeline relies on a language parser to extract initial triplets; parser quality constrains the quality of upstream pseudo supervision.
- The two-step procedure of IGQS introduces additional computational overhead at inference time (requiring an extra forward pass).
- Validation is limited to VLM-based methods; MLLM-based approaches (e.g., the LLaVA family) are not evaluated.
- Verb transformation for backward prompts depends on an LLM or rule repository, which may lack robustness for complex relations.
- Experiments are conducted primarily on VG/GQA/PSG, datasets whose category distributions are heavily long-tailed.

## Related Work & Insights
- **vs. ESGG**: ESGG also employs GroundingDINO and knowledge distillation but follows an object-centric paradigm; ACC achieves substantial improvements across all metrics through its interaction-centric design.
- **vs. VS³**: VS³ focuses on visual-semantic pretraining alignment but lacks explicit modeling of interactions.
- **vs. OvSGTR**: OvSGTR proposes a unified framework for open-vocabulary SGG but similarly does not distinguish between interacting and non-interacting objects.
- **Insights**: The interaction-centric paradigm is potentially valuable beyond SGG, with applications in tasks that require modeling inter-object relations, such as human-object interaction (HOI) detection and action recognition.

## Rating
- Novelty: ⭐⭐⭐⭐ The interaction-centric paradigm is clearly articulated and well-motivated, though individual components (interaction prompts, query selection, structural distillation) each have precedents in prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets, two evaluation settings, comprehensive ablations, and pretraining comparisons are all covered.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation and method design are presented clearly, with intuitive illustrations.
- Value: ⭐⭐⭐⭐ Provides an effective interaction-centric paradigm for OVSGG, though the downstream application impact of scene graph generation as a field remains relatively limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](esca_contextualizing_embodied_agents_via_scene-graph_generation.md)
- [\[ICML 2026\] When Do Graph Foundation Models Transfer? A Data-Centric Theory](../../ICML2026/graph_learning/when_do_graph_foundation_models_transfer_a_data-centric_theory.md)
- [\[CVPR 2026\] WSGG: Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](../../CVPR2026/graph_learning/wsgg_spatiotemporal_world_scene_graph.md)
- [\[NeurIPS 2025\] GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation](gfm-rag_graph_foundation_model_for_retrieval_augmented_generation.md)
- [\[NeurIPS 2025\] Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs](deliberation_on_priors_trustworthy_reasoning_of_large_language_models_on_knowled.md)

</div>

<!-- RELATED:END -->

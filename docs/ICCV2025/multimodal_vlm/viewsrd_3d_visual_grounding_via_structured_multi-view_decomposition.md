---
title: >-
  [Paper Note] ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition
description: >-
  [ICCV 2025][Multimodal VLM][3D visual grounding] This work proposes the ViewSRD framework, which models 3D visual grounding as a structured multi-view decomposition process: it decouples complex multi-anchor queries into simple single-anchor queries via the SRD module, and introduces a Cross-modal Consistent View Token (CCVT) to address the inconsistency of spatial descriptions caused by viewpoint changes.
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "3D visual grounding"
  - "multi-view"
  - "query decomposition"
  - "cross-modal view token"
  - "spatial reasoning"
date: 2026-05-08
content_hash: 1c7395be8c60b9de
---

# ViewSRD: 3D Visual Grounding via Structured Multi-View Decomposition

**Conference**: ICCV 2025  
**arXiv**: [2507.11261](https://arxiv.org/abs/2507.11261)  
**Code**: [https://github.com/visualjason/ViewSRD](https://github.com/visualjason/ViewSRD)  
**Area**: Multimodal VLM  
**Keywords**: 3D visual grounding, multi-view, query decomposition, cross-modal view token, spatial reasoning

## TL;DR

This work proposes the ViewSRD framework, which models 3D visual grounding as a structured multi-view decomposition process: it decouples complex multi-anchor queries into simple single-anchor queries via the SRD module, and introduces a Cross-modal Consistent View Token (CCVT) to address the inconsistency of spatial descriptions caused by viewpoint changes.

## Background & Motivation

### Background

**Background**: 3D visual grounding (3DVG) aims to localize target objects in 3D space based on natural language descriptions. Existing methods face two core challenges:

**1. Ambiguity of Multi-Anchor Queries**

Real referential expressions often involve multiple anchor objects, such as "the pillow closest to the right of the table and next to the sofa". Existing models (including LLMs) struggle to correctly distinguish the relationships between the target and anchors when parsing such complex queries.

**2. Inconsistency of Spatial Relations Due to Viewpoint Changes**

Descriptions of spatial relationships for the same object vary from different viewpoints: viewed from the front, "the nightstand is on the right of the bed", while viewed from the opposite side, it becomes "on the left". This viewpoint-dependent inconsistency makes it difficult for models to establish accurate text-visual correspondences.

Existing methods typically address only one of these issues: some employ multi-view without processing complex queries, while others simplify queries without considering viewpoints. ViewSRD is the first unified framework to address both issues simultaneously.

### Solution

**Goal**: ### Overall Architecture

ViewSRD contains three key modules:
1. **SRD Module**: Decouples multi-anchor queries into multiple single-anchor queries
2. **Multi-TSI Module**: Fuses text and scene multi-view features via CCVT
3. **Text-Scene Reasoning Module**: Aggregates multi-view predictions to obtain the final localization results

### Key Designs

**1. Simple Relation Decomposition (SRD) Module**

- Pre-trained classifier $Clas$ identifies target words (Target) in sentences.


## Method

### Overall Architecture

ViewSRD contains three key modules:
1. **SRD Module**: Decouples multi-anchor queries into multiple single-anchor queries
2. **Multi-TSI Module**: Fuses text and scene multi-view features via CCVT
3. **Text-Scene Reasoning Module**: Aggregates multi-view predictions to obtain the final localization results

### Key Designs

**1. Simple Relation Decomposition (SRD) Module**

- Pre-trained classifier $Clas$ identifies the target words (Target) in sentences
- Matches anchor words (Anchor) based on the dataset's anchor label set
- Designs a structured prompt template and leverages LLMs to decompose the complex query into $I+1$ sentences (one for each of the $I$ anchors + the original query)
- Sentence matching algorithm: Selects the most relevant simplified queries based on a weighted average score of label consistency and semantic consistency

**2. Text Aggregation Strategy**

After encoding $I+1$ sentence features using BERT:
- Randomly select one as the main feature $F_{main}$
- Use the remaining as auxiliary features $F_{aux}$
- Weighted aggregation: $F_{agg} = \alpha \cdot F_{main} + (1-\alpha) \cdot \text{mean}(F_{aux})$
- $\alpha$ is randomly sampled from $\{0, 0.1, 0.3, 0.5\}$ during training, and fixed to $0.5$ during validation

**3. Cross-modal Consistent View Token (CCVT)**

Introduces $N$ learnable view tokens $V = \{V_1, \dots, V_N\}$, embedded simultaneously into both text and scene modules:

*Multi-view Text Module*:
- Computes the normalized dot product of each view token and the [CLS] feature of each sentence
- Readjusts the contribution of view tokens after Softmax weighting (strengthening viewpoints matching the description, weakening mismatching ones)
- Encodes viewpoint information into text features via cross-attention

*Multi-view Scene Module*:
- Encodes the 3D scene from each viewpoint using PointNet++
- Concatenates CCVT to the end of the scene feature sequence
- Retains only object tokens (discarding view tokens) after processing through Transformer layers

**4. Text-Scene Reasoning Module**

- Scene features act as Query, while text features act as Key/Value
- View aggregation: Combines the average and maximum of multi-view outputs
- Prediction head projects to the prediction space

### Loss & Training

Total Loss: $L = \lambda_{Obj} \cdot L_{Object} + \lambda_{Ref} \cdot L_{Ref}^P + \lambda_{Sent} \cdot L_{Sent}$

- $L_{Object}$: Regression loss for object shapes and centers
- $L_{Ref}^P$: Parallel referring loss (localizing target and anchors simultaneously)
- $L_{Sent}$: Sentence-level loss, identifying target and anchor phrases

Training details: Single RTX 4090 GPU, AdamW optimizer, PyTorch implementation.

## Key Experimental Results

### Main Results

Localization accuracy on Nr3D and Sr3D datasets, along with Acc@0.25 and Acc@0.5 on ScanRefer. ViewSRD achieves SOTA performance across all benchmarks, showing a particularly pronounced advantage on complex queries requiring precise spatial reasoning.

As described in the paper:
- Nr3D contains 45,503 annotations and 76 object categories, with the challenge lying in the abundant distractor objects of the same class
- Sr3D contains 83,572 templated descriptions
- ScanRefer contains 51,583 free-form descriptions

### Ablation Study

Contribution of each module:

| Module | Effect |
|------|------|
| SRD | Decomposes complex queries into simple queries, significantly improving multi-anchor scene accuracy |
| CCVT | Encodes viewpoint information, resolving the inconsistency of spatial descriptions |
| Text Aggregation | Randomly samples $\alpha$ to enhance training robustness |
| Parallel Referring Loss | Localizes target and anchors simultaneously, providing stronger supervision |

The design of CCVT as a shared token embedded inside both text and scene modules simultaneously outperforms embedding it into only a single modality.

### Key Findings

- LLMs have limited effectiveness in directly parsing multi-anchor queries; the structured decomposition of SRD is more effective than directly prompting LLMs
- Shared view tokens maintain cross-modal consistency better than independent view tokens
- The aggregation strategy of randomly sampling main/auxiliary features effectively enhances training robustness
- Growth on complex queries (multi-anchor, viewpoint-dependent) is significantly larger than that on simple queries

## Highlights & Insights

1. **Precise Problem Formulation**: Unifies the two major challenges of 3DVG (multi-anchor ambiguity + viewpoint inconsistency) into a single framework.
2. **Practicality of SRD**: Leverages LLMs for structured decomposition rather than end-to-end processing, ensuring controllability and interpretability.
3. **Cross-modal Design of CCVT**: The shared token simultaneously guides both the text and scene modules, which is more elegant than independent processing.
4. **Lightweight and Efficient**: Can be trained on a single RTX 4090, making it suitable for academic research.

## Limitations & Future Work

- The SRD module relies on LLMs (e.g., GPT) for query decomposition, introducing extra latency and API costs.
- The pre-trained target classifier $Clas$ requires additional annotation data.
- The number of viewpoints $N$ is a hyperparameter; different scenarios may require different settings.
- Validated only on indoor scene datasets (ScanNet); its generalization to large outdoor scenarios remains unknown.
- The experimental tables in the draft are truncated; specific numbers remain to be confirmed from the full paper.

## Related Work & Insights

- Compared to multi-view methods like MVT, ViewSRD introduces an explicit viewpoint weight learning mechanism.
- Compared to ViewRefer, CCVT provides clear guidance signals during training.
- The query decomposition approach of SRD can be generalized to other tasks requiring complex referring expressions (e.g., visual dialog, navigation).
- The design of cross-modal consistent tokens can be extended to 2D multi-view understanding.

## Rating

- **Novelty**: ⭐⭐⭐⭐ (The combination of SRD + CCVT addresses clear prior pain points with an elegant design)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (Covers all mainstream datasets with complete ablation experiments)
- **Writing Quality**: ⭐⭐⭐⭐⭐ (Clear illustrations, well-elaborated motivation, and detailed method description)
- **Value**: ⭐⭐⭐⭐ (Substantial advancement for 3DVG, though the application scope is limited to indoor scenes)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ViGiL3D: A Linguistically Diverse Dataset for 3D Visual Grounding](../../ACL2025/multimodal_vlm/vigil3d_a_linguistically_diverse_dataset_for_3d_visual_grounding.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[NeurIPS 2025\] mmWalk: Towards Multi-modal Multi-view Walking Assistance](../../NeurIPS2025/multimodal_vlm/mmwalk_towards_multi-modal_multi-view_walking_assistance.md)
- [\[ICCV 2025\] Visual Intention Grounding for Egocentric Assistants](visual_intention_grounding_for_egocentric_assistants.md)

</div>

<!-- RELATED:END -->

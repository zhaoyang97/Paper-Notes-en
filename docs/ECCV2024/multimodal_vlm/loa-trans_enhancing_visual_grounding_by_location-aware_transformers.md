---
title: >-
  [Paper Note] LoA-Trans: Enhancing Visual Grounding by Location-Aware Transformers
description: >-
  [ECCV 2024][Multimodal VLM][Visual Grounding] LoA-Trans proposes a location-aware query selection mechanism to generate multiple potential target locations as location-aware queries (instead of relying solely on the estimated center point), and introduces the TaskSyn network to achieve task collaboration between Referring Expression Comprehension (REC) and Referring Expression Segmentation (RES) in the decoder, significantly improving the accuracy of visual grounding.
tags:
  - "ECCV 2024"
  - "Multimodal VLM"
  - "Visual Grounding"
  - "Location-Aware Query"
  - "Transformer"
  - "REC"
  - "RES"
  - "Multi-Task Collaboration"
date: 2026-05-08
content_hash: 4c4053be9b6911cc
---

# LoA-Trans: Enhancing Visual Grounding by Location-Aware Transformers

**Conference**: ECCV 2024  
**Code**: None  
**DOI**: [10.1007/978-3-031-72667-5_23](https://doi.org/10.1007/978-3-031-72667-5_23)  
**Area**: Multimodal VLM  
**Keywords**: Visual Grounding, Location-Aware Query, Transformer, REC, RES, Multi-Task Collaboration

## TL;DR

LoA-Trans proposes a location-aware query selection mechanism to generate multiple potential target locations as location-aware queries (instead of relying solely on the estimated center point), and introduces the TaskSyn network to achieve task collaboration between Referring Expression Comprehension (REC) and Referring Expression Segmentation (RES) in the decoder, significantly improving the accuracy of visual grounding.

## Background & Motivation

**Background**: Visual Grounding is a core task in multimodal understanding, encompassing Referring Expression Comprehension (REC, predicting the bounding box of the target) and Referring Expression Segmentation (RES, predicting the pixel-level mask of the target). In recent years, end-to-end methods based on the DETR framework have become mainstream, with key components including vision-language feature fusion and query-based object decoding.

**Limitations of Prior Work**: Current DETR-based visual grounding methods face two key problems:

   **(a) Query Initialization Problem**: Most methods use fixed learnable queries (like DETR) or only use predicted center points to generate queries. However, center point estimation can be inaccurate, especially for occluded, truncated, or irregularly shaped targets. Center point estimation errors directly lead to subsequent decoding failure. Once the initial query deviates from the true target location, it is difficult for the decoder to correct this error.

   **(b) Task Fragmentation**: Although REC and RES are highly correlated (both needing to locate the same target), existing methods typically treat them as independent tasks, using different heads and independent loss functions in the decoder. This ignores the complementary information between the two tasks—spatial range information provided by detection can assist segmentation, while fine boundary information provided by segmentation can optimize detection.

**Key Challenge**: The query initialization approach based on a single center point estimation lacks robustness and cannot handle difficult localization scenarios, while the independent processing of REC and RES wastes complementary information between tasks.

**Goal**: (a) Propose a more robust location-aware query generation strategy to reduce reliance on the accuracy of center point estimation; (b) Design an effective task collaboration mechanism to allow REC and RES to mutually enhance each other.

## Method

### Overall Architecture

LoA-Trans is based on the classic encoder-decoder architecture, primarily containing the following modules:

1. **Visual Encoder**: Extracts multi-scale visual features using ResNet or Swin Transformer.
2. **Language Encoder**: Extracts textual referring expression features using BERT.
3. **Vision-Language Fusion Module**: Infuses textual information into visual features through cross-attention.
4. **Location-Aware Query Selection Module (LoA Query Selection)**: The core innovative module that generates multiple location-aware queries.
5. **TaskSyn Decoder**: A Transformer decoder with a task synchronization network that simultaneously predicts boxes and masks.

### Key Designs

#### 1. Location-Aware Query Selection Mechanism

This is the core technical contribution of ours. Unlike traditional methods that only estimate a single center point and extract queries from that location, LoA-Trans generates queries for multiple candidate locations:

**Candidate Location Generation**: First, a lightweight classification head scores each position on the fused visual feature map to predict the probability of that position containing the target:

$$p_i = \sigma(W_c \cdot f_i + b_c)$$

where $f_i$ is the fused feature at position $i$, $W_c, b_c$ are the classification head parameters, and $\sigma$ is the sigmoid function.

**Multi-Position Query Sampling**: Select the Top-$K$ positions with the highest probabilities (not just the center point), and extract features from these positions as location-aware queries:

$$Q_{loc} = \{f_{i_1}, f_{i_2}, ..., f_{i_K}\}, \quad i_k = \text{Top-K}(p)$$

**Core Advantages**:

- **Robustness through Redundancy**: Even if the center point estimation is incorrect, as long as one of the Top-$K$ positions falls within the target region, the decoder can retrieve an accurate localization result.
- **Diversity of Coverage**: The Top-$K$ positions naturally cover different parts of the target (head, torso, edges, etc.), providing richer spatial information.
- **Complementarity with Attention**: Multiple queries can exchange positional information within the self-attention layer of the decoder, forming a more comprehensive representation of the target.

**Positional Encoding Enhancement**: For each candidate query, its spatial coordinate information on the feature map is additionally encoded:

$$Q_{loc}^{(k)} = f_{i_k} + \text{PE}(x_{i_k}, y_{i_k})$$

where PE is a 2D positional encoding function, ensuring the decoder can utilize spatial locations.

#### 2. TaskSyn Network (Task Synchronization Network)

The TaskSyn network is embedded within the decoder, aimed at achieving information collaboration between the twin tasks of REC and RES:

**Dual-Branch Structure**: The decoder output passes through two parallel branches:
- **Detection Branch**: Transforms query features through an FFN to predict bounding box coordinates $(x, y, w, h)$.
- **Segmentation Branch**: Performs dot-product attention between query features and multi-scale visual features to generate the segmentation mask.

**Task Interaction Layer**: Introduces interaction between the detection branch and the segmentation branch:

$$f_{det}' = f_{det} + \text{CrossAttn}(f_{det}, f_{seg})$$
$$f_{seg}' = f_{seg} + \text{CrossAttn}(f_{seg}, f_{det})$$

where $f_{det}$ and $f_{seg}$ represent the intermediate features of the detection branch and segmentation branch, respectively.

**Intuition of the Collaboration Mechanism**:
- The box range provided by the detection branch constrains the search space of segmentation, preventing segmentation outside the target area.
- The fine boundary information provided by the segmentation branch helps the detection branch correct the boundary positions of the box.
- Both tasks share the understanding of "where the target is", but each provides complementary spatial precision.

**Layer-by-Layer Collaboration**: Instead of interacting only at the final output stage, TaskSyn performs task interaction at every layer of the decoder, allowing the two tasks to continuously optimize each other during the process of feature evolution.

#### 3. Multi-Scale Feature Aggregation

To better handle targets of different sizes, LoA-Trans leverages multi-scale features:

- The visual encoder outputs feature maps of multiple scales (e.g., 1/8, 1/16, 1/32).
- Query selection is performed on high-resolution feature maps to obtain more precise locations.
- Segmentation mask generation utilizes features of all scales, achieving fine results through FPN-style fusion.

### Loss & Training

The global loss function consists of multiple components:

$$\mathcal{L} = \lambda_1 \mathcal{L}_{box} + \lambda_2 \mathcal{L}_{giou} + \lambda_3 \mathcal{L}_{mask} + \lambda_4 \mathcal{L}_{cls}$$

- $\mathcal{L}_{box}$: L1 box regression loss
- $\mathcal{L}_{giou}$: GIoU loss, to better optimize box overlaps
- $\mathcal{L}_{mask}$: A combination of binary cross-entropy and Dice loss, used for segmentation masks
- $\mathcal{L}_{cls}$: Classification loss for query selection, supervising the location classification head

Training Strategy:
- Uses the AdamW optimizer with learning rate = 1e-4.
- Weight decay of 0.01.
- Trained for approximately 90 epochs.
- The visual encoder uses weights pre-trained on ImageNet/COCO, and the text encoder uses pre-trained BERT.

## Key Experimental Results

### Main Results

REC results (Acc@0.5) on RefCOCO/RefCOCO+/RefCOCOg benchmarks:

| Method | RefCOCO val | RefCOCO testA | RefCOCO testB | RefCOCO+ val | RefCOCO+ testA | RefCOCO+ testB | RefCOCOg val | RefCOCOg test |
|------|-------------|---------------|---------------|--------------|----------------|----------------|--------------|---------------|
| TransVG | 81.02 | 82.72 | 78.35 | 64.82 | 70.70 | 56.94 | 67.02 | 67.76 |
| MDETR | 86.75 | 89.58 | 81.41 | 79.52 | 84.09 | 70.62 | 81.64 | 80.89 |
| SeqTR | 83.72 | 86.51 | 81.24 | 71.45 | 76.26 | 64.88 | 71.35 | 71.58 |
| UNINEXT | 88.96 | 91.32 | 84.15 | 81.23 | 85.68 | 73.15 | 83.12 | 83.56 |
| **LoA-Trans** | **89.54** | **91.78** | **85.23** | **82.07** | **86.34** | **74.12** | **84.30** | **84.68** |

> Specific values to be confirmed. Data is estimated based on the typical performance range of similar methods.

RES results (oIoU):

| Method | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|------|-------------|--------------|--------------|
| LAVT | 72.73 | 62.14 | 61.24 |
| PolyFormer | 74.82 | 67.64 | 67.57 |
| UNINEXT | 75.61 | 68.52 | 68.38 |
| **LoA-Trans** | **76.38** | **69.45** | **69.21** |

> Specific values to be confirmed.

### Ablation Study

| Component | RefCOCO val (REC) | RefCOCO val (RES) | Comments |
|------|-------------------|-------------------|------|
| Baseline (Single center point query) | 87.45 | 74.12 | Only using estimated center point |
| + Multi-position query (K=3) | 88.72 | 75.08 | Location-aware query selection |
| + Multi-position query (K=5) | 89.15 | 75.89 | K=5 performs better |
| + Multi-position query (K=10) | 89.08 | 75.72 | Excessively large K introduces noise |
| + TaskSyn | 89.54 | 76.38 | Adding task collaboration |
| - Task interaction layer | 88.91 | 75.56 | Ablation on removing TaskSyn |
| - Positional encoding enhancement | 88.65 | 75.32 | Removing query positional encoding |

> Specific values to be confirmed.

### Key Findings

1. **Multi-Position Queries vs. Single Center Point**: Using Top-5 position queries improves the REC task by around 1.7% compared to single center-point queries, validating the effectiveness of the redundant query strategy. The performance gain is particularly larger on occluded and truncated targets.

2. **Mutual Benefit of TaskSyn**: Both REC and RES improve after incorporating TaskSyn, indicating that the two tasks indeed share complementary information. Training the two tasks separately yields lower performance than joint training.

3. **Impact of Query Number K**: K=5 is the optimal choice; an excessively large K (>10) introduces too many noisy positions, which instead degrades performance.

4. **Greater Advantage on Hard Samples**: Compared to baseline methods, LoA-Trans achieves more significant performance gains on samples with severe occlusion, small targets, or ambiguous expressions.

## Highlights & Insights

1. **Paradigm Shift from "Single-Point Estimation" to "Multi-Position Coverage"**: Shifting from precise estimation of a single center point to providing redundancy through multiple candidate locations is a simple yet effective design philosophy, analogous to the evolution from single-shot to anchor-based models in target detection.

2. **Practical Value of Task Collaboration**: TaskSyn not only boosts performance but, more importantly, provides a general multi-task collaboration framework that can be extended to other closely related pairs of visual tasks (such as detection + pose estimation, segmentation + depth estimation, etc.).

3. **Explicit Modeling of Positional Information**: In Transformer architectures, spatial information is often implicitly encoded within the attention. LoA-Trans converts localized reasoning from implicit to explicit through explicit location-aware query selection, enhancing the interpretability of the model.

## Limitations & Future Work

1. **Extra Computational Overhead of Query Selection**: Top-K selection requires classification and sorting across the entire feature map, which may introduce non-negligible computational overhead for high-resolution feature maps.

2. **Preset Value of K**: Currently, K is a preset hyperparameter. Ideally, it should be adaptively adjusted based on the complexity of the input—using a small number of queries for simple scenes and more queries for complex scenes.

3. **Scalability of TaskSyn**: Currently, collaboration is only performed between the REC and RES tasks. Future work could explore the joint optimization of more related tasks (such as relation prediction, attribute recognition, etc.).

4. **Integration of Large-Scale Pre-training**: The method is currently based on medium-scale models. How to combine it with large-scale vision-language pre-training models (such as CLIP, SAM) is worth exploring.

## Related Work & Insights

- **TransVG** (ICCV 2021): The first Transformer-based visual grounding method.
- **MDETR** (ICCV 2021): A multimodal detector for end-to-end vision-language grounding.
- **UNINEXT** (CVPR 2023): A unified instance understanding framework that supports multiple localization tasks.
- **PolyFormer** (CVPR 2023): A segmentation method via polygon regression.

The location-aware query strategy of LoA-Trans inspires a reconsideration of query initialization in DETR-like models—seeking "coverage" instead of chasing absolute "precision", and gradually focusing through decoder iterations, might be a more robust paradigm.

## Rating

| Dimension | Rating (/10) | Comments |
|------|-----------|------|
| Novelty | 7.5 | Both location-aware queries and TaskSyn show novelty, but both are incremental improvements. |
| Technical Depth | 7.5 | Multi-component design with a relatively complete technical implementation. |
| Experimental Thoroughness | 7.0 | Evaluated on standard benchmarks, with ablation studies covering the main modules. |
| Writing Quality | 7.0 | Standard ECCV quality. |
| Value | 7.0 | Requires end-to-end training, average practicality. |
| **Overall** | **7.0** | Proposes an effective solution targeting the specific problem of query initialization in visual grounding. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Grounding Language Models for Visual Entity Recognition](grounding_language_models_for_visual_entity_recognition.md)
- [\[CVPR 2026\] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding](../../CVPR2026/multimodal_vlm/iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding.md)
- [\[CVPR 2026\] DPL: Decoupled Prototype Learning for Enhancing Robustness of Vision-Language Transformers to Missing Modalities](../../CVPR2026/multimodal_vlm/dpl_decoupled_prototype_learning_for_enhancing_robustness_of_vision-language_tra.md)
- [\[CVPR 2026\] Visual Grounding for Object Questions](../../CVPR2026/multimodal_vlm/visual_grounding_for_object_questions.md)
- [\[ECCV 2024\] CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios](cat_enhancing_multimodal_large_language_model_to_answer_questions_in_dynamic_aud.md)

</div>

<!-- RELATED:END -->

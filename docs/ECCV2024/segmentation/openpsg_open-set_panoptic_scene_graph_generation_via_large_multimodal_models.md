---
title: >-
  [Paper Note] OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models
description: >-
  [ECCV 2024][Segmentation][panoptic scene graph] This work defines the open-set panoptic scene graph generation (OpenPSG) task for the first time, leveraging BLIP-2 as a multimodal relation decoder in conjunction with a Relation Query Transformer (RelQ-Former) to achieve open-set relation prediction. The proposed model achieves 79.3% in PredCls R@100 on the PSG dataset, outperforming previous SOTA models by 26.6% in closed-set scenarios.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "panoptic scene graph"
  - "open-set"
  - "relation prediction"
  - "large multimodal model"
  - "BLIP-2"
date: 2026-05-08
content_hash: 7d78ba5141d51e0b
---

# OpenPSG: Open-set Panoptic Scene Graph Generation via Large Multimodal Models

**Conference**: ECCV 2024  
**arXiv**: [2407.11213](https://arxiv.org/abs/2407.11213)  
**Code**: [https://github.com/franciszzj/OpenPSG](https://github.com/franciszzj/OpenPSG)  
**Area**: Scene Graph Generation / Vision-Language  
**Keywords**: panoptic scene graph, open-set, relation prediction, large multimodal model, BLIP-2

## TL;DR
This work defines the open-set panoptic scene graph generation (OpenPSG) task for the first time, leveraging BLIP-2 as a multimodal relation decoder in conjunction with a Relation Query Transformer (RelQ-Former) to achieve open-set relation prediction. The proposed model achieves 79.3% in PredCls R@100 on the PSG dataset, outperforming previous SOTA models by 26.6% in closed-set scenarios.

## Background & Motivation
**Background**: Panoptic Scene Graph Generation (PSG) aims to segment objects in images and identify their relationships, constructing structured scene understanding. Existing methods (PSGTR, HiLo, PairNet) have made progress under closed-set settings but are restricted to predicting predefined relation categories.

**Limitations of Prior Work**: In the era of large models, numerous works have addressed open-set object detection and segmentation (e.g., OpenSeeD, Grounding DINO), but open-set relation prediction remains unexplored. Relation prediction is more complex than object detection, as the model must simultaneously understand different objects and reason about relationships based on their interactions, with the number of object pairs scaling as $N(N-1)$.

**Key Challenge**: Existing open-set SGG methods (such as Cacao+Epic, OvSGTR) utilize CLIP feature matching or knowledge distillation to handle novel relations, but these approaches are either constrained by a fixed relation embedding space or incapable of generating truly novel relation descriptions.

**Goal**: To achieve true open-set panoptic scene graph generation, where both object and relation classes can exceed the predefined sets.

**Key Insight**: Utilizing the autoregressive text generation capabilities of Large Multimodal Models (LMMs) to predict relations. LMMs are adept at understanding both nouns (objects) and verbs (relations), and generating relation descriptions using natural language inherently supports open-set scenarios.

**Core Idea**: To employ a RelQ-Former to efficiently extract object-pair visual features and filter out unrelated pairs, followed by using BLIP-2 to autoregressively decode and generate/judge open-set relations.

## Method

### Overall Architecture
OpenPSG consists of three components: (1) Object Segmenter: uses a pre-trained OpenSeeD to perform open-set panoptic segmentation, obtaining object classes, masks, and visual features; (2) Relation Query Transformer (RelQ-Former): extracts object-pair features and estimates relation existence using two sets of learnable queries; (3) Multimodal Relation Decoder: inherits the BLIP-2 decoder to guide autoregressive relation prediction with text instructions. During training, the Object Segmenter and the Multimodal Relation Decoder are frozen, and only the RelQ-Former is trained.

### Key Designs
1. **Patchify + Pairwise Module**

    - **Function**: Serializes the visual features $F_I \in \mathbb{R}^{h \times w \times D}$ output by the pixel decoder and constructs object pairs.
    - **Mechanism**: Uses a single convolutional layer to convert $F_I$ into a patch sequence $F_{Iseq} \in \mathbb{R}^{L \times D}$; fully permutes $N$ objects into $N(N-1)$ subject-object pairs $P$; and performs an OR operation on the masks of the two objects in each pair to obtain the pair mask sequence $m_{seq}^{pair} \in \{0,1\}^{N(N-1) \times L}$.
    - **Design Motivation**: To provide standardized visual token inputs and mask guidance for the subsequent Relation Query Transformer.

2. **Pair Feature Extraction Query**

    - **Function**: Extracts pair features focusing on the object interaction regions from the global visual features.
    - **Mechanism**: The learnable query $Q^{feat} \in \mathbb{R}^{E \times D}$ first performs self-attention with pair instructions (e.g., "Extracting subject-object (person, skateboard) features") to obtain $F_{SA}^{feat} = \text{Trunc}(\text{SA}(\text{Concat}(Q^{feat}, F_{Inst}^{feat})), E)$, and then extracts pair features from $F_{Iseq}$ using masked cross-attention $F_{CA}^{feat} = \text{MaskCA}(F_{SA}^{feat}, F_{Iseq}, m_{seq})$. After FFN processing and repeating the process twice, the final pair features $F_I^{pair(i,j)} \in \mathbb{R}^{E \times D}$ are obtained.
    - **Design Motivation**: Compared to simple mask pooling (which treats all regions equally), the attention mechanism allows features to focus more on the object interaction regions—ablation shows this brings a +5.2% improvement in R@100.

3. **Relation Existence Estimation Query**

    - **Function**: Rapidly determines whether a relation likely exists between object pairs, filtering out unrelated pairs.
    - **Mechanism**: A single-token query $Q^{exist} \in \mathbb{R}^{1 \times D}$ interacts with the instruction "Is there a relation between $o_i$ and $o_j$?" through a similar pipeline, and the output is processed via a 2-layer MLP + sigmoid to obtain a score in [0,1]. A threshold of $\theta=0.35$ is used for filtering.
    - **Design Motivation**: The majority of the $N(N-1)$ pairs have no relations; sending all of them to the LMM for decoding would be extremely slow. The relation existence filtering achieves a **20× speedup**.

4. **Generation + Judgement Dual-Instruction Design**

    - **Function**: Performs open-set relation prediction using two complementary instruction types.
    - **Mechanism**:
        - Generation Instruction: "What are the relations between $c_i$ and $c_j$?" → Autoregressively generates all possible relations, separating multiple relations using [SEP]: $r_{i,j} = \text{Dec}(\text{Concat}(F_I^{pair(i,j)}, F_{inst}^{gen}))$.
        - Judgement Instruction: "Please judge between $c_i$ and $c_j$ whether there is a relation $r_k$" → Predicts Yes/No for each candidate relation. By using KV-cache to store prefix representations: $F_{prefix}^{(i,j)} = \text{Dec}(\text{Concat}(F_I^{pair(i,j)}, F_{inst}^{judge}))$, the process only needs to compute the relation name tokens for each target relation.
    - **Design Motivation**: Generation excels at discovering new relations but tends to favor high-frequency relations; Judgement leverages the LMM's discriminative ability to handle low-frequency and rare relations, while maintaining a comparable inference speed to Generation via prefix caching.

### Loss & Training
- **Loss Function**: $\mathcal{L} = \lambda \mathcal{L}_{exist} + \mathcal{L}_{LM}$, where $\mathcal{L}_{exist}$ is the binary cross-entropy (for relation existence prediction), $\mathcal{L}_{LM}$ is the standard language modeling cross-entropy, and $\lambda=10$.
- **Training Settings**: Freeze the Object Segmenter and Multimodal Relation Decoder, and only train the RelQ-Former; AdamW optimizer, lr=1e-4, weight decay=5e-2; trained for 12 epochs, with lr dropping to 1e-5 at the 8th epoch; utilizing 4×A100 GPUs.
- **Open-set Split**: Base relations : Novel relations = 7:3, with training restricted to base relation data.

## Key Experimental Results

### Main Results (PSG Dataset)

| Method | Setting | PredCls R@100 | PredCls mR@100 | SGDet R@100 | SGDet mR@100 |
|------|------|--------------|----------------|-------------|--------------|
| HiLo | Closed-set | - | - | 43.0 | 33.1 |
| PairNet | Closed-set | - | - | 39.6 | 30.6 |
| PSGTR | Closed-set | - | - | 36.3 | 22.1 |
| **OpenPSG** | **Closed-set** | **79.3** | **63.8** | **52.0** | **50.1** |
| OpenPSG | Open-set | 61.5 | 46.0 | 36.7 | 25.4 |

### PredCls on VG Dataset

| Method | Setting | R@100 | mR@100 |
|------|------|-------|--------|
| VCTree | Closed-set | 68.1 | 19.4 |
| Cacao+Epic | Closed-set | - | 40.8 |
| **OpenPSG** | **Closed-set** | **71.4** | **50.3** |
| OvSGTR | Open-set | 26.7 | - |
| **OpenPSG** | **Open-set** | **30.6** | **27.2** |

### Ablation Study

| Configuration | PredCls R@100 | PredCls mR@100 | Explanation |
|------|--------------|----------------|------|
| Mask Pooling Feature Extraction | 74.1 | 59.1 | Simple pooling |
| **RelQ-Former Attention Extraction** | **79.3** | **63.8** | +5.2 / +4.7 |
| No relation existence filtering | 79.3 | 63.8 | All-pair inference (20× slower) |
| With relation existence filtering | 78.8 | 63.0 | Mild performance drop but 20× speedup |
| Generation Instruction (Open-set) | 59.8 | 41.6 | Biased towards high-frequency relations |
| **Judgement Instruction (Open-set)** | **61.5** | **46.0** | Significantly better mR |

### Key Findings
- OpenPSG trained under the open-set setting even outperforms all previous closed-set methods on PredCls.
- The Judgement instruction significantly outperforms the Generation instruction on mR@K (+4.4 on mR@100), suggesting that the LMM's discriminative judgment is better suited for low-frequency relations than its generative ability.
- The attention mechanism of the RelQ-Former achieves a 5%+ improvement compared to mask pooling, demonstrating the importance of focusing on interaction regions.
- Relation existence filtering has negligible impact on performance (-0.5 R@100) while yielding a 20× speedup.

## Highlights & Insights
- **First Definitions of the OpenPSG Task**: Extends the open-set concept from object detection/segmentation to relation prediction, filling an important gap in scene graph generation. Leveraging the free-form text generation capability of LMMs naturally addresses the open-vocabulary challenge of open-set relations.
- **Efficient RelQ-Former Design**: Two sets of queries are responsible for feature extraction and relation filtering, respectively. The former replaces simple mask pooling by focusing on interaction regions, while the latter filters out a vast number of unrelated pairs at an $O(1)$ cost, striking an overall balance between quality and efficiency.
- **Generation + Judgement Dual-Instruction Design**: The former is used for discovering relations, and the latter for making precise judgments. This dual-instruction strategy provides a valuable paradigm for applying LMMs to structured visual reasoning.

## Limitations & Future Work
- The current Object Segmenter is completely frozen, causing segmentation errors to propagate directly to relation prediction. End-to-end joint training might yield greater improvements.
- The threshold $\theta=0.35$ for relation existence filtering is manually set, and an adaptive threshold could be more effective.
- Under the open-set setting, the SGDet performance (36.7 R@100) still has a large gap compared to closed-set performance, indicating that the propagation of novel relations throughout the full pipeline remains limited.

## Related Work & Insights
- **vs HiLo**: HiLo designs high- and low-frequency branches to handle imbalanced relations but is limited to closed-sets; OpenPSG significantly outperforms it in mR through Judgement instructions.
- **vs OvSGTR**: OvSGTR utilizes CLIP to match visual-text relation features for open-set scenarios, while OpenPSG employs LMM autoregressive generation/judgment, boosting R@100 by 3.9% on the VG dataset.
- **vs Cacao+Epic**: Transfers relation knowledge via external knowledge graphs, which is limited by the graph's coverage; OpenPSG does not rely on external knowledge.

## Rating
- Novelty: ⭐⭐⭐⭐ Defines the open-set PSG task for the first time; the combination of LMMs and the Relation Query Transformer is creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Double datasets (PSG and VG), double settings (closed-set and open-set), and detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definitions and detailed method descriptions.
- Value: ⭐⭐⭐⭐ Open-set relation prediction is a crucial direction, and this work establishes a baseline for future research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](../../CVPR2026/segmentation/dsflash_panoptic_scene_graph_realtime.md)
- [\[CVPR 2025\] Learning 4D Panoptic Scene Graph Generation from Rich 2D Visual Scene](../../CVPR2025/segmentation/learning_4d_panoptic_scene_graph_generation_from_rich_2d_visual_scene.md)
- [\[ICCV 2025\] SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation](../../ICCV2025/segmentation/spade_spatial-aware_denoising_network_for_open-vocabulary_panoptic_scene_graph_g.md)
- [\[ECCV 2024\] VISA: Reasoning Video Object Segmentation via Large Language Models](visa_reasoning_video_object_segmentation_via_large_language_models.md)
- [\[ECCV 2024\] Diffusion Models for Open-Vocabulary Segmentation](diffusion_models_for_open-vocabulary_segmentation.md)

</div>

<!-- RELATED:END -->

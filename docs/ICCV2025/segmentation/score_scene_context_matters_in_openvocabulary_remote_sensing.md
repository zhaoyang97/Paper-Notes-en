---
title: >-
  [Paper Note] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation
description: >-
  [ICCV 2025][Segmentation][Open-vocabulary instance segmentation] This paper proposes SCORE, a framework that injects multi-granularity scene knowledge from a remote sensing-specific CLIP into an open-vocabulary instance…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Open-vocabulary instance segmentation"
  - "remote sensing"
  - "scene context"
  - "vision-language models"
  - "cross-dataset generalization"
date: 2026-05-08
content_hash: 4f03c4027c85a4d7
---

# SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation

**Conference**: ICCV 2025
**arXiv**: [2507.12857](https://arxiv.org/abs/2507.12857)
**Code**: [https://github.com/HuangShiqi128/SCORE](https://github.com/HuangShiqi128/SCORE)
**Area**: Image Segmentation
**Keywords**: Open-vocabulary instance segmentation, remote sensing, scene context, vision-language models, cross-dataset generalization

## TL;DR
This paper proposes SCORE, a framework that injects multi-granularity scene knowledge from a remote sensing-specific CLIP into an open-vocabulary instance segmentation pipeline via two modules — Region-Aware Integration (RAI) and Global Context Adaptation (GCA) — achieving an average mAP improvement of 5.53% over the previous state of the art in cross-dataset evaluation across multiple remote sensing benchmarks.

## Background & Motivation
Remote sensing instance segmentation is a fundamental task in Earth observation, with broad applications in disaster monitoring, urban development, and agricultural planning. Existing methods are predominantly trained and evaluated under closed-vocabulary settings, where models can only recognize categories seen during training and fail to generalize to novel categories or transfer across datasets. This severely limits their practical utility in the diverse and dynamic conditions of real-world Earth observation.

Although open-vocabulary (OV) segmentation has been extensively studied for natural images (e.g., FC-CLIP, ODISE), direct transfer to remote sensing is ineffective for three reasons: (1) remote sensing images exhibit highly diverse landscapes and large seasonal variations; (2) objects in bird's-eye views are small and ambiguous, with shape-similar categories (e.g., cars and ships are both elongated) being difficult to distinguish; and (3) text embeddings from general-purpose CLIP lack remote sensing domain knowledge and struggle with large intra-class appearance variation and resolution discrepancy.

A key observation motivates this work: in remote sensing, objects are strongly correlated with their surrounding environments. Ships appear near coastlines, cars appear in parking lots, and aircraft appear near airports — such **regional scene context** serves as a critical cue for recognizing aerial objects, yet is entirely unexploited by existing OV segmentation models.

## Core Problem
How can the correlation between objects and their surrounding environments (i.e., scene context) be leveraged in an open-vocabulary setting to improve cross-dataset remote sensing instance segmentation? Two sub-problems must be addressed: (1) on the visual side — how to incorporate regional context into category embeddings to enhance object discriminability; and (2) on the textual side — how to infuse general-purpose CLIP text embeddings with remote sensing domain knowledge to improve classifier adaptability.

## Method

### Overall Architecture
SCORE comprises three branches:
- **Instance branch** (orange): Uses a frozen ConvNeXt-Large CLIP backbone to extract features and generates 300 query-based category embeddings and mask proposals via Mask2Former.
- **Semantic branch** (yellow): A frozen CLIP text encoder that encodes category names inserted into remote sensing prompt templates (e.g., "satellite imagery of ...") to produce text embeddings as classifiers.
- **Context branch** (blue): A frozen RemoteCLIP ViT-L/14 that extracts multi-granularity scene context — the [CLS] token as global context and patch embeddings as spatially dense features.

The three branches interact through RAI and GCA: RAI injects regional context into category embeddings, while GCA injects global context into text embeddings. Final classification is performed by matching the enhanced category embeddings against the adapted text embeddings.

### Key Designs
1. **Region-Aware Integration (RAI)**: The core idea is to enhance object representations using surrounding environmental cues. The process involves three steps:

    - **Adaptive region formation**: Given a predicted mask proposal, a learnable dilation factor $\delta$ (initialized to 1) controls the max-pooling kernel size $k=3+\text{clamp}(\delta,0,10)$, adaptively expanding the mask to cover the surrounding region.
    - **Regional context extraction**: The expanded mask is used to perform weighted pooling over the final-layer patch embeddings of RemoteCLIP, obtaining semantic features from the object's surrounding region.
    - **Regional context fusion**: Through $l$ Transformer layers, the regional context (scaled by a temperature coefficient $\lambda$) is injected into the category embeddings, producing region-aware category embeddings $\hat{\mathbf{V}}$.

2. **Global Context Adaptation (GCA)**: Addresses the lack of remote sensing knowledge in general-purpose CLIP text embeddings. The [CLS] token from RemoteCLIP (i.e., global context) is used as the query in multi-head cross-attention with text embeddings $\mathbf{T}$: $\hat{\mathbf{T}} = \text{MHA}(W_Q \mathbf{F}_{\text{CLS}}, W_K \mathbf{T}, W_V \mathbf{T})$. This allows the text embeddings to retain their OV generalization capability while being enriched with remote sensing visual priors, bridging the semantic gap between the general and remote sensing domains.

3. **Open-vocabulary inference**: An ensemble strategy is adopted for in-vocabulary and out-of-vocabulary classification. In-vocabulary inference uses learned region-aware category embeddings with a domain-adapted classifier; out-of-vocabulary inference relies on general-purpose CLIP, which empirically outperforms remote sensing CLIP in generalization — attributed to the substantially larger pretraining data of general CLIP (400M pairs) compared to remote sensing CLIP.

### Loss & Training
- AdamW optimizer with learning rate $1.25 \times 10^{-5}$.
- Ablation models trained for 50 epochs, batch size 2, input resized to 512×512.
- Training performed on a single L40S GPU.
- Weights of all CLIP models (RemoteCLIP and general CLIP) are frozen; only the RAI and GCA modules together with the Mask2Former components are trained.

## Key Experimental Results

**Trained on iSAID, cross-dataset evaluation (mAP):**

| Dataset | Metric | Ours (SCORE) | Prev. SOTA (FC-CLIP) | Gain |
|--------|------|------|----------|------|
| NWPU | mAP | 67.59 | 60.67 | +6.92 |
| SOTA | mAP | 42.57 | 33.62 | +8.95 |
| FAST | mAP | 13.67 | 11.88 | +1.79 |
| SIOR | mAP | 30.90 | 26.79 | +4.11 |
| **Average** | mAP | **38.68** | **33.24** | **+5.44** |

**Trained on SIOR, cross-dataset evaluation (mAP):**

| Dataset | Metric | Ours (SCORE) | Prev. SOTA (ZoRI) | Gain |
|--------|------|------|----------|------|
| NWPU | mAP | 69.17 | 59.77 | +9.40 |
| SOTA | mAP | 23.68 | 20.26 | +3.42 |
| FAST | mAP | 10.33 | 9.58 | +0.75 |
| iSAID | mAP | 27.15 | 23.46 | +3.69 |
| **Average** | mAP | **32.59** | **28.27** | **+4.32** |

SCORE also achieves state-of-the-art performance on open-vocabulary remote sensing **semantic segmentation**, with an average mIoU of 29.76 versus GSNet's 28.63 (+1.13%), and an improvement of up to 9.62% on the FLAIR dataset.

### Ablation Study
- **RAI and GCA are complementary**: RAI alone yields an average improvement of ~3.66% (trained on iSAID); GCA alone yields ~3.42%; combining both achieves the best result of +5.43%, confirming that visual-side and text-side enhancements are mutually complementary.
- **Remote sensing CLIP selection**: RemoteCLIP > GeoRSCLIP > SkyCLIP > general CLIP, indicating that domain-specific pretraining is critical for scene context extraction.
- **Context type in RAI**: Regional context > [CLS] token > intermediate-layer patch embeddings. The [CLS] token has a global bias toward dominant components; intermediate-layer patches emphasize texture and introduce noise; regional context adaptively focuses on the object's immediate surroundings.
- **GCA injection mechanism**: Multi-head cross-attention (MHA) >> addition > concatenation. Direct addition and concatenation disrupt cross-modal alignment due to misalignment between visual and text embedding spaces.
- **OV classifier selection**: General CLIP > remote sensing CLIP, as remote sensing CLIP pretraining data (0.8M–5M pairs) is far smaller than general CLIP (400M pairs), resulting in insufficient generalization capacity.

## Highlights & Insights
- **Insightful observation**: The correlation between aerial objects and their environments (ships near water, cars in parking lots) is formalized as a learnable regional context, yielding a clear and compelling motivation.
- **Dual-side enhancement**: The framework simultaneously improves both the visual side (RAI for category embeddings) and the textual side (GCA for the classifier), with their complementarity verified experimentally.
- **Adaptive dilation mechanism**: The learnable dilation factor allows the regional scope to adjust automatically during training, eliminating the need for manually designed hyperparameters.
- **First OV remote sensing instance segmentation benchmark**: A systematic cross-dataset evaluation protocol is established (train on one dataset, test on four others), filling a gap in the field.
- **Generality**: The framework also applies to semantic segmentation tasks, demonstrating its transferability.

## Limitations & Future Work
- **Computational overhead**: The additional RemoteCLIP branch increases inference cost; no runtime comparison is reported.
- **Limited generalization of remote sensing CLIP**: Experiments show that remote sensing CLIP underperforms general CLIP in OV classification, fundamentally due to insufficient remote sensing image–text pair data. If remote sensing CLIP pretraining data scales up in the future, the performance ceiling of the entire framework may improve further.
- **Limited dataset scale**: The largest evaluation dataset contains only 37 categories, which remains far from the category diversity of a true open-world setting.
- **Mask quality dependency**: Performance relies on the quality of Mask2Former mask proposals and remains limited for extremely small objects (e.g., FAST dataset with 37 fine-grained categories), achieving only 13.67 mAP.
- **Potential directions**: Combining regional context with SAM, introducing multi-scale contextual granularities, or transferring this paradigm to remote sensing object detection.

## Related Work & Insights
- **vs. FC-CLIP**: FC-CLIP employs a frozen CNN CLIP backbone for general OV segmentation, but entirely ignores remote sensing domain knowledge and object–environment correlations. SCORE introduces remote sensing scene context via RemoteCLIP, improving average mAP by over 5%.
- **vs. ZoRI (AAAI 2025)**: A prior work from the same group addressing zero-shot remote sensing instance segmentation, but limited to the zero-shot setting (requiring category attribute annotations). SCORE targets the more general open-vocabulary setting without requiring additional attribute information, and achieves superior performance.
- **vs. GSNet/OVRS**: These works address OV remote sensing semantic segmentation — GSNet fuses features from general and remote sensing backbones, and OVRS incorporates object orientation. However, both are confined to semantic segmentation. SCORE is the first to extend OV segmentation to the instance level in remote sensing, while also surpassing these methods on semantic segmentation.

- **Transferability of cross-domain generalization**: The RAI paradigm of "using surrounding context to assist object recognition" is not limited to remote sensing; it may also be effective in medical imaging (relationships between organs and surrounding tissue) and autonomous driving (relationships between vehicles and roads/parking lots).
- **Remote sensing OV detection**: As SCORE addresses segmentation only, the regional context idea can be directly transferred to open-vocabulary object detection in remote sensing.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — First to define the OV remote sensing instance segmentation task and provide a systematic solution; the use of regional context is intuitive and effective, though the individual components (learnable dilation, cross-attention injection) are not technically novel in isolation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive cross-dataset evaluation across two training sets and four test sets; ablations cover all modules, VLM choices, context types, injection mechanisms, and OV classifiers; generalization to semantic segmentation is also validated. Runtime comparison is absent.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly conveyed via the ship/car example in Figure 1; the paper is well-structured with thorough ablation analysis. Minor notation inconsistencies exist (GCA is sometimes referred to as VCA in the main text).
- **Value**: ⭐⭐⭐⭐ — Establishes a new benchmark and strong baseline for the task, with pioneering significance for the remote sensing OV segmentation community; the regional context paradigm is transferable to other remote sensing downstream tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] CAVIS: Context-Aware Video Instance Segmentation](cavis_context-aware_video_instance_segmentation.md)
- [\[ICCV 2025\] SPADE: Spatial-Aware Denoising Network for Open-vocabulary Panoptic Scene Graph Generation](spade_spatial-aware_denoising_network_for_open-vocabulary_panoptic_scene_graph_g.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)
- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Personalized OVSS: Understanding Personal Concept in Open-Vocabulary Semantic Segmentation](understanding_personal_concept_in_open-vocabulary_semantic_segmentation.md)

</div>

<!-- RELATED:END -->

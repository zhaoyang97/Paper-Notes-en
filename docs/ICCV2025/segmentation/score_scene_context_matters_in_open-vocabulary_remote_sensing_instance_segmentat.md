---
title: >-
  [Paper Note] SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation
description: >-
  [ICCV 2025][Segmentation][Open-vocabulary segmentation] This paper proposes the SCORE framework, which leverages multi-granularity scene context (regional context + global context) to enhance open-vocabulary remote sensing instance segmentation. Two dedicated modules — Region-Aware Integration (RAI) and Global Context Adaptation (GCA) — are introduced to strengthen visual and textual representations, respectively.
tags:
  - ICCV 2025
  - Segmentation
  - Open-vocabulary segmentation
  - remote sensing
  - scene context
  - CLIP
  - instance segmentation
date: 2026-05-08
content_hash: 382149dd66f1dcf1
---

# SCORE: Scene Context Matters in Open-Vocabulary Remote Sensing Instance Segmentation

**Conference**: ICCV 2025
**arXiv**: [2507.12857](https://arxiv.org/abs/2507.12857)
**Code**: [https://github.com/HuangShiqi128/SCORE](https://github.com/HuangShiqi128/SCORE)
**Area**: Remote Sensing Image Instance Segmentation / Open-Vocabulary
**Keywords**: Open-vocabulary segmentation, remote sensing, scene context, CLIP, instance segmentation

## TL;DR

This paper proposes the SCORE framework, which leverages multi-granularity scene context (regional context + global context) to enhance open-vocabulary remote sensing instance segmentation. Two dedicated modules — Region-Aware Integration (RAI) and Global Context Adaptation (GCA) — are introduced to strengthen visual and textual representations, respectively.

## Background & Motivation

- **State of the Field**: Existing remote sensing instance segmentation methods are predominantly closed-set, lacking the ability to recognize novel categories or generalize across datasets.
- **Limitations of Prior Work**: Open-vocabulary (OV) segmentation models developed for natural images face unique challenges when applied to remote sensing: diverse terrain, seasonal variation, and an abundance of small or ambiguous objects. Furthermore, frozen text embeddings from generic CLIP lack domain adaptation for remote sensing, making it difficult to capture high intra-class variation and resolution differences.
- **Root Cause**: A key observation in remote sensing is that **objects are strongly correlated with their surrounding environment** — ships appear near water, vehicles near parking lots, and aircraft near airports. Generic models fail to exploit this contextual prior.
- **Paper Goals**: OV instance segmentation in remote sensing remains unexplored (prior work is limited to semantic segmentation). This paper establishes the task, benchmark, and a strong baseline.

## Method

### Overall Architecture

SCORE consists of three branches: (1) a **context branch** (blue) — employs a remote sensing CLIP (RemoteCLIP) to extract multi-granularity scene context; (2) a **semantic branch** (yellow) — uses a frozen CLIP text encoder to generate text embeddings as classifiers; (3) an **instance branch** (orange) — employs a frozen generic CLIP image encoder to extract features, generating class embeddings and mask proposals via Mask2Former. The three branches interact through the RAI and GCA modules.

### Key Designs

1. **Scene Context Extraction**: A frozen RemoteCLIP ViT-L/14 encodes the input image to obtain context at two granularities:

    - Global context $\mathbf{F}^{final}_{CLS} \in \mathbb{R}^{1 \times C}$: the [CLS] token encodes global image semantics.
    - Regional context $\mathbf{F}^{final}_{HW} \in \mathbb{R}^{\frac{H}{14} \times \frac{W}{14} \times C}$: patch embeddings provide spatially dense features.

2. **Region-Aware Integration (RAI)**: Enhances class embeddings using contextual information from the surroundings of each object.

    - **Adaptive Region Formation**: The predicted mask is expanded via a learnable dilation factor $\delta$, with dilation kernel size $k = 3 + \text{clamp}(\delta, 0, 10)$, implemented via max-pooling.
    - **Regional Context Extraction**: Weighted pooling of RemoteCLIP patch embeddings within the expanded mask yields $F_{region}$.
    - **Regional Context Integration**: Regional context is injected into class embeddings through $l$ Transformer layers: $\mathbf{V}_{i+1} = \text{TransLayer}_i(\mathbf{V}_i, \lambda \cdot \mathbf{F}_{region})$.

3. **Global Context Adaptation (GCA)**: Injects remote sensing domain-specific global visual context into text embeddings.

    - Global context $\mathbf{F}^{final}_{CLS}$ serves as Query; text embeddings $\mathbf{T}$ serve as Key/Value.
    - Multi-head cross-attention is applied: $\hat{\mathbf{T}} = \text{MHA}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
    - Learnable linear projection matrices $\mathbf{w}_Q, \mathbf{w}_K, \mathbf{w}_V$ are used to mitigate the alignment gap between RS CLIP and generic CLIP.

### Loss & Training

- Standard Mask2Former segmentation losses are adopted.
- Trained for 50 epochs with batch size 2 on a single L40S GPU.
- Learning rate $1.25 \times 10^{-5}$, AdamW optimizer.
- Input images resized to $512 \times 512$.
- 300 object queries.
- At inference, an ensemble strategy combining in-vocabulary and out-of-vocabulary classification is employed.

## Key Experimental Results

### Main Results

**Cross-dataset evaluation trained on iSAID (mAP %):**

| Method | NWPU | SOTA | FAST | SIOR | Avg. |
|:---:|:---:|:---:|:---:|:---:|:---:|
| ODISE | 36.40 | 13.91 | 4.65 | 13.68 | 17.16 |
| FC-CLIP | 60.67 | 33.62 | 11.88 | 26.79 | 33.24 |
| MAFT+ | 35.32 | 6.63 | 5.52 | 9.84 | 14.33 |
| ZoRI | 62.06 | 30.02 | 12.65 | 26.27 | 32.75 |
| **SCORE** | **67.59** | **42.57** | **13.67** | **30.90** | **38.68** |

**Cross-dataset evaluation trained on SIOR (mAP %):**

| Method | NWPU | SOTA | FAST | iSAID | Avg. |
|:---:|:---:|:---:|:---:|:---:|:---:|
| FC-CLIP | 60.69 | 19.84 | 8.67 | 22.24 | 27.86 |
| ZoRI | 59.77 | 20.26 | 9.58 | 23.46 | 28.27 |
| **SCORE** | **69.17** | **23.68** | **10.33** | **27.15** | **32.59** |

### Ablation Study

**Component ablation (trained on iSAID, average mAP):**

| RAI | GCA | NWPU | SOTA | FAST | SIOR | Avg. |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 58.59 | 36.44 | 11.56 | 26.43 | 33.25 |
| ✓ | ✗ | 66.32 | 39.55 | 12.85 | 28.91 | 36.91 |
| ✗ | ✓ | 67.21 | 38.14 | 12.37 | 28.96 | 36.67 |
| **✓** | **✓** | **67.59** | **42.57** | **13.67** | **30.90** | **38.68** |

**VLM selection ablation (trained on iSAID):**

| VLM | NWPU | SOTA | FAST | SIOR |
|:---:|:---:|:---:|:---:|:---:|
| CLIP (generic) | 64.03 | 38.74 | 11.42 | 28.11 |
| SkyCLIP | 65.04 | 33.57 | 12.43 | 28.96 |
| GeoRSCLIP | 64.72 | 39.33 | 12.56 | 28.13 |
| **RemoteCLIP** | **67.59** | **42.57** | **13.67** | **30.90** |

### Key Findings

- SCORE surpasses the best prior method by **5.53% average mAP** (trained on iSAID) and **4.32%** (trained on SIOR).
- RAI and GCA are complementary: each individually yields ~3.5% gain, while their combination achieves a 5.4% gain.
- **Remote sensing-specific CLIP outperforms generic CLIP for scene context extraction**: RemoteCLIP achieves the best results across all datasets.
- **Regional context > global context > patch embeddings**, confirming that adaptive local context is more effective than the global [CLS] token.
- **The choice of GCA injection method is critical**: simple addition or concatenation disrupts pre-trained alignment, whereas multi-head cross-attention performs best.
- RemoteCLIP still underperforms generic CLIP on out-of-vocabulary classification; thus, generic CLIP is used for out-of-vocabulary classification at inference.

## Highlights & Insights

- This work is the **first to formulate, benchmark, and evaluate open-vocabulary remote sensing instance segmentation**.
- The observation that objects correlate with their surrounding environment — ships near water, vehicles in parking lots — is intuitive yet effective, and is operationalized via learnable dilated masks.
- The dual-modality enhancement strategy (visual-side RAI + text-side GCA) consistently outperforms single-sided augmentation.
- The paradigm of injecting domain-specific prior knowledge via a specialized CLIP model is generalizable to other specialized domains.

## Limitations & Future Work

- Gains on the FAST dataset remain limited (+1.02%); segmentation across its 37 fine-grained categories remains challenging.
- Training is conducted on a single L40S GPU; large-scale training may yield further improvements.
- RAI relies on mask quality; inaccurate initial mask predictions may introduce noisy contextual signals.
- RemoteCLIP's out-of-vocabulary capability remains weaker than that of generic CLIP, and the hybrid inference strategy increases framework complexity.
- The impact of remote sensing-specific text prompt templates on performance has not been explored.

## Related Work & Insights

- FC-CLIP's frozen CNN CLIP backbone combined with Mask2Former provides an effective baseline.
- Domain-specific knowledge from remote sensing VLMs (RemoteCLIP, SkyCLIP, GeoRSCLIP) can serve as plug-and-play context enhancement modules.
- This work demonstrates the value of a **collaborative paradigm between domain-specific CLIP and generic CLIP**.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to establish the OV remote sensing instance segmentation task and benchmark; RAI/GCA designs are well-motivated, though not breakthrough contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multi-dataset training/evaluation with comprehensive ablations (components, VLM selection, context type, injection strategy, OV classification).
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation; the ship/vehicle discrimination example in Figure 1 is intuitive and compelling.
- **Value**: ⭐⭐⭐⭐ Establishes a foundation for OV segmentation in remote sensing, though performance is constrained by the generalization capacity of remote sensing CLIP models.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity](legrad_an_explainability_method_for_vision_transformers_via_feature_formation_se.md)
- [\[ICCV 2025\] FLOSS: Free Lunch in Open-vocabulary Semantic Segmentation](floss_free_lunch_in_openvocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] Stepping Out of Similar Semantic Space for Open-Vocabulary Segmentation](stepping_out_of_similar_semantic_space_for_open-vocabulary_segmentation.md)
- [\[ICCV 2025\] Training-Free Class Purification for Open-Vocabulary Semantic Segmentation](training-free_class_purification_for_open-vocabulary_semantic_segmentation.md)
- [\[ICCV 2025\] CorrCLIP: Reconstructing Patch Correlations in CLIP for Open-Vocabulary Semantic Segmentation](corrclip_reconstructing_patch_correlations_in_clip_for_openv.md)

<!-- RELATED:END -->

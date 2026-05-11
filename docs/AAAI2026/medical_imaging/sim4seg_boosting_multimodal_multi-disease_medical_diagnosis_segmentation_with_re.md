---
title: >-
  [Paper Note] Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks
description: >-
  [AAAI 2026][Medical Imaging][Medical Diagnosis Segmentation] This paper introduces the Medical Diagnosis Segmentation (MDS) task along with the M3DS dataset, and proposes the Sim4Seg framework…
tags:
  - "AAAI 2026"
  - "Medical Imaging"
  - "Medical Diagnosis Segmentation"
  - "Vision-Language Similarity"
  - "Chain-of-Thought Diagnosis"
  - "Multimodal Multi-disease"
  - "Test-Time Scaling"
date: 2026-05-08
content_hash: c2d689973a938fbe
---

# Sim4Seg: Boosting Multimodal Multi-disease Medical Diagnosis Segmentation with Region-Aware Vision-Language Similarity Masks

**Conference**: AAAI 2026
**arXiv**: [2511.06665](https://arxiv.org/abs/2511.06665)
**Code**: [https://github.com/SLR567/Sim4Seg](https://github.com/SLR567/Sim4Seg)
**Area**: Medical Imaging / Vision-Language Models
**Keywords**: Medical Diagnosis Segmentation, Vision-Language Similarity, Chain-of-Thought Diagnosis, Multimodal Multi-disease, Test-Time Scaling

## TL;DR

This paper introduces the Medical Diagnosis Segmentation (MDS) task along with the M3DS dataset, and proposes the Sim4Seg framework, which leverages **Region-aware Vision-Language Similarity Masks (RVLS2M)** derived from LVLM hidden states to prompt SAM for segmentation while simultaneously generating diagnostic chain-of-thought reasoning. Combined with a test-time scaling strategy, Sim4Seg comprehensively outperforms baselines on both segmentation and diagnosis.

## Background & Motivation

1. **Background**: Medical image segmentation models perform well on specific tasks but lack the ability to provide interpretable diagnostic outputs. Reasoning segmentation has recently been proposed to integrate textual reasoning with visual segmentation.
2. **Limitations of Prior Work**: (a) Existing medical LVLMs focus either on segmentation or textual diagnosis, lacking a unified model; (b) general-purpose reasoning segmentation methods (e.g., LISA) are not optimized for medical images and yield insufficient segmentation accuracy; (c) no unified medical dataset exists that contains both segmentation masks and diagnostic chain-of-thought annotations.
3. **Key Challenge**: Medical diagnosis requires simultaneously producing **pixel-level segmentation results** and **interpretable diagnostic reasoning**, yet these two tasks remain decoupled in existing frameworks.
4. **Goal**: To formally define the MDS task and construct a corresponding dataset, and to design a unified framework that jointly outputs segmentation masks and diagnostic results.
5. **Key Insight**: The similarity between image token embeddings and special token embeddings in the last hidden layer of an LVLM is exploited to generate region-aware mask prompts for precise SAM-based segmentation.
6. **Core Idea**: The hidden states of an LVLM naturally encode correspondences between textually described targets and image regions; leveraging such similarity to generate mask prompts provides richer spatial information than relying solely on special token embeddings.

## Method

### Overall Architecture

An LVLM receives a medical image and a query text, and generates diagnostic reasoning text containing special tokens. Image token embeddings $\mathbf{E}_{img}$ and special token embeddings $\mathbf{E}_{seg}$ are extracted from the last hidden layer. The RVLS2M module computes their similarity and produces a region mask $\mathbf{M}_{region}$, which is then fed alongside $\mathbf{E}_{seg}$ and visual features $\mathbf{F}$ into the SAM decoder to generate the final segmentation mask.

### Key Designs

1. **Region-Aware Vision-Language Similarity Mask Module (RVLS2M)**

    - **Function**: Mines spatial priors of target regions from LVLM hidden states and generates binary masks to guide SAM segmentation.
    - **Mechanism**: The dot-product similarity between image token embeddings and special token embeddings is computed as $\text{Sim} = \mathbf{E}_{img} \cdot (\mathbf{E}_{seg})^T$. After softmax normalization, the result is reshaped into a 2D similarity map, partitioned into a $g \times g$ grid, and mean-pooled within each cell to obtain a region similarity matrix $\mathcal{R}$. An adaptive threshold binarizes this matrix to produce $\mathbf{M}_{region}$, which is provided as an additional prompt to the SAM decoder.
    - **Design Motivation**: LISA prompts SAM solely with special token embeddings, discarding spatial positional information. After processing both image and text, LVLM hidden states already encode semantic correspondences; exploiting these correspondences yields more precise region priors.

2. **M3DS Dataset and CoT Generation Pipeline**

    - **Function**: Provides unified training data containing segmentation masks and diagnostic chain-of-thought annotations.
    - **Mechanism**: Ten sub-datasets are integrated, covering four imaging modalities (X-ray, dermoscopy, endoscopy, ultrasound, and fundus photography) and multiple diseases. HuatuoGPT-Vision is used as a medical assistant to generate CoT diagnostic reasoning following the pipeline of modality identification → image analysis → diagnostic conclusion. A critic assistant evaluates quality, and human-assisted review ensures reliability. The dataset comprises 16,148 samples.
    - **Design Motivation**: Existing datasets provide either segmentation annotations or VQA annotations, but not a unified resource combining segmentation with diagnostic reasoning.

3. **MDS Test-Time Scaling Strategy**

    - **Function**: Improves segmentation and diagnostic quality at inference time through multi-path reasoning.
    - **Mechanism**: $m$ diverse diagnostic reasoning paths are generated; each path produces a region mask via RVLS2M. Combined with $n$ random perturbations, $m \times n$ candidate segmentation masks are obtained in total. The candidate with the highest evaluation metric $\mathcal{Q}$ (mean of gIoU and cIoU) is selected as the final output.
    - **Design Motivation**: Multiple sampling from an LLM yields reasoning paths of varying quality; best-of-N selection is a simple yet effective strategy.

### Loss & Training

$$\mathcal{L} = \lambda_{txt}\mathcal{L}_{txt} + \lambda_{mask}(\lambda_{bce}\mathcal{L}_{bce} + \lambda_{dice}\mathcal{L}_{dice})$$

The model is fine-tuned on the LISA framework using CoT data.

## Key Experimental Results

### Main Results

Results on the M3DS test set:

| Method | gIoU | cIoU | Diagnostic Accuracy |
|--------|------|------|---------------------|
| LISA (zero-shot) | 32.43 | 31.83 | 4.71% |
| LISA (ft-CoT) | 45.90 | 45.92 | 58.05% |
| Sim4Seg (ft-CoT) | 51.86 | 53.90 | 69.04% |
| Sim4Seg + Test-Time Scaling | **53.11** | **55.83** | **82.63%** |

### Ablation Study

| Configuration | gIoU | cIoU | Notes |
|---------------|------|------|-------|
| LISA baseline | 45.90 | 45.92 | Without RVLS2M |
| + RVLS2M | 51.86 | 53.90 | Significant segmentation gain |
| + Test-time scaling | 53.11 | 55.83 | Further improvement |
| Without CoT training | 51.00 | 54.06 | Low diagnostic accuracy |

### Key Findings

- RVLS2M yields the most significant segmentation gains (gIoU +5.96, cIoU +7.98), validating the effectiveness of vision-language similarity masks.
- CoT training substantially boosts diagnostic accuracy (54.33%→69.04%) but has limited impact on segmentation (+0.86 gIoU).
- The test-time scaling strategy provides an additional gain of 1.25 gIoU and 13.59% in diagnostic accuracy (69.04%→82.63%).
- The model demonstrates robust generalization across datasets and imaging modalities.

## Highlights & Insights

- **Exploiting spatial information in LVLM hidden states**: After processing both image and text, LVLM hidden layers encode semantic-spatial correspondences. Leveraging this freely available information to generate region masks is an elegant observation.
- **MDS task formulation**: The formal definition of a unified task combining segmentation and diagnosis helps advance interpretability research in medical AI.
- **M3DS dataset**: A unified dataset covering five imaging modalities and multiple diseases fills an important gap in the field.

## Limitations & Future Work

- Test-time scaling requires multiple inference passes, increasing computational cost.
- CoT quality depends on the generative capability of HuatuoGPT-Vision, which may introduce medical knowledge errors.
- The choice of grid size $g$ affects results; an adaptive grid may be more appropriate.
- Only SAM is used as the segmentation backbone; more advanced segmentation models could potentially yield further improvements.

## Related Work & Insights

- **vs. LISA**: LISA prompts SAM solely with special token embeddings; Sim4Seg additionally leverages vision-language similarity masks to provide region priors.
- **vs. READ**: READ employs point prompts; Sim4Seg uses region mask prompts, providing denser spatial guidance.
- **vs. SAM-Med2D**: Segmentation-dedicated models lack diagnostic capability; Sim4Seg unifies both.

## Rating

- Novelty: ⭐⭐⭐⭐ — MDS task formulation + RVLS2M module + M3DS dataset
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multimodal multi-disease evaluation, though clinical expert evaluation is absent
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with detailed algorithmic descriptions
- Value: ⭐⭐⭐⭐⭐ — Significant contribution to interpretability in medical AI

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MAPI-GNN: Multi-Activation Plane Interaction Graph Neural Network for Multimodal Medical Diagnosis](mapi-gnn_multi-activation_plane_interaction_graph_neural_network_for_multimodal_.md)
- [\[ICLR 2026\] Boosting Medical Visual Understanding From Multi-Granular Language Learning](../../ICLR2026/medical_imaging/boosting_medical_visual_understanding_from_multi-granular_language_learning.md)
- [\[AAAI 2026\] PulseMind: A Multi-Modal Medical Model for Real-World Clinical Diagnosis](pulsemind_a_multi-modal_medical_model_for_real-world_clinical_diagnosis.md)
- [\[ICCV 2025\] Boosting Vision Semantic Density with Anatomy Normality Modeling for Medical Vision-language Pre-training](../../ICCV2025/medical_imaging/boosting_vision_semantic_density_with_anatomy_normality_modeling_for_medical_vis.md)
- [\[AAAI 2026\] Bridging Vision and Language for Robust Context-Aware Surgical Point Tracking: The VL-SurgPT Dataset and Benchmark](bridging_vision_and_language_for_robust_context-aware_surgical_point_tracking_th.md)

</div>

<!-- RELATED:END -->
